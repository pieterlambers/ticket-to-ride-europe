"""PPO self-play: improve the network by playing games against itself, past versions and the heuristic.

    python -m train.ppo --init models/bc.pt --iters 300 --games-per-iter 128 --workers 7
"""
import argparse
import csv
import multiprocessing as mp
import os
import random
import shutil
import time
from pathlib import Path

import numpy as np

from ttr.ai import features, graph
from ttr.ai.arena import make_bot, play_game
from ttr.ai.macro import macros, to_action

_models: dict = {}  # per-worker cache: checkpoint path -> (mtime, model)


def _worker_init():
    import torch

    torch.set_num_threads(1)


def _model(path):
    from ttr.ai.model import load_checkpoint

    mtime = os.path.getmtime(path)
    if path not in _models or _models[path][0] != mtime:
        _models[path] = (mtime, load_checkpoint(path))
    return _models[path][1]


class Learner:
    """Samples from the policy (temperature 1) and records everything PPO needs."""

    uses_full_state = True

    def __init__(self, model, seed, potential_coef=0.0):
        import torch

        self.torch, self.model, self.potential_coef = torch, model, potential_coef
        self.rng = np.random.default_rng(seed)
        self.steps = []  # (state, acts, idx, logp, value, phi)

    def choose(self, game, legal):
        torch = self.torch
        me = game.actor
        plan = graph.plan(game, me)
        ms = macros(game, legal)
        if len(ms) == 1:
            return to_action(game, ms[0], legal, plan["need"])
        state = features.encode_state(game, me, plan)
        acts = features.encode_actions(game, me, ms, plan)
        with torch.no_grad():
            logits, value = self.model(torch.from_numpy(state)[None], torch.from_numpy(acts),
                                       torch.zeros(len(ms), dtype=torch.long))
            logp = torch.log_softmax(logits, 0).numpy().astype(np.float64)
        p = np.exp(logp)
        i = int(self.rng.choice(len(ms), p=p / p.sum()))
        m = ms[i]
        # potential for shaping: fewer trains still to build and more ticket points done is better
        phi = 0.0
        if self.potential_coef:
            phi = -plan["cost"] / 45 + sum(game.board.tickets[t].points for t in plan["done"]) / 100
        self.steps.append((state, acts, i, float(logp[i]), float(value[0]), phi))
        return to_action(game, m, legal, plan["need"])


def terminal_reward(my_total, opp_total):
    """+1 for a win, -1 for a loss, plus a small margin term as a tiebreaker."""
    margin = my_total - opp_total
    return float(np.sign(margin)) + 0.3 * margin / 50


def rollout(job):
    learner_path, opp, seed, potential_coef = job
    learner = Learner(_model(learner_path), seed, potential_coef)
    if opp.startswith("bot:"):  # a rule-based opponent, e.g. "bot:heuristic"
        other = make_bot(opp[4:], seed)
    else:
        from ttr.ai.neural import NeuralPlayer

        other = NeuralPlayer(opp, temperature=1.0, seed=seed, model=_model(opp))
    seat = seed % 2
    g = play_game([learner, other] if seat == 0 else [other, learner], seed=seed)
    totals = [r["total"] for r in g.state.final_scores]
    margin = (totals[seat] - totals[1 - seat]) / 100
    st = learner.steps
    phis = np.array([s[5] for s in st] + [0.0], np.float32)  # potential is 0 after the last decision
    rewards = learner.potential_coef * (phis[1:] - phis[:-1])
    rewards[-1] += terminal_reward(totals[seat], totals[1 - seat])
    return {
        "states": np.stack([s[0] for s in st]), "acts": np.concatenate([s[1] for s in st]),
        "counts": np.array([len(s[1]) for s in st], np.int32), "idx": np.array([s[2] for s in st], np.int64),
        "logp": np.array([s[3] for s in st], np.float32), "value": np.array([s[4] for s in st], np.float32),
        "reward": rewards, "margin": margin, "win": 1.0 if margin > 0 else 0.5 if margin == 0 else 0.0,
    }


def gae(reward, value, gamma=1.0, lam=0.95):
    adv = np.zeros_like(reward)
    last = 0.0
    for t in reversed(range(len(reward))):
        nxt = value[t + 1] if t + 1 < len(reward) else 0.0
        delta = reward[t] + gamma * nxt - value[t]
        last = delta + gamma * lam * last
        adv[t] = last
    return adv, adv + value


def build_batch(trajs):
    advs, rets = zip(*(gae(t["reward"], t["value"]) for t in trajs))
    counts = np.concatenate([t["counts"] for t in trajs])
    return {
        "states": np.concatenate([t["states"] for t in trajs]), "acts": np.concatenate([t["acts"] for t in trajs]),
        "counts": counts, "offsets": np.cumsum(counts) - counts,
        "idx": np.concatenate([t["idx"] for t in trajs]), "logp": np.concatenate([t["logp"] for t in trajs]),
        "adv": np.concatenate(advs), "ret": np.concatenate(rets),
    }


def ppo_update(model, opt, b, epochs=4, minibatch=512, clip=0.2, vcoef=0.5, ecoef=0.01, target_kl=0.02):
    import torch

    from ttr.ai.model import pad_logits

    adv = (b["adv"] - b["adv"].mean()) / (b["adv"].std() + 1e-8)
    n = len(b["idx"])
    stats = np.zeros(4)
    batches = 0
    epochs_done = 0
    model.train()
    for _ in range(epochs):
        epoch_kl, epoch_batches = 0.0, 0
        order = np.random.permutation(n)
        for s in range(0, n, minibatch):
            ids = order[s:s + minibatch]
            acts = np.concatenate([b["acts"][b["offsets"][i]:b["offsets"][i] + b["counts"][i]] for i in ids])
            owner = torch.from_numpy(np.repeat(np.arange(len(ids)), b["counts"][ids]))
            logits, values = model(torch.from_numpy(b["states"][ids]), torch.from_numpy(acts), owner)
            padded, _ = pad_logits(logits, owner, len(ids))
            logp_all = torch.log_softmax(padded, 1)
            logp = logp_all[torch.arange(len(ids)), torch.from_numpy(b["idx"][ids])]
            ratio = torch.exp(logp - torch.from_numpy(b["logp"][ids]))
            a = torch.from_numpy(adv[ids])
            policy_loss = -torch.min(ratio * a, torch.clamp(ratio, 1 - clip, 1 + clip) * a).mean()
            value_loss = ((values - torch.from_numpy(b["ret"][ids])) ** 2).mean()
            entropy = -(logp_all.exp() * logp_all.masked_fill(torch.isinf(logp_all), 0)).sum(1).mean()
            loss = policy_loss + vcoef * value_loss - ecoef * entropy
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            opt.step()
            kl = (torch.from_numpy(b["logp"][ids]) - logp).mean().item()
            stats += (policy_loss.item(), value_loss.item(), entropy.item(), kl)
            epoch_kl += kl
            epoch_batches += 1
            batches += 1
        epochs_done += 1
        if epoch_kl / epoch_batches > 1.5 * target_kl:  # the policy moved far enough: stop early
            break
    model.eval()
    return *(stats / batches), epochs_done


def main():
    import torch

    from ttr.ai.model import load_checkpoint, save_checkpoint
    from train.evaluate import evaluate

    ap = argparse.ArgumentParser()
    ap.add_argument("--init", default="models/bc.pt")
    ap.add_argument("--resume")
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--games-per-iter", type=int, default=512)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--out", default="runs/ppo")
    ap.add_argument("--eval-every", type=int, default=10)
    ap.add_argument("--eval-games", type=int, default=600)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--entropy", type=float, default=0.003)
    ap.add_argument("--entropy-decay-iters", type=int, default=150)
    ap.add_argument("--target-kl", type=float, default=0.02)
    ap.add_argument("--potential-coef", type=float, default=0.0)
    ap.add_argument("--opp-heuristic", default="heuristic")
    ap.add_argument("--separate-value", action="store_true", help="give the value head its own encoder")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    model = load_checkpoint(args.resume or args.init)
    if args.separate_value:
        model.add_separate_value()
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    start = 0
    if args.resume:
        ck = torch.load(args.resume, map_location="cpu", weights_only=True)
        start = ck.get("iter", 0)
        if "opt" in ck:
            opt.load_state_dict(ck["opt"])
    current = str(out / "current.pt")
    save_checkpoint(model, current)
    pool_paths = sorted(str(p) for p in out.glob("ckpt_*.pt"))
    if not pool_paths:
        first = str(out / f"ckpt_{start:04d}.pt")
        save_checkpoint(model, first, iter=start, opt=opt.state_dict())
        pool_paths = [first]
    best = Path("models/best.pt")
    if not best.exists():
        best.parent.mkdir(exist_ok=True)
        shutil.copy(pool_paths[-1], best)

    log_path = out / "log.csv"
    new_log = not log_path.exists()
    log = open(log_path, "a", newline="")
    writer = csv.writer(log)
    if new_log:
        writer.writerow(["iter", "policy_loss", "value_loss", "entropy", "kl", "epochs_done", "train_margin", "train_win",
                         "win_vs_heuristic", "ci_low_vs_heuristic", "win_vs_best", "ci_low_vs_best", "seconds"])
    rng = random.Random(args.seed + start)
    ctx = mp.get_context("spawn")
    with ctx.Pool(args.workers, initializer=_worker_init) as pool:
        for it in range(start + 1, start + args.iters + 1):
            t0 = time.time()
            jobs = []
            for k in range(args.games_per_iter):
                r = rng.random()  # 20% latest learner, 40% rule-based bot, 30% recent checkpoint, 10% best
                opp = (current if r < 0.2 else f"bot:{args.opp_heuristic}" if r < 0.6
                       else rng.choice(pool_paths[-10:]) if r < 0.9 else str(best))
                jobs.append((current, opp, it * 100000 + k, args.potential_coef))
            trajs = pool.map(rollout, jobs, chunksize=max(1, len(jobs) // (args.workers * 4)))
            ecoef = args.entropy * max(0.0, 1 - it / args.entropy_decay_iters)
            pl, vl, ent, kl, epochs_done = ppo_update(model, opt, build_batch(trajs), ecoef=ecoef, target_kl=args.target_kl)
            save_checkpoint(model, current)
            margin = float(np.mean([t["margin"] for t in trajs]))
            win = float(np.mean([t["win"] for t in trajs]))
            wh = wb = clh = clb = ""
            if it % args.eval_every == 0:
                ck = str(out / f"ckpt_{it:04d}.pt")
                save_checkpoint(model, ck, iter=it, opt=opt.state_dict())
                pool_paths.append(ck)
                rh = evaluate("neural", args.opp_heuristic, args.eval_games, ck, None, args.workers)
                wh, clh = rh["win_rate"], rh["ci_low"]
                rb = evaluate("neural", "neural", args.eval_games, ck, str(best), args.workers)
                wb, clb = rb["win_rate"], rb["ci_low"]
                if rb["ci_low"] > 0.5:  # promote only when clearly better than the saved best
                    archive = Path("models/archive")
                    archive.mkdir(parents=True, exist_ok=True)
                    shutil.copy(best, archive / f"best_{int(time.time())}.pt")
                    shutil.copy(ck, best)
                    print(f"  new best (win rate {wb:.3f}, ci_low {rb['ci_low']:.3f} vs previous best)")
            f3 = lambda x: x if x == "" else f"{x:.3f}"  # noqa: E731
            writer.writerow([it, f"{pl:.4f}", f"{vl:.4f}", f"{ent:.4f}", f"{kl:.4f}", epochs_done, f"{margin:.3f}",
                             f"{win:.3f}", f3(wh), f3(clh), f3(wb), f3(clb), f"{time.time() - t0:.0f}"])
            log.flush()
            print(f"iter {it}: pg {pl:.4f} v {vl:.4f} ent {ent:.3f} kl {kl:.4f} ep {epochs_done} margin {margin:+.2f} win {win:.2f}"
                  + (f" | vs heuristic {wh:.3f} vs best {wb:.3f}" if wh != "" else "") + f" ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
