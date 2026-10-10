"""Behaviour cloning: train the network to copy the heuristic bot (supervised backpropagation).

    python -m train.imitate --games 20000 --epochs 5 --out models/bc.pt
"""
import argparse
import multiprocessing as mp
import os
import shutil
import time
from pathlib import Path

import numpy as np

from ttr.ai import features, graph
from ttr.ai.arena import play_game
from ttr.ai.heuristic import HeuristicPlayer
from ttr.ai.macro import from_action, macros
from ttr.players import RandomPlayer


class Recorder:
    """A heuristic bot that records (state, action matrix, chosen index) for each real decision."""

    uses_full_state = True

    def __init__(self, seed):
        self.bot = HeuristicPlayer(seed)
        self.rows = []

    def choose(self, game, legal):
        action = self.bot.choose(game, legal)
        ms = macros(game, legal)
        if len(ms) > 1:  # forced moves teach nothing
            me = game.actor
            plan = graph.plan(game, me)
            self.rows.append((
                features.encode_state(game, me, plan).astype(np.float16),
                features.encode_actions(game, me, ms, plan).astype(np.float16),
                ms.index(from_action(action)),
            ))
        return action


def make_shard(args):
    idx, games, out_dir = args
    states, acts, counts, labels, margins = [], [], [], [], []
    for i in range(games):
        gid = idx * games + i
        if gid % 5 == 4:  # 20% heuristic vs random, heuristic seat alternates
            seat = (gid // 5) % 2
            rec = Recorder(gid)
            players = [rec, RandomPlayer(gid)] if seat == 0 else [RandomPlayer(gid), rec]
            recorders = {seat: rec}
        else:
            recorders = {0: Recorder(gid * 2), 1: Recorder(gid * 2 + 1)}
            players = [recorders[0], recorders[1]]
        g = play_game(players, seed=gid)
        totals = [r["total"] for r in g.state.final_scores]
        for seat, rec in recorders.items():
            margin = (totals[seat] - totals[1 - seat]) / 100
            for st, ac, lab in rec.rows:
                states.append(st)
                acts.append(ac)
                counts.append(len(ac))
                labels.append(lab)
                margins.append(margin)
    path = Path(out_dir) / f"shard_{idx:04d}.npz"
    np.savez_compressed(
        path, states=np.stack(states), acts=np.concatenate(acts), counts=np.array(counts, np.int32),
        labels=np.array(labels, np.int32), margins=np.array(margins, np.float32),
    )
    return str(path), len(labels)


def generate(games, shard_games, out_dir, workers):
    out_dir.mkdir(parents=True, exist_ok=True)
    jobs = [(i, shard_games, out_dir) for i in range(games // shard_games)]
    t = time.time()
    total = 0
    with mp.Pool(workers) as pool:
        for k, (path, n) in enumerate(pool.imap_unordered(make_shard, jobs), 1):
            total += n
            print(f"  shard {k}/{len(jobs)}: {n} decisions ({time.time() - t:.0f}s)", flush=True)
    print(f"generated {total} decisions from {len(jobs) * shard_games} games")


class Shard:
    def __init__(self, path):
        z = np.load(path)
        self.states, self.acts, self.labels, self.margins = z["states"], z["acts"], z["labels"], z["margins"]
        self.counts = z["counts"]
        self.offsets = np.cumsum(self.counts) - self.counts

    def __len__(self):
        return len(self.labels)

    def batch(self, idx):
        import torch

        counts = self.counts[idx]
        acts = np.concatenate([self.acts[self.offsets[i]:self.offsets[i] + self.counts[i]] for i in idx])
        return (
            torch.from_numpy(self.states[idx].astype(np.float32)),
            torch.from_numpy(acts.astype(np.float32)),
            torch.from_numpy(np.repeat(np.arange(len(idx)), counts)),
            torch.from_numpy(self.labels[idx].astype(np.int64)),
            torch.from_numpy(self.margins[idx]),
        )


def run_batch(model, shard, idx):
    import torch
    import torch.nn.functional as F

    from ttr.ai.model import pad_logits

    states, acts, owner, labels, margins = shard.batch(idx)
    logits, values = model(states, acts, owner)
    padded, _ = pad_logits(logits, owner, len(idx))
    ce = F.cross_entropy(padded, labels)
    mse = F.mse_loss(values, margins)
    acc = (padded.argmax(1) == labels).float().mean()
    return ce, mse, acc


def train(shard_paths, epochs, out, batch=256, lr=3e-4, seed=0):
    import torch

    from ttr.ai.model import PolicyValueNet, save_checkpoint

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    val_n = max(1, round(len(shard_paths) * 0.05))
    val_paths, train_paths = shard_paths[-val_n:], shard_paths[:-val_n]
    val = [Shard(p) for p in val_paths]
    model = PolicyValueNet(features.STATE_DIM, features.ACT_DIM)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(1, epochs + 1):
        model.train()
        tot = n = 0
        t = time.time()
        for p in rng.permutation(train_paths):
            shard = Shard(p)
            order = rng.permutation(len(shard))
            for s in range(0, len(order), batch):
                ce, mse, _ = run_batch(model, shard, order[s:s + batch])
                loss = ce + 0.5 * mse
                opt.zero_grad()
                loss.backward()
                opt.step()
                tot += loss.item() * len(order[s:s + batch])
                n += len(order[s:s + batch])
        model.eval()
        vce = vmse = vacc = vn = 0
        with torch.no_grad():
            for shard in val:
                for s in range(0, len(shard), 1024):
                    idx = np.arange(s, min(s + 1024, len(shard)))
                    ce, mse, acc = run_batch(model, shard, idx)
                    vce += float(ce) * len(idx); vmse += float(mse) * len(idx); vacc += float(acc) * len(idx)
                    vn += len(idx)
        print(f"epoch {epoch}/{epochs}: train loss {tot / n:.4f} | val ce {vce / vn:.4f} mse {vmse / vn:.4f} "
              f"top-1 acc {vacc / vn:.4f} ({time.time() - t:.0f}s)", flush=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    save_checkpoint(model, out)
    print(f"saved {out}")
    return vacc / vn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=20000)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--out", default="models/bc.pt")
    ap.add_argument("--data", default="runs/bc_data")
    ap.add_argument("--shard-games", type=int, default=250)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--skip-gen", action="store_true", help="train on the shards already in --data")
    args = ap.parse_args()
    data = Path(args.data)
    if not args.skip_gen:
        generate(args.games, args.shard_games, data, args.workers)
    paths = sorted(str(p) for p in data.glob("shard_*.npz"))
    train(paths, args.epochs, Path(args.out))
    best = Path("models/best.pt")
    if not best.exists():
        best.parent.mkdir(exist_ok=True)
        shutil.copy(args.out, best)
        print(f"copied to {best}")


if __name__ == "__main__":
    main()
