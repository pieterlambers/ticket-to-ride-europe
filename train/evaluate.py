"""Play two kinds of bot against each other and print a win-rate table.

Every deal is played twice with the seats swapped (paired seeds), so deal luck cancels out.

    python -m train.evaluate --a heuristic --b random --games 1000
"""
import argparse
import math
import multiprocessing as mp
import os

from ttr.ai.arena import make_bot, play_game

COMPONENTS = ("routes", "tickets", "stations", "express")


def run_one(args):
    kind_a, kind_b, ckpt_a, ckpt_b, i, offset = args
    a = make_bot(kind_a, i * 2, ckpt_a)
    b = make_bot(kind_b, i * 2 + 1, ckpt_b)
    seat_a = i % 2  # the two games of a pair share the seed and swap the seats
    players = [a, b] if seat_a == 0 else [b, a]
    g = play_game(players, seed=offset + i // 2)
    rows = g.state.final_scores
    ra, rb = rows[seat_a], rows[1 - seat_a]
    held = [len(p.tickets) for p in g.state.players]
    win = 0.5 if ra["total"] == rb["total"] and ra["completed"] == rb["completed"] and ra["stations"] == rb["stations"] \
        else float(seat_a in g.winners())
    out = {
        "win": win, "a": ra["total"], "b": rb["total"],
        "a_done": ra["completed"], "b_done": rb["completed"],
        "a_failed": held[seat_a] - ra["completed"], "b_failed": held[1 - seat_a] - rb["completed"],
    }
    for c in COMPONENTS:
        out[f"a_{c}"], out[f"b_{c}"] = ra[c], rb[c]
    return out


def win_ci(wins):
    """Win rate with a 95% interval. `wins` lists game results in pair order (games 2k and 2k+1 form a pair)."""
    pairs = [(wins[k] + wins[k + 1]) / 2 for k in range(0, len(wins) - 1, 2)]
    n = len(pairs)
    mean = sum(pairs) / n
    if n < 2:
        return mean, 0.0, 1.0
    var = sum((p - mean) ** 2 for p in pairs) / (n - 1)
    half = 1.96 * math.sqrt(var / n)
    return mean, mean - half, mean + half


def run_pool(fn, jobs, workers):
    workers = workers or max(1, (os.cpu_count() or 2) - 1)
    if workers == 1:
        return [fn(j) for j in jobs]
    with mp.Pool(workers) as pool:
        return pool.map(fn, jobs, chunksize=max(1, len(jobs) // (workers * 4)))


def evaluate(kind_a, kind_b, games, ckpt_a=None, ckpt_b=None, workers=None, seed_offset=0):
    games += games % 2  # whole pairs only
    jobs = [(kind_a, kind_b, ckpt_a, ckpt_b, i, seed_offset) for i in range(games)]
    results = run_pool(run_one, jobs, workers)
    n = len(results)
    mean = lambda k: sum(r[k] for r in results) / n  # noqa: E731
    win_rate, ci_low, ci_high = win_ci([r["win"] for r in results])
    out = {
        "games": n, "win_rate": win_rate, "ci_low": ci_low, "ci_high": ci_high,
        "a_score": mean("a"), "b_score": mean("b"),
        "margin": mean("a") - mean("b"), "a_done": mean("a_done"), "b_done": mean("b_done"),
        "a_failed": mean("a_failed"), "b_failed": mean("b_failed"),
    }
    for c in COMPONENTS:
        out[f"a_{c}"], out[f"b_{c}"] = mean(f"a_{c}"), mean(f"b_{c}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--games", type=int, default=1000)
    ap.add_argument("--a-ckpt")
    ap.add_argument("--b-ckpt")
    ap.add_argument("--workers", type=int)
    ap.add_argument("--seed-offset", type=int, default=0, help="use other deals than the default ones")
    args = ap.parse_args()
    r = evaluate(args.a, args.b, args.games, args.a_ckpt, args.b_ckpt, args.workers, args.seed_offset)
    print(f"{args.a} (A) vs {args.b} (B), {r['games']} games (paired seeds, seats swapped)")
    print(f"  A win rate      {r['win_rate']:.3f}   95% CI [{r['ci_low']:.3f}, {r['ci_high']:.3f}]   (ties count 1/2)")
    print(f"  mean score      A {r['a_score']:.1f}   B {r['b_score']:.1f}   margin {r['margin']:+.1f}")
    for c in COMPONENTS:
        print(f"    {c:<9}     A {r[f'a_{c}']:6.1f}   B {r[f'b_{c}']:6.1f}")
    print(f"  tickets done    A {r['a_done']:.2f}   B {r['b_done']:.2f}")
    print(f"  tickets failed  A {r['a_failed']:.2f}   B {r['b_failed']:.2f}")


if __name__ == "__main__":
    main()
