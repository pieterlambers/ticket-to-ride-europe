"""Play two kinds of bot against each other and print a win-rate table.

    python -m train.evaluate --a heuristic --b random --games 500
"""
import argparse
import multiprocessing as mp
import os

from ttr.ai.arena import make_bot, play_game


def run_one(args):
    kind_a, kind_b, ckpt_a, ckpt_b, i = args
    a = make_bot(kind_a, i * 2, ckpt_a)
    b = make_bot(kind_b, i * 2 + 1, ckpt_b)
    seat_a = i % 2  # swap who starts every game
    players = [a, b] if seat_a == 0 else [b, a]
    g = play_game(players, seed=i)
    rows = g.state.final_scores
    ra, rb = rows[seat_a], rows[1 - seat_a]
    held = [len(p.tickets) for p in g.state.players]
    win = 0.5 if ra["total"] == rb["total"] and ra["completed"] == rb["completed"] and ra["stations"] == rb["stations"] \
        else float(seat_a in g.winners())
    return {
        "win": win, "a": ra["total"], "b": rb["total"],
        "a_done": ra["completed"], "b_done": rb["completed"],
        "a_failed": held[seat_a] - ra["completed"], "b_failed": held[1 - seat_a] - rb["completed"],
    }


def evaluate(kind_a, kind_b, games, ckpt_a=None, ckpt_b=None, workers=None):
    jobs = [(kind_a, kind_b, ckpt_a, ckpt_b, i) for i in range(games)]
    workers = workers or max(1, (os.cpu_count() or 2) - 1)
    if workers == 1:
        results = [run_one(j) for j in jobs]
    else:
        with mp.Pool(workers) as pool:
            results = pool.map(run_one, jobs, chunksize=max(1, games // (workers * 4)))
    n = len(results)
    mean = lambda k: sum(r[k] for r in results) / n  # noqa: E731
    return {
        "games": n, "win_rate": mean("win"), "a_score": mean("a"), "b_score": mean("b"),
        "margin": mean("a") - mean("b"), "a_done": mean("a_done"), "b_done": mean("b_done"),
        "a_failed": mean("a_failed"), "b_failed": mean("b_failed"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--games", type=int, default=200)
    ap.add_argument("--a-ckpt")
    ap.add_argument("--b-ckpt")
    ap.add_argument("--workers", type=int)
    args = ap.parse_args()
    r = evaluate(args.a, args.b, args.games, args.a_ckpt, args.b_ckpt, args.workers)
    print(f"{args.a} (A) vs {args.b} (B), {r['games']} games, seats alternate")
    print(f"  A win rate      {r['win_rate']:.3f}   (ties count 1/2)")
    print(f"  mean score      A {r['a_score']:.1f}   B {r['b_score']:.1f}   margin {r['margin']:+.1f}")
    print(f"  tickets done    A {r['a_done']:.2f}   B {r['b_done']:.2f}")
    print(f"  tickets failed  A {r['a_failed']:.2f}   B {r['b_failed']:.2f}")


if __name__ == "__main__":
    main()
