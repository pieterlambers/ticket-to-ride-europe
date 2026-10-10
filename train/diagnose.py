"""Find where a bot loses points: per-side game statistics beyond the win rate.

    python -m train.diagnose --a neural --b heuristic --games 1000
"""
import argparse
import re

from ttr.ai import graph
from ttr.ai.arena import make_bot, play_game
from ttr.state import Phase

from .evaluate import COMPONENTS, run_pool, win_ci

LAST_ROUND = re.compile(r"^(.*) has \d+ trains left: last round!$")


class Probe:
    """Forwards to a bot and records the routes it planned and how many own turns it took."""

    def __init__(self, bot):
        self.bot = bot
        self.uses_full_state = getattr(bot, "uses_full_state", False)
        self.turns = 0
        self.planned: set = set()

    def choose(self, game, legal):
        if game.state.phase == Phase.TURN:
            self.turns += 1
            self.planned |= graph.plan(game, game.actor)["routes"]
        return self.bot.choose(game, legal)


def _taken(s, twins, rid, seat):
    """True if an opponent owns the route, or its twin while it is still unowned."""
    owner, twin = s.route_owner.get(rid), s.route_owner.get(twins.get(rid))
    return (owner is not None and owner != seat) or (owner is None and twin is not None and twin != seat)


def run_one(args):
    kind_a, kind_b, ckpt_a, ckpt_b, i = args[:5]
    probes = [Probe(make_bot(kind_a, i * 2, ckpt_a)), Probe(make_bot(kind_b, i * 2 + 1, ckpt_b))]
    seat_a = i % 2
    if seat_a == 1:
        probes.reverse()
    g = play_game(probes, seed=i // 2)
    s = g.state
    rows = s.final_scores
    twins = graph._twin_ids(g.board)
    ender = next((m.group(1) for m in map(LAST_ROUND.match, s.log) if m), None)
    out = {"win": 0.5 if rows[seat_a]["total"] == rows[1 - seat_a]["total"] else float(seat_a in g.winners())}
    for side, seat in (("a", seat_a), ("b", 1 - seat_a)):
        p = s.players[seat]
        lost = sum(1 for r in probes[seat].planned if _taken(s, twins, r, seat))
        out.update({
            f"{side}_kept": len(p.tickets), f"{side}_done": rows[seat]["completed"],
            f"{side}_failed": len(p.tickets) - rows[seat]["completed"],
            f"{side}_trains": p.trains, f"{side}_stations": p.stations_left,
            f"{side}_turns": probes[seat].turns, f"{side}_ends": float(ender == p.name),
            f"{side}_lost": lost,
        })
        for c in COMPONENTS:
            out[f"{side}_{c}"] = rows[seat][c]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--games", type=int, default=1000)
    ap.add_argument("--a-ckpt")
    ap.add_argument("--b-ckpt")
    ap.add_argument("--workers", type=int)
    args = ap.parse_args()
    games = args.games + args.games % 2
    results = run_pool(run_one, [(args.a, args.b, args.a_ckpt, args.b_ckpt, i) for i in range(games)], args.workers)
    n = len(results)
    mean = lambda k: sum(r[k] for r in results) / n  # noqa: E731
    wr, lo, hi = win_ci([r["win"] for r in results])
    print(f"{args.a} (A) vs {args.b} (B), {n} games: A win rate {wr:.3f} [{lo:.3f}, {hi:.3f}]")
    rows = [("tickets kept", "kept"), ("tickets completed", "done"), ("tickets failed", "failed"),
            ("trains left", "trains"), ("stations left", "stations"), ("own turns", "turns"),
            ("ends the game", "ends"), ("planned routes lost", "lost")] + [(f"points: {c}", c) for c in COMPONENTS]
    print(f"  {'':<22}{'A':>8}{'B':>8}")
    for label, key in rows:
        print(f"  {label:<22}{mean('a_' + key):8.2f}{mean('b_' + key):8.2f}")


if __name__ == "__main__":
    main()
