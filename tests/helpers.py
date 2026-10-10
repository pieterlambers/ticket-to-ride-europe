from collections import Counter

from ttr.actions import KeepTickets
from ttr.engine import Game
from ttr.state import Phase


def new_game(n=2, seed=0):
    return Game.new([(f"P{i}", "human") for i in range(n)], seed=seed)


def started_game(n=2, seed=0):
    """A game past ticket setup (everyone keeps all offered tickets), player 0 to move."""
    g = new_game(n, seed)
    while g.state.phase == Phase.SETUP_TICKETS:
        g.apply(KeepTickets(frozenset(g.state.pending_tickets)))
    return g


def give(g, i, **cards):
    """Replace player i's hand (keeps the deck untouched, so conservation tests don't apply)."""
    g.state.players[i].hand = Counter(cards)


def route_id(g, a, b, color=None):
    for r in g.board.routes:
        if {r.a, r.b} == {a, b} and (color is None or r.color == color):
            return r.id
    raise KeyError((a, b, color))
