import random
import time
from collections import Counter

import numpy as np

from ttr.ai import features, graph
from ttr.ai.macro import macros, to_action
from ttr.engine import Game
from ttr.state import Phase


def _walk(seed, n=2):
    """Yield (game, me) at every decision of a random game."""
    g = Game.new([(f"P{i}", "bot") for i in range(n)], seed=seed)
    rng = random.Random(seed)
    while g.state.phase != Phase.GAME_OVER:
        yield g, g.actor
        legal = g.legal_actions()
        g.apply(rng.choice(legal))


def test_shapes_are_constant():
    for seed in range(20):
        for g, me in _walk(seed):
            plan = graph.plan(g, me)
            s = features.encode_state(g, me, plan)
            assert s.shape == (features.STATE_DIM,) and s.dtype == np.float32 and np.isfinite(s).all()
            ms = macros(g, g.legal_actions())
            a = features.encode_actions(g, me, ms, plan)
            assert a.shape == (len(ms), features.ACT_DIM) and np.isfinite(a).all()


def test_no_hidden_information_leaks():
    """Scrambling the opponent's hand composition and ticket ids must not change anything."""
    for seed in range(5):
        for g, me in _walk(seed):
            ms = macros(g, g.legal_actions())
            plan = graph.plan(g, me)
            before = (features.encode_state(g, me, plan), features.encode_actions(g, me, ms, plan))
            opp = g.state.players[1 - me]
            n = sum(opp.hand.values())
            opp.hand, saved_hand = Counter({"hidden": n}), opp.hand
            opp.tickets, saved_tickets = [-1] * len(opp.tickets), opp.tickets
            try:
                plan2 = graph.plan(g, me)
                after = (features.encode_state(g, me, plan2), features.encode_actions(g, me, ms, plan2))
            finally:
                opp.hand, opp.tickets = saved_hand, saved_tickets
            assert np.array_equal(before[0], after[0]) and np.array_equal(before[1], after[1])


def test_encoding_time(capsys):
    g = Game.new([("A", "bot"), ("B", "bot")], seed=1)
    rng = random.Random(1)
    for _ in range(80):
        g.apply(rng.choice(g.legal_actions()))
    me = g.actor
    t = time.perf_counter()
    for _ in range(100):
        plan = graph.plan(g, me)
        features.encode_state(g, me, plan)
        features.encode_actions(g, me, macros(g, g.legal_actions()), plan)
    with capsys.disabled():
        print(f"\nstate + actions encoding: {(time.perf_counter() - t) * 10:.2f} ms")
