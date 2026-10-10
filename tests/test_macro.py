import random

from ttr.ai import graph
from ttr.ai.macro import from_action, macros, to_action
from ttr.engine import Game
from ttr.state import Phase


def test_macros_map_to_legal_actions_and_round_trip():
    kinds = set()
    for seed in range(50):
        g = Game.new([("A", "bot"), ("B", "bot")], seed=seed)
        rng = random.Random(seed)
        while g.state.phase != Phase.GAME_OVER:
            legal = g.legal_actions()
            plan = graph.plan(g, g.actor) if g.state.phase != Phase.SETUP_TICKETS else {"need": {}}
            ms = macros(g, legal)
            assert len(ms) == len(set(ms))
            for m in ms:
                a = to_action(g, m, legal, plan["need"])
                assert a in legal
                assert from_action(a) == m  # claim/station compare by route or city, not payment
                kinds.add(m.kind)
            g.apply(to_action(g, rng.choice(ms), legal, plan["need"]))
    required = {"claim", "station", "faceup", "blind", "tickets", "keep", "tunnel_pay", "tunnel_decline"}
    assert required <= kinds  # "pass" only occurs when a player has no other move
