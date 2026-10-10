"""The optimised legal_actions must equal the original implementation at every step."""
import random

import pytest

from ttr import rules
from ttr.actions import BuildStation, ClaimRoute, DrawBlind, DrawFaceUp, DrawTickets, Pass
from ttr.cards import COLORS
from ttr.engine import Game
from ttr.rules import _check, _payments, station_cost
from ttr.state import Phase


def _candidates_reference(game):
    """Copy of the original _candidates (before the affordability pruning)."""
    s, board = game.state, game.board
    ph = s.phase
    if ph not in (Phase.TURN, Phase.SECOND_DRAW):
        return rules._candidates(game)
    out = [DrawFaceUp(i) for i in range(len(s.deck.face_up))] + [DrawBlind()]
    if ph == Phase.TURN:
        p = s.players[s.current]
        out.append(DrawTickets())
        for r in board.routes:
            if r.id in s.route_owner or p.trains < r.length:
                continue
            colors = COLORS if r.color == "grey" else (r.color,)
            out += [ClaimRoute(r.id, c, k) for c, k in _payments(p, r.length, colors, r.engines)]
        if p.stations_left > 0:
            cost = station_cost(p)
            pays = _payments(p, cost, COLORS)
            out += [BuildStation(c, col, k) for c in board.cities if c not in s.station_owner for col, k in pays]
    return out


def _legal_actions_reference(game):
    acts = [a for a in _candidates_reference(game) if not _check(game, a)]
    if not acts and game.state.phase in (Phase.TURN, Phase.SECOND_DRAW):
        return [Pass()]
    return acts


@pytest.mark.parametrize("n", [2, 3, 4, 5])
def test_legal_actions_match_reference(n):
    for seed in range(8):
        g = Game.new([(f"R{i}", "random") for i in range(n)], seed=seed)
        rng = random.Random(seed)
        while g.state.phase != Phase.GAME_OVER:
            legal = g.legal_actions()
            assert legal == _legal_actions_reference(g)
            g.apply(rng.choice(legal))
