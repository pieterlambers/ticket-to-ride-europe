import pytest

from ttr.engine import Game
from ttr.players import RandomPlayer
from ttr.state import Phase

MAX_ACTIONS = 3000


def play(n, seed):
    g = Game.new([(f"R{i}", "random") for i in range(n)], seed=seed)
    bots = [RandomPlayer(seed * 10 + i) for i in range(n)]
    steps = 0
    while g.state.phase != Phase.GAME_OVER:
        steps += 1
        assert steps < MAX_ACTIONS, f"game did not finish (n={n}, seed={seed})"
        legal = g.legal_actions()
        assert legal
        action = bots[g.actor].choose(g.view_for(g.actor), legal)
        assert action in legal
        g.apply(action)
        s = g.state
        in_hands = sum(sum(p.hand.values()) for p in s.players)
        assert in_hands + s.deck.total() == 110
        assert all(p.trains >= 0 for p in s.players)
    return g


@pytest.mark.parametrize("n", [2, 3, 4, 5])
@pytest.mark.parametrize("seed", range(50))
def test_random_game_finishes(n, seed):
    g = play(n, seed)
    rows = g.state.final_scores
    for r, p in zip(rows, g.state.players):
        assert r["routes"] == p.score
    assert g.winners()
