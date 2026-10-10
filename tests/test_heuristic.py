import pytest

from ttr.ai.arena import play_game
from ttr.ai.heuristic import HeuristicPlayer
from ttr.engine import Game
from ttr.players import RandomPlayer
from ttr.state import Phase


class ViewOnly:
    """Wraps a bot so it gets a view_for copy (hidden opponent info), as the GUI does."""

    def __init__(self, bot):
        self.bot = bot

    def choose(self, view, legal):
        return self.bot.choose(view, legal)


def _bots(n, seed, view):
    bots = [HeuristicPlayer(seed + i) for i in range(n)]
    return [ViewOnly(b) for b in bots] if view else bots


@pytest.mark.parametrize("view", [False, True])
@pytest.mark.parametrize("n", [2, 3, 4, 5])
def test_heuristic_games_finish_with_legal_moves(n, view):
    for seed in range(20):
        g = play_game(_bots(n, seed, view), seed=seed, n=n)  # play_game raises on an illegal action
        assert g.state.phase == Phase.GAME_OVER


def test_view_and_full_state_choose_the_same_moves():
    g = Game.new([("A", "bot"), ("B", "bot")], seed=3)
    bot = HeuristicPlayer(0)
    for _ in range(60):
        if g.state.phase == Phase.GAME_OVER:
            break
        legal = g.legal_actions()
        full = bot.choose(g, legal)
        assert full == bot.choose(g.view_for(g.actor), legal)
        g.apply(full)


def test_heuristic_beats_random():
    wins = 0
    for seed in range(100):
        h = seed % 2
        bots = [HeuristicPlayer(seed), RandomPlayer(seed)] if h == 0 else [RandomPlayer(seed), HeuristicPlayer(seed)]
        g = play_game(bots, seed=seed)
        wins += h in g.winners()
    assert wins >= 90
