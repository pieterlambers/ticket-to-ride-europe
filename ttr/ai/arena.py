"""Fast game runner for bots. Bots flagged `uses_full_state` get the real game (no deepcopy)."""
from ttr.engine import Game
from ttr.state import Phase


def play_game(players, seed, n=2, max_steps=3000, game=None):
    """Play one game to the end. `players` is a list of n objects with choose(game, legal)."""
    g = game or Game.new([(f"P{i}", "bot") for i in range(n)], seed=seed)
    steps = 0
    while g.state.phase != Phase.GAME_OVER:
        steps += 1
        if steps > max_steps:
            raise RuntimeError(f"game did not finish (seed={seed})")
        legal = g.legal_actions()
        actor = g.actor
        bot = players[actor]
        view = g if getattr(bot, "uses_full_state", False) else g.view_for(actor)
        action = bot.choose(view, legal)
        if action not in legal:
            raise RuntimeError(f"illegal action from player {actor}: {action}")
        g.apply(action)
    return g
