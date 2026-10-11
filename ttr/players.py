"""Player interface. A computer player only needs to implement choose()."""
import random
from typing import Protocol

from .actions import ClaimRoute


class Player(Protocol):
    def choose(self, game_view, legal: list):
        """Return one action from `legal`. `game_view` hides other players' private info."""


class HumanPlayer:
    """Marker: actions come from the GUI."""

    def choose(self, game_view, legal):
        raise NotImplementedError("Human actions come from the GUI")


class RandomPlayer:
    uses_full_state = True  # ignores the view, so it can be handed the real game

    def __init__(self, seed=None):
        self.rng = random.Random(seed)

    def choose(self, game_view, legal):
        claims = [a for a in legal if isinstance(a, ClaimRoute)]
        if claims and self.rng.random() < 0.6:
            return self.rng.choice(claims)
        return self.rng.choice(legal)


def make_player(kind: str, seed=None):
    if kind == "random":
        return RandomPlayer(seed)
    if kind == "heuristic":
        from .ai.heuristic import HeuristicPlayer  # lazy: keeps this module light

        return HeuristicPlayer(seed)
    if kind == "neural":
        from .ai.neural import NeuralPlayer

        return NeuralPlayer(seed=seed)
    if kind == "neural_search":
        from .ai.search import SearchPlayer

        return SearchPlayer(seed=seed)
    return HumanPlayer()
