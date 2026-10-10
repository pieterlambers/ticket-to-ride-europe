"""NeuralPlayer: plays with a trained PolicyValueNet checkpoint."""
import warnings
from pathlib import Path

import numpy as np

from . import features, graph
from .heuristic import HeuristicPlayer
from .macro import macros, to_action

DEFAULT_CKPT = Path(__file__).resolve().parents[2] / "models" / "best.pt"


class NeuralPlayer:
    uses_full_state = True  # reads only public info and its own cards

    def __init__(self, ckpt=DEFAULT_CKPT, temperature=0.0, seed=None, model=None):
        import torch

        from .model import load_checkpoint

        self.torch = torch
        self.temperature = temperature
        self.rng = np.random.default_rng(seed)
        self.fallback = None
        if model is not None:
            self.model = model
        elif Path(ckpt).exists():
            self.model = load_checkpoint(ckpt)
        else:
            warnings.warn(f"checkpoint {ckpt} not found: playing with the heuristic bot instead")
            self.model = None
            self.fallback = HeuristicPlayer(seed)

    def choose(self, game, legal):
        if self.fallback is not None:
            return self.fallback.choose(game, legal)
        torch = self.torch
        me = game.actor
        plan = graph.plan(game, me)
        ms = macros(game, legal)
        if len(ms) == 1:
            return to_action(game, ms[0], legal, plan["need"])
        state = torch.from_numpy(features.encode_state(game, me, plan)).unsqueeze(0)
        acts = torch.from_numpy(features.encode_actions(game, me, ms, plan))
        with torch.no_grad():
            logits, _ = self.model(state, acts, torch.zeros(len(ms), dtype=torch.long))
        if self.temperature > 0:
            p = torch.softmax(logits / self.temperature, 0).numpy().astype(np.float64)
            i = int(self.rng.choice(len(ms), p=p / p.sum()))
        else:
            i = int(torch.argmax(logits))
        return to_action(game, ms[i], legal, plan["need"])
