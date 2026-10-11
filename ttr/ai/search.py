"""SearchPlayer: the network proposes moves, determinized rollouts decide between close ones.

For a ticket choice, or a normal turn where the network's top moves are close, each candidate is tried
in several sampled worlds (hidden cards and tickets re-dealt) and played out to the end by a fast bot.
The candidate with the best average result is played. All candidates share the same sampled worlds,
so differences between them are not drowned in deal noise."""
import copy
import random

import numpy as np

from ttr.state import Phase

from .arena import make_bot, play_game
from .determinize import sample_world
from .macro import to_action
from .neural import NeuralPlayer

CLOSE_GAP = 2.0  # search a turn only when the top two logits are this close


class SearchPlayer:
    uses_full_state = True  # reads only public info and its own cards (worlds are sampled from view_for)

    def __init__(self, ckpt=None, k=4, n=12, rollout="heuristic", seed=None, model=None):
        import torch

        torch.set_num_threads(1)
        self.net = NeuralPlayer(ckpt, seed=seed, model=model) if ckpt else NeuralPlayer(seed=seed, model=model)
        self.k, self.n, self.rollout = k, n, rollout
        self.rng = random.Random(seed)
        self.searched = 0  # number of decisions that used search (for timing)

    def choose(self, game, legal):
        net = self.net
        if net.fallback is not None:
            return net.fallback.choose(game, legal)
        ms, logits, plan = net.score(game, legal)
        if len(ms) == 1:
            return to_action(game, ms[0], legal, plan["need"])
        order = np.argsort(-logits)
        phase = game.state.phase
        if phase in (Phase.SETUP_TICKETS, Phase.CHOOSE_TICKETS):
            cands = [ms[i] for i in order]
        elif phase == Phase.TURN and self.k > 1 and logits[order[0]] - logits[order[1]] < CLOSE_GAP:
            cands = [ms[i] for i in order[:self.k]]
        else:
            return to_action(game, ms[order[0]], legal, plan["need"])
        self.searched += 1
        return to_action(game, self._best(game, cands, plan), legal, plan["need"])

    def _best(self, game, cands, plan):
        me = game.actor
        n_players = len(game.state.players)
        totals = np.zeros(len(cands))
        for _ in range(self.n):
            world = sample_world(game, me, self.rng)
            seed = self.rng.randrange(1 << 30)
            for c, macro in enumerate(cands):
                w = copy.deepcopy(world, {id(world.board): world.board})
                w.apply(to_action(w, macro, w.legal_actions(), plan["need"]))
                bots = [make_bot(self.rollout, seed + i) for i in range(n_players)]
                play_game(bots, seed=seed, n=n_players, game=w)
                rows = w.state.final_scores
                margin = rows[me]["total"] - max(r["total"] for i, r in enumerate(rows) if i != me)
                totals[c] += np.sign(margin) + margin / 100
        return cands[int(np.argmax(totals))]
