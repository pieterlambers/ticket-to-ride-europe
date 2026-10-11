import random
import time
from collections import Counter

import pytest

from ttr.ai.arena import make_bot, play_game
from ttr.ai.determinize import sample_world
from ttr.cards import CARDS_PER_COLOR, COLORS, LOCO, LOCOS
from ttr.engine import Game
from ttr.state import Phase

TOTAL_CARDS = len(COLORS) * CARDS_PER_COLOR + LOCOS


def _midgame(seed, n=2, plies=60):
    """A game played for a while by heuristic bots, stopped at a TURN of player 0 if possible."""
    g = Game.new([(f"P{i}", "bot") for i in range(n)], seed=seed)
    bots = [make_bot("heuristic", seed + i) for i in range(n)]
    steps = 0
    while g.state.phase != Phase.GAME_OVER and (steps < plies or g.actor != 0 or g.state.phase != Phase.TURN):
        g.apply(bots[g.actor].choose(g, g.legal_actions()))
        steps += 1
    return g


def _cards(world):
    s = world.state
    total = Counter(s.deck.draw_pile) + Counter(s.deck.discard) + Counter(c for c in s.deck.face_up if c)
    for p in s.players:
        total += p.hand
    return total


@pytest.mark.parametrize("n", [2, 3])
def test_sampled_world_is_consistent(n):
    for seed in range(5):
        g = _midgame(seed, n)
        w = sample_world(g, 0, random.Random(seed))
        total = _cards(w)
        assert sum(total.values()) == TOTAL_CARDS
        assert all(total[c] == CARDS_PER_COLOR for c in COLORS) and total[LOCO] == LOCOS
        mine, theirs = g.state.players[0], w.state.players[0]
        assert theirs.hand == mine.hand and theirs.tickets == mine.tickets
        assert w.state.route_owner == g.state.route_owner and w.state.deck.face_up == g.state.deck.face_up
        for i in range(1, n):
            assert sum(w.state.players[i].hand.values()) == sum(g.state.players[i].hand.values())
            assert len(w.state.players[i].tickets) == len(g.state.players[i].tickets)
            assert -1 not in w.state.players[i].tickets and "hidden" not in w.state.players[i].hand
        assert "hidden" not in w.state.deck.draw_pile and -1 not in w.state.ticket_deck
        w.legal_actions()  # playable


def test_sample_does_not_depend_on_hidden_information():
    g1 = _midgame(3)
    g2 = _midgame(3)
    s2 = g2.state
    # change what player 0 cannot see: player 1's hand and tickets, the draw pile order, the long pool
    other = s2.players[1]
    pool = list(other.hand.elements()) + s2.deck.draw_pile
    random.Random(99).shuffle(pool)
    k = sum(other.hand.values())
    other.hand = Counter(pool[:k])
    s2.deck.draw_pile = pool[k:]
    s2.ticket_deck = list(reversed(s2.ticket_deck))
    g2._long_pool = list(reversed(g2._long_pool))
    w1, w2 = sample_world(g1, 0, random.Random(5)), sample_world(g2, 0, random.Random(5))
    assert w1.state.players[1].hand == w2.state.players[1].hand
    assert w1.state.players[1].tickets == w2.state.players[1].tickets
    assert w1.state.deck.draw_pile == w2.state.deck.draw_pile
    assert w1.state.ticket_deck == w2.state.ticket_deck and w1._long_pool == w2._long_pool


@pytest.mark.parametrize("n", [2, 3])
def test_search_player_plays_legal_games(n):
    for seed in range(3):
        bots = [make_bot("search", seed) if i == 0 else make_bot("heuristic", seed + i) for i in range(n)]
        bots[0].n, bots[0].k = 2, 2
        g = play_game(bots, seed=seed, n=n)
        assert g.state.phase == Phase.GAME_OVER
