"""Sample a complete game consistent with what one player can see (for search rollouts).

The sample is built from `view_for(me)`, so nothing the player should not know can leak in: the
opponents' hands, their tickets, the draw pile order and the ticket deck are dealt anew from the cards
and tickets that are not visible to `me`."""
import random
from collections import Counter

from ttr.cards import CARDS_PER_COLOR, COLORS, LOCO, LOCOS

LONG_KEPT = 0.7  # chance that an opponent kept the long ticket offered at the start


def _unseen_cards(state, me):
    pool = Counter({c: CARDS_PER_COLOR for c in COLORS})
    pool[LOCO] = LOCOS
    pool.subtract(state.players[me].hand)
    pool.subtract(c for c in state.deck.face_up if c is not None)
    pool.subtract(state.deck.discard)
    return list(pool.elements())


def sample_world(game, me, rng):
    """A full game state, playable to the end, that matches everything `me` can see."""
    world = game.view_for(me)
    s, board = world.state, world.board
    opps = [i for i in range(len(s.players)) if i != me]

    cards = _unseen_cards(s, me)
    rng.shuffle(cards)
    assert len(cards) == len(s.deck.draw_pile) + sum(sum(s.players[i].hand.values()) for i in opps)
    for i in opps:
        n = sum(s.players[i].hand.values())
        s.players[i].hand = Counter(cards[:n])
        del cards[:n]
    s.deck.draw_pile = cards

    mine = set(s.players[me].tickets) | set(s.pending_tickets if s.pending_tickets and me == world.actor else ())
    longs = [t.id for t in board.tickets if t.long and t.id not in mine]
    regular = [t.id for t in board.tickets if not t.long and t.id not in mine]
    rng.shuffle(longs)
    rng.shuffle(regular)
    deck_len = len(s.ticket_deck)
    for i in opps:
        want = len(s.players[i].tickets)
        n_long = 1 if want and longs and rng.random() < LONG_KEPT else 0
        got = [longs.pop() for _ in range(n_long)] + [regular.pop() for _ in range(min(want - n_long, len(regular)))]
        s.players[i].tickets = got
    s.ticket_deck = regular[:deck_len]
    world._long_pool = longs

    world.rng = random.Random(rng.random())
    s.deck.rng = world.rng
    return world
