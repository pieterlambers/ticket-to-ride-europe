import random

from ttr.cards import Deck, LOCO


def test_full_deck_is_110():
    d = Deck(random.Random(1))
    assert d.total() == 110


def test_take_face_up_refills():
    d = Deck(random.Random(2))
    d.take_face_up(0)
    assert all(c is not None for c in d.face_up)
    assert d.total() == 109


def test_reshuffle_discard_when_empty():
    d = Deck(random.Random(3))
    d.discard = d.draw_pile
    d.draw_pile = []
    assert d.draw() is not None
    assert d.discard == []


def test_three_loco_rule_replaces_row():
    d = Deck(random.Random(4), deal_face_up=False)
    d.face_up = [LOCO, LOCO, LOCO, "red", "blue"]
    d.draw_pile = [c for c in d.draw_pile if c != LOCO] + [LOCO] * 0
    d.refill_face_up()
    assert sum(c == LOCO for c in d.face_up) < 3


def test_no_infinite_loop_only_locos():
    d = Deck(random.Random(5), deal_face_up=False)
    d.draw_pile = [LOCO] * 14
    d.discard = []
    d.refill_face_up()  # must terminate
    assert d.total() == 14


def test_reveal_conserves_total():
    d = Deck(random.Random(6))
    cards = d.reveal(3)
    d.discard_cards(cards)
    assert d.total() == 110
