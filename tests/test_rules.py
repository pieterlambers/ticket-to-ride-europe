import pytest

from ttr.actions import (
    BuildStation, ClaimRoute, DeclineTunnel, DrawBlind, DrawFaceUp, DrawTickets, KeepTickets, Pass, PayTunnel,
)
from ttr.cards import LOCO
from ttr.engine import IllegalMove
from ttr.rules import legal_actions, validate
from ttr.state import Phase

from .helpers import give, new_game, route_id, started_game


def ok(g, a):
    return validate(g, a)[0]


def test_setup_keep_at_least_two():
    g = new_game()
    ids = list(g.state.pending_tickets)
    assert len(ids) == 4
    assert not ok(g, KeepTickets(frozenset(ids[:1])))
    assert ok(g, KeepTickets(frozenset(ids[:2])))
    assert not ok(g, KeepTickets(frozenset({999})))


def test_keep_at_least_one_later():
    g = started_game()
    g.apply(DrawTickets())
    assert g.state.phase == Phase.CHOOSE_TICKETS
    assert not ok(g, KeepTickets(frozenset()))
    assert ok(g, KeepTickets(frozenset(g.state.pending_tickets[:1])))


def test_face_up_loco_ends_turn_and_not_as_second_card():
    g = started_game()
    g.state.deck.face_up = [LOCO, "red", "blue", LOCO, "pink"]
    g.apply(DrawFaceUp(0))  # first card, face-up loco: turn ends
    assert g.state.current == 1 and g.state.phase == Phase.TURN
    g.apply(DrawFaceUp(1))  # first card (non-loco)
    assert g.state.phase == Phase.SECOND_DRAW
    loco_slots = [i for i, c in enumerate(g.state.deck.face_up) if c == LOCO]
    for i in loco_slots:
        assert not ok(g, DrawFaceUp(i))
    assert ok(g, DrawBlind())


def test_claim_needs_matching_color_and_cards():
    g = started_game()
    r = route_id(g, "Dieppe", "Brest")  # orange, 2
    give(g, 0, orange=1)
    assert not ok(g, ClaimRoute(r, "orange", 0))
    assert ok(g, ClaimRoute(r, "orange", 1)) is False  # no loco in hand
    give(g, 0, orange=1, loco=1)
    assert ok(g, ClaimRoute(r, "orange", 1))
    assert not ok(g, ClaimRoute(r, "red", 1))


def test_grey_route_accepts_any_single_color():
    g = started_game()
    r = route_id(g, "Venezia", "Zagrab")  # grey, 2
    give(g, 0, red=2, blue=1)
    assert ok(g, ClaimRoute(r, "red", 0))
    assert not ok(g, ClaimRoute(r, "blue", 0))


def test_ferry_needs_locomotives():
    g = started_game()
    r = route_id(g, "London", "Amsterdam")  # ferry: 2 locos, length 2
    give(g, 0, red=5, loco=1)
    assert not ok(g, ClaimRoute(r, "red", 1))
    give(g, 0, loco=2)
    assert ok(g, ClaimRoute(r, LOCO, 2))


def test_not_enough_trains():
    g = started_game()
    g.state.players[0].trains = 1
    r = route_id(g, "Dieppe", "Brest")
    give(g, 0, orange=2)
    assert "trains" in validate(g, ClaimRoute(r, "orange", 0))[1]


def test_double_route_closed_in_2p_open_in_4p():
    for n, expect in ((2, False), (4, True)):
        g = started_game(n)
        a = route_id(g, "Paris", "Bruxelles", "yellow")
        b = route_id(g, "Paris", "Bruxelles", "red")
        give(g, 0, yellow=2)
        g.apply(ClaimRoute(a, "yellow", 0))
        give(g, 1, red=2)
        assert ok(g, ClaimRoute(b, "red", 0)) is expect


def test_cannot_own_both_double_routes():
    g = started_game(4)
    a = route_id(g, "Paris", "Bruxelles", "yellow")
    b = route_id(g, "Paris", "Bruxelles", "red")
    give(g, 0, yellow=2, red=2)
    g.apply(ClaimRoute(a, "yellow", 0))
    for _ in range(3):
        g.apply(DrawBlind())
        g.apply(DrawBlind())
    give(g, 0, red=2)
    assert g.state.current == 0
    assert not ok(g, ClaimRoute(b, "red", 0))


def test_station_cost_escalates_and_city_occupied():
    g = started_game()
    give(g, 0, red=1)
    assert ok(g, BuildStation("Paris", "red", 0))
    g.apply(BuildStation("Paris", "red", 0))
    assert g.state.players[0].stations_left == 2
    assert not ok(g, BuildStation("Paris", "red", 0))  # occupied (also not your turn / cost)
    g.apply(DrawBlind()); g.apply(DrawBlind())  # P1 moves on
    give(g, 0, red=1)
    assert not ok(g, BuildStation("Roma", "red", 0))  # 2nd station costs 2
    give(g, 0, red=2)
    assert ok(g, BuildStation("Roma", "red", 0))


def test_apply_rejects_illegal_with_reason():
    g = started_game()
    give(g, 0)
    with pytest.raises(IllegalMove):
        g.apply(ClaimRoute(0, "black", 0))


def test_pass_only_when_nothing_else():
    g = started_game()
    assert not ok(g, Pass())
    assert Pass() not in legal_actions(g)


def test_tunnel_phase_actions():
    g = started_game()
    r = route_id(g, "Pamplona", "Barcelona")  # grey tunnel, 2
    give(g, 0, red=4)
    g.state.deck.draw_pile += ["blue", "red", "red"]  # draw() pops from the end
    g.apply(ClaimRoute(r, "red", 0))
    assert g.state.phase == Phase.TUNNEL
    acts = legal_actions(g)
    assert DeclineTunnel() in acts and PayTunnel(0) in acts
