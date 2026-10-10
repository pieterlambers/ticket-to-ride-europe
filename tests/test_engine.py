from collections import Counter

from ttr.actions import ClaimRoute, DeclineTunnel, DrawBlind, DrawFaceUp, PayTunnel
from ttr.cards import LOCO
from ttr.state import Phase

from .helpers import give, new_game, route_id, started_game


def test_setup_counts():
    g = new_game(3)
    s = g.state
    assert all(sum(p.hand.values()) == 4 for p in s.players)
    assert all(c is not None for c in s.deck.face_up)
    assert s.deck.total() + 12 == 110
    assert len(s.pending_tickets) == 4
    assert sum(g.board.tickets[t].long for t in s.pending_tickets) == 1


def test_setup_then_play():
    g = started_game(3)
    assert g.state.phase == Phase.TURN and g.state.current == 0
    assert all(len(p.tickets) == 4 for p in g.state.players)


def test_turn_passes_after_two_draws():
    g = started_game()
    g.state.deck.face_up = ["red", "blue", "pink", "white", "black"]
    g.apply(DrawBlind())
    assert g.state.current == 0 and g.state.phase == Phase.SECOND_DRAW
    g.apply(DrawFaceUp(0))
    assert g.state.current == 1 and g.state.phase == Phase.TURN


def _tunnel_game():
    g = started_game()
    r = route_id(g, "Pamplona", "Barcelona")  # grey tunnel, length 2
    give(g, 0, red=4)
    g.state.deck.draw_pile += ["blue", "red", "red"]  # revealed: red, red, blue -> +2
    g.apply(ClaimRoute(r, "red", 0))
    return g, r


def test_tunnel_pay():
    g, r = _tunnel_game()
    assert g.state.phase == Phase.TUNNEL and g.state.pending_tunnel.extra == 2
    g.apply(PayTunnel(0))
    p = g.state.players[0]
    assert g.state.route_owner[r] == 0 and sum(p.hand.values()) == 0 and p.trains == 43
    assert g.state.current == 1


def test_tunnel_decline_keeps_cards():
    g, r = _tunnel_game()
    g.apply(DeclineTunnel())
    p = g.state.players[0]
    assert r not in g.state.route_owner and p.hand["red"] == 4 and p.trains == 45
    assert g.state.current == 1


def test_tunnel_pay_with_locos():
    g = started_game()
    r = route_id(g, "Pamplona", "Barcelona")
    give(g, 0, red=2, loco=2)
    g.state.deck.draw_pile += ["blue", "red", "red"]
    g.apply(ClaimRoute(r, "red", 0))
    g.apply(PayTunnel(2))
    assert sum(g.state.players[0].hand.values()) == 0


def test_end_game_gives_everyone_one_more_turn():
    g = started_game(3)
    g.state.players[0].trains = 3
    r = route_id(g, "Dieppe", "Paris")  # length 1
    give(g, 0, pink=1)
    g.apply(ClaimRoute(r, "pink", 0))  # P0 now has 2 trains -> final round
    assert g.state.turns_left == 3
    for expected in (1, 2, 0):
        assert g.state.current == expected and g.state.phase == Phase.TURN
        g.apply(DrawBlind()); g.apply(DrawBlind())
    assert g.state.phase == Phase.GAME_OVER
    assert g.state.final_scores is not None
