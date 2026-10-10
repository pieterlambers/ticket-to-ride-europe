from ttr.scoring import connected, longest_trail, final_scores

from .helpers import give, route_id, started_game


def test_connected():
    assert connected([("a", "b"), ("b", "c")], "a", "c")
    assert not connected([("a", "b"), ("c", "d")], "a", "d")


def test_longest_trail_loop_beats_simple_path():
    # triangle a-b-c plus tail c-d: trail a-b-c-a... uses all edges once from d
    routes = [("a", "b", 2), ("b", "c", 2), ("c", "a", 2), ("c", "d", 3)]
    assert longest_trail(routes) == 9


def test_longest_trail_branching():
    routes = [("a", "b", 1), ("b", "c", 2), ("b", "d", 3)]
    assert longest_trail(routes) == 5  # c-b-d


def _own(g, i, rid):
    g.state.route_owner[rid] = i
    g.state.players[i].routes.append(rid)


def test_unconnected_ticket_subtracts():
    g = started_game()
    s = g.state
    t = g.board.tickets[0]  # Venezia - Constantinople
    s.players[0].tickets = [t.id]
    assert final_scores(g.board, s)[0]["tickets"] == -t.points


def test_express_tied():
    g = started_game()
    s = g.state
    for p in s.players:
        p.tickets = []
    _own(g, 0, route_id(g, "Dieppe", "Brest"))
    _own(g, 1, route_id(g, "Lisboa", "Cadiz"))
    rows = final_scores(g.board, s)
    assert rows[0]["express"] == 10 and rows[1]["express"] == 10


def test_station_borrows_opponent_route():
    g = started_game()
    s = g.state
    # Find a ticket and craft a network: P1 owns the link, P0 has a station at its end.
    t = next(t for t in g.board.tickets if {t.a, t.b} == {"Madrid", "Zurich"})
    s.players[0].tickets = [t.id]
    # Madrid-Pamplona (P0), Pamplona-Marseille (P0), Marseille-Zurich owned by P1; P0 station in Marseille.
    _own(g, 0, route_id(g, "Madrid", "Pamplona", "black"))
    _own(g, 0, route_id(g, "Pamplona", "Marseille"))
    _own(g, 1, route_id(g, "Marseille", "Zurich"))
    row = final_scores(g.board, s)[0]
    assert row["tickets"] == -t.points
    s.players[0].stations = ["Marseille"]
    s.players[0].stations_left = 2
    row = final_scores(g.board, s)[0]
    assert row["tickets"] == t.points and row["completed"] == 1
