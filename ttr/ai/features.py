"""Numeric encoding of the game for the neural network. numpy only, no torch.

encode_state reads only public information plus the acting player's own hand and tickets."""
import numpy as np

from ttr.board import load_board
from ttr.cards import COLORS, LOCO
from ttr.scoring import ROUTE_POINTS
from ttr.state import Phase

from . import graph
from .macro import KINDS

_BOARD = load_board()
NR = len(_BOARD.routes)
CITIES = list(_BOARD.cities)
CITY_INDEX = {c: i for i, c in enumerate(CITIES)}
CARD_KINDS = COLORS + (LOCO,)  # 9 card kinds
PHASES = list(Phase)
ROUTE_COLORS = COLORS + ("grey",)
STATIC = np.array([[r.length / 8, float(r.tunnel), r.engines / 3] for r in _BOARD.routes], dtype=np.float32)
ROUTE_FEATS = 8
N_SCALARS = 19 + len(PHASES) + 6
STATE_DIM = NR * ROUTE_FEATS + len(CARD_KINDS) + 5 * (len(CARD_KINDS) + 1) + N_SCALARS + len(CITIES) * 2
ACT_DIM = len(KINDS) + 15 + 4 + (len(CARD_KINDS) + 1 + 1) + 4 + 2


def encode_state(game, me, plan=None) -> np.ndarray:
    s, board = game.state, game.board
    plan = plan or graph.plan(game, me)
    p = s.players[me]
    opps = [q for i, q in enumerate(s.players) if i != me]
    costs = plan["costs"]

    routes = np.zeros((NR, ROUTE_FEATS), np.float32)
    routes[:, 4:7] = STATIC
    for r in board.routes:
        owner = s.route_owner.get(r.id)
        if owner is None:
            routes[r.id, 3 if costs[r.id] == graph.INF else 0] = 1  # free / blocked for me
        else:
            routes[r.id, 1 if owner == me else 2] = 1
    for rid in plan["routes"]:
        routes[rid, 7] = 1

    hand = np.array([p.hand[c] for c in CARD_KINDS], np.float32) / 12
    face = np.zeros((5, len(CARD_KINDS) + 1), np.float32)
    for i, card in enumerate(s.deck.face_up):
        face[i, CARD_KINDS.index(card) if card is not None else len(CARD_KINDS)] = 1

    tickets = [board.tickets[t] for t in p.tickets]
    done = set(plan["done"])
    pts_done = sum(t.points for t in tickets if t.id in done)
    pts_open = sum(t.points for t in tickets if t.id not in done)
    phase = np.zeros(len(PHASES), np.float32)
    phase[PHASES.index(s.phase)] = 1
    scalars = np.array([
        p.trains / 45, p.stations_left / 3, p.score / 100,
        sum(q.trains for q in opps) / (45 * len(opps)), sum(q.stations_left for q in opps) / (3 * len(opps)),
        max(q.score for q in opps) / 100,
        sum(sum(q.hand.values()) for q in opps) / 30, sum(len(q.tickets) for q in opps) / 10,
        len(s.deck.draw_pile) / 110, len(s.deck.discard) / 110, len(s.ticket_deck) / 46,
        float(s.turns_left is not None), (s.turns_left or 0) / 5,
        len(tickets) / 10, len(done) / 10, pts_done / 50, pts_open / 50, plan["cost"] / 45,
        len(plan["unreachable"]) / 5,
        *phase,
        len(plan["too_expensive"]) / 5, len(p.stations) / 3, plan["engines"] / 5,
        len(plan["routes"]) / 20, len(s.players) / 5, float(len(s.players) <= 3),
    ], np.float32)
    assert len(scalars) == N_SCALARS

    cities = np.zeros((len(CITIES), 2), np.float32)
    for rid in plan["routes"]:
        r = board.routes[rid]
        cities[CITY_INDEX[r.a], 0] = cities[CITY_INDEX[r.b], 0] = 1
    for c in s.station_owner:
        cities[CITY_INDEX[c], 1] = 1
    return np.concatenate([routes.ravel(), hand, face.ravel(), scalars, cities.ravel()])


def _deficit(plan, hand, card):
    have = hand[card]
    want = plan["engines"] if card == LOCO else plan["need"].get(card, 0)
    return float(np.clip((want - have) / 5, -1, 1))


def encode_action(game, me, macro, plan) -> np.ndarray:
    s, board = game.state, game.board
    p = s.players[me]
    v = np.zeros(ACT_DIM, np.float32)
    o = 0
    v[KINDS.index(macro.kind)] = 1
    o += len(KINDS)
    if macro.kind == "claim":
        r = board.routes[macro.arg]
        twin = graph._twin_ids(board).get(r.id)
        in_plan = r.id in plan["routes"] or twin in plan["routes"]
        uses = plan["uses"][r.id] + (plan["uses"][twin] if twin is not None else 0)
        v[o:o + 4] = (r.length / 8, ROUTE_POINTS[r.length] / 21, float(in_plan), uses / 3)
        v[o + 4 + ROUTE_COLORS.index(r.color)] = 1
        v[o + 13:o + 15] = (float(r.tunnel), r.engines / 3)
    o += 15
    if macro.kind == "station":
        city = macro.arg
        endpoints = {c for t in plan["unreachable"] for c in (board.tickets[t].a, board.tickets[t].b)}
        in_plan = any(city in (board.routes[r].a, board.routes[r].b) for r in plan["routes"])
        opp_routes = sum(1 for r in board.routes_by_city[city] if s.route_owner.get(r.id, me) != me)
        v[o:o + 4] = (float(in_plan), float(city in endpoints), opp_routes / 5, (4 - p.stations_left) / 3)
    o += 4
    if macro.kind in ("faceup", "blind"):
        if macro.kind == "faceup":
            card = s.deck.face_up[macro.arg]
            v[o + CARD_KINDS.index(card)] = 1
            v[o + len(CARD_KINDS) + 1] = _deficit(plan, p.hand, card)
        else:
            v[o + len(CARD_KINDS)] = 1  # unknown card
    o += len(CARD_KINDS) + 1 + 1
    if macro.kind == "keep":
        ids = macro.arg
        pl = graph.plan(game, me, extra_tickets=ids, budget=max(0, p.trains - 5))
        bad = set(pl["unreachable"]) | set(pl["too_expensive"])
        v[o:o + 4] = (
            len(ids) / 3, sum(board.tickets[t].points for t in ids) / 50,
            float(np.clip((pl["cost"] - plan["cost"]) / 45, -1, 1)), len(bad & (set(ids) | set(p.tickets))) / 3,
        )
    o += 4
    if macro.kind == "tunnel_pay":
        v[o] = macro.arg / 3
    if macro.kind in ("tunnel_pay", "tunnel_decline") and s.pending_tunnel is not None:
        v[o + 1] = s.pending_tunnel.extra / 3
    return v


def encode_actions(game, me, macro_list, plan) -> np.ndarray:
    return np.stack([encode_action(game, me, m, plan) for m in macro_list])
