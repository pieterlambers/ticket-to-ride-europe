"""Route planning: shortest paths and a shared route plan for a player's tickets. Pure python."""
import heapq
from collections import Counter

from ttr.cards import LOCO  # noqa: F401  (re-exported for callers building payments)
from ttr.rules import MAX_PLAYERS_DOUBLE_CLOSED

INF = float("inf")
_twins: dict = {}  # id(board) -> {route id: twin route id}


def _twin_ids(board) -> dict:
    key = id(board)
    if key not in _twins:
        by_double: dict = {}
        for r in board.routes:
            if r.double_id is not None:
                by_double.setdefault(r.double_id, []).append(r.id)
        _twins[key] = {a: b for ids in by_double.values() if len(ids) == 2 for a, b in (ids, ids[::-1])}
    return _twins[key]


def route_cost(game, me, route, stations=frozenset()) -> float:
    """Trains still to spend to use `route`: 0 if mine (or borrowed through one of my stations),
    INF if I cannot claim it, else its length."""
    s = game.state
    owner = s.route_owner.get(route.id)
    if owner == me:
        return 0
    if owner is not None:
        return 0 if (route.a in stations or route.b in stations) else INF
    twin = _twin_ids(game.board).get(route.id)
    if twin is not None and twin in s.route_owner:
        if s.route_owner[twin] == me or len(s.players) <= MAX_PLAYERS_DOUBLE_CLOSED:
            return INF
    if route.length > s.players[me].trains:
        return INF
    return route.length


def shortest_path(board, a, b, costs, free=frozenset()):
    """Dijkstra on cities. `costs` maps route id -> cost; routes in `free` cost 0.
    Returns (cost, route ids along the path); (INF, []) if unreachable."""
    if a == b:
        return 0, []
    dist = {a: 0}
    prev = {}
    heap = [(0, a)]
    while heap:
        d, city = heapq.heappop(heap)
        if d > dist.get(city, INF):
            continue
        if city == b:
            break
        for r in board.routes_by_city[city]:
            c = 0 if r.id in free else costs[r.id]
            if c == INF:
                continue
            nxt = r.b if r.a == city else r.a
            nd = d + c
            if nd < dist.get(nxt, INF):
                dist[nxt] = nd
                prev[nxt] = (city, r.id)
                heapq.heappush(heap, (nd, nxt))
    if b not in dist:
        return INF, []
    path, city = [], b
    while city != a:
        city, rid = prev[city]
        path.append(rid)
    return dist[b], path[::-1]


def plan(game, me, extra_tickets=(), extra_stations=(), budget=None) -> dict:
    """Plan the cheapest shared set of routes for my tickets (highest points first).

    Returns routes (unclaimed route ids to build), need (colour -> cards, grey under "grey"),
    engines (locomotives needed), uses (route id -> tickets using it), done / unreachable /
    too_expensive (ticket ids) and cost (trains still to build).
    """
    board, s = game.board, game.state
    p = s.players[me]
    stations = frozenset(p.stations) | frozenset(extra_stations)
    costs = {r.id: route_cost(game, me, r, stations) for r in board.routes}
    budget = p.trains if budget is None else budget
    tids = list(dict.fromkeys(list(p.tickets) + list(extra_tickets)))
    planned: set = set()
    uses: Counter = Counter()
    done, unreachable, too_expensive = [], [], []
    total = 0
    for t in sorted((board.tickets[i] for i in tids), key=lambda t: -t.points):
        c, path = shortest_path(board, t.a, t.b, costs, planned)
        if c == INF:
            unreachable.append(t.id)
        elif c == 0 and not any(costs[r] > 0 and r not in planned for r in path):
            done.append(t.id)
            uses.update(r for r in path if costs[r] > 0)
        elif total + c > budget:
            too_expensive.append(t.id)
        else:
            total += c
            for rid in path:
                if costs[rid] > 0:
                    planned.add(rid)
                    uses[rid] += 1
    need: Counter = Counter()
    engines = 0
    for rid in planned:
        r = board.routes[rid]
        need["grey" if r.color == "grey" else r.color] += r.length
        engines += r.engines
    return {
        "routes": planned, "need": need, "engines": engines, "uses": uses, "done": done,
        "unreachable": unreachable, "too_expensive": too_expensive, "cost": total, "costs": costs,
    }
