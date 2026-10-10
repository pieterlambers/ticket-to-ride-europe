"""Scoring: route points, tickets (with stations), longest trail."""
from itertools import product

ROUTE_POINTS = {1: 1, 2: 2, 3: 4, 4: 7, 6: 15, 8: 21}
STATION_POINTS = 4
EXPRESS_POINTS = 10


def connected(edges, a: str, b: str) -> bool:
    """edges: iterable of (city, city)."""
    adj: dict[str, set] = {}
    for x, y in edges:
        adj.setdefault(x, set()).add(y)
        adj.setdefault(y, set()).add(x)
    seen, stack = {a}, [a]
    while stack:
        c = stack.pop()
        if c == b:
            return True
        for n in adj.get(c, ()):
            if n not in seen:
                seen.add(n)
                stack.append(n)
    return a == b


def longest_trail(routes) -> int:
    """routes: list of (a, b, length). Longest trail (edges used once, cities may repeat)."""
    adj: dict[str, list] = {}
    for i, (a, b, ln) in enumerate(routes):
        adj.setdefault(a, []).append((b, ln, i))
        adj.setdefault(b, []).append((a, ln, i))
    best = 0

    def dfs(city, used, total):
        nonlocal best
        best = max(best, total)
        for nxt, ln, i in adj.get(city, ()):
            if i not in used:
                used.add(i)
                dfs(nxt, used, total + ln)
                used.remove(i)

    for c in adj:
        dfs(c, set(), 0)
    return best


def ticket_result(board, game_state, player_index):
    """Return (points, completed_count) for a player's tickets, using the best station borrowing."""
    p = game_state.players[player_index]
    own = [(board.routes[r].a, board.routes[r].b) for r in p.routes]
    options = []
    for city in p.stations:
        opts = [None]
        for r in board.routes_by_city[city]:
            owner = game_state.route_owner.get(r.id)
            if owner is not None and owner != player_index:
                opts.append((r.a, r.b))
        options.append(opts)
    best = None
    for combo in product(*options) if options else [()]:
        edges = own + [e for e in combo if e is not None]
        pts = done = 0
        for tid in p.tickets:
            t = board.tickets[tid]
            if connected(edges, t.a, t.b):
                pts += t.points
                done += 1
            else:
                pts -= t.points
        if best is None or (pts, done) > best:
            best = (pts, done)
    return best


def final_scores(board, game_state):
    rows = []
    for i, p in enumerate(game_state.players):
        tickets, done = ticket_result(board, game_state, i)
        trail = longest_trail([(board.routes[r].a, board.routes[r].b, board.routes[r].length) for r in p.routes])
        rows.append(
            {
                "player": i,
                "routes": p.score,
                "tickets": tickets,
                "completed": done,
                "stations": STATION_POINTS * p.stations_left,
                "trail": trail,
                "express": 0,
            }
        )
    top = max(r["trail"] for r in rows)
    for r in rows:
        if r["trail"] == top and top > 0:
            r["express"] = EXPRESS_POINTS
        r["total"] = r["routes"] + r["tickets"] + r["stations"] + r["express"]
    return rows


def winners(game_state, rows):
    def key(r):
        return (r["total"], r["completed"], r["stations"])

    best = max(key(r) for r in rows)
    return [r["player"] for r in rows if key(r) == best]
