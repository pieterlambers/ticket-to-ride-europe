"""The board: cities and routes drawn on a Tk canvas from longitude/latitude."""
import math
import tkinter as tk

CARD_COLORS = {
    "red": "#d62728", "orange": "#ff9f1c", "yellow": "#f2d600", "green": "#3a9d3a",
    "blue": "#2f7fd1", "pink": "#e377c2", "white": "#f7f7f7", "black": "#2b2b2b",
    "grey": "#a8a8a8", "loco": "#8a5cc2",
}
PAD = 40
CITY_R = 6
GAP = 12  # distance from a city centre where carriages start
HALF_WIDTH = 4
DOUBLE_OFFSET = 6


class MapCanvas(tk.Canvas):
    def __init__(self, master, game, on_route_click, on_city_click, on_hover, **kw):
        super().__init__(master, bg="#e8dfc8", highlightthickness=0, **kw)
        self.game = game
        self.on_route_click = on_route_click
        self.on_city_click = on_city_click
        self.on_hover = on_hover
        self.claimable: set = set()
        self.pick_city = False
        self.bind("<Configure>", lambda e: self.redraw())

    # --- geometry --------------------------------------------------------
    def _project(self):
        cities = self.game.board.cities.values()
        lon0 = min(c.lon for c in cities)
        lat1 = max(c.lat for c in cities)
        xs = [(c.lon - lon0) * math.cos(math.radians(48)) for c in cities]
        ys = [lat1 - c.lat for c in cities]
        w, h = max(self.winfo_width(), 300), max(self.winfo_height(), 300)
        k = min((w - 2 * PAD) / max(xs), (h - 2 * PAD) / max(ys))
        ox = (w - max(xs) * k) / 2
        oy = (h - max(ys) * k) / 2
        return {
            c.name: (ox + (c.lon - lon0) * math.cos(math.radians(48)) * k, oy + (lat1 - c.lat) * k)
            for c in cities
        }

    def _route_geometry(self, route):
        (x1, y1), (x2, y2) = self.pos[route.a], self.pos[route.b]
        dx, dy = x2 - x1, y2 - y1
        length = math.hypot(dx, dy) or 1
        ux, uy = dx / length, dy / length
        px, py = -uy, ux
        off = 0
        if route.double_id is not None:
            twin = self.game.board.twin(route)
            off = -DOUBLE_OFFSET if route.id < twin.id else DOUBLE_OFFSET
        return x1 + px * off, y1 + py * off, ux, uy, px, py, length

    # --- drawing ---------------------------------------------------------
    def redraw(self, claimable=None, pick_city=None):
        if claimable is not None:
            self.claimable = claimable
        if pick_city is not None:
            self.pick_city = pick_city
        self.delete("all")
        self.pos = self._project()
        state = self.game.state
        for route in self.game.board.routes:
            self._draw_route(route, state)
        for name in self.game.board.cities:
            self._draw_city(name, state)

    def _draw_route(self, route, state):
        x, y, ux, uy, px, py, length = self._route_geometry(route)
        owner = state.route_owner.get(route.id)
        closed = False
        if owner is None and route.double_id is not None:
            twin = self.game.board.twin(route)
            closed = twin.id in state.route_owner and len(state.players) <= 3
        span = max(length - 2 * GAP, 10)
        piece = span / route.length
        tag = f"r{route.id}"
        if owner is not None:
            fill = state.players[owner].color
        elif closed:
            fill = "#d9d2bd"
        else:
            fill = CARD_COLORS[route.color]
        if route.id in self.claimable:
            outline, width = "#ffd400", 3
        elif route.tunnel:
            outline, width = "#000000", 2
        else:
            outline, width = "#555555", 1
        if owner is not None:
            outline, width = "#ffffff", 2
        for i in range(route.length):
            s0 = GAP + i * piece + 1.5
            s1 = GAP + (i + 1) * piece - 1.5
            pts = []
            for s, side in ((s0, 1), (s1, 1), (s1, -1), (s0, -1)):
                pts += [x + ux * s + px * HALF_WIDTH * side, y + uy * s + py * HALF_WIDTH * side]
            self.create_polygon(
                pts, fill=fill, outline=outline, width=width,
                dash=(3, 2) if route.tunnel and owner is None else None, tags=(tag, "route"),
            )
            if route.engines and i < route.engines:
                cx, cy = x + ux * (s0 + s1) / 2, y + uy * (s0 + s1) / 2
                self.create_oval(cx - 3, cy - 3, cx + 3, cy + 3, fill=CARD_COLORS["loco"],
                                 outline="white", tags=(tag, "route"))
        if owner is not None:
            mx, my = x + ux * length / 2, y + uy * length / 2
            self.create_oval(mx - 7, my - 7, mx + 7, my + 7, fill=state.players[owner].color,
                             outline="white", width=2, tags=(tag, "route"))
            self.create_text(mx, my, text=state.players[owner].name[:1].upper(), fill="white",
                             font=("Helvetica", 8, "bold"), tags=(tag, "route"))
        self.tag_bind(tag, "<Button-1>", lambda e, r=route.id: self.on_route_click(r))
        self.tag_bind(tag, "<Enter>", lambda e, r=route: self.on_hover(self._describe(r, owner, closed)))
        self.tag_bind(tag, "<Leave>", lambda e: self.on_hover(""))

    def _describe(self, r, owner, closed):
        kind = [r.color if r.color != "grey" else "grey (any colour)", f"{r.length} trains"]
        if r.tunnel:
            kind.append("tunnel")
        if r.engines:
            kind.append(f"ferry: {r.engines} locomotive(s)")
        state = ""
        if owner is not None:
            state = f" - claimed by {self.game.state.players[owner].name}"
        elif closed:
            state = " - closed (double route)"
        return f"{r.a} - {r.b}: " + ", ".join(kind) + state

    def _draw_city(self, name, state):
        x, y = self.pos[name]
        tag = f"c_{name}"
        fill = "#fff3a0" if self.pick_city and name not in state.station_owner else "#ffffff"
        self.create_oval(x - CITY_R, y - CITY_R, x + CITY_R, y + CITY_R, fill=fill,
                         outline="#222222", width=2, tags=(tag, "city"))
        self.create_text(x, y + CITY_R + 8, text=name, font=("Helvetica", 9, "bold"),
                         fill="#222222", tags=(tag, "city"))
        owner = state.station_owner.get(name)
        if owner is not None:
            col = state.players[owner].color
            self.create_polygon(x, y - CITY_R - 12, x - 6, y - CITY_R - 1, x + 6, y - CITY_R - 1,
                                fill=col, outline="white", width=1, tags=(tag, "city"))
        self.tag_bind(tag, "<Button-1>", lambda e, n=name: self.on_city_click(n))
