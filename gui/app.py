"""Tk application: ties the engine to the widgets. The GUI only submits actions to the engine."""
import tkinter as tk
from tkinter import ttk

from ttr import rules
from ttr.actions import (
    BuildStation, ClaimRoute, DeclineTunnel, KeepTickets, PayTunnel,
)
from ttr.cards import LOCO
from ttr.engine import Game, IllegalMove
from ttr.players import make_player
from ttr.state import Phase

from . import dialogs
from .map_canvas import MapCanvas
from .side_panel import SidePanel


def describe_payment(length, color, locos):
    if color == LOCO:
        return f"{locos} x locomotive"
    text = f"{length - locos} x {color}"
    return text + (f" + {locos} x locomotive" if locos else "")


class App:
    def __init__(self, root, allow_random=False, seed=None, bot_delay_ms=400):
        self.root = root
        self.allow_random = allow_random
        self.seed = seed
        self.bot_delay_ms = bot_delay_ms
        self.game = None
        self.bots = {}
        self.revealed = None
        self.station_mode = False
        self.legal = []
        self._bot_scheduled = False
        self._over_shown = False
        self._busy = False
        self.on_game_over = self._default_game_over

        root.title("Ticket to Ride: Europe")
        root.geometry("1450x860")
        self.status = tk.StringVar()
        ttk.Label(root, textvariable=self.status, anchor="w", relief="sunken", padding=4).pack(side="bottom", fill="x")
        self.body = tk.Frame(root)
        self.body.pack(fill="both", expand=True)
        self.map = None
        self.panel = None

    # --- game lifecycle ----------------------------------------------------
    def start(self, specs=None, seed=None):
        if specs is None:
            answer = dialogs.ask_setup(self.root, self.allow_random, self.seed)
            if answer is None:
                self.root.destroy()
                return
            specs, seed = answer
        self.game = Game.new(specs, seed=seed)
        self.bots = {i: make_player(kind, (seed or 0) * 10 + i) for i, (_, kind) in enumerate(specs)}
        self.humans = [i for i, (_, kind) in enumerate(specs) if kind == "human"]
        self.revealed = self.humans[0] if len(self.humans) == 1 else None
        self.station_mode = False
        self._over_shown = False
        for w in self.body.winfo_children():
            w.destroy()
        self.overlay = None
        self.panel = SidePanel(self.body, self.do, self.toggle_station)
        self.panel.pack(side="right", fill="y")
        self.overlay = tk.Frame(self.panel, bg="#222222")
        self.overlay_label = tk.Label(self.overlay, fg="white", bg="#222222", font=("Helvetica", 18, "bold"))
        self.overlay_label.pack(expand=True, pady=(200, 10))
        self.overlay_btn = ttk.Button(self.overlay, text="I'm ready")
        self.overlay_btn.pack(pady=10)
        self.map = MapCanvas(self.body, self.game, self.route_clicked, self.city_clicked,
                             lambda text: self.status.set(text))
        self.map.pack(side="left", fill="both", expand=True)
        self.root.after(50, self.refresh)

    def quit(self):
        self.root.destroy()

    # --- main refresh / turn loop -------------------------------------------
    def refresh(self):
        if self._busy:
            return
        g = self.game
        s = g.state
        self.legal = g.legal_actions()
        actor = g.actor
        is_human = s.players[actor].kind == "human"
        self.panel.update_panel(g, self.legal, self.revealed if is_human else None, self.station_mode)
        self.map.redraw(claimable=rules.claimable_routes(g) if is_human and self.revealed == actor else set(),
                        pick_city=self.station_mode)
        if s.phase == Phase.GAME_OVER:
            self.overlay.place_forget()
            if not self._over_shown:
                self._over_shown = True
                self.root.after(50, self.on_game_over)
            return
        if not is_human:
            self.overlay.place_forget()
            self.status.set(f"{s.players[actor].name} (computer) is playing...")
            if not self._bot_scheduled:
                self._bot_scheduled = True
                self.root.after(self.bot_delay_ms, self._bot_step)
            return
        if len(self.humans) > 1 and self.revealed != actor:
            self.overlay_label.configure(text=f"Pass the device to\n{s.players[actor].name}")
            self.overlay_btn.configure(command=lambda: self._reveal(actor))
            self.overlay.place(x=0, y=0, relwidth=1, relheight=1)
            return
        self.overlay.place_forget()
        self.root.after_idle(self._prompt_if_needed)

    def _reveal(self, actor):
        self.revealed = actor
        self.refresh()

    def _prompt_if_needed(self):
        """Open the modal dialog required by the current phase, if any."""
        if self._busy or self.game.state.phase == Phase.GAME_OVER:
            return
        g = self.game
        s = g.state
        who = s.players[g.actor].name
        if s.phase in (Phase.SETUP_TICKETS, Phase.CHOOSE_TICKETS):
            if not self.legal or not isinstance(self.legal[0], KeepTickets):
                return
            min_keep = min(len(a.ticket_ids) for a in self.legal)
            self._busy = True
            try:
                keep = dialogs.ask_tickets(self.root, g.board, s.pending_tickets, min_keep, who)
            finally:
                self._busy = False
            self.do(KeepTickets(keep))
        elif s.phase == Phase.TUNNEL:
            t = s.pending_tunnel
            options = []
            for a in self.legal:
                if isinstance(a, PayTunnel):
                    options.append((f"Pay {t.extra} extra card(s) ({a.locos} locomotive(s))", a))
                elif isinstance(a, DeclineTunnel):
                    options.append(("Give up (keep your cards)", a))
            self._busy = True
            try:
                action = dialogs.ask_tunnel(self.root, t.revealed, t.extra, options)
            finally:
                self._busy = False
            self.do(action)
        else:
            self.status.set("")

    def _bot_step(self):
        self._bot_scheduled = False
        g = self.game
        if g.state.phase == Phase.GAME_OVER:
            return
        actor = g.actor
        legal = g.legal_actions()
        action = self.bots[actor].choose(g.view_for(actor), legal)
        g.apply(action)
        self.refresh()

    # --- user actions ---------------------------------------------------------
    def do(self, action):
        try:
            self.game.apply(action)
        except IllegalMove as e:  # the buttons only offer legal moves, so this is a safety net
            self.status.set(f"Illegal move: {e}")
            return
        self.station_mode = False
        self.status.set("")
        self.refresh()

    def toggle_station(self):
        self.station_mode = not self.station_mode
        self.status.set("Click the city where you want to build a station" if self.station_mode else "")
        self.refresh()

    def _human_may_act(self):
        g = self.game
        return (g.state.phase != Phase.GAME_OVER and g.state.players[g.actor].kind == "human"
                and self.revealed == g.actor and not self._busy)

    def route_clicked(self, rid):
        if not self._human_may_act() or self.station_mode:
            return
        acts = [a for a in self.legal if isinstance(a, ClaimRoute) and a.route_id == rid]
        if not acts:
            self.status.set(rules.explain_route(self.game, rid))
            return
        route = self.game.board.routes[rid]
        if len(acts) == 1:
            return self.do(acts[0])
        options = [(describe_payment(route.length, a.color, a.locos), a) for a in acts]
        self._busy = True
        try:
            chosen = dialogs.ask_choice(self.root, "Claim route",
                                        f"{route.a} - {route.b}: choose your payment", options)
        finally:
            self._busy = False
        if chosen is not None:
            self.do(chosen)

    def city_clicked(self, city):
        if not self.station_mode or not self._human_may_act():
            return
        acts = [a for a in self.legal if isinstance(a, BuildStation) and a.city == city]
        if not acts:
            self.status.set(rules.explain_station(self.game, city))
            return
        cost = rules.station_cost(self.game.state.players[self.game.state.current])
        if len(acts) == 1:
            return self.do(acts[0])
        options = [(describe_payment(cost, a.color, a.locos), a) for a in acts]
        self._busy = True
        try:
            chosen = dialogs.ask_choice(self.root, "Build station", f"Station in {city}: choose your payment", options)
        finally:
            self._busy = False
        if chosen is not None:
            self.do(chosen)

    # --- end of game ------------------------------------------------------------
    def _default_game_over(self):
        s = self.game.state
        dialogs.show_game_over(self.root, s.players, s.final_scores, self.game.winners(),
                               on_new_game=lambda: self.start(), on_quit=self.quit)


def main(allow_random=False, seed=None):
    root = tk.Tk()
    root.withdraw()
    app = App(root, allow_random=allow_random, seed=seed)
    root.deiconify()
    app.start()
    root.mainloop()
