"""Right-hand panel: current player's hand, card row, buttons, tickets, scoreboard and log."""
import tkinter as tk
from tkinter import ttk

from ttr import scoring
from ttr.actions import BuildStation, DrawBlind, DrawFaceUp, DrawTickets
from ttr.cards import COLORS, LOCO

from .map_canvas import CARD_COLORS

DARK_TEXT = {"red", "blue", "black", "green", "loco", "pink"}


def card_fg(color):
    return "white" if color in DARK_TEXT else "black"


class SidePanel(tk.Frame):
    def __init__(self, master, on_action, on_toggle_station, **kw):
        super().__init__(master, padx=8, pady=8, **kw)
        self.on_action = on_action
        self.on_toggle_station = on_toggle_station

        self.header = tk.Label(self, font=("Helvetica", 16, "bold"), anchor="w", padx=6)
        self.header.pack(fill="x")
        self.stats = ttk.Label(self)
        self.stats.pack(anchor="w", pady=(2, 6))
        self.hint = ttk.Label(self, wraplength=300, foreground="#444")
        self.hint.pack(anchor="w", pady=(0, 6))

        ttk.Label(self, text="Your cards").pack(anchor="w")
        self.hand = tk.Frame(self)
        self.hand.pack(anchor="w", pady=3)
        self.hand_chips = {}
        for c in COLORS + (LOCO,):
            lbl = tk.Label(self.hand, text="0", width=4, bg=CARD_COLORS[c], fg=card_fg(c), relief="ridge")
            lbl.pack(side="left", padx=1)
            self.hand_chips[c] = lbl

        ttk.Label(self, text="Face-up cards (click to draw)").pack(anchor="w", pady=(8, 0))
        row = tk.Frame(self)
        row.pack(anchor="w")
        self.face_buttons = []
        for i in range(5):
            b = tk.Button(row, width=7, command=lambda i=i: self.on_action(DrawFaceUp(i)))
            b.pack(side="left", padx=1)
            self.face_buttons.append(b)
        actions = tk.Frame(self)
        actions.pack(anchor="w", pady=6, fill="x")
        self.deck_btn = ttk.Button(actions, command=lambda: self.on_action(DrawBlind()))
        self.deck_btn.pack(side="left", padx=2)
        self.tickets_btn = ttk.Button(actions, command=lambda: self.on_action(DrawTickets()))
        self.tickets_btn.pack(side="left", padx=2)
        self.station_btn = ttk.Button(actions, command=self.on_toggle_station)
        self.station_btn.pack(side="left", padx=2)

        ttk.Label(self, text="Your destination tickets").pack(anchor="w")
        self.tickets = tk.Listbox(self, height=6, width=42, activestyle="none")
        self.tickets.pack(anchor="w", pady=3)

        ttk.Label(self, text="Scoreboard").pack(anchor="w", pady=(6, 0))
        self.board_frame = tk.Frame(self)
        self.board_frame.pack(anchor="w", fill="x")

        ttk.Label(self, text="Log").pack(anchor="w", pady=(6, 0))
        self.log = tk.Text(self, height=8, width=44, state="disabled", wrap="word", font=("Helvetica", 9))
        self.log.pack(anchor="w", fill="both", expand=True)

    def update_panel(self, game, legal, revealed, station_mode):
        s = game.state
        actor = game.actor
        p = s.players[actor]
        self.header.configure(text=p.name, bg=p.color, fg="white")
        show = revealed == actor
        self.stats.configure(
            text=f"Trains {p.trains}   Stations left {p.stations_left}   Route points {p.score}")
        for c, lbl in self.hand_chips.items():
            lbl.configure(text=str(p.hand[c]) if show else "?")
        self.tickets.delete(0, "end")
        if show:
            own = [(game.board.routes[r].a, game.board.routes[r].b) for r in p.routes]
            for tid in p.tickets:
                t = game.board.tickets[tid]
                mark = "OK " if scoring.connected(own, t.a, t.b) else "   "
                self.tickets.insert("end", f"{mark}{t.a} -> {t.b} ({t.points})")
        self.hint.configure(text=_hint(s.phase.value, show))

        can_face = {a.slot for a in legal if isinstance(a, DrawFaceUp)}
        for i, b in enumerate(self.face_buttons):
            c = s.deck.face_up[i]
            if c is None:
                b.configure(text="-", bg="#cccccc", state="disabled")
            else:
                b.configure(text=c, bg=CARD_COLORS[c], fg=card_fg(c), activebackground=CARD_COLORS[c],
                            state="normal" if i in can_face and show else "disabled")
        n = len(s.deck.draw_pile) + len(s.deck.discard)
        self.deck_btn.configure(text=f"Deck ({n})",
                                state="normal" if any(isinstance(a, DrawBlind) for a in legal) and show else "disabled")
        self.tickets_btn.configure(text=f"Tickets ({len(s.ticket_deck)})",
                                   state="normal" if any(isinstance(a, DrawTickets) for a in legal) and show else "disabled")
        can_station = any(isinstance(a, BuildStation) for a in legal) and show
        self.station_btn.configure(text="Station: click a city" if station_mode else "Build station",
                                   state="normal" if (can_station or station_mode) else "disabled")
        self._update_scoreboard(s)
        self.log.configure(state="normal")
        self.log.delete("1.0", "end")
        self.log.insert("end", "\n".join(s.log[-40:]))
        self.log.see("end")
        self.log.configure(state="disabled")

    def _update_scoreboard(self, s):
        for w in self.board_frame.winfo_children():
            w.destroy()
        for i, p in enumerate(s.players):
            marker = ">" if i == s.current else " "
            tk.Label(self.board_frame, fg=p.color, anchor="w", font=("Helvetica", 10, "bold"),
                     text=f"{marker} {p.name}: {p.score} pts | {p.trains} trains | "
                          f"{sum(p.hand.values())} cards | {len(p.tickets)} tickets").pack(anchor="w")


def _hint(phase, show):
    if not show:
        return ""
    return {
        "turn": "Your turn: draw cards, claim a route (click it on the map), draw tickets or build a station.",
        "second_draw": "Draw your second card (a face-up locomotive is not allowed).",
        "setup_tickets": "Choose your starting tickets.",
        "choose_tickets": "Choose which tickets to keep.",
        "tunnel": "Tunnel: pay the extra cards or give up.",
        "game_over": "Game over.",
    }.get(phase, "")
