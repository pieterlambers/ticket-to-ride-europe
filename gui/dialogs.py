"""Modal dialogs: game setup, ticket choice, payment choice, tunnel, game over."""
import tkinter as tk
from tkinter import ttk

from ttr.state import PLAYER_COLORS

from .map_canvas import CARD_COLORS


def _modal(root, title, closable=False):
    win = tk.Toplevel(root)
    win.title(title)
    win.transient(root)
    if not closable:
        win.protocol("WM_DELETE_WINDOW", lambda: None)
    return win


def _show(win, root):
    win.update_idletasks()
    x = root.winfo_rootx() + (root.winfo_width() - win.winfo_width()) // 2
    y = root.winfo_rooty() + (root.winfo_height() - win.winfo_height()) // 3
    win.geometry(f"+{max(x, 0)}+{max(y, 0)}")
    win.grab_set()
    win.wait_visibility()
    win.focus_set()
    root.wait_window(win)


def ask_setup(root, allow_random=False, seed=None):
    """Returns (player_specs, seed) or None if cancelled."""
    win = _modal(root, "Ticket to Ride: Europe - new game", closable=True)
    result = {}
    frm = ttk.Frame(win, padding=14)
    frm.pack()
    ttk.Label(frm, text="Number of players").grid(row=0, column=0, sticky="w")
    count = tk.IntVar(value=2)
    rows = []
    kinds = ["Human"] + (["Computer (random)"] if allow_random else [])

    def rebuild(*_):
        for name, kind, swatch in rows:
            name.grid_remove(); kind.grid_remove(); swatch.grid_remove()
        for i in range(count.get()):
            for w in rows[i]:
                w.grid()

    spin = ttk.Spinbox(frm, from_=2, to=5, textvariable=count, width=4, command=rebuild)
    spin.grid(row=0, column=1, sticky="w")
    for i in range(5):
        swatch = tk.Label(frm, width=2, bg=PLAYER_COLORS[i])
        name = ttk.Entry(frm, width=18)
        name.insert(0, f"Player {i + 1}")
        kind = ttk.Combobox(frm, values=kinds, state="readonly", width=18)
        kind.current(0)
        swatch.grid(row=i + 1, column=0, pady=3)
        name.grid(row=i + 1, column=1, pady=3, padx=4)
        kind.grid(row=i + 1, column=2, pady=3)
        rows.append((name, kind, swatch))
    if not allow_random:
        ttk.Label(frm, text="Computer players: coming later", foreground="#777").grid(
            row=6, column=0, columnspan=3, sticky="w", pady=(6, 0))
    ttk.Label(frm, text="Seed (optional)").grid(row=7, column=0, sticky="w", pady=(8, 0))
    seed_var = tk.StringVar(value="" if seed is None else str(seed))
    ttk.Entry(frm, textvariable=seed_var, width=10).grid(row=7, column=1, sticky="w", pady=(8, 0))

    def start():
        specs = []
        for i in range(count.get()):
            n = rows[i][0].get().strip() or f"Player {i + 1}"
            specs.append((n, "random" if rows[i][1].get().startswith("Computer") else "human"))
        sd = seed_var.get().strip()
        result["value"] = (specs, int(sd) if sd.lstrip("-").isdigit() else None)
        win.destroy()

    ttk.Button(frm, text="Start game", command=start).grid(row=8, column=0, columnspan=3, pady=(12, 0))
    rebuild()
    _show(win, root)
    return result.get("value")


def ask_tickets(root, board, ticket_ids, min_keep, who):
    """Returns the frozenset of ticket ids to keep."""
    win = _modal(root, f"{who}: choose destination tickets")
    frm = ttk.Frame(win, padding=14)
    frm.pack()
    ttk.Label(frm, text=f"Keep at least {min_keep} ticket(s):").pack(anchor="w")
    vars_ = {}
    for tid in ticket_ids:
        t = board.tickets[tid]
        v = tk.BooleanVar(value=True)
        vars_[tid] = v
        label = f"{t.a}  ->  {t.b}   ({t.points} pts)" + ("   [long route]" if t.long else "")
        ttk.Checkbutton(frm, text=label, variable=v, command=lambda: update()).pack(anchor="w", pady=2)
    ok = ttk.Button(frm, text="OK")
    ok.pack(pady=(10, 0))
    result = {}

    def update():
        ok.state(["!disabled"] if sum(v.get() for v in vars_.values()) >= min_keep else ["disabled"])

    def done():
        result["keep"] = frozenset(t for t, v in vars_.items() if v.get())
        win.destroy()

    ok.configure(command=done)
    update()
    _show(win, root)
    return result["keep"]


def ask_choice(root, title, text, options, cancellable=True):
    """options: list of (label, value). Returns the chosen value or None."""
    win = _modal(root, title, closable=cancellable)
    frm = ttk.Frame(win, padding=14)
    frm.pack()
    ttk.Label(frm, text=text).pack(anchor="w", pady=(0, 6))
    result = {}
    for label, value in options:
        def pick(v=value):
            result["v"] = v
            win.destroy()
        ttk.Button(frm, text=label, command=pick).pack(fill="x", pady=2)
    _show(win, root)
    return result.get("v")


def ask_tunnel(root, revealed, extra, options):
    """options: list of (label, action). Shows the three revealed cards."""
    win = _modal(root, "Tunnel!")
    frm = ttk.Frame(win, padding=14)
    frm.pack()
    ttk.Label(frm, text="Cards revealed from the deck:").pack(anchor="w")
    row = ttk.Frame(frm)
    row.pack(pady=6)
    for c in revealed:
        tk.Label(row, text=c, width=8, bg=CARD_COLORS[c],
                 fg="white" if c in ("black", "blue", "loco", "red", "green") else "black",
                 relief="raised").pack(side="left", padx=3)
    ttk.Label(frm, text=f"Extra cards needed: {extra}").pack(anchor="w", pady=(0, 6))
    result = {}
    for label, action in options:
        def pick(a=action):
            result["a"] = a
            win.destroy()
        ttk.Button(frm, text=label, command=pick).pack(fill="x", pady=2)
    _show(win, root)
    return result["a"]


def show_game_over(root, players, rows, winners, on_new_game, on_quit):
    win = _modal(root, "Game over", closable=True)
    frm = ttk.Frame(win, padding=14)
    frm.pack()
    heads = ["Player", "Routes", "Tickets", "Stations", "Express", "Total"]
    for c, h in enumerate(heads):
        ttk.Label(frm, text=h, font=("Helvetica", 11, "bold")).grid(row=0, column=c, padx=8)
    for r, row in enumerate(rows, start=1):
        p = players[row["player"]]
        tk.Label(frm, text=p.name, fg=p.color, font=("Helvetica", 11, "bold")).grid(row=r, column=0, padx=8)
        for c, key in enumerate(["routes", "tickets", "stations", "express", "total"], start=1):
            ttk.Label(frm, text=str(row[key])).grid(row=r, column=c, padx=8)
    names = " & ".join(players[i].name for i in winners)
    ttk.Label(frm, text=f"Winner: {names}", font=("Helvetica", 14, "bold")).grid(
        row=len(rows) + 1, column=0, columnspan=6, pady=10)
    btns = ttk.Frame(frm)
    btns.grid(row=len(rows) + 2, column=0, columnspan=6)
    ttk.Button(btns, text="New game", command=lambda: (win.destroy(), on_new_game())).pack(side="left", padx=6)
    ttk.Button(btns, text="Quit", command=lambda: (win.destroy(), on_quit())).pack(side="left", padx=6)
    _show(win, root)
