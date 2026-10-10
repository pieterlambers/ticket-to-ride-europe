"""Headless-ish GUI smoke test: random bots play a whole game through the real widgets.

Run:  .venv/bin/python tests/smoke_gui.py   (needs a display; not collected by pytest)
"""
import sys
import time
import tkinter as tk
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gui.app import App  # noqa: E402
from ttr.state import Phase  # noqa: E402


def run(n=3, seed=1):
    root = tk.Tk()
    app = App(root, bot_delay_ms=0)
    done = []
    app.on_game_over = lambda: done.append(True)
    app.start([(f"Bot{i}", "random") for i in range(n)], seed=seed)
    t0 = time.time()
    while not done and time.time() - t0 < 120:
        root.update()
    assert app.game.state.phase == Phase.GAME_OVER, "game did not finish"
    print("finished", app.game.state.final_scores, "winners", app.game.winners())
    root.destroy()


if __name__ == "__main__":
    run()
