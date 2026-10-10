"""Print random 2-player games per second."""
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from ttr.ai.arena import play_game  # noqa: E402
from ttr.players import RandomPlayer  # noqa: E402

N = 100
t = time.time()
for s in range(N):
    play_game([RandomPlayer(s * 10), RandomPlayer(s * 10 + 1)], seed=s)
print(f"{N / (time.time() - t):.1f} random games/s (full-state arena)")
