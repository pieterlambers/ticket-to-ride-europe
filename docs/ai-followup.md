# Follow-up: where the Ticket to Ride AI stands and what to try next

Written when work stopped, with branch `ai-m9` as the latest. Read `ttr-ai-handover.md` (in the home folder when it was written) and `docs/ai-strength-plan.md` for the background and the exact numbers.

## What we have now
- `models/best.pt` is the strongest network: the end of one long PPO run (about 3200 iterations in `runs/ppo4`, which is gitignored).
- Strength, 2 players, 1000 games each on fresh deals (paired seeds, seats swapped):

| Opponent | Win rate of `best.pt` | 95% interval |
|---|---|---|
| heuristic (v1) | 0.874 | [0.854, 0.894] |
| the original M4 network | 0.808 | [0.786, 0.830] |
| the run-1 network (iteration 120) | 0.797 | [0.773, 0.821] |
| random (400 games) | 1.000 | |

- The starting point at the beginning of this work was 0.564 against the heuristic.
- Search on top of it (`SearchPlayer`, 12 imagined worlds) scored 0.509 against the plain network, so it was not adopted. It is available as `make_bot("search")` / `make_player("neural_search")` but is not offered in the GUI.

## What made the difference
1. Measurement first (M5): paired deals plus a confidence interval. The old 200-game evaluations could not tell progress from noise. `train/evaluate.py` and `train/diagnose.py` now report the interval and the score split.
2. A cleaner PPO signal (M6): reward +1/-1 for win/loss with a small margin term, no route-points bonus, KL early stop, entropy that fades out, and a separate encoder for the value head.
3. Time. Run 1 (200 iterations) only reached 0.594. The big gains (0.60 to 0.85) came between iterations 200 and 2600 of the continuation run.

## Caveats about the 0.87
- The net trained against the heuristic in 40% of its games. Against it, the heuristic fails about 1.3 tickets per game and scores about 7 ticket points (it was 0.6 and 20 before). The net partly wins by taking the routes the heuristic is committed to. That is legal play, but it makes 0.87 look better than the general strength.
- The 0.80 against older networks is the better yardstick, though those are copies of the heuristic's style too.
- The plateau: from iteration about 400 to 3200 the win rate against the heuristic stayed between 0.77 and 0.86 and the win rate against the previous best stayed near 0.5. Training this way has stopped paying off.
- Nobody has played it as a human in a long session yet. Do that first: `.venv/bin/python play.py`, pick "Computer (AI)".
- A seed overlap: evaluation offset 500000 overlaps the deals of training iteration 5. The effect is small, but use another offset (for example 7654321) next time.

## Why search did not help
Candidate moves were judged by playing the rest of the game with the v1 heuristic for both sides. The real network plays differently and much better, so the simulated continuation says little about the real value of a move. Timing: a searched decision costs about 1 s at 16 worlds.

## Things to try next (in the order I would do them)
1. **Judge candidates with the network.** Let the rollouts use the greedy network instead of the heuristic (`make_bot("neural")` for both seats, or use the value head after a few plies). Time it first: each simulated game gets slower. Try 400 games before a full 1000.
2. **Replace the benchmark.** The heuristic is saturated. Build a league: heuristic, old bests (`models/archive/`), run-1 net, random. Report win rates against all of them. It also catches the "exploits one opponent" problem.
3. **Train against a broader pool.** Lower the heuristic's share, add the archive nets, and add random noise to the heuristic's choices so it cannot be exploited by a single pattern.
4. **Heuristic v2 (M7 in the plan).** Only worth it as a better benchmark and teacher now. Biggest gaps seen: stations are never built (about 12 points of unused station value on both sides), longest-trail play, end-game timing.
5. **More features (M8).** Cost of losing a route, what the opponent is building, longest-trail standing, a per-route embedding. Retraining is needed because the feature size changes.
6. **3 to 5 players (M10).** Only 2 players were ever trained or measured. The bots play legally with more players but are untuned.
7. **Expert iteration.** Once search beats the plain network, train the network on search results.

## How to pick this up
```bash
git checkout ai-m9
.venv/bin/pytest -q                                   # 269 tests
.venv/bin/python play.py                              # play against "Computer (AI)"
.venv/bin/python -m train.evaluate --a neural --b heuristic --games 1000 --seed-offset 7654321
.venv/bin/python -m train.diagnose --a neural --b heuristic --games 1000
# continue training (about 7 seconds per iteration; resume needs the ckpt of the earlier run, which is gitignored)
.venv/bin/python -m train.ppo --init models/best.pt --separate-value --out runs/ppo5 --iters 500
```
- `runs/` and `models/archive/` are not in git. If this is a fresh clone, `models/best.pt` is all you have; train from it with `--init models/best.pt --separate-value`.
- The machine has 18 cores. Training uses 16 workers by default. For long runs keep the Mac awake (`caffeinate -ims`).
- Promotion rule in `train/ppo.py`: a checkpoint replaces `models/best.pt` only when its lower interval bound against the current best is above 0.5 over 600 paired games. The old best is archived first.

## Branches
`ai-m5` (measurement), `ai-m6` (PPO fixes, new `best.pt`), `ai-m9` (search, this file). They are stacked on `ai-m4`. No PR has been opened. M7 and M8 were not done.
