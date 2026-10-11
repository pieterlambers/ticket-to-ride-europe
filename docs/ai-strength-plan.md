# Plan: make the computer opponent play better (for Sonnet 5.5)

> **Written to be executed by Sonnet 5.5, one milestone at a time.** Do the milestones in order. Each milestone ends with an acceptance check; do not start the next until it passes, or until its stop rule fires.
>
> - **Branches:** `ai-m5`, `ai-m6`, … Each is branched from the previous one; `ai-m5` comes from `ai-m4`. Commit at the end of each milestone, after `.venv/bin/pytest -q` passes. Do not push and do not open PRs unless the user asks.
> - **Commit messages** end with `Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>`.
> - **Never change game rules** in `ttr/rules.py`, `ttr/engine.py`, `ttr/scoring.py`, `ttr/cards.py` or `ttr/state.py`.
> - **Never report a number you did not measure.** Paste the real CLI output into the "Results" section at the end of this file.
> - **Long runs** (PPO, imitation, evaluations over 5 minutes): run them in the background, for example `nohup … > runs/<name>.log 2>&1 &`, and watch the log. Don't block on them.
> - **Before overwriting `models/best.pt`**, copy it to `models/archive/best_<branch>.pt`. That folder is gitignored, which is fine.
> - **Stop rules:** if a milestone's acceptance check fails after the attempts listed, stop. Record the numbers, commit what works and ask the user how to continue. Do not invent extra experiments.
> - This machine has 18 cores. Use `--workers 16`.

## Why the AI is stuck at about 56% (background, verified)
1. **Evaluation is too noisy.** 200 games give about ±3.5 points of standard error. The PPO rule "promote at ≥ 0.55 vs best over 200 games" fires on noise.
2. **PPO gets a weak signal.**
   - The reward is `margin / 100`, so value loss stays flat at about 0.03.
   - Entropy *rose* from 0.3 to 0.9 over training, so the policy got flatter, not sharper.
   - The reward is the margin, not winning.
3. **The net only sees the heuristic's view.**
   - Its features come from one shortest-path plan (`graph.plan`), and payment is fixed (`macro.best_payment`).
   - It has no opponent awareness, no route-loss cost and no longest-trail feature.
4. **The heuristic it copies has gaps:**
   - no European Express play;
   - no contested-route priority;
   - tickets are judged by trains, not turns;
   - no end-game clock;
   - it pays every tunnel.
5. **There is no search.**

## Verified facts about the code (do not re-derive)
- **Game loop:** `ttr/ai/arena.py: play_game(players, seed, n=2, max_steps=3000, game=None)`.
  - Passing `game=` plays an existing `Game` to the end. Search rollouts use this.
  - `make_bot(kind, seed, ckpt)` builds bots by name. GUI kinds go through `ttr/players.py: make_player`, and the labels are in `gui/dialogs.py:40-45` (`kind_by_label`).
- **Copying a game:** `copy.deepcopy(game, {id(game.board): game.board})` takes about 0.07 ms. `Game.view_for(i)` hides:
  - opponent hands, as `Counter({"hidden": n})`;
  - opponent tickets, as `-1`;
  - the draw pile, as `"hidden"`;
  - the ticket deck, as `-1`.
  - It does **not** hide `game._long_pool`, the unused long tickets. The determinizer must resample it.
- **Card conservation:** 110 cards in total: 8 colours × 12 (`cards.COLORS`, `CARDS_PER_COLOR`) plus 14 `LOCO`.
  - Paid and revealed cards always go to `deck.discard`; none leave the game.
  - So the unseen cards = full deck − my hand − non-None face-up cards − discard. Those are exactly the opponents' hands plus the draw pile.
- **Ending:** `engine.TRAIN_THRESHOLD = 2`. When a player ends a turn with ≤ 2 trains, `turns_left = n_players` and the final round starts.
- **Scoring:** `final_scores` rows have the keys `routes`, `tickets`, `completed`, `stations`, `trail`, `express` and `total`. Also `scoring.longest_trail(list of (a, b, length))`, `STATION_POINTS = 4` and `EXPRESS_POINTS = 10`.
- **Public log** (`state.log`, plain strings):
  - `"{name} took a face-up {card} card"`, `"{name} drew a card from the deck"`, `"{name} claimed {a}-{b} (+pts)"`;
  - `"{name} has {n} trains left: last round!"`.
  - Match players by `state.players[j].name`. Arena names are `P0`, `P1`, …
- **Speed:** a heuristic-vs-heuristic game takes about 32 ms, roughly 31 games/s on one core. 128 PPO games per iteration took 6 to 8 s with 7 workers.
- **Planning:** `graph.plan(game, me, extra_tickets=(), extra_stations=(), budget=None)` returns `routes`, `need`, `engines`, `uses`, `done`, `unreachable`, `too_expensive`, `cost` and `costs`. Twin routes come from `graph._twin_ids(board)`.
- **Features:** `features.STATE_DIM` and `ACT_DIM` are module constants. Checkpoints store `state_dim`, `act_dim` and `hidden` (`ttr/ai/model.py: save_checkpoint` / `load_checkpoint`), and `model.VERSION = 1`.
- `models/best.pt` is committed. Everything else under `models/` and `runs/` is gitignored.

---

## M5: trustworthy measurement
1. **`train/evaluate.py`: paired seeds.** In `run_one`, job `i` uses `seed = i // 2` and `seat_a = i % 2`, so both seatings of the same deal are played.
   - `evaluate()` rounds `games` up to an even number.
   - It also returns:
     - `ci_low` and `ci_high`: 95% interval from the per-pair mean wins, `mean ± 1.96 · std(pair_means, ddof=1) / sqrt(n_pairs)`;
     - the mean per-component score for A and B: `routes`, `tickets`, `stations`, `express`.
   - Change the CLI default `--games` to 1000. Print the CI and a component table.
2. **New `train/diagnose.py`.** CLI: `--a`, `--b`, `--games`, `--a-ckpt`, `--b-ckpt` and `--workers`, the same as evaluate, with paired seeds.
   - Wrap each bot in a `Probe` that forwards `uses_full_state` and `choose`. On each of its own `TURN` decisions it records:
     - the turn count;
     - `graph.plan(game, me)["routes"]`, collected into a union set.
   - Report per side:
     - tickets kept, completed and failed;
     - trains left at the end;
     - stations left;
     - game length in own turns;
     - `ends_game`: share of games where this side triggered the last round (parse the "last round!" log line);
     - `lost_routes`: planned routes that ended up owned by the opponent, either the route or its twin.
   - Reuse `run_one`'s structure and the multiprocessing pool from `evaluate.py`. Do not duplicate the CI code; import it.
3. **`train/ppo.py` promotion.** Default `--eval-games` is 600. Promote to `models/best.pt` only if `ci_low > 0.5` against best; archive first, see the rules above.
4. **Tests:** add `tests/test_evaluate.py`.
   - Run `evaluate("heuristic", "random", 4, workers=1)` and check that it returns `ci_low ≤ win_rate ≤ ci_high` and the component keys.
   - Run `diagnose` on 2 games with `workers=1`.
- **Accept:**
  - Run `evaluate --a neural --b heuristic --games 1000`.
  - Run `diagnose` for neural vs heuristic and for heuristic vs heuristic (1000 games each).
  - Paste all three outputs under "Results → M5".
  - Below them, write 3 to 5 bullet points saying where points are lost, for example "express decides 30% of losses" or "lost_routes ≈ 2.1 per game".
  - These findings decide the order of the steps in M7.

## M6: fix the PPO signal (`train/ppo.py`)
1. **Reward.** Add `terminal_reward(my_total, opp_total)`: `sign(margin) + 0.3 * margin / 50`. In `rollout`, use it for `rewards[-1]` instead of `margin / 100`. Keep `margin` and `win` in the returned dict.
2. **Shaping.** Delete the route-points shaping (`shaping` in jobs and in `Learner`).
   - Add `--potential-coef`, default 0.
   - When it is above 0, `Learner` stores `phi = -plan["cost"]/45 + sum(points of plan["done"] tickets)/100` per decision.
   - The step reward is `coef · (phi_next − phi)`, with `phi = 0` after the last decision.
3. **Entropy.** Add `--entropy` (default 0.003), annealed linearly to 0 over `--entropy-decay-iters` (default 150).
4. **KL early stop.** In `ppo_update`, compute `approx_kl = (old_logp − logp).mean()` per minibatch. When it exceeds `1.5 · target_kl` (`--target-kl`, default 0.02), stop the remaining epochs. Log `kl` and `epochs_done`.
5. **Defaults.** `--games-per-iter 512`, `--lr 5e-5`, `--workers 16`.
6. **Opponent pool, per game:**
   - 20% current learner;
   - 40% `--opp-heuristic` (default `"heuristic"`, through `make_bot`);
   - 30% a random checkpoint from the last 10 in `pool_paths`;
   - 10% `models/best.pt`.
7. Add the columns `kl`, `ci_low_vs_heuristic` and `ci_low_vs_best` to `log.csv`.
8. **Run:** `nohup .venv/bin/python -m train.ppo --init models/best.pt --out runs/ppo3 --iters 200 > runs/ppo3.log 2>&1 &` (about 30 s/iteration, roughly 2 hours). Check `log.csv` every 10 iterations.
9. **Only if** value loss has not dropped below 70% of its iteration-10 value by iteration 50:
   - add `separate_value: bool` to `PolicyValueNet`. When true, the value head has its own encoder with the same shape as `enc`, copied from `enc` at init.
   - Store `separate_value` in the checkpoint. `load_checkpoint` reads it with `ck.get("separate_value", False)`.
   - Restart the run once with it.
- **Accept:**
  - entropy does not rise over the run;
  - `evaluate --a neural --a-ckpt <best ckpt> --b heuristic --games 1000` gives a win rate ≥ 0.60.
  - Promote that checkpoint to `models/best.pt`.
- **Stop rule:** at most 2 full runs, the second one with the step 9 change or `--lr 2e-5`. If 0.60 is still not reached, keep the best checkpoint whose `ci_low` vs the old best is above 0.5, record the numbers and continue to M7. M7 does not depend on M6 passing.

## M7: heuristic v2, a stronger teacher
- Add a new file `ttr/ai/heuristic_v2.py` with `class HeuristicV2Player(HeuristicPlayer)`, overriding methods. **Do not edit `heuristic.py`.** v1 stays the baseline.
- Register `"heuristic2"` in `arena.make_bot` and `players.make_player`.
- **Work in increments.** Do the steps in the order the M5 findings suggest; the default order is below.
  - After each step, run `evaluate --a heuristic2 --b heuristic --games 1000`.
  - Keep the step only if the win rate does not drop. Record each number under "Results → M7".
- Put new helpers in `ttr/ai/graph.py`, pure Python and unit-tested in `tests/test_graph.py`.

1. **Contested routes first.**
   - Add `blocked=frozenset()` to `graph.plan`. It sets those route ids, and their twins, to `INF` cost.
   - Add `detour_cost(game, me, plan, rid)`, defined as `new_plan.cost − plan.cost + 2 · points of tickets that become unreachable or too expensive`, capped at 30.
   - Add `opp_cities(game, me)`: cities on any route owned by an opponent, or with an opponent station.
   - The new `_priority` is `ROUTE_POINTS + 0.5 · detour_cost + 3 · (route touches opp_cities)`. Remove v1's ad-hoc bottleneck term only if the evaluation says so.
   - Compute detour costs only for routes in `plan["routes"]`. If a decision takes over 5 ms, cache them per `(frozenset(route_owner.items()), my tickets)`.
2. **End-game clock.**
   - Add `est_turns_left(game)`. Calibrate it first: write `scripts/calibrate_clock.py`, which plays 300 heuristic-vs-heuristic games, logs `(min trains over players, own turns actually left)` and fits a linear rule.
   - Write the fitted coefficients into the function with a comment giving the fit.
   - When `est ≤ 4`: never `DrawTickets`, and claim any affordable planned route before drawing.
   - When `est ≤ 2`: claim the highest `ROUTE_POINTS` affordable route.
   - If my estimated total (route score + done tickets − open tickets + 4 · stations left) beats the opponent's public route score plus 15, and claiming takes me to ≤ 2 trains, claim to trigger the end.
3. **Tickets by turns.**
   - In `_keep`, a ticket is feasible if `cost/2.5 + max(0, missing cards)/2 ≤ est_turns_left − 2`. Here `cost` is the extra plan cost and missing cards = `need − hand`, summed.
   - In `_turn`, draw tickets when every open ticket's remaining plan cost is ≤ 6, `trains ≥ 20`, `est_turns_left ≥ 8` and the deck is not empty. v1 only draws once the plan is empty.
4. **European Express.**
   - Add `trail_gain(game, me, rid)`: `longest_trail` of my routes plus `rid`, minus the current value.
   - When the plan is empty, choose the claim by `ROUTE_POINTS + 0.7 · trail_gain`, instead of v1's `_points_claim`.
   - When the end is near (`est ≤ 6`) and the opponent's public trail is within 6 of mine, add `+5` for any trail gain.
5. **Tunnels and stations.**
   - Pay a tunnel only if the route is planned (or its twin is) or `extra ≤ 1`. Otherwise `DeclineTunnel`.
   - Build a station only if `2 · ticket points saved > STATION_POINTS + cards it costs`.
6. **Optional: opponent colour signal.**
   - Add `opp_faceup_colors(game, me, last=10)`, parsed from `state.log`.
   - In step 1's priority, add `+1` if the opponent recently took the route's colour.
- Add `tests/test_heuristic.py` cases for `heuristic2`:
  - legal moves for 2 to 5 players × 10 seeds, both with full state and with `view_for`;
  - beats random ≥ 0.9 over 100 seeds.
- **Retrain from v2:**
  - Add `--teacher {heuristic,heuristic2}` to `train/imitate.py`. `Recorder` builds the teacher through `make_bot`.
  - Run `imitate --teacher heuristic2 --games 20000 --epochs 5 --out models/bc2.pt`.
  - Then run M6's PPO with `--init models/bc2.pt --opp-heuristic heuristic2 --out runs/ppo4`.
- **Accept:**
  - `heuristic2` vs `heuristic` win rate ≥ 0.60 over 1000 games;
  - the new net vs `heuristic2` ≥ 0.55;
  - the new net vs the old `models/best.pt` has `ci_low > 0.5`.
  - Promote it, and change the GUI's "Computer (heuristic)" label to map to `heuristic2`.
- **Stop rule:** if `heuristic2` is below 0.60 after all 6 steps, record the numbers and ask the user.

## M8: features beyond the heuristic's plan (`ttr/ai/features.py`)
1. **`ROUTE_FEATS` goes from 8 to 12.** Add per route:
   - `detour_cost / 30`, only for routes in my plan, else 0;
   - touches `opp_cities`;
   - `opp_interest`;
   - `trail_gain / 8`.
   - `opp_interest` is defined over the unseen tickets: all ticket ids minus mine. It is the weighted share of those tickets whose cheapest path, from the opponent's point of view, uses the route.
     - Path cost: `shortest_path` with `costs = {r: route_cost(game, opp, r)}`.
     - Weight is 3 if a ticket endpoint is in `opp_cities`, else 1.
     - It reads only public route ownership and trains.
     - Cache it per `frozenset(route_owner.items())` in a module-level dict of at most 256 entries.
2. **`N_SCALARS` grows by 12:**
   - my longest trail / 45 and the opponent's / 45;
   - `est_turns_left / 30`;
   - the opponent's face-up colour picks over the last 10 log lines, 9 values / 3.
3. **Action encoding.**
   - Append a one-hot route id to the claim block (`NR` = 101 values) and a one-hot city id to the station block (47 values). `ACT_DIM` grows by 148.
   - This works as a learned per-route embedding with no model change.
4. **Version check.**
   - Set `model.VERSION = 2`.
   - `NeuralPlayer` checks that the checkpoint's `state_dim` and `act_dim` match `features.STATE_DIM` and `ACT_DIM`. On a mismatch it warns and falls back to `HeuristicV2Player`, so the old `best.pt` never crashes the GUI.
5. **Tests.**
   - Extend `tests/test_features.py`'s leak test: changing the opponent's hand, tickets, `draw_pile` order or `_long_pool` must not change `encode_state`.
   - Shapes stay constant.
   - Print the encode time. Over 5 ms per state plus actions means fixing the cache before training.
6. **Retrain.**
   - Regenerate the BC data, since the feature dims changed: `imitate --teacher heuristic2 --out models/bc3.pt`.
   - Then run PPO (`--init models/bc3.pt --opp-heuristic heuristic2 --out runs/ppo5`).
- **Accept:**
  - BC val top-1 ≥ 0.88;
  - after PPO, vs `heuristic2` ≥ 0.58;
  - the new net's win rate vs `heuristic2` is higher than the M7 net's number in "Results". The old checkpoint can't run on the new features, so compare recorded numbers rather than playing the two nets against each other.
- **Stop rule:** one PPO run. If it does not beat the M7 net, keep M7's `best.pt`, record the numbers and continue to M9. Search works with either net.

## M9: decision-time search with determinized rollouts
1. **New file `ttr/ai/determinize.py`, function `sample_world(game, me, rng) -> Game`.**
   - Always start from `game.view_for(me)`, even when given the full game. This guarantees no leak.
   - Cards:
     - unseen = full deck − my hand − face-up − discard. Assert that its size equals `len(draw_pile)` plus the opponents' hidden counts.
     - Shuffle the unseen cards. Deal each opponent `n` cards as a `Counter`. The rest becomes `draw_pile`.
   - Tickets:
     - unseen = all ids − my tickets − my `pending_tickets`.
     - Give each opponent `len(tickets)` tickets from the unseen set, with at most one long ticket each.
     - The ticket deck is the remaining unseen regular tickets, shuffled and cut to the original length.
     - `_long_pool` is the remaining unseen long tickets.
   - Set `world.rng = world.state.deck.rng = random.Random(rng.random())`.
2. **New file `ttr/ai/search.py`, `class SearchPlayer(uses_full_state=True)`.**
   - Constructor: `__init__(self, ckpt=None, k=4, n=16, rollout="heuristic2", seed=None)`.
   - Refactor `NeuralPlayer` to expose `score(game, legal) -> (macros, logits, plan)`. `choose` keeps its behaviour.
   - **Search only:**
     - `SETUP_TICKETS` and `CHOOSE_TICKETS`: all `keep` options;
     - `TURN`: the top-k macros by logit, when `k > 1` and the gap between the top two logits is < 2.0.
   - Every other decision uses plain `NeuralPlayer`.
   - **For each candidate**, for `j` in `range(n)`:
     1. Sample `world_j` with seed `j`. Use the same worlds for every candidate (common random numbers).
     2. Copy it.
     3. Apply `to_action(world, macro, world.legal_actions(), plan["need"])`.
     4. Play to the end with `play_game([make_bot(rollout, …)] * n_players, seed, game=world)`. Use two separate bot instances, not one shared instance.
     5. Score the result as `sign(margin) + margin/100` for me.
   - Pick the candidate with the highest mean score.
   - Print the time per searched decision once in a debug test.
3. **Wiring:**
   - `make_bot("search", seed, ckpt)` builds the search bot;
   - `make_player("neural_search")` does the same for the GUI;
   - add the GUI label "Computer (AI + search)", only when `models/best.pt` exists.
4. **Tests: `tests/test_search.py`.**
   - Card totals: the sampled world has 110 cards.
   - Opponent hand sizes and ticket counts are kept, and my hand, tickets, route owners and face-up row are unchanged.
   - **No leak:** take two real games that differ only in the opponent's hand and the draw pile order. Under the same seed they produce the same sampled world.
   - `SearchPlayer(n=2, k=2)` plays legal 2- and 3-player games for 3 seeds.
- **Tuning:** pick the largest `n` that keeps the mean time per searched decision ≤ 1 s on one core. Report it.
- **Accept** (1000 games each; expect about 40 minutes with 16 workers; run in the background):
  - `search` vs `neural` ≥ 0.60;
  - `search` vs `heuristic2` ≥ 0.65.
- **Stop rule:** if either check fails at the 1 s budget, try once with `n` doubled, to see whether more budget helps. Record both numbers and stop.

## After M9: stop and report
Summarise the "Results" section for the user. These next steps are **out of scope until the user approves them**:
- ISMCTS with PUCT (policy prior, value leaves);
- expert iteration;
- 3 to 5 player training.

## Results
(Sonnet fills this in per milestone, with the real CLI output.)
### M5
Measured on branch `ai-m5`, 1000 paired games each (`evaluate`, then `diagnose`):

```
neural (A) vs heuristic (B), 1000 games (paired seeds, seats swapped)
  A win rate      0.564   95% CI [0.537, 0.590]   (ties count 1/2)
  mean score      A 106.7   B 97.6   margin +9.1
    routes        A   62.7   B   57.5
    tickets       A   26.4   B   23.4
    stations      A   11.9   B   11.8
    express       A    5.8   B    4.9
  tickets done    A 3.12   B 3.01
  tickets failed  A 0.58   B 0.69

neural (A) vs heuristic (B), 1000 games: A win rate 0.565 [0.538, 0.591]
                               A       B
  tickets kept              3.70    3.70
  tickets completed         3.12    3.01
  tickets failed            0.58    0.69
  trains left               2.26    4.56
  stations left            11.84   11.82
  own turns                42.42   42.31
  ends the game             0.61    0.39
  planned routes lost       1.32    1.46
  points: routes           62.73   57.49
  points: tickets          26.48   23.53
  points: stations         11.84   11.82
  points: express           5.80    4.87

heuristic (A) vs heuristic (B), 1000 games: A win rate 0.500 [0.500, 0.500]
                               A       B
  tickets kept              3.70    3.70
  tickets completed         3.15    3.15
  tickets failed            0.55    0.55
  trains left               3.35    3.34
  stations left            11.82   11.82
  own turns                42.77   42.77
  ends the game             0.50    0.50
  planned routes lost       1.35    1.35
  points: routes           59.70   59.71
  points: tickets          27.27   27.33
  points: stations         11.82   11.82
  points: express           5.36    5.37
```

Findings (these set the order of the M7 steps):
- Neither side builds stations: both end with about 11.8 station points (3 unused stations).
- The net spends its trains (2.3 left vs 4.6 for the heuristic) and ends the game 61% of the time. The heuristic strands trains, so the end-game clock is the first M7 step.
- Planned routes lost are similar (1.3 vs 1.5 per game), so contested-route play is worth less than expected: do it second.
- Both sides fail about 0.6 tickets per game, so ticket choice by turns is third.
- Express: net 5.8 vs 4.9 points, so longest-trail play still has headroom (step 4).
- Heuristic vs heuristic gives exactly 0.500 with identical per-side stats, so the paired evaluation is unbiased.
### M6
Reward = win/loss (+ small margin term), no route-points shaping, KL early stop, entropy 0.003 annealed, 512 games/iter, lr 5e-5, opponent pool 20% latest / 40% heuristic / 30% recent checkpoints / 10% best.
- Run 1 (`runs/ppo3`, 200 iters, shared encoder): `best.pt` went from 0.564 to 0.594 vs heuristic (1000 games, CI [0.569, 0.619]). Value loss stayed about 0.17, so the separate value head was added (`PolicyValueNet.add_separate_value`, `--separate-value`).
- Run 2 (`runs/ppo4`, 200 iters, separate value head) and then an overnight continuation of the same run (`--resume`, up to iteration 3200, 19 promotions). Win rate vs heuristic climbed from 0.60 to about 0.85 and stayed there.
- Final `models/best.pt`, measured on fresh deals (`--seed-offset 500000`, 1000 games unless noted):
  - vs heuristic: **0.874**, CI [0.854, 0.894], score margin +49.2 (net tickets +27.6 vs +6.8, routes 73.2 vs 49.2, express 7.3 vs 3.1).
  - vs the original M4 net: 0.808, CI [0.786, 0.830].
  - vs the run-1 net: 0.797, CI [0.773, 0.821].
  - vs random (400 games): 1.000.
- Caveat: the heuristic only completes 2.4 tickets and fails 1.3 per game against this net (it fails 0.6 against the old net). The net partly wins by blocking the routes the heuristic is committed to. Its wins over the older nets show the gain is not only that. M7 steps 1 (contested routes) and 3 (tickets by turns) should therefore still help the teacher.
### M7
### M8
### M9

