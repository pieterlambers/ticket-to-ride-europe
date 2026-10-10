# Analysing Ticket to ride: Europe

This repository contains the data of the map and an analysis of the routes.

To import the data, download the files:
- ```cities.csv```
- ```destinations.csv```
- ```routes.csv```

You can also read my analysis in the file [analysis.ipynb](https://github.com/leonsi7/les-aventuries-du-rail/blob/main/analysis.ipynb) You need to know the game and its rules tu understand my analysis.

Enjoy your reading ;)

## Play the game

A Tkinter GUI for 2-5 players on one computer (hot-seat). Every move goes through a rules validator, so only legal moves are possible.

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python play.py            # add --allow-random to enable random computer players
.venv/bin/pytest                    # run the tests
```

- `ttr/` is the game logic (no GUI code): `rules.py` is the validator, `engine.py` applies actions, `scoring.py` does final scoring.
- `gui/` is the Tkinter interface. It only submits actions to the engine.
- `ttr/data/routes_full.csv` is the complete route table (colours and double routes), built from `map.jpg`.
- A computer player only has to implement `Player.choose(game_view, legal_actions)` in `ttr/players.py`.

## Computer opponents & training

New-game dialog options: **Computer (heuristic)** (rule-based) and **Computer (AI)** (neural network, needs `models/best.pt`).

- `ttr/ai/heuristic.py`: plans the cheapest routes for its tickets, collects the colours it needs, claims planned routes.
- `ttr/ai/model.py`, `ttr/ai/neural.py`: policy/value network that scores each legal move; `ttr/ai/features.py` encodes the game (own hand and tickets plus public info only).
- `models/best.pt` is the committed checkpoint. Training logs and data go to `runs/` (ignored by git).

```bash
.venv/bin/python -m train.evaluate --a heuristic --b random --games 500   # win-rate table
.venv/bin/python -m train.imitate --games 20000 --epochs 5 --out models/bc.pt   # learn from the heuristic
.venv/bin/python -m train.ppo --init models/bc.pt --iters 300               # self-play improvement
.venv/bin/python -m train.evaluate --a neural --b heuristic --games 400
```

Current strength (2 players): heuristic beats random 100%; the AI beats random 100% and the heuristic about 56%.
