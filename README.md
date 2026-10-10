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
