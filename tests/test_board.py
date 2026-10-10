import csv
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _rows(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


FULL = _rows(ROOT / "ttr" / "data" / "routes_full.csv")
OLD = _rows(ROOT / "routes.csv")
CITIES = {r["City"] for r in _rows(ROOT / "cities.csv")}


def pair(a, b):
    return frozenset((a, b))


def test_counts():
    assert len(FULL) == 101
    doubles = Counter(r["double_id"] for r in FULL if r["double_id"])
    assert len(doubles) == 11
    assert all(n == 2 for n in doubles.values())


def test_cities_exist():
    for r in FULL:
        assert r["source"] in CITIES and r["target"] in CITIES, r


def test_pairs_match_old_csv():
    new = {pair(r["source"], r["target"]) for r in FULL}
    old = {pair(r["Source"], r["Target"]) for r in OLD}
    old = (old - {pair("Palermo", "Athina")}) | {pair("Palermo", "Smyrna")}
    assert new == old


def test_attributes_match_old_csv():
    old = {pair(r["Source"], r["Target"]): r for r in OLD}
    old[pair("Palermo", "Smyrna")] = old.pop(pair("Palermo", "Athina"))
    for r in FULL:
        o = old[pair(r["source"], r["target"])]
        # routes.csv wrongly lists Berlin-Danzic as 3; the board (map.jpg) shows 4 carriages
        if pair(r["source"], r["target"]) == pair("Berlin", "Danzic"):
            assert r["length"] == "4"
        else:
            assert r["length"] == o["Carriages"], r
        assert r["tunnel"] == o["Tunnel"], r
        assert r["engines"] == o["Engine"], r
        is_grey = r["color"] == "grey"
        if pair(r["source"], r["target"]) == pair("Zurich", "Marseille"):
            assert not is_grey
        else:
            assert is_grey == (o["Colored"] == "false"), r


def test_lengths():
    assert {int(r["length"]) for r in FULL} <= {1, 2, 3, 4, 6, 8}
