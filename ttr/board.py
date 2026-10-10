"""Static game data: cities, routes and destination tickets."""
import csv
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class City:
    name: str
    lon: float
    lat: float


@dataclass(frozen=True)
class Route:
    id: int
    a: str
    b: str
    length: int
    color: str  # a card colour or "grey"
    tunnel: bool
    engines: int
    double_id: int | None


@dataclass(frozen=True)
class Ticket:
    id: int
    a: str
    b: str
    points: int
    long: bool


@dataclass
class Board:
    cities: dict[str, City]
    routes: list[Route]
    tickets: list[Ticket]
    routes_by_city: dict[str, list[Route]] = field(default_factory=dict)

    def twin(self, route: Route) -> Route | None:
        """The parallel route of a double route, if any."""
        if route.double_id is None:
            return None
        for r in self.routes:
            if r.double_id == route.double_id and r.id != route.id:
                return r
        return None


def _read(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


@cache
def load_board() -> Board:
    cities = {
        r["City"]: City(r["City"], float(r["Longitude"]), float(r["Latitude"]))
        for r in _read(ROOT / "cities.csv")
    }
    routes = [
        Route(
            id=int(r["id"]),
            a=r["source"],
            b=r["target"],
            length=int(r["length"]),
            color=r["color"],
            tunnel=r["tunnel"] == "true",
            engines=int(r["engines"]),
            double_id=int(r["double_id"]) if r["double_id"] else None,
        )
        for r in _read(ROOT / "ttr" / "data" / "routes_full.csv")
    ]
    tickets = [
        Ticket(i, r["Source"], r["Target"], int(r["Points"]), int(r["Points"]) >= 20)
        for i, r in enumerate(_read(ROOT / "destinations.csv"))
    ]
    by_city: dict[str, list[Route]] = {c: [] for c in cities}
    for r in routes:
        by_city[r.a].append(r)
        by_city[r.b].append(r)
    return Board(cities, routes, tickets, by_city)
