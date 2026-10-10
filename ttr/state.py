"""Mutable game state. Pure data: all rule logic lives in rules.py / engine.py."""
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum

from .cards import Deck

PLAYER_COLORS = ["#d62728", "#1f77b4", "#2ca02c", "#9467bd", "#ff7f0e"]
START_TRAINS = 45
START_STATIONS = 3


class Phase(Enum):
    SETUP_TICKETS = "setup_tickets"
    TURN = "turn"
    SECOND_DRAW = "second_draw"
    CHOOSE_TICKETS = "choose_tickets"
    TUNNEL = "tunnel"
    GAME_OVER = "game_over"


@dataclass
class PlayerState:
    name: str
    color: str
    kind: str = "human"
    hand: Counter = field(default_factory=Counter)
    trains: int = START_TRAINS
    stations_left: int = START_STATIONS
    stations: list = field(default_factory=list)  # city names
    routes: list = field(default_factory=list)  # route ids
    tickets: list = field(default_factory=list)  # ticket ids
    score: int = 0  # route points so far


@dataclass
class TunnelPending:
    route_id: int
    color: str  # base payment colour ("loco" if locomotives only)
    locos: int  # locomotives in the base payment
    revealed: list
    extra: int


@dataclass
class GameState:
    players: list
    deck: Deck
    ticket_deck: list  # regular ticket ids, top first
    phase: Phase = Phase.SETUP_TICKETS
    current: int = 0
    setup_player: int = 0
    pending_tickets: list = field(default_factory=list)
    route_owner: dict = field(default_factory=dict)  # route id -> player index
    station_owner: dict = field(default_factory=dict)  # city -> player index
    pending_tunnel: TunnelPending | None = None
    turns_left: int | None = None  # set when the final round starts
    final_scores: list | None = None
    log: list = field(default_factory=list)
