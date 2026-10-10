"""Actions a player can submit to the engine. All frozen so they can be compared/hashed."""
from dataclasses import dataclass


@dataclass(frozen=True)
class DrawFaceUp:
    slot: int


@dataclass(frozen=True)
class DrawBlind:
    pass


@dataclass(frozen=True)
class DrawTickets:
    pass


@dataclass(frozen=True)
class KeepTickets:
    ticket_ids: frozenset


@dataclass(frozen=True)
class ClaimRoute:
    route_id: int
    color: str  # the one non-loco colour used, or "loco" when paying with locomotives only
    locos: int  # number of locomotives in the payment


@dataclass(frozen=True)
class BuildStation:
    city: str
    color: str
    locos: int


@dataclass(frozen=True)
class PayTunnel:
    locos: int  # how many of the extra tunnel cards are paid with locomotives


@dataclass(frozen=True)
class DeclineTunnel:
    pass


@dataclass(frozen=True)
class Pass:
    """Only legal when the player has no other legal action (avoids a stuck game)."""
