"""Macro actions: the small set of choices a learned player makes, mapped to raw legal actions."""
from dataclasses import dataclass

from ttr.actions import (
    BuildStation, ClaimRoute, DeclineTunnel, DrawBlind, DrawFaceUp, DrawTickets, KeepTickets, Pass, PayTunnel,
)
from ttr.cards import LOCO

KINDS = ("claim", "station", "faceup", "blind", "tickets", "keep", "tunnel_pay", "tunnel_decline", "pass")


@dataclass(frozen=True)
class Macro:
    kind: str
    arg: object = None


def best_payment(options, hand, plan_need):
    """Pick one payment among legal claim/station actions for the same target.

    Fewest locomotives first; then spend the colour we have the most spare cards of
    (cards still needed by the plan count as not spare)."""
    def key(a):
        spare = 0 if a.color == LOCO else hand[a.color] - plan_need.get(a.color, 0)
        return (a.locos, -spare)

    return min(options, key=key)


def from_action(action) -> Macro:
    if isinstance(action, ClaimRoute):
        return Macro("claim", action.route_id)
    if isinstance(action, BuildStation):
        return Macro("station", action.city)
    if isinstance(action, DrawFaceUp):
        return Macro("faceup", action.slot)
    if isinstance(action, DrawBlind):
        return Macro("blind")
    if isinstance(action, DrawTickets):
        return Macro("tickets")
    if isinstance(action, KeepTickets):
        return Macro("keep", frozenset(action.ticket_ids))
    if isinstance(action, PayTunnel):
        return Macro("tunnel_pay", action.locos)
    if isinstance(action, DeclineTunnel):
        return Macro("tunnel_decline")
    if isinstance(action, Pass):
        return Macro("pass")
    raise ValueError(f"unknown action {action!r}")


def macros(game, legal) -> list:
    """Distinct macro choices, in the order of the first legal action producing each."""
    return list(dict.fromkeys(from_action(a) for a in legal))


def to_action(game, macro, legal, plan_need):
    """The raw legal action for a macro; claims and stations pick their payment with best_payment."""
    if macro.kind in ("claim", "station"):
        kind, field = (ClaimRoute, "route_id") if macro.kind == "claim" else (BuildStation, "city")
        options = [a for a in legal if isinstance(a, kind) and getattr(a, field) == macro.arg]
        hand = game.state.players[game.actor].hand
        return best_payment(options, hand, plan_need)
    for a in legal:
        if from_action(a) == macro:
            return a
    raise ValueError(f"{macro} is not legal")
