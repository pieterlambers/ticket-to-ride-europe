"""Choices shared by the computer players. (Macro actions are added in milestone M2.)"""
from ttr.cards import LOCO


def best_payment(options, hand, plan_need):
    """Pick one payment among legal claim/station actions for the same target.

    Fewest locomotives first; then spend the colour we have the most spare cards of
    (cards still needed by the plan count as not spare)."""
    def key(a):
        spare = 0 if a.color == LOCO else hand[a.color] - plan_need.get(a.color, 0)
        return (a.locos, -spare)

    return min(options, key=key)
