"""Rule-based computer player: plans routes for its tickets, collects the colours it needs
and claims planned routes. Reads only public information plus its own hand and tickets."""
import random
from collections import Counter

from ttr.actions import (
    BuildStation, ClaimRoute, DeclineTunnel, DrawBlind, DrawFaceUp, DrawTickets, KeepTickets, PayTunnel,
)
from ttr.cards import COLORS, LOCO
from ttr.scoring import ROUTE_POINTS
from ttr.state import Phase

from . import graph
from .macro import best_payment

KEEP_MARGIN = 5  # trains kept free when judging whether a ticket fits
FAIL_WEIGHT = 1.5  # an infeasible ticket counts this many times its points against an option
HAND_TICKET_TRAINS = 25  # draw more tickets only with at least this many trains left


class HeuristicPlayer:
    uses_full_state = True  # reads only public info and its own cards, so it may get the real game

    def __init__(self, seed=None):
        self.rng = random.Random(seed)

    def choose(self, game, legal):
        me = game.actor
        s = game.state
        ph = s.phase
        if ph in (Phase.SETUP_TICKETS, Phase.CHOOSE_TICKETS):
            return self._keep(game, me, legal)
        if ph == Phase.TUNNEL:
            pays = [a for a in legal if isinstance(a, PayTunnel)]
            return min(pays, key=lambda a: a.locos) if pays else DeclineTunnel()
        plan = graph.plan(game, me)
        if ph == Phase.TURN:
            action = self._turn(game, me, legal, plan)
            if action is not None:
                return action
        return self._draw(game, me, legal, plan)

    # --- tickets -----------------------------------------------------------
    def _keep(self, game, me, legal):
        p = game.state.players[me]
        budget = max(0, p.trains - KEEP_MARGIN)
        held = set(p.tickets)

        def score(a):
            pl = graph.plan(game, me, extra_tickets=a.ticket_ids, budget=budget)
            bad = set(pl["unreachable"]) | set(pl["too_expensive"])
            pts = 0
            for t in held | set(a.ticket_ids):
                points = game.board.tickets[t].points
                pts += -FAIL_WEIGHT * points if t in bad else points
            return (pts, -len(a.ticket_ids))

        return max((a for a in legal if isinstance(a, KeepTickets)), key=score)

    # --- main turn -----------------------------------------------------------
    def _turn(self, game, me, legal, plan):
        s = game.state
        p = s.players[me]
        claims = [a for a in legal if isinstance(a, ClaimRoute)]
        by_route: dict = {}
        for a in claims:
            by_route.setdefault(a.route_id, []).append(a)
        twins = graph._twin_ids(game.board)

        def planned(rid):
            return rid in plan["routes"] or twins.get(rid) in plan["routes"]

        wanted = [rid for rid in by_route if planned(rid)]
        if wanted:
            return best_payment(by_route[max(wanted, key=lambda r: self._priority(game, plan, r))], p.hand, plan["need"])
        if s.turns_left is not None and by_route:
            return self._points_claim(game, by_route, p, plan)
        if not plan["routes"]:  # every feasible ticket is done
            if p.trains >= HAND_TICKET_TRAINS and any(isinstance(a, DrawTickets) for a in legal):
                return DrawTickets()
            if by_route:
                return self._points_claim(game, by_route, p, plan)
        return self._station(game, me, legal, plan)

    def _priority(self, game, plan, rid):
        board = game.board
        r = board.routes[rid]
        twin = graph._twin_ids(board).get(rid)
        pr = ROUTE_POINTS[r.length]
        if plan["uses"][rid] + (plan["uses"][twin] if twin is not None else 0) >= 2:
            pr += 3
        for city in (r.a, r.b):
            free = sum(1 for x in board.routes_by_city[city] if x.id not in game.state.route_owner)
            if free <= 2:
                pr += 2
                break
        return pr

    def _points_claim(self, game, by_route, p, plan):
        rid = max(by_route, key=lambda r: (ROUTE_POINTS[game.board.routes[r].length], -r))
        return best_payment(by_route[rid], p.hand, plan["need"])

    def _station(self, game, me, legal, plan):
        """Build a station when it makes an unreachable ticket reachable."""
        if not plan["unreachable"]:
            return None
        builds = [a for a in legal if isinstance(a, BuildStation)]
        best_city, best_gain = None, 0
        for city in {a.city for a in builds}:
            after = graph.plan(game, me, extra_stations=(city,))
            bad = set(after["unreachable"]) | set(after["too_expensive"])
            gain = sum(game.board.tickets[t].points for t in plan["unreachable"] if t not in bad)
            if gain > best_gain:
                best_city, best_gain = city, gain
        if best_city is None:
            return None
        return best_payment([a for a in builds if a.city == best_city], game.state.players[me].hand, plan["need"])

    # --- drawing cards -------------------------------------------------------
    def _draw(self, game, me, legal, plan):
        s = game.state
        hand = s.players[me].hand
        need = Counter({c: v for c, v in plan["need"].items() if c != "grey"})
        grey = plan["need"].get("grey", 0)
        if grey:  # a grey route can use whichever colour we already hold most of
            need[max(COLORS, key=lambda c: (hand[c] - need[c], c))] += grey
        has_need = sum(need.values()) > 0
        if has_need:
            wanted = max(COLORS, key=lambda c: (need[c] - hand[c], hand[c], c))
            deficit = need[wanted] - hand[wanted]
        else:
            wanted = max(COLORS, key=lambda c: (hand[c], c))
            deficit = 1 if hand[wanted] > 0 else 0  # build a pile of our best colour for a long route
        faceup = [(a, s.deck.face_up[a.slot]) for a in legal if isinstance(a, DrawFaceUp)]
        if deficit > 0:
            for a, card in faceup:
                if card == wanted:
                    return a
        loco_missing = plan["engines"] - hand[LOCO] > 0
        if (has_need and deficit >= 3) or loco_missing:
            for a, card in faceup:
                if card == LOCO:
                    return a
        if any(isinstance(a, DrawBlind) for a in legal):
            return DrawBlind()
        if faceup:
            return faceup[0][0]
        return legal[0]
