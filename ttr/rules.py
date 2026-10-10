"""The validator. Pure functions: nothing here changes the game."""
from itertools import combinations

from .actions import (
    BuildStation, ClaimRoute, DeclineTunnel, DrawBlind, DrawFaceUp, DrawTickets,
    KeepTickets, Pass, PayTunnel,
)
from .cards import COLORS, LOCO
from .state import Phase

MAX_PLAYERS_DOUBLE_CLOSED = 3  # with <= 3 players only one route of a double route is open


def actor_index(game) -> int:
    s = game.state
    return s.setup_player if s.phase == Phase.SETUP_TICKETS else s.current


def station_cost(player) -> int:
    return 3 - player.stations_left + 1


def _payment_error(player, color, locos, count, allowed_colors, min_locos=0):
    """Check a payment of `count` cards: (count - locos) of `color` plus `locos` locomotives."""
    if locos < 0 or locos > count:
        return "Invalid number of locomotives"
    if color == LOCO:
        if locos != count:
            return "A locomotive-only payment must use only locomotives"
    elif color not in allowed_colors:
        return f"Wrong colour: {color}"
    if locos < min_locos:
        return f"Need at least {min_locos} locomotive(s)"
    if player.hand[LOCO] < locos:
        return "Not enough locomotives"
    if color != LOCO and player.hand[color] < count - locos:
        return f"Not enough {color} cards"
    return ""


def _claim_error(game, a: ClaimRoute) -> str:
    s, board = game.state, game.board
    if not 0 <= a.route_id < len(board.routes):
        return "Unknown route"
    route = board.routes[a.route_id]
    p = s.players[s.current]
    if a.route_id in s.route_owner:
        return "Route already claimed"
    if p.trains < route.length:
        return f"Not enough trains ({p.trains} left, {route.length} needed)"
    twin = board.twin(route)
    if twin is not None and twin.id in s.route_owner:
        if s.route_owner[twin.id] == s.current:
            return "You cannot claim both routes of a double route"
        if len(s.players) <= MAX_PLAYERS_DOUBLE_CLOSED:
            return "Double route closed: only one is open with 2-3 players"
    allowed = COLORS if route.color == "grey" else (route.color,)
    return _payment_error(p, a.color, a.locos, route.length, allowed, route.engines)


def _station_error(game, a: BuildStation) -> str:
    s = game.state
    p = s.players[s.current]
    if a.city not in game.board.cities:
        return "Unknown city"
    if a.city in s.station_owner:
        return "This city already has a station"
    if p.stations_left <= 0:
        return "No stations left"
    return _payment_error(p, a.color, a.locos, station_cost(p), COLORS)


def _tunnel_error(game, a: PayTunnel) -> str:
    s = game.state
    t = s.pending_tunnel
    p = s.players[s.current]
    route = game.board.routes[t.route_id]
    if a.locos < 0 or a.locos > t.extra:
        return "Invalid number of locomotives"
    total_locos = t.locos + a.locos
    if p.hand[LOCO] < total_locos:
        return "Not enough locomotives"
    if t.color == LOCO:
        if a.locos != t.extra:
            return "Only locomotives can pay the extra cost of a locomotive-only payment"
        return ""
    coloured = (route.length - t.locos) + (t.extra - a.locos)
    if p.hand[t.color] < coloured:
        return f"Not enough {t.color} cards"
    return ""


def _check(game, a) -> str:
    """Return '' if the action is valid, otherwise the reason it is not."""
    s = game.state
    ph = s.phase
    if ph == Phase.GAME_OVER:
        return "The game is over"
    p = s.players[actor_index(game)]
    if isinstance(a, (DrawFaceUp, DrawBlind)):
        if ph not in (Phase.TURN, Phase.SECOND_DRAW):
            return "You cannot draw cards now"
        if isinstance(a, DrawBlind):
            return "" if s.deck.available() > 0 else "The deck is empty"
        if not 0 <= a.slot < len(s.deck.face_up) or s.deck.face_up[a.slot] is None:
            return "No card in that slot"
        if ph == Phase.SECOND_DRAW and s.deck.face_up[a.slot] == LOCO:
            return "You cannot take a face-up locomotive as your second card"
        return ""
    if isinstance(a, DrawTickets):
        if ph != Phase.TURN:
            return "You cannot draw tickets now"
        return "" if s.ticket_deck else "No destination tickets left"
    if isinstance(a, KeepTickets):
        if ph not in (Phase.SETUP_TICKETS, Phase.CHOOSE_TICKETS):
            return "No tickets to choose from"
        if not set(a.ticket_ids) <= set(s.pending_tickets):
            return "Those tickets were not offered"
        need = 2 if ph == Phase.SETUP_TICKETS else 1
        need = min(need, len(s.pending_tickets))
        return "" if len(a.ticket_ids) >= need else f"Keep at least {need} ticket(s)"
    if isinstance(a, ClaimRoute):
        return "You cannot claim a route now" if ph != Phase.TURN else _claim_error(game, a)
    if isinstance(a, BuildStation):
        return "You cannot build a station now" if ph != Phase.TURN else _station_error(game, a)
    if isinstance(a, (PayTunnel, DeclineTunnel)):
        if ph != Phase.TUNNEL:
            return "No tunnel to resolve"
        return _tunnel_error(game, a) if isinstance(a, PayTunnel) else ""
    return "Unknown action"


def _payments(player, count, colors, min_locos=0):
    """Candidate (color, locos) payments of `count` cards."""
    out = []
    for color in colors:
        lo = max(min_locos, count - player.hand[color])
        hi = min(count - 1, player.hand[LOCO])
        out.extend((color, k) for k in range(lo, hi + 1))
    if player.hand[LOCO] >= count and count >= min_locos:
        out.append((LOCO, count))
    return out


def _candidates(game):
    s, board = game.state, game.board
    ph = s.phase
    if ph in (Phase.SETUP_TICKETS, Phase.CHOOSE_TICKETS):
        lo = min(2 if ph == Phase.SETUP_TICKETS else 1, len(s.pending_tickets))
        ids = s.pending_tickets
        return [KeepTickets(frozenset(c)) for n in range(lo, len(ids) + 1) for c in combinations(ids, n)]
    if ph == Phase.TUNNEL:
        t = s.pending_tunnel
        return [PayTunnel(k) for k in range(t.extra + 1)] + [DeclineTunnel()]
    if ph in (Phase.TURN, Phase.SECOND_DRAW):
        out = [DrawFaceUp(i) for i in range(len(s.deck.face_up))] + [DrawBlind()]
        if ph == Phase.TURN:
            p = s.players[s.current]
            out.append(DrawTickets())
            for r in board.routes:
                if r.id in s.route_owner or p.trains < r.length:
                    continue
                colors = COLORS if r.color == "grey" else (r.color,)
                out += [ClaimRoute(r.id, c, k) for c, k in _payments(p, r.length, colors, r.engines)]
            if p.stations_left > 0:
                cost = station_cost(p)
                pays = _payments(p, cost, COLORS)
                out += [BuildStation(c, col, k) for c in board.cities if c not in s.station_owner for col, k in pays]
        return out
    return []


def validate(game, action):
    """Return (True, '') if legal, else (False, reason)."""
    if isinstance(action, Pass):
        if game.state.phase not in (Phase.TURN, Phase.SECOND_DRAW):
            return False, "You cannot pass now"
        if any(not _check(game, a) for a in _candidates(game)):
            return False, "You still have legal moves"
        return True, ""
    err = _check(game, action)
    return (not err), err


def legal_actions(game) -> list:
    acts = [a for a in _candidates(game) if not _check(game, a)]
    if not acts and game.state.phase in (Phase.TURN, Phase.SECOND_DRAW):
        return [Pass()]
    return acts


def claimable_routes(game) -> set:
    return {a.route_id for a in legal_actions(game) if isinstance(a, ClaimRoute)}


def explain_route(game, route_id: int) -> str:
    """Human-readable reason why a route cannot be claimed right now ('' if it can)."""
    s = game.state
    if s.phase != Phase.TURN:
        return "You cannot claim a route now"
    route = game.board.routes[route_id]
    p = s.players[s.current]
    if any(isinstance(a, ClaimRoute) and a.route_id == route_id for a in legal_actions(game)):
        return ""
    # Probe with the most generous payment to surface the structural reason first.
    probe = ClaimRoute(route_id, LOCO, route.length)
    err = _claim_error(game, probe)
    if err.startswith(("Route already", "Not enough trains", "You cannot claim both", "Double route")):
        return err
    if route.engines:
        return f"Needs {route.engines} locomotive(s) and {route.length - route.engines} more matching cards"
    colour = "any single colour" if route.color == "grey" else route.color
    return f"Not enough cards: need {route.length} cards of {colour} (locomotives are wild)"


def explain_station(game, city: str) -> str:
    """Human-readable reason why a station cannot be built in `city` ('' if it can)."""
    s = game.state
    if s.phase != Phase.TURN:
        return "You cannot build a station now"
    p = s.players[s.current]
    if city in s.station_owner:
        return "This city already has a station"
    if p.stations_left <= 0:
        return "You have no stations left"
    if any(isinstance(a, BuildStation) and a.city == city for a in legal_actions(game)):
        return ""
    return f"Need {station_cost(p)} card(s) of one colour (locomotives are wild)"
