"""The game engine: setup, applying validated actions, turn flow and end of game."""
import copy
import random
from collections import Counter

from . import rules, scoring
from .actions import (
    BuildStation, ClaimRoute, DeclineTunnel, DrawBlind, DrawFaceUp, DrawTickets,
    KeepTickets, Pass, PayTunnel,
)
from .board import Board, load_board
from .cards import LOCO, Deck
from .state import PLAYER_COLORS, GameState, Phase, PlayerState, TunnelPending

TRAIN_THRESHOLD = 2


class IllegalMove(Exception):
    pass


class Game:
    def __init__(self, board: Board, state: GameState, rng: random.Random):
        self.board = board
        self.state = state
        self.rng = rng

    # --- setup -----------------------------------------------------------
    @classmethod
    def new(cls, player_specs, seed=None, board=None):
        """player_specs: list of (name, kind) with 2-5 entries."""
        if not 2 <= len(player_specs) <= 5:
            raise ValueError("2 to 5 players")
        board = board or load_board()
        rng = random.Random(seed)
        deck = Deck(rng, deal_face_up=False)
        players = [
            PlayerState(name, PLAYER_COLORS[i], kind) for i, (name, kind) in enumerate(player_specs)
        ]
        for p in players:
            for _ in range(4):
                p.hand[deck.draw()] += 1
        deck.refill_face_up()
        longs = [t.id for t in board.tickets if t.long]
        regular = [t.id for t in board.tickets if not t.long]
        rng.shuffle(longs)
        rng.shuffle(regular)
        state = GameState(players=players, deck=deck, ticket_deck=regular)
        game = cls(board, state, rng)
        game._long_pool = longs
        game._offer_setup_tickets(0)
        return game

    def _offer_setup_tickets(self, i):
        s = self.state
        s.setup_player = i
        offer = [self._long_pool.pop()] + [s.ticket_deck.pop(0) for _ in range(3)]
        s.pending_tickets = offer

    # --- accessors -------------------------------------------------------
    @property
    def actor(self) -> int:
        return rules.actor_index(self)

    def legal_actions(self):
        return rules.legal_actions(self)

    def view_for(self, i):
        """A copy of the game with the other players' private information hidden."""
        g = copy.deepcopy(self, {id(self.board): self.board})  # the board is static: share it
        s = g.state
        for j, p in enumerate(s.players):
            if j != i:
                p.hand = Counter({"hidden": sum(p.hand.values())})
                p.tickets = [-1] * len(p.tickets)
        s.deck.draw_pile = ["hidden"] * len(s.deck.draw_pile)
        s.ticket_deck = [-1] * len(s.ticket_deck)
        if i != g.actor:
            s.pending_tickets = []
        return g

    # --- applying actions ------------------------------------------------
    def apply(self, action):
        ok, reason = rules.validate(self, action)
        if not ok:
            raise IllegalMove(reason)
        s = self.state
        p = s.players[s.current]
        if isinstance(action, KeepTickets):
            self._keep_tickets(action)
        elif isinstance(action, DrawFaceUp):
            card = s.deck.take_face_up(action.slot)
            p.hand[card] += 1
            self._log(f"{p.name} took a face-up {card} card")
            self._after_draw(card, first=s.phase == Phase.TURN)
        elif isinstance(action, DrawBlind):
            card = s.deck.draw()
            p.hand[card] += 1
            self._log(f"{p.name} drew a card from the deck")
            self._after_draw(card, first=s.phase == Phase.TURN, blind=True)
        elif isinstance(action, DrawTickets):
            n = min(3, len(s.ticket_deck))
            s.pending_tickets = [s.ticket_deck.pop(0) for _ in range(n)]
            s.phase = Phase.CHOOSE_TICKETS
            self._log(f"{p.name} drew destination tickets")
        elif isinstance(action, ClaimRoute):
            self._claim(action)
        elif isinstance(action, BuildStation):
            self._build_station(action)
        elif isinstance(action, PayTunnel):
            self._pay_tunnel(action)
        elif isinstance(action, DeclineTunnel):
            t = s.pending_tunnel
            s.pending_tunnel = None
            self._log(f"{p.name} gave up on the tunnel")
            self._end_turn()
        elif isinstance(action, Pass):
            self._log(f"{p.name} passed")
            self._end_turn()

    def _log(self, msg):
        self.state.log.append(msg)

    # --- tickets -----------------------------------------------------------
    def _keep_tickets(self, action):
        s = self.state
        keep = set(action.ticket_ids)
        setup = s.phase == Phase.SETUP_TICKETS
        idx = s.setup_player if setup else s.current
        p = s.players[idx]
        p.tickets.extend(t for t in s.pending_tickets if t in keep)
        for t in s.pending_tickets:
            if t not in keep and not self.board.tickets[t].long:
                s.ticket_deck.append(t)  # bottom of the deck; unused long tickets leave the game
        s.pending_tickets = []
        self._log(f"{p.name} kept {len(keep)} ticket(s)")
        if setup:
            if idx + 1 < len(s.players):
                self._offer_setup_tickets(idx + 1)
            else:
                s.phase = Phase.TURN
                s.current = 0
        else:
            self._end_turn()

    # --- drawing -----------------------------------------------------------
    def _after_draw(self, card, first, blind=False):
        s = self.state
        if first:
            if card == LOCO and not blind:
                return self._end_turn()
            s.phase = Phase.SECOND_DRAW
            if not any(not rules._check(self, a) for a in rules._candidates(self)):
                self._end_turn()
        else:
            self._end_turn()

    # --- claiming ----------------------------------------------------------
    def _pay(self, p, color, locos, count):
        cards = [LOCO] * locos + ([color] * (count - locos) if color != LOCO else [])
        for c in cards:
            p.hand[c] -= 1
        self.state.deck.discard_cards(cards)

    def _claim(self, a: ClaimRoute):
        s = self.state
        route = self.board.routes[a.route_id]
        p = s.players[s.current]
        if route.tunnel:
            revealed = s.deck.reveal(3)
            extra = sum(1 for c in revealed if c == LOCO or c == a.color)
            s.deck.discard_cards(revealed)
            self._log(f"{p.name} tries tunnel {route.a}-{route.b}: revealed {', '.join(revealed)}; +{extra}")
            if extra:
                s.pending_tunnel = TunnelPending(a.route_id, a.color, a.locos, revealed, extra)
                s.phase = Phase.TUNNEL
                return
        self._pay(p, a.color, a.locos, route.length)
        self._finish_claim(a.route_id)

    def _pay_tunnel(self, a: PayTunnel):
        s = self.state
        t = s.pending_tunnel
        p = s.players[s.current]
        route = self.board.routes[t.route_id]
        s.pending_tunnel = None
        self._pay(p, t.color, t.locos + a.locos, route.length + t.extra)
        self._finish_claim(t.route_id)

    def _finish_claim(self, route_id):
        s = self.state
        route = self.board.routes[route_id]
        p = s.players[s.current]
        p.trains -= route.length
        p.routes.append(route_id)
        p.score += scoring.ROUTE_POINTS[route.length]
        s.route_owner[route_id] = s.current
        self._log(f"{p.name} claimed {route.a}-{route.b} (+{scoring.ROUTE_POINTS[route.length]})")
        self._end_turn()

    def _build_station(self, a: BuildStation):
        s = self.state
        p = s.players[s.current]
        cost = rules.station_cost(p)
        self._pay(p, a.color, a.locos, cost)
        p.stations_left -= 1
        p.stations.append(a.city)
        s.station_owner[a.city] = s.current
        self._log(f"{p.name} built a station in {a.city}")
        self._end_turn()

    # --- turn flow ---------------------------------------------------------
    def _end_turn(self):
        s = self.state
        p = s.players[s.current]
        if s.turns_left is None:
            if p.trains <= TRAIN_THRESHOLD:
                s.turns_left = len(s.players)
                self._log(f"{p.name} has {p.trains} trains left: last round!")
        else:
            s.turns_left -= 1
            if s.turns_left <= 0:
                return self._game_over()
        s.current = (s.current + 1) % len(s.players)
        s.phase = Phase.TURN

    def _game_over(self):
        s = self.state
        s.phase = Phase.GAME_OVER
        s.final_scores = scoring.final_scores(self.board, s)
        self._log("Game over")

    def winners(self):
        return scoring.winners(self.state, self.state.final_scores)
