"""Train cards: colours, the deck, the discard pile and the face-up row."""
import random

COLORS = ("red", "orange", "yellow", "green", "blue", "pink", "white", "black")
LOCO = "loco"
CARDS_PER_COLOR = 12
LOCOS = 14
FACE_UP = 5


class Deck:
    def __init__(self, rng: random.Random | None = None, deal_face_up: bool = True):
        self.rng = rng or random.Random()
        self.draw_pile: list[str] = [c for c in COLORS for _ in range(CARDS_PER_COLOR)]
        self.draw_pile += [LOCO] * LOCOS
        self.rng.shuffle(self.draw_pile)
        self.discard: list[str] = []
        self.face_up: list[str | None] = [None] * FACE_UP
        if deal_face_up:
            self.refill_face_up()

    # --- basic pile operations -------------------------------------------
    def available(self) -> int:
        return len(self.draw_pile) + len(self.discard)

    def draw(self) -> str | None:
        if not self.draw_pile and self.discard:
            self.draw_pile = self.discard
            self.discard = []
            self.rng.shuffle(self.draw_pile)
        return self.draw_pile.pop() if self.draw_pile else None

    def reveal(self, n: int) -> list[str]:
        """Draw up to n cards from the deck (tunnel check). Caller discards them."""
        cards = []
        for _ in range(n):
            c = self.draw()
            if c is None:
                break
            cards.append(c)
        return cards

    def discard_cards(self, cards) -> None:
        self.discard.extend(cards)

    # --- face-up row -----------------------------------------------------
    def refill_face_up(self) -> None:
        for i in range(FACE_UP):
            if self.face_up[i] is None:
                self.face_up[i] = self.draw()
        self._apply_loco_rule()

    def _apply_loco_rule(self) -> None:
        for _ in range(5):
            if sum(1 for c in self.face_up if c == LOCO) < 3:
                return
            # Not enough cards to replace the row: leave it as is.
            if self.available() < FACE_UP:
                return
            self.discard.extend(c for c in self.face_up if c is not None)
            self.face_up = [None] * FACE_UP
            for i in range(FACE_UP):
                self.face_up[i] = self.draw()

    def take_face_up(self, slot: int) -> str:
        card = self.face_up[slot]
        assert card is not None
        self.face_up[slot] = self.draw()
        self._apply_loco_rule()
        return card

    # --- bookkeeping -----------------------------------------------------
    def total(self) -> int:
        """Cards in the pile, discard and face-up row (not in hands)."""
        return len(self.draw_pile) + len(self.discard) + sum(c is not None for c in self.face_up)
