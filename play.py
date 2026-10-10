"""Start the Ticket to Ride: Europe GUI."""
import argparse

from gui.app import main

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=None, help="default seed shown in the setup dialog")
    ap.add_argument("--allow-random", action="store_true", help="enable random computer players (testing)")
    args = ap.parse_args()
    main(allow_random=args.allow_random, seed=args.seed)
