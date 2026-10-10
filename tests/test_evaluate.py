from train import diagnose
from train.evaluate import COMPONENTS, evaluate, win_ci


def test_evaluate_reports_ci_and_components():
    r = evaluate("heuristic", "random", 4, workers=1)
    assert r["games"] == 4
    assert r["ci_low"] <= r["win_rate"] <= r["ci_high"]
    for c in COMPONENTS:
        assert f"a_{c}" in r and f"b_{c}" in r


def test_odd_game_count_rounds_up_to_pairs():
    assert evaluate("heuristic", "random", 3, workers=1)["games"] == 4


def test_win_ci_pairs():
    mean, lo, hi = win_ci([1, 0, 1, 0, 1, 0, 1, 0])  # every pair splits: no variance
    assert mean == 0.5 and lo == hi == 0.5


def test_diagnose_runs():
    for i in range(2):
        out = diagnose.run_one(("heuristic", "random", None, None, i))
        assert out["a_turns"] > 0 and "b_lost" in out
