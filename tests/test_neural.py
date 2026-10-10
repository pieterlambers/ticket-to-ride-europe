import pytest

torch = pytest.importorskip("torch")

from ttr.ai import features  # noqa: E402
from ttr.ai.arena import play_game  # noqa: E402
from ttr.ai.model import PolicyValueNet, pad_logits, save_checkpoint  # noqa: E402
from ttr.ai.neural import NeuralPlayer  # noqa: E402


def test_pad_logits():
    logits = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    owner = torch.tensor([0, 0, 1, 2, 2, 2])
    padded, pos = pad_logits(logits, owner, 3)
    assert padded.shape == (3, 3)
    assert padded[0].tolist()[:2] == [1.0, 2.0] and padded[0, 2] == float("-inf")
    assert padded[2].tolist() == [4.0, 5.0, 6.0]
    assert pos.tolist() == [0, 1, 0, 0, 1, 2]


def test_untrained_model_plays_a_legal_game(tmp_path):
    ckpt = tmp_path / "net.pt"
    save_checkpoint(PolicyValueNet(features.STATE_DIM, features.ACT_DIM), ckpt)
    for seed in range(3):
        g = play_game([NeuralPlayer(ckpt, seed=seed), NeuralPlayer(ckpt, temperature=1.0, seed=seed)], seed=seed)
        assert g.state.final_scores


def test_missing_checkpoint_falls_back_to_heuristic(tmp_path):
    with pytest.warns(UserWarning):
        bot = NeuralPlayer(tmp_path / "nope.pt")
    assert play_game([bot, bot], seed=0).state.final_scores
