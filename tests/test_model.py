import torch

from ttr.ai.model import PolicyValueNet, load_checkpoint, save_checkpoint


def test_separate_value_head_roundtrip(tmp_path):
    net = PolicyValueNet(10, 4, hidden=16)
    states, acts, owner = torch.randn(3, 10), torch.randn(5, 4), torch.tensor([0, 0, 1, 2, 2])
    before = net(states, acts, owner)
    net.add_separate_value()
    after = net(states, acts, owner)
    assert torch.allclose(before[0], after[0]) and torch.allclose(before[1], after[1])  # copy changes nothing
    path = tmp_path / "m.pt"
    save_checkpoint(net, path)
    loaded = load_checkpoint(path)
    assert loaded.separate_value
    assert torch.allclose(loaded(states, acts, owner)[1], after[1])


def test_old_checkpoint_has_shared_encoder(tmp_path):
    net = PolicyValueNet(10, 4, hidden=16)
    path = tmp_path / "m.pt"
    save_checkpoint(net, path)
    assert not load_checkpoint(path).separate_value
