"""Policy/value network. Actions are scored one by one, so any number of legal moves works."""
import copy

import torch
from torch import nn

VERSION = 1


class PolicyValueNet(nn.Module):
    def __init__(self, state_dim, act_dim, hidden=256, separate_value=False):
        super().__init__()
        self.state_dim, self.act_dim, self.hidden = state_dim, act_dim, hidden
        self.separate_value = False
        self.enc = nn.Sequential(
            nn.Linear(state_dim, 512), nn.ReLU(), nn.LayerNorm(512), nn.Linear(512, hidden), nn.ReLU(),
        )
        self.scorer = nn.Sequential(nn.Linear(hidden + act_dim, 128), nn.ReLU(), nn.Linear(128, 1))
        self.value = nn.Sequential(nn.Linear(hidden, 128), nn.ReLU(), nn.Linear(128, 1))
        if separate_value:
            self.add_separate_value()

    def add_separate_value(self):
        """Give the value head its own encoder (a copy of the shared one), so value noise cannot disturb the policy."""
        if not self.separate_value:
            self.value_enc = copy.deepcopy(self.enc)
            self.separate_value = True

    def forward(self, states, actions, owner):
        """states [B,S], actions [N,A], owner [N] (non-decreasing: the state each action belongs to).
        Returns logits [N] and values [B]."""
        h = self.enc(states)
        logits = self.scorer(torch.cat([h[owner], actions], dim=1)).squeeze(1)
        vh = self.value_enc(states) if self.separate_value else h
        return logits, self.value(vh).squeeze(1)


def pad_logits(logits, owner, batch):
    """Scatter flat logits [N] into [B, maxA] padded with -inf. Returns (padded, position in row)."""
    counts = torch.bincount(owner, minlength=batch)
    starts = torch.cumsum(counts, 0) - counts
    pos = torch.arange(len(owner), device=owner.device) - starts[owner]
    padded = torch.full((batch, int(counts.max())), float("-inf"), device=logits.device)
    padded = padded.index_put((owner, pos), logits)
    return padded, pos


def save_checkpoint(model, path, **extra):
    """`extra` (e.g. iter, optimizer state) is stored alongside the weights and ignored by load_checkpoint."""
    torch.save({"state_dict": model.state_dict(), "state_dim": model.state_dim, "act_dim": model.act_dim,
                "hidden": model.hidden, "separate_value": model.separate_value, "version": VERSION, **extra}, path)


def load_checkpoint(path):
    ck = torch.load(path, map_location="cpu", weights_only=True)
    model = PolicyValueNet(ck["state_dim"], ck["act_dim"], ck.get("hidden", 256), ck.get("separate_value", False))
    model.load_state_dict(ck["state_dict"])
    model.eval()
    return model
