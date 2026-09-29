from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


def _init(layer: nn.Linear, std: float = float(np.sqrt(2))) -> nn.Linear:
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, 0.0)
    return layer


def _mlp(inp: int, out: int, out_std: float) -> nn.Sequential:
    return nn.Sequential(
        _init(nn.Linear(inp, 128)), nn.Tanh(),
        _init(nn.Linear(128, 128)), nn.Tanh(),
        _init(nn.Linear(128, 64)), nn.Tanh(),
        _init(nn.Linear(64, out), out_std),
    )


class Actor(nn.Module):
    def __init__(self, obs_dim: int = 26, act_dim: int = 9):
        super().__init__()
        self.net = _mlp(obs_dim, act_dim, 0.01)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


class Critic(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.net = _mlp(in_dim, 1, 1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class RunningMeanStd:
    def __init__(self, eps: float = 1e-4):
        self.mean, self.var, self.count = 0.0, 1.0, eps

    def update(self, x: np.ndarray) -> None:
        bm, bv, bc = float(x.mean()), float(x.var()), x.size
        delta, tot = bm - self.mean, self.count + bc
        self.mean += delta * bc / tot
        m2 = self.var * self.count + bv * bc + delta ** 2 * self.count * bc / tot
        self.var, self.count = m2 / tot, tot

    @property
    def std(self) -> float:
        return float(np.sqrt(self.var) + 1e-8)

    def state_dict(self):
        return {"mean": self.mean, "var": self.var, "count": self.count}

    def load_state_dict(self, d):
        self.mean, self.var, self.count = d["mean"], d["var"], d["count"]
