"""A small gated recurrent network decoder.

One GRU layer reads the features bin by bin and a linear layer maps its hidden
state to behaviour. A GRU keeps a memory of the recent past in its hidden state
(Cho et al., 2014), which is the job the lag window does for the ridge readout
and the dynamics model does for the Kalman filter, but learned and nonlinear.

Training details, each chosen to keep a small network honest on a few minutes
of data:

* Long trials are cut into chunks so every batch has the same length and
  gradients do not have to travel through thousands of steps.
* The last 15 % of training trials are held out to decide when to stop; the
  weights from the best epoch on them are kept.
* Everything is seeded and runs on the CPU by default, so a result can be
  reproduced exactly. The network is small enough that the CPU is also faster
  than a GPU here.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from torch import nn

from neurochain.recording import FloatArray, IntArray


class _Network(nn.Module):
    """GRU followed by a linear readout."""

    def __init__(self, n_features: int, n_targets: int, hidden: int, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(n_features, hidden, batch_first=True)
        self.readout = nn.Linear(hidden, n_targets)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map ``(batch, time, features)`` to ``(batch, time, targets)``."""
        hidden, _ = self.gru(self.dropout(x))
        out: torch.Tensor = self.readout(hidden)
        return out


def _segments(groups: IntArray) -> list[tuple[int, int]]:
    """Return ``[start, stop)`` index pairs of each run of equal group labels."""
    edges = np.flatnonzero(np.diff(groups) != 0) + 1
    bounds = [0, *edges.tolist(), len(groups)]
    return list(itertools.pairwise(bounds))


def _chunks(segments: list[tuple[int, int]], length: int) -> list[tuple[int, int]]:
    """Cut each segment into pieces of at most ``length`` bins."""
    out = []
    for start, stop in segments:
        for a in range(start, stop, length):
            out.append((a, min(a + length, stop)))
    return out


@dataclass
class GruDecoder:
    """A one-layer GRU with a linear readout.

    Attributes:
        hidden: Size of the GRU's hidden state.
        epochs: Largest number of passes over the training data.
        patience: Epochs without improvement on the validation trials before
            training stops.
        learning_rate: Adam step size.
        weight_decay: L2 penalty on the weights.
        dropout: Share of input features zeroed at random during training.
        chunk: Longest sequence, in bins, fed to the network in training.
        batch_size: Sequences per gradient step.
        seed: Seed for initial weights, dropout and batch order.
        device: Torch device; the CPU keeps results exactly reproducible.
    """

    hidden: int = 64
    epochs: int = 60
    patience: int = 8
    learning_rate: float = 3.0e-3
    weight_decay: float = 1.0e-4
    dropout: float = 0.3
    chunk: int = 100
    batch_size: int = 32
    seed: int = 0
    device: str = "cpu"
    _network: Any = field(default=None, repr=False)
    _target_mean: FloatArray = field(default_factory=lambda: np.empty(0), repr=False)
    _target_scale: FloatArray = field(default_factory=lambda: np.empty(0), repr=False)
    epochs_run: int = 0

    def _batch(
        self, x: FloatArray, y: FloatArray, pieces: list[tuple[int, int]]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Pad a list of chunks into ``(inputs, targets, mask)`` tensors."""
        longest = max(b - a for a, b in pieces)
        xb = np.zeros((len(pieces), longest, x.shape[1]), dtype=np.float32)
        yb = np.zeros((len(pieces), longest, y.shape[1]), dtype=np.float32)
        mask = np.zeros((len(pieces), longest, 1), dtype=np.float32)
        for i, (a, b) in enumerate(pieces):
            xb[i, : b - a] = x[a:b]
            yb[i, : b - a] = y[a:b]
            mask[i, : b - a] = 1.0
        return (
            torch.from_numpy(xb).to(self.device),
            torch.from_numpy(yb).to(self.device),
            torch.from_numpy(mask).to(self.device),
        )

    def fit(self, features: FloatArray, targets: FloatArray, groups: IntArray) -> None:
        """Train the network.

        Args:
            features: ``(n_bins, n_features)``, already standardised.
            targets: ``(n_bins, n_targets)`` behaviour.
            groups: ``(n_bins,)`` trial label; the hidden state never crosses a
                trial boundary.
        """
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        self._target_mean = targets.mean(axis=0)
        self._target_scale = targets.std(axis=0) + 1e-12
        y = (targets - self._target_mean) / self._target_scale
        segments = _segments(groups)
        n_validation = max(1, round(0.15 * len(segments)))
        train = _chunks(segments[:-n_validation], self.chunk)
        validation = _chunks(segments[-n_validation:], self.chunk)
        network = _Network(features.shape[1], targets.shape[1], self.hidden, self.dropout)
        network.to(self.device)
        optimiser = torch.optim.Adam(
            network.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )
        xv, yv, mv = self._batch(features, y, validation)
        best_loss, best_state, stale = np.inf, None, 0
        for epoch in range(self.epochs):
            network.train()
            order = rng.permutation(len(train))
            for start in range(0, len(order), self.batch_size):
                pieces = [train[i] for i in order[start : start + self.batch_size]]
                xb, yb, mb = self._batch(features, y, pieces)
                optimiser.zero_grad()
                loss = (((network(xb) - yb) ** 2) * mb).sum() / (mb.sum() * yb.shape[2])
                loss.backward()
                nn.utils.clip_grad_norm_(network.parameters(), 1.0)
                optimiser.step()
            network.eval()
            with torch.no_grad():
                validation_loss = float(
                    (((network(xv) - yv) ** 2) * mv).sum() / (mv.sum() * yv.shape[2])
                )
            self.epochs_run = epoch + 1
            if validation_loss < best_loss - 1e-5:
                best_loss, stale = validation_loss, 0
                best_state = {k: v.detach().clone() for k, v in network.state_dict().items()}
            else:
                stale += 1
                if stale >= self.patience:
                    break
        if best_state is not None:
            network.load_state_dict(best_state)
        network.eval()
        self._network = network

    def predict(self, features: FloatArray, groups: IntArray) -> FloatArray:
        """Decode behaviour, ``(n_bins, n_targets)``; each trial starts from a zero state."""
        out = np.empty((len(features), self._target_mean.size))
        with torch.no_grad():
            for start, stop in _segments(groups):
                x = torch.from_numpy(features[start:stop].astype(np.float32)).to(self.device)
                out[start:stop] = self._network(x[None])[0].cpu().numpy()
        return out * self._target_scale + self._target_mean
