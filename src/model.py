"""SoundBridge Encoder: the neural network at the centre of the platform.

A multi-task MLP that maps a track's audio features to a 32-dimensional, L2-normalised
"sound embedding". It is trained jointly on three tasks using labels that exist in the
real dataset:

* contrastive  - tracks by the same artist should be close (InfoNCE, in-batch negatives)
* genre        - predict the track's genre labels from its embedding
* popularity   - predict popularity from sound (an exposure proxy, not quality)

Only audio/structural features are inputs: artist, genre and popularity are training
targets and are never fed to the encoder, so the embedding describes how a track sounds.
"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


class SoundBridgeEncoder(nn.Module):
    def __init__(self, in_dim: int, n_genres: int, hidden: int = 256, emb_dim: int = 32, dropout: float = 0.1):
        super().__init__()
        self.config = dict(in_dim=in_dim, n_genres=n_genres, hidden=hidden, emb_dim=emb_dim, dropout=dropout)
        self.backbone = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, emb_dim),
        )
        # cosine classifier: genre prototypes live in the embedding space
        self.genre_prototypes = nn.Parameter(torch.randn(max(n_genres, 1), emb_dim) * 0.1)
        self.genre_scale = nn.Parameter(torch.tensor(10.0))
        self.pop_head = nn.Sequential(nn.Linear(emb_dim, 64), nn.GELU(), nn.Linear(64, 1))

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.backbone(x), dim=-1)

    def genre_logits(self, z: torch.Tensor) -> torch.Tensor:
        return self.genre_scale * z @ F.normalize(self.genre_prototypes, dim=-1).T

    def popularity(self, z: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.pop_head(z)).squeeze(-1) * 100.0

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z = self.embed(x)
        return z, self.genre_logits(z), self.popularity(z)

    def save(self, path: str | Path, extra: dict | None = None) -> None:
        torch.save({"config": self.config, "state_dict": self.state_dict(), "extra": extra or {}}, path)

    @classmethod
    def load(cls, path: str | Path) -> tuple["SoundBridgeEncoder", dict]:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        model = cls(**ckpt["config"])
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        return model, ckpt.get("extra", {})


@torch.no_grad()
def encode(model: SoundBridgeEncoder, x, batch_size: int = 8192):
    """Embed, predict genre probabilities and popularity for a numpy input matrix."""
    import numpy as np

    model.eval()
    embs, probs, pops = [], [], []
    for i in range(0, len(x), batch_size):
        xb = torch.from_numpy(np.asarray(x[i : i + batch_size], dtype=np.float32))
        z, logits, pop = model(xb)
        embs.append(z.numpy())
        probs.append(torch.softmax(logits, dim=-1).numpy())
        pops.append(pop.numpy())
    return (
        np.vstack(embs).astype(np.float32),
        np.vstack(probs).astype(np.float32),
        np.concatenate(pops).astype(np.float32),
    )


def similarity_attribution(
    model: SoundBridgeEncoder, x, target_embedding, baseline=None, steps: int = 32
):
    """Integrated gradients of cos(embed(x), target) w.r.t. each input feature.

    Attributions sum (approximately) to sim(x) - sim(baseline). The baseline defaults to
    the all-zero input, which in quantile-normal space is the catalogue's median track.
    """
    import numpy as np

    model.eval()
    x = torch.as_tensor(np.asarray(x, dtype=np.float32))
    base = torch.zeros_like(x) if baseline is None else torch.as_tensor(np.asarray(baseline, dtype=np.float32))
    target = F.normalize(torch.as_tensor(np.asarray(target_embedding, dtype=np.float32)), dim=-1)
    alphas = torch.linspace(0.0, 1.0, steps).view(-1, 1)
    path = (base + alphas * (x - base)).requires_grad_(True)
    sims = (model.embed(path) * target).sum(-1)
    grads = torch.autograd.grad(sims.sum(), path)[0]
    attributions = ((x - base) * grads.mean(0)).detach().numpy()
    with torch.no_grad():
        delta = float((model.embed(x[None]) * target).sum() - (model.embed(base[None]) * target).sum())
    return attributions, delta
