"""Training and evaluation of the SoundBridge Encoder (CPU only).

Evaluation is performed on *held-out artists*: no track of a test artist is seen during
training, so retrieval scores measure generalisation to unseen music rather than
memorisation.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.neighbors import NearestNeighbors

from .model import SoundBridgeEncoder, encode


@dataclass
class TrainResult:
    model: SoundBridgeEncoder
    history: list[dict] = field(default_factory=list)
    train_idx: np.ndarray | None = None
    test_idx: np.ndarray | None = None
    seconds: float = 0.0


def artist_split(artist_codes: np.ndarray, test_fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    artists = np.unique(artist_codes)
    test_artists = rng.choice(artists, size=int(len(artists) * test_fraction), replace=False)
    is_test = np.isin(artist_codes, test_artists)
    return np.where(~is_test)[0], np.where(is_test)[0]


class _PositiveSampler:
    """Samples, for each anchor, another track by the same primary artist when one exists."""

    def __init__(self, artist_codes: np.ndarray, rng: np.random.Generator):
        self.rng = rng
        order = np.argsort(artist_codes, kind="stable")
        sorted_codes = artist_codes[order]
        starts = np.r_[0, np.flatnonzero(np.diff(sorted_codes)) + 1]
        sizes = np.diff(np.r_[starts, len(sorted_codes)])
        self.order = order
        pos_in_sorted = np.empty_like(order)
        pos_in_sorted[order] = np.arange(len(order))
        group_of = np.repeat(np.arange(len(starts)), sizes)
        self.start = starts[group_of][pos_in_sorted]
        self.size = sizes[group_of][pos_in_sorted]

    def sample(self, anchors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        size = self.size[anchors]
        offs = (self.rng.random(len(anchors)) * size).astype(np.int64)
        pos = self.order[self.start[anchors] + offs]
        has_pair = (size > 1) & (pos != anchors)
        return pos, has_pair


def train_encoder(
    X: np.ndarray,
    genre_targets: np.ndarray,
    popularity: np.ndarray,
    artist_codes: np.ndarray,
    train_idx: np.ndarray,
    *,
    hidden: int = 256,
    emb_dim: int = 32,
    dropout: float = 0.1,
    epochs: int = 40,
    batch_size: int = 1024,
    lr: float = 2e-3,
    weight_decay: float = 1e-4,
    temperature: float = 0.1,
    loss_weights: dict | None = None,
    noise: float = 0.08,
    seed: int = 42,
    verbose: bool = True,
) -> TrainResult:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    lw = {"contrastive": 1.0, "genre": 1.0, "popularity": 0.5, **(loss_weights or {})}
    model = SoundBridgeEncoder(X.shape[1], genre_targets.shape[1], hidden, emb_dim, dropout)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    steps_per_epoch = int(np.ceil(len(train_idx) / batch_size))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * steps_per_epoch)

    Xt = torch.from_numpy(X.astype(np.float32))
    gsum = genre_targets.sum(1, keepdims=True)
    G = torch.from_numpy((genre_targets / np.maximum(gsum, 1)).astype(np.float32))
    has_genre = torch.from_numpy(gsum[:, 0] > 0)
    P = torch.from_numpy(popularity.astype(np.float32))
    A = torch.from_numpy(artist_codes.astype(np.int64))

    # positives are drawn only from training tracks
    local_codes = artist_codes[train_idx]
    sampler = _PositiveSampler(local_codes, rng)

    history = []
    t0 = time.time()
    for epoch in range(epochs):
        model.train()
        perm = rng.permutation(len(train_idx))
        totals = {"loss": 0.0, "contrastive": 0.0, "genre": 0.0, "popularity": 0.0}
        for s in range(steps_per_epoch):
            local = perm[s * batch_size : (s + 1) * batch_size]
            pos_local, has_pair = sampler.sample(local)
            a_idx = torch.from_numpy(train_idx[local])
            p_idx = torch.from_numpy(train_idx[np.where(has_pair, pos_local, local)])
            xa = Xt[a_idx] + noise * torch.randn(len(a_idx), X.shape[1])
            xp = Xt[p_idx] + noise * torch.randn(len(p_idx), X.shape[1])
            za, logits, pop = model(xa)
            zp = model.embed(xp)

            sim = za @ zp.T / temperature
            same_artist = A[a_idx][:, None] == A[p_idx][None, :]
            eye = torch.eye(len(a_idx), dtype=torch.bool)
            sim = sim.masked_fill(same_artist & ~eye, float("-inf"))  # no false negatives
            target = torch.arange(len(a_idx))
            l_con = 0.5 * (F.cross_entropy(sim, target) + F.cross_entropy(sim.T, target))

            hg = has_genre[a_idx]
            l_gen = (
                -(G[a_idx][hg] * F.log_softmax(logits[hg], -1)).sum(-1).mean()
                if hg.any()
                else torch.tensor(0.0)
            )
            pmask = P[a_idx] > 0  # popularity 0 is mostly duplicate-release noise
            l_pop = (
                F.smooth_l1_loss(pop[pmask] / 100.0, P[a_idx][pmask] / 100.0, beta=0.1)
                if pmask.any()
                else torch.tensor(0.0)
            )
            loss = lw["contrastive"] * l_con + lw["genre"] * l_gen + lw["popularity"] * l_pop
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            sched.step()
            for k, v in (("loss", loss), ("contrastive", l_con), ("genre", l_gen), ("popularity", l_pop)):
                totals[k] += float(v.detach())
        row = {"epoch": epoch + 1, **{k: v / steps_per_epoch for k, v in totals.items()}}
        history.append(row)
        if verbose:
            print(
                f"epoch {epoch + 1:3d}  loss {row['loss']:.4f}  con {row['contrastive']:.4f}  "
                f"genre {row['genre']:.4f}  pop {row['popularity']:.4f}  ({time.time() - t0:.0f}s)",
                flush=True,
            )
    model.eval()
    return TrainResult(model=model, history=history, train_idx=train_idx, seconds=time.time() - t0)


def retrieval_metrics(
    space: np.ndarray,
    idx: np.ndarray,
    artist_codes: np.ndarray,
    genre_multi_hot: np.ndarray,
    k: int = 10,
    max_queries: int = 5000,
    seed: int = 0,
) -> dict[str, float]:
    """Nearest-neighbour quality inside a subset (e.g. held-out artists).

    * artist_precision@k - share of the k neighbours by the query's primary artist
    * artist_hit@k       - queries with at least one same-artist neighbour in the top k
    * genre_agreement@k  - share of neighbours sharing at least one genre label
    Queries are restricted to tracks whose artist has another track in the subset.
    """
    sub = space[idx]
    codes = artist_codes[idx]
    gm = genre_multi_hot[idx]
    counts = np.bincount(codes)
    eligible = np.where(counts[codes] > 1)[0]
    rng = np.random.default_rng(seed)
    q = rng.choice(eligible, size=min(max_queries, len(eligible)), replace=False)
    nn = NearestNeighbors(n_neighbors=k + 1, metric="cosine", algorithm="brute").fit(sub)
    _, nbrs = nn.kneighbors(sub[q])
    nbrs = nbrs[:, 1:]  # drop the query itself
    same_artist = codes[nbrs] == codes[q][:, None]
    genre_overlap = (gm[nbrs] * gm[q][:, None, :]).sum(-1) > 0
    # chance level for genre agreement: random neighbours from the subset
    rand = rng.integers(0, len(idx), size=(len(q), k))
    chance_genre = ((gm[rand] * gm[q][:, None, :]).sum(-1) > 0).mean()
    return {
        f"artist_precision@{k}": float(same_artist.mean()),
        f"artist_hit@{k}": float(same_artist.any(1).mean()),
        f"genre_agreement@{k}": float(genre_overlap.mean()),
        f"genre_agreement_chance@{k}": float(chance_genre),
        "n_queries": int(len(q)),
    }


def evaluate_heads(model: SoundBridgeEncoder, X, genre_multi_hot, popularity, idx) -> dict[str, float]:
    _, probs, pred_pop = encode(model, X[idx])
    gm = genre_multi_hot[idx]
    top1 = probs.argmax(1)
    top5 = np.argsort(-probs, 1)[:, :5]
    has = gm.sum(1) > 0
    acc1 = gm[np.arange(len(idx)), top1][has].mean()
    acc5 = (np.take_along_axis(gm, top5, 1).sum(1) > 0)[has].mean()
    pop = popularity[idx]
    m = pop > 0
    return {
        "genre_top1_accuracy": float(acc1),
        "genre_top5_accuracy": float(acc5),
        "popularity_r2": float(r2_score(pop[m], pred_pop[m])),
        "popularity_mae": float(mean_absolute_error(pop[m], pred_pop[m])),
        "popularity_mae_mean_baseline": float(np.abs(pop[m] - pop[m].mean()).mean()),
    }
