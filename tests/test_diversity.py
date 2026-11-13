import numpy as np

from src.diversity_ranking import greedy_rerank, mmr


def _setup():
    # four near-identical tracks by artist "a", and two different tracks by b and c
    v = np.array([[1, 0, 0], [1, 0.01, 0], [1, 0.02, 0], [1, 0.03, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
    rel = np.array([0.99, 0.98, 0.97, 0.96, 0.6, 0.5])
    artists = np.array(["a", "a", "a", "a", "b", "c"])
    return np.arange(6), v, rel, artists


def test_pure_relevance_equals_sorting():
    idx, v, rel, art = _setup()
    r = greedy_rerank(idx, v, rel, art, 4, max_per_artist=None, dup_threshold=1.1)
    assert list(r.idx) == [0, 1, 2, 3]


def test_artist_cap_is_enforced():
    idx, v, rel, art = _setup()
    r = greedy_rerank(idx, v, rel, art, 5, max_per_artist=2, dup_threshold=1.1)
    assert (art[r.idx] == "a").sum() == 2
    assert len(r.idx) == 4  # only 2 + b + c possible


def test_near_duplicates_removed():
    idx, v, rel, art = _setup()
    r = greedy_rerank(idx, v, rel, art, 6, max_per_artist=None, dup_threshold=0.99)
    assert (art[r.idx] == "a").sum() == 1


def test_mmr_diversity_strength():
    idx, v, rel, art = _setup()
    low = mmr(idx, v, rel, art, 3, lam=1.0, max_per_artist=None, dup_threshold=1.1)
    high = mmr(idx, v, rel, art, 3, lam=0.3, max_per_artist=None, dup_threshold=1.1)
    assert set(art[high.idx]) == {"a", "b", "c"}
    assert len(set(art[low.idx])) == 1


def test_scores_follow_formula():
    idx, v, rel, art = _setup()
    nov = np.linspace(0, 1, 6)
    r = greedy_rerank(idx, v, rel, art, 1, novelty=nov, w_rel=0.5, w_nov=0.5, max_per_artist=None)
    j = int(np.argmax(0.5 * rel + 0.5 * nov))
    assert r.idx[0] == j
    assert np.isclose(r.score[0], 0.5 * rel[j] + 0.5 * nov[j])


def test_penalty_demotes():
    idx, v, rel, art = _setup()
    pen = np.array([1, 0, 0, 0, 0, 0], dtype=float)
    r = greedy_rerank(idx, v, rel, art, 1, penalty=pen, w_pen=0.5, max_per_artist=None)
    assert r.idx[0] == 1
