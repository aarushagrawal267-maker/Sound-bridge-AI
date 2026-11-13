import pytest

from src.feedback import (
    FeedbackEvent, FeedbackStore, PersonalPreferenceModel, SessionFeedback, StudyStore, adjust_weights, apply_event,
    feedback_summary,
)
from src.hybrid_recommender import RankingWeights


def _ev(i, kind, artist="X", strategy="hybrid"):
    return FeedbackEvent(session_id="s1", track_idx=i, track_id=f"t{i}", feedback=kind, artist=artist, strategy=strategy, rank=i)


def test_apply_event_updates_state():
    st = SessionFeedback()
    apply_event(st, _ev(1, "like"))
    apply_event(st, _ev(2, "dislike"))
    apply_event(st, _ev(3, "not_my_genre"), primary_genre="metal")
    apply_event(st, _ev(1, "dislike"))  # changing one's mind moves the track
    assert st.liked == []
    assert st.disliked == [2, 3, 1]
    assert st.penalised_genres == ["metal"]
    assert len(st.events) == 4


def test_adjust_weights_bounded():
    st = SessionFeedback(known_artists=list("abcdefghij"), variety_requests=20, more_like_this=1)
    w = adjust_weights(RankingWeights(relevance=0.5, novelty=0.1, diversity=0.1), st)
    assert w.novelty == pytest.approx(0.3)   # +0.2 max
    assert w.diversity == pytest.approx(0.4)  # +0.3 max
    assert w.relevance == pytest.approx(0.54)
    assert w.max_per_artist == 1


def test_store_roundtrip(tmp_path):
    store = FeedbackStore(tmp_path)
    store.record(_ev(1, "like"))
    store.record(_ev(2, "dislike"))
    df = store.load()
    assert list(df["feedback"]) == ["like", "dislike"]
    s = feedback_summary(df)
    assert s["n_events"] == 2 and s["like_rate"] == 0.5
    store.clear()
    assert store.load().empty


def test_store_rejects_unknown_type(tmp_path):
    with pytest.raises(ValueError):
        FeedbackStore(tmp_path).record(_ev(1, "love"))


def test_study_store(tmp_path):
    s = StudyStore(tmp_path)
    s.append({"type": "pilot_response", "preferred": "hybrid"})
    assert s.load().iloc[0]["preferred"] == "hybrid"


def test_empty_summary():
    import pandas as pd

    assert feedback_summary(pd.DataFrame()) == {"n_events": 0}


def test_personal_model_threshold(rng):
    e = rng.normal(size=(100, 4))
    pm = PersonalPreferenceModel(min_labels=20)
    assert not pm.fit(e, list(range(5)), list(range(5, 10)))
    liked = [i for i in range(100) if e[i, 0] > 0][:15]
    disliked = [i for i in range(100) if e[i, 0] <= 0][:15]
    assert pm.fit(e, liked, disliked)
    p = pm.predict(e, [liked[0], disliked[0]])
    assert p[liked[0]] > p[disliked[0]]


def test_feedback_changes_recommendations(engine, seeds):
    from src.engine import FeedbackState
    from src.hybrid_recommender import Recommender

    rec = Recommender(engine)
    base = rec.recommend(engine.profile(seeds), "hybrid", k=10)
    top = int(base.iloc[0]["idx"])
    after = rec.recommend(engine.profile(seeds, FeedbackState(disliked=[top])), "hybrid", k=10)
    assert top not in set(after["idx"])
