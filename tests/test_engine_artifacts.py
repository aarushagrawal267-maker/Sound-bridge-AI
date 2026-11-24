"""The engine must report incomplete or inconsistent artifacts as ArtifactsMissing.

app.py only starts its automatic rebuild for ArtifactsMissing, so any other exception here
(FileNotFoundError, IndexError ...) would show visitors a traceback instead of repairing itself.
"""
import shutil

import numpy as np
import pytest

from conftest import ROOT, _artifacts_ready

pytestmark = pytest.mark.skipif(not _artifacts_ready(), reason="model artifacts not built")


@pytest.fixture
def artifact_dirs(tmp_path, monkeypatch):
    """A private copy of models/ and data/processed/ that the engine reads instead of the real ones."""
    import src.engine as engine_mod

    models, processed = tmp_path / "models", tmp_path / "processed"
    models.mkdir()
    processed.mkdir()
    sources = [(f, models / f.name) for f in (ROOT / "models").iterdir() if f.is_file()]
    sources.append((ROOT / "data/processed/tracks.parquet", processed / "tracks.parquet"))
    for src, dst in sources:
        try:
            dst.symlink_to(src)
        except OSError:  # e.g. Windows without symlink permission
            shutil.copy(src, dst)
    dirs = {"models_dir": models, "processed_dir": processed}
    monkeypatch.setattr(engine_mod, "path", lambda key: dirs[key])
    return models


def test_copied_artifacts_load(artifact_dirs):
    from src.engine import SoundBridgeEngine

    eng = SoundBridgeEngine()
    assert len(eng.embeddings) == len(eng.tracks)


@pytest.mark.parametrize("name", ["coords_2d.npy", "clusters.npy", "feature_pipeline.joblib", "test_idx.npy"])
def test_missing_artifact_is_reported(artifact_dirs, name):
    from src.engine import ArtifactsMissing, SoundBridgeEngine

    (artifact_dirs / name).unlink()
    with pytest.raises(ArtifactsMissing, match=name):
        SoundBridgeEngine(load_model=False)


def test_stale_arrays_are_reported(artifact_dirs):
    from src.engine import ArtifactsMissing, SoundBridgeEngine

    coords = np.load(artifact_dirs / "coords_2d.npy")
    (artifact_dirs / "coords_2d.npy").unlink()
    np.save(artifact_dirs / "coords_2d.npy", coords[:-5])
    with pytest.raises(ArtifactsMissing, match="out of date"):
        SoundBridgeEngine(load_model=False)


def test_corrupt_checkpoint_is_reported(artifact_dirs):
    from src.engine import ArtifactsMissing, SoundBridgeEngine

    ckpt = artifact_dirs / "soundbridge_encoder.pt"
    data = ckpt.read_bytes()
    ckpt.unlink()
    ckpt.write_bytes(data[: len(data) // 3])  # e.g. an interrupted download
    with pytest.raises(ArtifactsMissing, match="could not be loaded"):
        SoundBridgeEngine()


def test_reuse_weights_falls_back_on_corrupt_checkpoint(tmp_path):
    import sys

    sys.path.insert(0, str(ROOT / "scripts"))
    from train_model import load_existing_encoder

    (tmp_path / "soundbridge_encoder.pt").write_bytes(b"not a checkpoint")
    assert load_existing_encoder(tmp_path, tmp_path, [], [], {}) is None
