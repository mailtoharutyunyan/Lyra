import pytest

from voicetotext import config
from voicetotext.models import download, registry


@pytest.fixture
def model_root(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "_DATA_OVERRIDE", tmp_path / "data")
    return config.models_dir()


def test_registry_has_expected_specs():
    assert registry.NLLB.repo_id == "JustFrederik/nllb-200-distilled-600M-ct2-int8"
    assert registry.PARAKEET.approx_bytes > 0
    assert set(registry.ALL) == {"parakeet", "silero_vad", "nllb"}


def test_ensure_model_invokes_downloader_and_marks_complete(model_root):
    calls = []

    def fake_dl(repo_id, local_dir):
        calls.append((repo_id, local_dir))
        # simulate a downloaded file
        (local_dir / "dummy.bin").write_bytes(b"x")

    path = download.ensure_model(registry.NLLB, _downloader=fake_dl)
    assert path.is_dir()
    assert (path / ".complete").exists()
    assert calls and calls[0][0] == registry.NLLB.repo_id


def test_ensure_model_is_idempotent(model_root):
    n = {"count": 0}

    def fake_dl(repo_id, local_dir):
        n["count"] += 1

    download.ensure_model(registry.NLLB, _downloader=fake_dl)
    download.ensure_model(registry.NLLB, _downloader=fake_dl)
    assert n["count"] == 1  # second call short-circuits on .complete


def test_insufficient_disk_raises(model_root, monkeypatch):
    huge = registry.ModelSpec(key="huge", repo_id="x/y", approx_bytes=10**18, kind="hf_snapshot")
    with pytest.raises(download.InsufficientDiskError):
        download.ensure_model(huge, _downloader=lambda *a: None)
