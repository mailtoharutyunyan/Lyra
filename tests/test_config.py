import json
from voicetotext import config


def test_dirs_are_created_and_nested(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "_DATA_OVERRIDE", tmp_path / "data")
    monkeypatch.setattr(config, "_CONFIG_OVERRIDE", tmp_path / "cfg")
    assert config.data_dir().is_dir()
    assert config.models_dir() == config.data_dir() / "models"
    assert config.models_dir().is_dir()
    assert config.config_dir().is_dir()


def test_settings_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "_CONFIG_OVERRIDE", tmp_path / "cfg")
    assert config.load_settings() == {}
    config.save_settings({"target_lang": "rus_Cyrl", "font_size": 22})
    assert config.load_settings() == {"target_lang": "rus_Cyrl", "font_size": 22}
    # written as readable JSON
    raw = (config.config_dir() / "settings.json").read_text()
    assert json.loads(raw)["target_lang"] == "rus_Cyrl"
