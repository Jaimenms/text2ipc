from text2ipc import config


def test_repo_data_dir_detection(tmp_path, monkeypatch):
    monkeypatch.delenv("TEXT2IPC_HOME", raising=False)
    assert config._repo_data_dir(tmp_path) is None
    (tmp_path / "pyproject.toml").write_text('[project]\nname = "text2ipc"\n')
    assert config._repo_data_dir(tmp_path) is None  # data/ must exist
    (tmp_path / "data").mkdir()
    assert config._repo_data_dir(tmp_path) == tmp_path / "data"
    (tmp_path / "notebooks").mkdir()
    assert config._repo_data_dir(tmp_path / "notebooks") == tmp_path / "data"  # walks up
    (tmp_path / "pyproject.toml").write_text('[project]\nname = "other"\n')
    assert config._repo_data_dir(tmp_path) is None


def test_home_prefers_env(tmp_path, monkeypatch):
    monkeypatch.setenv("TEXT2IPC_HOME", str(tmp_path / "elsewhere"))
    assert config.home() == tmp_path / "elsewhere"
