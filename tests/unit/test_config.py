from novel_rag.config import Settings


def _settings(monkeypatch, **env) -> Settings:
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    return Settings(_env_file=None)


def test_defaults_without_env(monkeypatch):
    s = _settings(monkeypatch)
    assert s.embedding.dim == 1024
    assert s.retrieval.rrf_k == 60
    assert {"extraction", "generation", "judge"} <= set(s.llm.tasks)


def test_nested_env_override(monkeypatch):
    s = _settings(monkeypatch, ES__URL="http://es:9201", EMBEDDING__BATCH_SIZE="4")
    assert s.es.url == "http://es:9201"
    assert s.embedding.batch_size == 4


def test_partial_task_override_keeps_other_tasks_and_fields(monkeypatch):
    default = _settings(monkeypatch).llm.tasks["generation"]
    s = _settings(monkeypatch, LLM__TASKS__GENERATION__MODEL="other-model")
    assert s.llm.tasks["generation"].model == "other-model"
    assert s.llm.tasks["generation"].extra_body == default.extra_body
    assert "extraction" in s.llm.tasks


def test_new_task_can_be_added_from_env(monkeypatch):
    s = _settings(monkeypatch, LLM__TASKS__SUMMARY__MODEL="m")
    assert s.llm.tasks["summary"].model == "m"


def test_snapshot_redacts_secrets(monkeypatch):
    s = _settings(monkeypatch, LLM__API_KEY="sk-secret-123", NEO4J__PASSWORD="pw-secret-456")
    snap = s.snapshot()
    text = repr(snap)
    assert "sk-secret-123" not in text and "pw-secret-456" not in text
    assert snap["llm"]["tasks"]["generation"]["model"]
    assert s.llm.api_key.get_secret_value() == "sk-secret-123"


def test_resolve_model_prefers_local_dir(tmp_path):
    from novel_rag.config import PathsConfig

    paths = PathsConfig(models_dir=tmp_path)
    assert paths.resolve_model("BAAI/bge-m3") == "BAAI/bge-m3"  # 本地没有 → 走 Hub
    (tmp_path / "bge-m3").mkdir()
    assert paths.resolve_model("BAAI/bge-m3") == str(tmp_path / "bge-m3")
    assert paths.resolve_model(str(tmp_path)) == str(tmp_path)  # 直接给路径
