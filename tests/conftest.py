import pytest


@pytest.fixture(autouse=True)
def _isolated_ai_settings(tmp_path, monkeypatch):
    """AI tests must not read or write the user's ~/.gfem/ai.json."""
    monkeypatch.setenv("GFEM_AI_CONFIG", str(tmp_path / "ai_settings.json"))
