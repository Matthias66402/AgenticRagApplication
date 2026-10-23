from pathlib import Path

from app.check_config import check, main
from app.config import Settings


def write_env(path: Path, **overrides: str) -> Path:
    """Env-Datei mit allen Einstellungen auf ihren Standardwerten, einzelne überschrieben."""
    values = {name.upper(): str(field.default) for name, field in Settings.model_fields.items()}
    values.update(overrides)
    path.write_text("".join(f"{name}={value}\n" for name, value in values.items()), encoding="utf-8")
    return path


def test_complete_file_passes(tmp_path: Path) -> None:
    path = write_env(tmp_path / ".env.example", OPENAI_API_KEY="")

    assert check(path) == []
    assert main([str(path)]) == 0


def test_missing_setting_is_reported(tmp_path: Path) -> None:
    path = write_env(tmp_path / ".env.example")
    lines = path.read_text(encoding="utf-8").splitlines()
    path.write_text("\n".join(line for line in lines if not line.startswith("LLM_MODEL=")), encoding="utf-8")

    assert check(path) == ["LLM_MODEL fehlt"]
    assert main([str(path)]) == 1


def test_unknown_variable_is_reported(tmp_path: Path) -> None:
    path = write_env(tmp_path / ".env.example", LLM_REQUEST_PER_MINUTE="5")

    assert check(path) == ["LLM_REQUEST_PER_MINUTE ist unbekannt (Tippfehler?)"]


def test_invalid_value_is_reported(tmp_path: Path) -> None:
    path = write_env(tmp_path / ".env.example", LLM_REQUESTS_PER_MINUTE="")

    assert check(path) == ["LLM_REQUESTS_PER_MINUTE: Input should be a valid integer, unable to parse string as an integer"]


def test_missing_file_is_reported(tmp_path: Path) -> None:
    assert check(tmp_path / "fehlt.env") == [f"{tmp_path / 'fehlt.env'} nicht gefunden"]
