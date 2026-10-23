"""Prüft eine Env-Datei (Standard: .env.example) gegen die Einstellungen in app/config.py.

- Jede Einstellung aus `Settings` steht in der Datei.
- Jede Variable der Datei ist eine Einstellung oder eine bekannte externe Variable (Tippfehler fallen sonst nicht
  auf, weil `Settings` unbekannte Variablen ignoriert).
- Die Werte lassen sich als `Settings` einlesen (z. B. Zahlen bei Zahlenfeldern).

Aufruf: python -m app.check_config [Datei]; Exit-Code 1 bei Fehlern. Läuft in GitHub Actions bei jedem Push.
"""

import argparse
import sys
from pathlib import Path

from dotenv import dotenv_values
from pydantic import ValidationError

from app.config import ROOT_DIR, Settings

# Variablen, die nicht über Settings, sondern direkt aus der Umgebung gelesen werden
EXTERNAL_VARIABLES = {
    "OPENAI_API_KEY",  # OpenAI-SDK
    "TEST_DATABASE_URL",  # tests/conftest.py
    "LANGFUSE_BASE_URL",  # Langfuse-SDK
    "LANGFUSE_TRACING_ENVIRONMENT",  # Langfuse-SDK
}


def check(path: Path) -> list[str]:
    """Gibt die gefundenen Fehler zurück; leer = in Ordnung."""
    if not path.is_file():
        return [f"{path} nicht gefunden"]
    values = dotenv_values(path)
    names = {name.upper() for name in Settings.model_fields}
    errors = [f"{name} fehlt" for name in sorted(names - values.keys())]
    errors += [f"{name} ist unbekannt (Tippfehler?)" for name in sorted(values.keys() - names - EXTERNAL_VARIABLES)]
    settings_values = {name.lower(): value for name, value in values.items() if name in names and value is not None}
    try:
        Settings(_env_file=None, **settings_values)
    except ValidationError as exc:
        errors += [f"{str(e['loc'][0]).upper()}: {e['msg']}" for e in exc.errors()]
    return errors


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("path", type=Path, nargs="?", default=ROOT_DIR / ".env.example", help="zu prüfende Datei")
    args = parser.parse_args(argv)
    errors = check(args.path)
    for error in errors:
        print(f"{args.path.name}: {error}")
    if not errors:
        print(f"{args.path.name}: in Ordnung")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
