# AgenticRAG-Abenteuerplaner

Eingeloggte Nutzer beschreiben ein Abenteuer in eigenen Worten, etwa „3 Tage Hüttenwanderung von Garmisch nach Innsbruck ab Freitag“. Ein Agent entscheidet dann selbst, welche Tools er aufruft: Wissensbasis (RAG), Geocoding, Routing, Wetter, Unterkunftssuche, Gehzeit und Sonnenzeiten. Fehlen Angaben, fragt er nach. Den fertigen Plan prüft er auf Plausibilität.

| Schicht | Technik |
|---|---|
| Frontend | React 19, TypeScript, Vite, React Router, Leaflet |
| Backend | FastAPI, SQLAlchemy 2 (async), Alembic, JWT-Auth |
| Agent | PydanticAI (anbieter-agnostisch, Modell per `.env`) |
| RAG | pgvector (HNSW, Kosinus), Embeddings über PydanticAI `Embedder` |
| Datenquellen | Nominatim (Geocoding, Unterkünfte), OpenRouteService + BRouter (Routing), Open-Meteo (Wetter), Overpass als Ersatz |

## Schnellstart

Voraussetzungen: Docker und Node 20+.

```bash
cp .env.example .env            # OPENAI_API_KEY, JWT_SECRET, optional ORS_API_KEY eintragen
docker compose up -d --build    # Postgres/pgvector (Port 5440) + Backend (Port 8010, Hot-Reload)
docker compose exec backend alembic upgrade head
docker compose exec backend python -m app.seed      # Admin-Konto + Beispieldokumente einlesen

cd frontend && npm install && npm run dev            # http://localhost:5180
```

- Anmeldung mit `ADMIN_EMAIL` / `ADMIN_PASSWORD` aus der `.env`. Neue Nutzer können sich auch selbst registrieren.
- API-Dokumentation: http://localhost:8010/docs

> **Warum läuft das Backend im Container?** Auf manchen Windows-Rechnern blockiert die Anwendungssteuerung (Smart App Control) die nativen Postgres-Treiber (`asyncpg`, `psycopg`) und die Cython-Module von SQLAlchemy. Im Linux-Container tritt das nicht auf. Der Ordner `backend/` ist per Volume eingebunden, Codeänderungen werden also sofort neu geladen.

## Tests

```bash
docker compose exec backend pytest -q
```

Die Tests laufen gegen die Datenbank `adventure_test`. Sie verwenden ein geskriptetes Modell (`FunctionModel`) und Fake-Embeddings, rufen also kein echtes LLM auf. HTTP-Aufrufe der Tools werden mit `respx` gemockt.

Für die Test-DB-Migration (einmalig):

```bash
docker compose exec backend sh -c 'ALEMBIC_DATABASE_URL=$TEST_DATABASE_URL alembic upgrade head'
```

## Wie der Agent arbeitet

Ausführliche Erklärung mit Code-Verweisen: [docs/agentic-rag.md](docs/agentic-rag.md)

`backend/app/agent/planner.py`

1. **Strukturierte Ausgabe:** Der Agent liefert entweder `final_plan` (ein `AdventurePlan`) oder `ask_user` (eine `ClarificationQuestion`).
2. **Pydantic-Validierung:** Bei mehrtägigen Touren ist die Übernachtungsart Pflicht, und es muss genau eine Etappe pro Tag geben. Schlägt die Validierung fehl, korrigiert sich das Modell selbst.
3. **Output-Validator:** Er verlangt vor jedem Plan mindestens eine Suche in der Wissensbasis und weist zu lange Etappen je Fortbewegungsart zurück (`MAX_STAGE_KM`).
4. **Fehler als Ergebnis:** Tools geben Fehler als lesbares Ergebnis zurück, statt Exceptions zu werfen. Der Agent kann darauf reagieren, etwa mit einem anderen Routing-Anbieter, einer Luftlinien-Schätzung als letztem Ausweg oder einem Hinweis im Plan.
5. **Gesprächsverlauf:** Der Verlauf wird pro Abenteuer gespeichert (`agent_history`). Folgenachrichten wie „Mach daraus 4 Tage“ passen deshalb den bestehenden Plan an.
6. **Streaming:** `POST /api/chat/{id}` streamt die Arbeit als Server-Sent Events (`token`, `tool_call`, `tool_result`, `question`, `plan`, `error`, `done`). Das Frontend zeigt die Tool-Aufrufe live an.

## Wissensbasis erweitern

- **Upload:** Als Admin unter *Wissensbasis* `.md`-, `.txt`- oder `.pdf`-Dateien hochladen. Gleicher Dateiname ersetzt die vorhandene Version.
- **Als Datei:** `.md` nach `backend/app/seed/documents/` legen und mit `python -m app.seed` einlesen (`--force` liest alle neu ein).
- **Wikivoyage-Import:** Artikel werden abgerufen, in Markdown umgewandelt (mit Quelle, Versionsdatum und Lizenzhinweis) und als `wikivoyage-<titel>.md` gespeichert sowie eingelesen.

```bash
docker compose exec backend python -m app.seed.wikivoyage --search "Radweg Bayern"   # Titel finden
docker compose exec backend python -m app.seed.wikivoyage "Bodensee-Radweg" "Europäischer Fernwanderweg E5"
docker compose exec backend python -m app.seed.wikivoyage --no-ingest "Donauradweg"  # nur speichern, später einlesen
docker compose exec backend python -m app.seed.wikivoyage --dry-run "Allgäu"         # nur anzeigen
```

Wikivoyage-Inhalte stehen unter CC BY-SA 4.0: Bei Weitergabe (z. B. in einem öffentlichen Repository) müssen Quelle und Lizenz genannt bleiben, Bearbeitungen stehen unter derselben Lizenz. Der Kopf jeder importierten Datei enthält die nötigen Angaben. Nutzt der Agent einen solchen Artikel für einen Plan, landet die vollständige Quellenangabe (Artikel, Version, Autoren über die Versionsgeschichte, Lizenz, Hinweis auf Bearbeitung) automatisch im Plan, in der Planansicht, im PDF und in den GPX-Metadaten. Ermittelt wird das aus dem Agent-Verlauf (`backend/app/rag/attribution.py`), nicht aus der Freitext-Quellenliste des LLM. Für ältere Pläne ergänzt `python -m app.seed` die Angaben.

Das Einlesen erzeugt Embeddings über `EMBEDDING_MODEL`. Ist der Anbieter nicht erreichbar, wird die Datei trotzdem gespeichert und kann später mit `python -m app.seed` eingelesen werden.

## Konfiguration (`.env`)

| Variable | Bedeutung |
|---|---|
| `LLM_MODEL` | PydanticAI-Modellname, z. B. `openai:gpt-5-mini`, `anthropic:claude-sonnet-5-5`, `ollama:llama3.3` |
| `EMBEDDING_MODEL` / `EMBEDDING_DIM` | Bei einem Wechsel: `EMBEDDING_DIM` anpassen, Migration für die Vektorspalte erzeugen, `python -m app.seed --force` |
| `ORS_API_KEY` | Optional, kostenlos unter openrouteservice.org (Free-Tarif: 200 Routen/Tag). Routing-Kette: ORS → BRouter (ohne Key) → weitere ORS-Profile → Luftlinien-Schätzung. Ohne Key wird direkt BRouter genutzt. |
| `OVERPASS_URLS` | Kommagetrennte Overpass-Instanzen; nur Ersatz, falls die Unterkunftssuche über Nominatim scheitert |

## Projektstruktur

```
backend/app/
  agent/      planner.py (Agent), runner.py (Event-Streaming), plan_schema.py, tools/
  rag/        embeddings.py, ingest.py (Chunking), retrieve.py (pgvector-Suche)
  api/        auth, users, adventures, chat (SSE), documents
  seed/       Admin-Seed + Beispieldokumente (documents/*.md)
frontend/src/
  pages/      Login/Registrierung, Abenteuerliste, Planer, Profil, Admin (Nutzer, Wissensbasis)
  components/ PlanCard, RouteMap (Leaflet), ToolCallBadge
```
