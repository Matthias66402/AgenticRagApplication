# AgenticRAG-Abenteuerplaner

Eingeloggte Nutzer beschreiben ein Abenteuer in eigenen Worten, etwa „3 Tage Hüttenwanderung von Garmisch nach Innsbruck ab Freitag, Anreise mit der Bahn aus München“. Ein Agent entscheidet dann selbst, welche Tools er aufruft: Wissensbasis (RAG), Geocoding, Routing, Wetter, Unterkunftssuche, Gehzeit, Sonnenzeiten und Bahnverbindungen. Fehlen Angaben, fragt er nach. Den fertigen Plan prüft er auf Plausibilität.

**Funktionen**

- **Planung im Chat:** Der Agent arbeitet live sichtbar (jeder Tool-Aufruf mit Argumenten und Ergebnis) und passt den Plan bei Folgenachrichten an („Mach daraus 4 Tage“).
- **Planansicht:** Karte mit Route, Etappen und Unterkünften, interaktives Höhenprofil, Etappen, Wetter, Packliste, Sicherheitshinweise und Quellen.
- **An- und Abreise mit der Bahn:** konkrete Verbindungen mit Zeiten, Linien und Gleisen. Auch reine Bahnauskünfte ohne Tour sind möglich.
- **Exporte:** PDF als Druckversion mit OpenStreetMap-Karte und Höhenprofil, GPX mit Track und Wegpunkten.
- **Nachvollziehbare Quellen:** Genutzte Wikivoyage-Artikel erscheinen mit Lizenzangabe (CC BY-SA) im Plan, im PDF und im GPX.
- **Messbare Suchqualität:** Eine Evaluation misst, wie gut die Wissensbasis passende Abschnitte findet und unpassende aussortiert.

| Schicht | Technik |
|---|---|
| Frontend | React 19, TypeScript, Vite, React Router, Leaflet |
| Backend | FastAPI, SQLAlchemy 2 (async), Alembic, JWT-Auth |
| Agent | PydanticAI (anbieter-agnostisch, Modell per `.env`) |
| RAG | pgvector (HNSW, Kosinus), Embeddings über PydanticAI `Embedder`, Relevanzprüfung der Treffer durch ein kleines LLM |
| Tracing | Langfuse (optional, OpenTelemetry über die PydanticAI-Instrumentierung) |
| Export | fpdf2 (PDF), Pillow (Kartenhintergrund aus OSM-Kacheln), GPX 1.1 |
| Datenquellen | Nominatim (Geocoding, Unterkünfte), OpenRouteService + BRouter (Routing), Open-Meteo (Wetter), Transitous (Bahnverbindungen), OpenStreetMap-Kacheln (PDF-Karte), Overpass als Ersatz |

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
- Datenbank direkt ansehen (z. B. mit DBeaver): PostgreSQL auf `localhost:5440`, Datenbank, Benutzer und Passwort jeweils `adventure`.

> **Warum läuft das Backend im Container?** Auf manchen Windows-Rechnern blockiert die Anwendungssteuerung (Smart App Control) die nativen Postgres-Treiber (`asyncpg`, `psycopg`) und die Cython-Module von SQLAlchemy. Im Linux-Container tritt das nicht auf. Der Ordner `backend/` ist per Volume eingebunden, Codeänderungen werden also sofort neu geladen.

## Tests

```bash
docker compose exec backend pytest -q
```

Die Tests laufen gegen die Datenbank `adventure_test`. Sie verwenden ein geskriptetes Modell (`FunctionModel`), Fake-Embeddings und eine Relevanzprüfung ohne LLM, rufen also kein echtes LLM auf. HTTP-Aufrufe der Tools (Routing, Wetter, Fahrplan, Kartenkacheln usw.) werden mit `respx` gemockt.

GitHub Actions führt die Tests bei jedem Push in Docker aus (`.github/workflows/docker.yml`): Image bauen, `docker compose up` mit `.env.example` als `.env`, beide Datenbanken migrieren, `/api/health` abfragen, `pytest`. So fällt auch auf, wenn Dockerfile, Compose-Datei oder Migrationen nicht mehr zusammenpassen. Das Frontend baut `.github/workflows/frontend.yml` mit `npm ci && npm run build` (Typprüfung mit `tsc` und Vite-Build).

Für die Test-DB-Migration (einmalig):

```bash
docker compose exec backend sh -c 'ALEMBIC_DATABASE_URL=$TEST_DATABASE_URL alembic upgrade head'
```

## Wie der Agent arbeitet

Ausführliche Erklärung mit Code-Verweisen: [docs/agentic-rag.md](docs/agentic-rag.md)

`backend/app/agent/planner.py`

1. **Strukturierte Ausgabe:** Der Agent liefert `final_plan` (ein `AdventurePlan`), `ask_user` (eine `ClarificationQuestion`) oder – bei Anfragen ohne Tour, etwa einer reinen Bahnauskunft – `answer_user` (eine `TextAnswer`). Gefundene Bahnverbindungen hängt das Backend an solche Antworten selbst an, das LLM muss sie nicht abschreiben. Wählt der Nutzer eine Verbindung aus, gibt der Agent über `final_train_trip` eine Reisekarte aus (ein `TrainTripPlan`, gespeichert mit `kind: "train_trip"`): rechts erscheint die Verbindung statt Etappen. Will der Nutzer mit dem Rad oder zu Fuß zum Bahnhof, enthält die Reisekarte diesen Weg mit Karte, Höhenprofil und Losfahrzeit (`access`, Route aus `plan_route`, Ankunft am Bahnhof vor der Zugabfahrt). Für die Exporte bringt `backend/app/export/trip.py` die Reisekarte in Tourform: Das PDF enthält Weg zum Bahnhof und Verbindung, das GPX den Weg zum Bahnhof (ohne diesen Weg gibt es nur das PDF, da Verbindungen keine Koordinaten haben).
2. **Pydantic-Validierung:** Bei mehrtägigen Touren ist die Übernachtungsart Pflicht, und es muss genau eine Etappe pro Tag geben. Schlägt die Validierung fehl, korrigiert sich das Modell selbst.
3. **Output-Validator:** Er verlangt vor jedem Plan mindestens eine Suche in der Wissensbasis, weist ein Startdatum in der Vergangenheit und zu lange Etappen je Fortbewegungsart zurück (`MAX_STAGE_KM`) und akzeptiert Bahnverbindungen nur, wenn sie aus der Fahrplanabfrage stammen. Verweist eine Antwort auf Verbindungen „unten“, ohne dass in dieser Nachricht welche abgefragt wurden, geht sie zurück an das Modell.
4. **Fehler als Ergebnis:** Tools geben Fehler als lesbares Ergebnis zurück, statt Exceptions zu werfen. Der Agent kann darauf reagieren, etwa mit einem anderen Routing-Anbieter, einer Luftlinien-Schätzung als letztem Ausweg oder einem Hinweis im Plan.
5. **Gesprächsverlauf:** Der Verlauf wird pro Abenteuer gespeichert (`agent_history`). Folgenachrichten wie „Mach daraus 4 Tage“ passen deshalb den bestehenden Plan an.
6. **Streaming:** `POST /api/chat/{id}` streamt die Arbeit als Server-Sent Events (`token`, `tool_call`, `tool_result`, `question`, `answer`, `plan`, `error`, `done`). Das Frontend zeigt die Tool-Aufrufe live an.
7. **Robuste Ausgabe:** Die Ausgaben laufen im strict-Modus mit einem Tokenlimit pro Anfrage. Ohne den strict-Modus hängt gpt-5-mini gelegentlich zehntausende Leerzeichen an die JSON-Argumente an und braucht dafür Minuten.

## Wie die Wissensbasis durchsucht wird

`backend/app/rag/` – das Tool `search_knowledge` durchläuft vier Schritte:

1. **Chunking beim Einlesen** (`ingest.py`): Dokumente werden an Markdown-Überschriften geteilt, Abschnitte haben höchstens 1.200 Zeichen. Jeder Abschnitt trägt seinen Überschriftenpfad und bettet ihn mit ein.
2. **Vektorsuche** (`retrieve.py`): Die Suchanfrage wird eingebettet. pgvector liefert die 4 ähnlichsten Abschnitte (Kosinus-Ähnlichkeit).
3. **Mindestähnlichkeit 0,5:** Treffer darunter fallen weg.
4. **Relevanzprüfung** (`relevance.py`): Ein kleines LLM (`RELEVANCE_MODEL`) bewertet in einem Aufruf jede Fundstelle: Beantwortet sie die Anfrage, oder ist sie nur thematisch ähnlich? Nur relevante Abschnitte erreichen den Agenten. Bleibt nichts übrig, erfährt der Agent, dass die Wissensbasis dazu nichts enthält. Fällt die Prüfung aus (Zeitlimit 10 s, z. B. bei einem Ratenlimit), gehen die Treffer ungeprüft weiter.

Schwelle und Relevanzprüfung sind mit der Evaluation unten begründet.

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

Das Einlesen erzeugt Embeddings über `EMBEDDING_MODEL`. Ist der Anbieter nicht erreichbar, wird die Datei trotzdem gespeichert und kann später mit `python -m app.seed` eingelesen werden. Neue Dokumente sollten Testfragen in der Evaluation bekommen.

## Retrieval-Qualität messen

`backend/app/evaluation/retrieval_cases.json` enthält Testfragen mit den Abschnitten der Wissensbasis, die sie beantworten, sowie Fragen, die die Wissensbasis nicht beantworten kann. Die Auswertung fragt jede Frage gegen die aktuelle Wissensbasis ab (kostet je Frage ein Embedding, mit `--grade` zusätzlich einen Aufruf von `RELEVANCE_MODEL`):

```bash
docker compose exec backend python -m app.evaluation.retrieval                          # Kennzahlen ausgeben
docker compose exec backend python -m app.evaluation.retrieval --output /app/report.json  # zusätzlich als JSON
docker compose exec backend python -m app.evaluation.retrieval --grade                  # inkl. Relevanzprüfung (LLM)
```

Kennzahlen: **Hit@1/Hit@4** (relevanter Abschnitt auf Platz 1 bzw. unter den ersten 4, so viele bekommt der Agent), **MRR** (mittlerer Kehrwert des Rangs), **Recall wie im Agenten** (Top 4 und über der Mindestähnlichkeit), **Fehlkontext** (nicht beantwortbare Fragen, für die trotzdem Abschnitte geliefert werden) und ein **Schwellen-Sweep** über die Mindestähnlichkeit. Mit `--grade` kommen Recall und Fehlkontext nach der Relevanzprüfung hinzu. Fehlgeschlagene Bewertungen (z. B. Ratenlimit) wiederholt die Auswertung nach einer Pause.

Stand 21.10.2026 (35 beantwortbare, 6 nicht beantwortbare Fragen; 10 Dokumente, 133 Abschnitte). Hit@1 liegt bei 91 %, Hit@4 bei 100 %, der MRR bei 0,95.

| Stufe | Recall | Fehlkontext |
|---|---|---|
| Mindestähnlichkeit 0,25 (ursprünglich) | 100 % | 100 % |
| Mindestähnlichkeit 0,5 | 100 % | 50 % |
| Mindestähnlichkeit 0,5 + Relevanzprüfung | 100 % | 17 % |

Die Testfragen wurden mit Kenntnis der Dokumente geschrieben und die Wissensbasis ist klein. Die Trefferwerte sind daher eher optimistisch. Bei nur 6 Fragen ohne Antwort entspricht eine einzelne Frage knapp 17 Prozentpunkten Fehlkontext.

## Planer messen

`backend/app/evaluation/planner_cases.json` enthält feste Anfragen an den Planungs-Agenten (Rad-Tagestour, mehrtägige Hüttenwanderung, Radtour mit Zug-Anreise, Bahnauskunft, mit dem Rad zum Bahnhof, unvollständige Anfrage). Die Auswertung schickt jede Anfrage wie im Chat an den Agenten, mit echtem Modell und echten Tools:

```bash
docker compose exec backend python -m app.evaluation.planner                                   # alle Fälle, je 2 Läufe
docker compose exec backend python -m app.evaluation.planner --case bahnauskunft --repeat 1    # einzelner Fall
docker compose exec -e LLM_REASONING_EFFORT=low backend python -m app.evaluation.planner --label low --output /app/low.json
```

Gemessen werden Laufzeit (davon Wartezeit auf Tools), Modellaufrufe, Eingabe-, Cache-, Ausgabe- und Reasoning-Tokens, Kosten und Wiederholungen durch die Selbstprüfung. Die Qualitätsprüfungen je Fall: erwarteter Ausgabetyp, beide Pflichtsuchen in der Wissensbasis, Packliste und Sicherheitshinweise, Quellen, Routen-IDs aus `plan_route`, Unterkünfte nur aus `find_accommodation`, An- und Rückreise bzw. Weg zum Bahnhof, wo verlangt. Ein Durchlauf kostet etwa 0,10-0,15 $ und dauert mit `LLM_REQUESTS_PER_MINUTE=5` rund 10-15 Minuten. Er belegt dasselbe Ratenlimit wie die App; währenddessen die App nicht nutzen, sonst scheitern Anfragen mit „Anfragelimit erreicht“.

Stand 23.10.2026 (gpt-5-mini, Standard-Reasoning; je Fall 2 bzw. 4 Läufe, Mittel je Anfrage). Vorher: Anweisungen ohne Hinweis auf gemeinsame Tool-Aufrufe. Nachher: mit diesem Hinweis und gedrosselt auf 5 Anfragen pro Minute.

| | vorher | nachher |
|---|---|---|
| Modellaufrufe | 5,1 | 3,8 (−25 %) |
| Eingabe-Tokens (davon ~80 % aus dem Cache) | 40.800 | 32.300 (−21 %) |
| Ausgabe-Tokens (davon ~70 % Reasoning) | 3.530 | 3.800 |
| Laufzeit | 67 s | 45 s |
| Kosten | 0,010 $ | 0,010 $ |
| Qualitätsprüfungen bestanden | 12/12 | 22/23 |

Am meisten bringt der Hinweis bei vielen unabhängigen Tools: Hüttenwanderung 12,5 → 8,3 Aufrufe, Rad zum Bahnhof 5,0 → 3,5. Die Werte streuen stark (Hüttenwanderung zwischen 6 und 12 Aufrufen). Die eine nicht bestandene Prüfung war eine berechtigte Rückfrage (die Anfrage nannte „nächsten Samstag“ an einem Freitag); der Fall ist inzwischen eindeutig formuliert.

Nicht übernommen: `LLM_REASONING_EFFORT=low` senkte die Ausgabe-Tokens auf etwa ein Viertel, ließ den Planer bei der Radtour mit Zug-Anreise aber in beiden Läufen nach Uhrzeiten fragen statt planen. Eine um ein Viertel gekürzte Fassung der Anweisungen sparte kaum (die Anweisungen kommen fast immer aus dem Prompt-Cache) und machte aus derselben Anfrage eine reine Bahnauskunft.

## Exporte

- `GET /api/adventures/{id}/pdf`: Druckversion mit Überblick, Karte (OpenStreetMap-Hintergrund, falls `MAP_TILE_URL` gesetzt), Höhenprofil, Etappen, An- und Abreise, Wetter, Packliste, Sicherheit und Quellen.
- `GET /api/adventures/{id}/gpx`: Track mit Höhen und Wegpunkten für Start, Etappenenden und Unterkünfte. Ohne berechnete Route enthält die Datei nur die Wegpunkte.

Beide Downloads gibt es als Schaltflächen in der Planansicht.

## Tracing mit Langfuse

`backend/app/observability.py` – optional. Mit `LANGFUSE_PUBLIC_KEY` und `LANGFUSE_SECRET_KEY` in der `.env` (Projekt anlegen auf [cloud.langfuse.com](https://cloud.langfuse.com), Keys unter *Settings → API Keys*) und einem Neustart des Backends (`docker compose up -d backend`) landet jede Chat-Nachricht als Trace in Langfuse:

- **Ein Trace je Nachricht** mit der Wurzel `chat-turn`: Eingabe ist die Nutzernachricht, Ausgabe die Antwort bzw. der Plantext, Metadaten enthalten das Ergebnis (`plan`, `question`, `answer`, `error`) und die Zahl der Tool-Aufrufe. Fehlgeschlagene Läufe sind als `ERROR` markiert.
- **Darunter:** der Agent `adventure-planner` mit jedem Modellaufruf als Generation (Modell, Tokens, Kosten, Reasoning-Zusammenfassung), die Tool-Aufrufe, die Wissenssuche `retrieve-knowledge` (Anfrage, Treffer mit Ähnlichkeit, Embedding) und die Relevanzprüfung `relevance-grader`.
- **Sessions und Nutzer:** Alle Nachrichten eines Abenteuers bilden die Session `adventure-<id>`; als Nutzer wird die numerische ID übertragen, nicht die E-Mail.
- **Maskierte Wohnadresse:** Straße und Hausnummer aus dem Profil werden vor dem Export durch `[Adresse]` bzw. `[Straße]` ersetzt (auch in Tool-Argumenten, Geocoding-Ergebnissen und Reasoning). Ort und Koordinaten bleiben sichtbar, abgekürzte Schreibweisen wie „Nelkenstr.“ werden nicht erkannt.

Tests senden nie Traces. Evaluation und Seed-Skripte werden nicht getraced.

## Konfiguration (`.env`)

Vorlage ist `.env.example`. Sie muss jede Einstellung aus `backend/app/config.py` enthalten und darf keine unbekannten Variablen enthalten, außerdem müssen sich ihre Werte einlesen lassen. Das prüft GitHub Actions bei jedem Push (`.github/workflows/config.yml`). Lokal geht das mit:

```bash
cd backend && python -m app.check_config
```

Wer eine Einstellung in `config.py` ergänzt, trägt sie also auch in `.env.example` ein. Variablen, die nicht über `config.py` gelesen werden (z. B. `OPENAI_API_KEY`), stehen in `EXTERNAL_VARIABLES` in `app/check_config.py`.

| Variable | Bedeutung |
|---|---|
| `LLM_MODEL` | PydanticAI-Modellname, z. B. `openai:gpt-5-mini`, `anthropic:claude-sonnet-5-5`, `ollama:llama3.3` |
| `LLM_REQUESTS_PER_MINUTE` | Anfragelimit des API-Keys für den Planer; er sendet höchstens so viele Modellaufrufe je Minute und wartet sonst. 0 = nicht drosseln. |
| `LLM_REASONING_EFFORT` | Optional: Reasoning-Aufwand des Planers bei Reasoning-Modellen (`minimal`, `low`, `medium`, `high`). Leer = Standard des Modells; `low` spart Tokens, verschlechterte aber die Planung (siehe „Planer messen“). |
| `RELEVANCE_MODEL` | Kleines Modell, das die Treffer von `search_knowledge` vor der Weitergabe an den Agenten auf Relevanz prüft (Standard `openai:gpt-4.1-mini`, ein Aufruf je Suche). Leer = nur Mindestähnlichkeit. Bei einem Fehler gehen die Treffer ungeprüft weiter. |
| `EMBEDDING_MODEL` / `EMBEDDING_DIM` | Bei einem Wechsel: `EMBEDDING_DIM` anpassen, Migration für die Vektorspalte erzeugen, `python -m app.seed --force` |
| `ORS_API_KEY` | Optional, kostenlos unter openrouteservice.org (Free-Tarif: 200 Routen/Tag). Routing-Kette: ORS → BRouter (ohne Key) → weitere ORS-Profile → Luftlinien-Schätzung. Ohne Key wird direkt BRouter genutzt. |
| `NOMINATIM_USER_AGENT` | User-Agent mit Kontaktangabe für Nominatim, OSM-Kacheln und Transitous; deren Nutzungsrichtlinien verlangen ihn |
| `OVERPASS_URLS` | Kommagetrennte Overpass-Instanzen; nur Ersatz, falls die Unterkunftssuche über Nominatim scheitert |
| `TRANSIT_API_URL` | Fahrplanauskunft für An- und Abreise mit Bahn/ÖPNV. Standard ist die offene MOTIS-Instanz von [Transitous](https://transitous.org) (ohne Key; verlangt einen aussagekräftigen User-Agent, dafür wird `NOMINATIM_USER_AGENT` mitgeschickt); jede andere MOTIS-Instanz funktioniert ebenso. Leer = aus. |
| `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` | Optional, aktivieren das Tracing mit Langfuse (siehe oben). `LANGFUSE_TRACING_ENABLED=false` schaltet es trotz Keys ab. |
| `LANGFUSE_BASE_URL` / `LANGFUSE_TRACING_ENVIRONMENT` | Langfuse-Instanz (Standard EU-Cloud `https://cloud.langfuse.com`) und Umgebung der Traces, z. B. `development` oder `production` |
| `MAP_TILE_URL` | Kartenkacheln für den PDF-Export (Standard: OpenStreetMap, Tile-Policy beachten). Leer = PDF ohne Kartenhintergrund. |

Hinweis zu OpenAI: Ein Projekt mit niedrigem Ratenlimit (z. B. 5 Anfragen pro Minute) bremst den Agenten spürbar, weil jeder Tool-Schritt eine LLM-Anfrage ist und jede Wissenssuche eine weitere für die Relevanzprüfung kostet.

## Projektstruktur

```
backend/app/
  agent/       planner.py (Agent), runner.py (Event-Streaming), plan_schema.py, deps.py,
               rate_limit.py (Drosselung auf das Anfragelimit)
    tools/     geocode, routing (+ Gehzeit), weather, accommodation, knowledge, sun, transit
  rag/         ingest.py (Chunking), embeddings.py, retrieve.py (pgvector-Suche), relevance.py (Relevanzprüfung),
               attribution.py + wikivoyage.py (Quellenangaben)
  evaluation/  retrieval.py + retrieval_cases.json (Messung der Suchqualität),
               planner.py + planner_cases.json (Aufwand und Ergebnisqualität des Planers)
  export/      pdf.py, gpx.py, basemap.py (OSM-Kartenhintergrund), filename.py
  api/         auth, users, adventures (inkl. PDF/GPX), chat (SSE), documents
  observability.py  Tracing mit Langfuse (inkl. Maskierung der Wohnadresse)
  check_config.py   Prüft .env.example gegen config.py (läuft in GitHub Actions)
  seed/        Admin-Seed, Wikivoyage-Import, Beispieldokumente (documents/*.md)
frontend/src/
  pages/       Login/Registrierung, Abenteuerliste, Planer, Profil, Admin (Nutzer, Wissensbasis)
  components/  PlanCard, RouteMap (Leaflet), ElevationProfile, JourneyCard, ToolCallBadge, ConfirmButton
docs/          agentic-rag.md (Agent und RAG im Detail)
```
