# Agentic RAG im Abenteuerplaner

Diese Dokumentation erklärt, wie der Planungs-Agent arbeitet, wie die Wissensbasis (RAG) eingebunden ist und wo der zugehörige Code liegt. Code-Verweise nennen Dateien und Funktionsnamen statt Zeilennummern, damit sie bei Änderungen gültig bleiben.

## Inhalt

1. [Klassisches RAG vs. Agentic RAG](#klassisches-rag-vs-agentic-rag)
2. [Ablauf einer Anfrage](#ablauf-einer-anfrage)
3. [Der Agent](#der-agent)
4. [Das Ergebnis-Schema](#das-ergebnis-schema)
5. [Die Tools](#die-tools)
6. [Die Wissensbasis (RAG)](#die-wissensbasis-rag)
7. [Streaming, Gedächtnis und Speicherung](#streaming-gedächtnis-und-speicherung)
8. [Robustheit gegenüber externen Diensten](#robustheit-gegenüber-externen-diensten)
9. [Tests ohne echtes LLM](#tests-ohne-echtes-llm)
10. [Erweitern](#erweitern)

---

## Klassisches RAG vs. Agentic RAG

**Klassisches RAG** folgt einem festen Ablauf:

```
Frage → Embedding → Top-k ähnliche Textstellen → Textstellen + Frage ans LLM → Antwort
```

Der Abruf passiert genau einmal und immer vor dem LLM-Aufruf, und gesucht wird mit der Originalfrage.

**Agentic RAG** macht das LLM zu einem **Agenten in einer Schleife**. Der Agent entscheidet selbst,

- **ob** er überhaupt etwas nachschlagen muss,
- **wo**: in der Wissensbasis oder über externe Tools (Routing, Wetter, Unterkünfte …),
- **wonach**: Die Suchanfragen formuliert er selbst, oft mehrere gezielte,
- **ob das Ergebnis reicht**, oder ob er nachfragen, umplanen oder sich korrigieren muss.

Die Wissensbasis ist dabei **ein Tool unter mehreren** (`search_knowledge`). Gerade das Zusammenspiel macht den Ansatz aus: Fakten aus Dokumenten (Regeln, Tipps) werden mit Live-Daten (Route, Wetter, Unterkünfte) zu einem strukturierten Plan kombiniert.

---

## Ablauf einer Anfrage

```
Nutzer: "3 Tage Radtour Ulm → Lindau, Camping"
   │
   ▼
POST /api/chat/{id}                           api/chat.py → chat()
   │  lädt den bisherigen Verlauf (agent_history) des Abenteuers
   ▼
run_planner()                                 agent/runner.py
   │  startet den Agenten und streamt dessen Events
   ▼
┌──────────────── Agenten-Schleife (PydanticAI) ────────────────┐
│ LLM liest Instructions + Verlauf + Tool-Beschreibungen        │
│   → entscheidet: Tool aufrufen? Welches, mit welchen Args?    │
│   → geocode("Ulm"), geocode("Lindau")          (parallel)     │
│   → plan_route([...], cycling)                                │
│   → find_accommodation(..., camping) je Etappenende           │
│   → plan_route mit Campingplätzen als Wegpunkte               │
│   → get_weather(...)                                          │
│   → search_knowledge("Wildcampen Deutschland Regeln")   ← RAG │
│   → final_plan(AdventurePlan)   oder   ask_user(Frage)        │
│                                                               │
│ Validierung schlägt fehl? → Fehlertext zurück ans LLM,        │
│   es korrigiert sich und gibt erneut aus                      │
└───────────────────────────────────────────────────────────────┘
   │  jedes Tool-Event geht live per Server-Sent Events ans Frontend
   ▼
save_outcome()                                api/chat.py
   speichert Plan, Chatnachrichten (inkl. Tool-Aufrufe) und Verlauf
```

Alle Pfade beziehen sich auf `backend/app/`.

---

## Der Agent

**Datei:** [`backend/app/agent/planner.py`](../backend/app/agent/planner.py)

| Element | Aufgabe |
|---|---|
| `INSTRUCTIONS` | System-Prompt: Pflichtangaben, empfohlenes Vorgehen, Regeln (z. B. „Unterkünfte nur aus Tool-Ergebnissen“, „Etappenenden sind reale Orte“) |
| `build_agent()` | Erzeugt den PydanticAI-`Agent` mit Modell aus `LLM_MODEL`, den Tools und den beiden möglichen Ausgaben |
| `ToolOutput(..., name="final_plan")` / `ToolOutput(..., name="ask_user")` | Die zwei Wege, die Schleife zu beenden: fertiger Plan oder Rückfrage. Was zutrifft, entscheidet der Agent selbst. |
| `@agent.instructions` → `current_context()` | Gibt bei jedem Lauf das heutige Datum und den Nutzernamen mit, damit Angaben wie „nächsten Samstag“ aufgelöst werden können |
| `@agent.output_validator` → `check_plausibility()` | Selbstprüfung vor der Ausgabe (siehe unten) |
| `used_tool()` | Prüft im Nachrichtenverlauf, ob ein bestimmtes Tool schon aufgerufen wurde |

### Selbstkorrektur

Ein Plan wird nur angenommen, wenn er mehrere Prüfungen besteht. Sonst wirft der Validator `ModelRetry` mit einer konkreten Begründung. PydanticAI schickt diese ans LLM zurück, das den Plan überarbeitet (bis zu `retries=3`):

- Vor jedem Plan muss **mindestens eine Suche in der Wissensbasis** stattgefunden haben. Das garantiert, dass RAG tatsächlich genutzt wird.
- Das **Startdatum** darf nicht in der Vergangenheit liegen.
- Keine Etappe darf länger sein als das Limit der Fortbewegungsart (`MAX_STAGE_KM` in `plan_schema.py`, z. B. Wandern 35 km, Rad 160 km).

### Modellwahl

Das Modell steht in der `.env` (`LLM_MODEL`, z. B. `openai:gpt-5-mini`). Da PydanticAI anbieter-agnostisch ist, lässt sich der Anbieter ohne Codeänderung wechseln, z. B. `groq:openai/gpt-oss-120b`. Voraussetzung ist, dass das Paket-Extra installiert und der API-Key gesetzt ist.

---

## Das Ergebnis-Schema

**Datei:** [`backend/app/agent/plan_schema.py`](../backend/app/agent/plan_schema.py)

- **`AdventurePlan`**: Titel, Zusammenfassung, Fortbewegungsart, Startdatum, Dauer, Start/Ziel, Übernachtungsart, Etappen (`Stage`), `route_ids`, Wetter, Packliste, Sicherheitshinweise, Quellen.
- **`ClarificationQuestion`**: Rückfrage und Liste der fehlenden Angaben.

Der `model_validator` in `AdventurePlan` erzwingt fachliche Regeln:

- Bei mehrtägigen Touren ist `accommodation_type` Pflicht.
- Es gibt genau eine Etappe pro Tag.

Verstößt das LLM dagegen, erhält es die Pydantic-Fehlermeldung und korrigiert sich. So entsteht immer ein **strukturierter, validierter** Plan statt Freitext, und Frontend, Karte und GPX-Export können sich darauf verlassen.

---

## Die Tools

**Verzeichnis:** [`backend/app/agent/tools/`](../backend/app/agent/tools/)

Jedes Tool ist eine normale Python-Funktion. PydanticAI erzeugt aus **Docstring und Typ-Hinweisen** die Tool-Beschreibung (JSON-Schema), die das LLM sieht. Die Docstrings sind deshalb Teil der Agentenlogik: Sie bestimmen mit, wann und wie das LLM ein Tool wählt.

| Tool | Datei | Quelle | Wann typischerweise |
|---|---|---|---|
| `search_knowledge` | `knowledge.py` | **eigene Wissensbasis (RAG)** | Regeln, Tipps, Ausrüstung, Sicherheit |
| `geocode` | `geocode.py` | Nominatim (OSM) | Start, Ziel, Etappenorte |
| `plan_route` | `routing.py` | OpenRouteService → BRouter → Schätzung | Strecke, Höhenmeter, Werte je Etappe |
| `find_accommodation` | `accommodation.py` | Nominatim-Kategoriesuche → Overpass | Hütten, Camping, Hotels nahe Etappenenden |
| `get_weather` | `weather.py` | Open-Meteo | Vorhersage (≤ 15 Tage) bzw. Klimamittel |
| `estimate_hiking_time` | `routing.py` | lokal (DAV-Formel) | Gehzeit je Wanderetappe |
| `sun_times` | `sun.py` | lokal (`astral`) | Tageslicht bei langen Etappen |

### Gemeinsamer Kontext: `AgentDeps`

**Datei:** [`backend/app/agent/deps.py`](../backend/app/agent/deps.py)

Jedes Tool erhält über `ctx.deps` den HTTP-Client, eine DB-Session-Factory, die Einstellungen und das heutige Datum. Eine Factory statt einer einzelnen Session ist nötig, weil Tools parallel laufen können.

`AgentDeps.routes` sammelt die Routengeometrien. Sie gehen **nicht** ans LLM, das würde nur Tokens kosten. Das LLM bekommt eine `route_id` und trägt sie in `AdventurePlan.route_ids` ein. Der Server ordnet beim Speichern die Geometrie zu, für Karte und GPX-Export.

### Fehler als Ergebnis

Tools werfen bei Problemen keine Exceptions, sondern geben einen lesbaren Fehler zurück (`tool_error()` in `common.py`). Das LLM kann darauf reagieren: anderen Ort wählen, ohne Unterkunft weiterplanen, den Nutzer fragen oder den Ausfall im Plan vermerken.

---

## Die Wissensbasis (RAG)

**Verzeichnis:** [`backend/app/rag/`](../backend/app/rag/)

### Einlesen: `ingest.py`

1. **Text extrahieren** (`extract_text`): Markdown, TXT oder PDF (`pypdf`).
2. **Zerlegen** (`split_markdown`): erst an Überschriften, dann an Absätzen, maximal ca. 1200 Zeichen pro Abschnitt. Jeder Abschnitt merkt sich seinen **Überschriftenpfad**, z. B. `Wildcampen … > Österreich`.
3. **Einbetten** (`embed_documents` in `embeddings.py`): Titel und Überschriftenpfad werden mit eingebettet, damit auch kurze Abschnitte ihren Kontext behalten.
4. **Speichern** (`ingest_document`): Tabelle `documents` (Titel, Quelle) und Tabelle `chunks` (Text, Metadaten, Vektor). Ein Dokument mit gleichem Dateinamen wird ersetzt.

### Embeddings: `embeddings.py`

`get_embedder()` liefert einen PydanticAI-`Embedder` für das Modell aus `EMBEDDING_MODEL`. `_check_dimensions()` stellt sicher, dass die Vektorlänge zur DB-Spalte passt (`EMBEDDING_DIM`). Bei einem Modellwechsel müssen Spalte und Daten neu aufgebaut werden.

### Suche: `retrieve.py`

`search_chunks()` bettet die Suchanfrage ein und sucht per **Kosinus-Distanz** über den HNSW-Index von pgvector. Zurück kommen die Top-4-Abschnitte oberhalb eines Mindest-Scores, mit Dokumenttitel und Überschrift.

### Als Tool: `agent/tools/knowledge.py`

`search_knowledge(query)` verbindet RAG und Agent. Entscheidend: **Den Suchtext formuliert das LLM**, passend zur aktuellen Planungssituation, z. B. „Wildcampen Österreich Biwak Regeln“ statt der Originalfrage des Nutzers. Das LLM kann auch mehrmals mit unterschiedlichen Anfragen suchen.

### Quellen der Wissensbasis

- `backend/app/seed/documents/*.md`: Beispieldokumente und importierte Wikivoyage-Artikel, eingelesen mit `python -m app.seed`.
- Upload über die Admin-Seite *Wissensbasis* (`api/documents.py`).
- Wikivoyage-Import: `python -m app.seed.wikivoyage "<Titel>"` (Code in `rag/wikivoyage.py` und `seed/wikivoyage.py`).

Details zum Erweitern stehen in der [README](../README.md#wissensbasis-erweitern).

---

## Streaming, Gedächtnis und Speicherung

### Event-Streaming: `agent/runner.py`

`run_planner()` startet den Agenten mit `run_stream_events()` und übersetzt die PydanticAI-Events in einfache UI-Events:

| PydanticAI-Event | UI-Event | Anzeige im Frontend |
|---|---|---|
| `PartStartEvent` / `PartDeltaEvent` (Text) | `token` | Text des Agenten |
| `FunctionToolCallEvent` | `tool_call` | Badge mit Spinner |
| `FunctionToolResultEvent` | `tool_result` | Badge mit ✓ oder ⚠️, aufklappbar |
| `AgentRunResultEvent` | `plan` / `question` | Plan-Karte bzw. Rückfrage |

Der Endpunkt `POST /api/chat/{id}` (`api/chat.py`) sendet diese Events als **Server-Sent Events**. Das Frontend liest den Stream in `streamChat()` (`frontend/src/api/client.ts`) und zeigt die Tool-Aufrufe live mit `ToolCallBadge` (`frontend/src/components/ToolCallBadge.tsx`).

### Gedächtnis

Der komplette PydanticAI-Nachrichtenverlauf wird als `agent_history` am Abenteuer gespeichert (`models/adventure.py`). Bei der nächsten Nachricht wird er als `message_history` übergeben. Folgenachrichten wie „Mach daraus 4 Tage“ oder „Bitte Route neu berechnen“ versteht der Agent deshalb im Kontext.

### Speicherung: `save_outcome()` in `api/chat.py`

- **Chatnachrichten:** Nutzer- und Agentennachricht werden in `chat_messages` gespeichert, inklusive der Tool-Aufrufe für die Anzeige.
- **Plan:** Er wird zusammen mit den Routengeometrien gespeichert (`build_stored_plan`). Hat das LLM vergessen, `route_ids` zu setzen, wird die zuletzt berechnete Route übernommen.

---

## Robustheit gegenüber externen Diensten

Öffentliche Dienste fallen zeitweise aus. Jedes Tool hat deshalb eine Rückfallebene:

| Tool | Strategie |
|---|---|
| `plan_route` | Kette ORS → BRouter → weitere ORS-Profile → Luftlinien-Schätzung; sporadische Timeouts werden wiederholt, hängende Profile 10 min übersprungen |
| `find_accommodation` | Nominatim; bei Fehler Overpass mit Circuit Breaker (2 min Pause nach Ausfall) |
| `geocode` | Drosselung auf 1 Anfrage/s (Nominatim-Richtlinie), Cache |
| `search_knowledge` | Fehler (z. B. Embedding-Anbieter nicht erreichbar) als Tool-Ergebnis; der Agent plant ohne RAG-Inhalte weiter |

---

## Tests ohne echtes LLM

**Datei:** [`backend/tests/test_agent.py`](../backend/tests/test_agent.py)

Die Agenten-Tests ersetzen das LLM durch ein **geskriptetes `FunctionModel`**. Es ruft Tools in festgelegter Reihenfolge auf und gibt bewusst fehlerhafte Pläne aus. So lässt sich die Schleife deterministisch prüfen:

- Rückfrage bei fehlenden Angaben (`ask_user`)
- Selbstkorrektur: erst Ablehnung ohne Wissenssuche, dann wegen zu langer Etappe, schließlich Annahme
- Folgenachrichten sehen den Verlauf der vorherigen
- Fremde Abenteuer sind nicht erreichbar

Die Tool-Tests (`test_tools.py`) mocken alle HTTP-Aufrufe mit `respx`, die RAG-Tests (`test_rag.py`) nutzen Fake-Embeddings. Alle Tests laufen ohne API-Kosten:

```bash
docker compose exec backend pytest -q
```

---

## Erweitern

### Ein neues Tool hinzufügen

1. Funktion in `backend/app/agent/tools/<name>.py` schreiben, mit `ctx: RunContext[AgentDeps]` als erstem Parameter, falls HTTP, DB oder Einstellungen gebraucht werden.
2. **Aussagekräftigen Docstring** mit `Args:`-Abschnitt verfassen. Daraus wird die Beschreibung für das LLM.
3. Fehler mit `tool_error(...)` zurückgeben statt Exceptions zu werfen.
4. In `build_agent()` (`planner.py`) zur Liste `tools=[...]` hinzufügen.
5. Optional in `INSTRUCTIONS` beschreiben, wann das Tool sinnvoll ist, und in `TOOL_LABELS` (`frontend/src/lib/labels.ts`) ein Symbol und eine Bezeichnung für die Anzeige ergänzen.
6. Test mit `respx`-Mocks in `backend/tests/test_tools.py` ergänzen.

### Weitere Regeln für den Plan

- Fachliche Pflichtregeln gehören in den `model_validator` von `AdventurePlan`.
- Plausibilitätsprüfungen, die den Kontext brauchen (Datum, Verlauf, genutzte Tools), gehören in `check_plausibility()`.
