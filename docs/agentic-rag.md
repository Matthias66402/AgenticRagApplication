# Agentic RAG im Abenteuerplaner

Diese Dokumentation erklärt, wie ein Chat-Durchlauf als LangGraph-Graph abläuft, wie der Planungs-Agent darin arbeitet, wie die Wissensbasis (RAG) eingebunden ist und wo der zugehörige Code liegt. Code-Verweise nennen Dateien und Funktionsnamen statt Zeilennummern, damit sie bei Änderungen gültig bleiben.

## Inhalt

1. [Klassisches RAG vs. Agentic RAG](#klassisches-rag-vs-agentic-rag)
2. [Ablauf einer Anfrage](#ablauf-einer-anfrage)
3. [Der Graph (LangGraph)](#der-graph-langgraph)
4. [Der Agent](#der-agent)
5. [Das Ergebnis-Schema](#das-ergebnis-schema)
6. [Die Tools](#die-tools)
7. [Die Wissensbasis (RAG)](#die-wissensbasis-rag)
8. [Streaming, Gedächtnis und Speicherung](#streaming-gedächtnis-und-speicherung)
9. [Robustheit gegenüber externen Diensten](#robustheit-gegenüber-externen-diensten)
10. [Tests ohne echtes LLM](#tests-ohne-echtes-llm)
11. [Erweitern](#erweitern)

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

Die Wissensbasis ist dabei **ein Tool unter mehreren** (`research_knowledge`, siehe [Der Recherche-Agent](#der-recherche-agent-agent-als-tool)). Gerade das Zusammenspiel macht den Ansatz aus: Fakten aus Dokumenten (Regeln, Tipps) werden mit Live-Daten (Route, Wetter, Unterkünfte) zu einem strukturierten Plan kombiniert.

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
   │  startet den LangGraph-Graphen und streamt dessen Events
   ▼
intake                                        agent/intake.py (kleines LLM)
   │  Orte zum Nachschlagen, Recherche-Auftrag, Startdatum
   ├──────────────────────────────┐
   ▼                              ▼            (parallel, nur wenn nötig)
prefetch_geocode                prefetch_research
   geocode("Ulm"), …               research_knowledge("Ulm", cycling, camping)  ← RAG
   │                              │  └ Recherche-Agent: search_knowledge(...) je Thema
   └──────────────┬───────────────┘
                  ▼   Vorab-Aufrufe stehen als Tool-Aufrufe im Verlauf des Planers
┌──────────────── Planer-Schleife (PydanticAI) ─────────────────┐
│ LLM liest Instructions + Verlauf + Tool-Beschreibungen        │
│   → entscheidet: Tool aufrufen? Welches, mit welchen Args?    │
│   → plan_route([...], cycling)                                │
│   → find_accommodation(..., camping) je Etappenende           │
│   → plan_route mit Campingplätzen als Wegpunkte               │
│   → get_weather(...)                                          │
│   → find_train_connections(...)   falls An-/Abreise per Bahn  │
│   → (geocode / research_knowledge selbst, falls noch nötig)   │
│   → final_plan(AdventurePlan) | ask_user(Frage)               │
│     | answer_user(Text, z. B. reine Bahnauskunft)             │
│     | final_train_trip(Reisekarte zur gewählten Verbindung)   │
│                                                               │
│ Validierung schlägt fehl? → Fehlertext zurück ans LLM,        │
│   es korrigiert sich und gibt erneut aus                      │
└───────────────────────────────────────────────────────────────┘
   │
   ▼
finish_plan | finish_question | finish_answer | finish_error
   │  jedes Tool-Event geht live per Server-Sent Events ans Frontend
   ▼
save_outcome()                                api/chat.py
   speichert Plan, Chatnachrichten (inkl. Tool-Aufrufe) und Verlauf
```

Alle Pfade beziehen sich auf `backend/app/`.

---

## Der Graph (LangGraph)

**Dateien:** [`backend/app/agent/graph.py`](../backend/app/agent/graph.py), [`backend/app/agent/intake.py`](../backend/app/agent/intake.py)

Ein Chat-Durchlauf ist ein LangGraph-`StateGraph` (`build_graph()`). Der Planer bleibt ein PydanticAI-Agent und ist einer der Knoten; der Graph legt fest, was vor und nach ihm passiert.

| Knoten | Aufgabe |
|---|---|
| `intake` | Ein kleines LLM (`INTAKE_MODEL`, Standard `gpt-4o-mini`) bestimmt in `plan_prefetch()`, was der Planer für diese Nachricht sicher braucht (`IntakeResult`): die Orte zum Nachschlagen, bei einer neuen Tour den Auftrag für die Recherche und das Startdatum. Es sieht die letzten Nutzernachrichten als Kontext und lässt aus, was der Verlauf schon enthält. Zeitlimit 10 s; ein Fehler ist unkritisch, der Planer startet dann ohne Vorab-Aufrufe. |
| `prefetch_geocode` / `prefetch_research` | Führen `geocode` bzw. `research_knowledge` ohne Planer-Runde aus (`_run_prefetched()`), parallel zueinander. `route_intake()` wählt nur die Knoten, für die das Intake etwas geliefert hat. Ausnahmen werden zu Tool-Fehlern, die der Planer sieht. |
| `planner` | Der PydanticAI-Planer. `planner_history()` fügt die Vorab-Aufrufe samt Ergebnissen so in den Verlauf ein, als hätte er sie selbst ausgeführt. Validatoren (z. B. „Recherche vor jedem Plan“), gespeicherter Verlauf und Evaluation funktionieren dadurch unverändert, und er kann jedes Tool trotzdem selbst (erneut) aufrufen. Das Startdatum aus dem Intake kommt als Hinweis in seine Anweisungen (`Prefetch.planner_note()`, `AgentDeps.intake_note`), nicht in die Nutzernachricht. |
| `finish_plan` / `finish_question` / `finish_answer` / `finish_error` | `route_output()` verzweigt nach Art des Ergebnisses. `finish_answer` hängt die gefundenen Bahnverbindungen an die Antwort an. |

**Warum vorab:** Ohne Intake verbrauchte der Planer, das teuerste Modell, seine ersten Runden mit Geocoding und Recherche, die bei einer neuen Tour ohnehin immer nötig sind. Vorab laufen sie parallel und mit kleinen Modellen; der Planer startet mit Koordinaten und Rechercheergebnis. In der Evaluation sanken die Aufrufe des Planers von 4,1 auf 2,4 je Anfrage, bei gleicher Qualität (README, „Planer messen“).

**Startdatum:** Relative Angaben wie „in acht Tagen“ las der Planer gelegentlich zugleich als Startdatum und als Dauer und fragte dann nach. Das Intake rechnet das Datum aus und gibt es als Hinweis mit. Die Dauer gibt es bewusst nicht mit: gpt-4o-mini erfand sie in 3 von 5 Läufen.

**Laufzeit-Kontext statt State:** HTTP-Client und Session-Factory (`AgentDeps`) sind nicht serialisierbar und kommen deshalb als `context` des Graphen statt im State. UI-Events schreiben die Knoten über den Stream-Writer (`stream_mode="custom"`), das Ergebnis steht im State.

---

## Der Agent

**Datei:** [`backend/app/agent/planner.py`](../backend/app/agent/planner.py)

| Element | Aufgabe |
|---|---|
| `INSTRUCTIONS` | System-Prompt: Pflichtangaben, empfohlenes Vorgehen, Regeln (z. B. „Unterkünfte nur aus Tool-Ergebnissen“, „Etappenenden sind reale Orte“) |
| `build_agent()` | Erzeugt den PydanticAI-`Agent` mit Modell aus `LLM_MODEL`, den Tools und den beiden möglichen Ausgaben |
| `ToolOutput(..., name="final_plan")` / `ToolOutput(..., name="final_train_trip")` / `ToolOutput(..., name="ask_user")` / `ToolOutput(..., name="answer_user")` | Die vier Wege, die Schleife zu beenden: fertiger Plan, Reisekarte einer reinen Bahnfahrt, Rückfrage oder Antwort ohne Plan (z. B. reine Bahnauskunft). Was zutrifft, entscheidet der Agent selbst. Bei `answer_user` hängt das Backend die in diesem Lauf gefundenen Bahnverbindungen formatiert an, das LLM muss sie nicht abschreiben. Die Reisekarte akzeptiert der Validator nur mit Verbindungen, die `find_train_connections` tatsächlich geliefert hat. |
| `@agent.instructions` → `current_context()` | Gibt bei jedem Lauf das heutige Datum und den Nutzernamen mit, damit Angaben wie „nächsten Samstag“ aufgelöst werden können, dazu den Hinweis des Intakes zum Startdatum |
| `@agent.output_validator` → `check_plausibility()` | Selbstprüfung vor der Ausgabe (siehe unten) |
| `used_tool()` | Prüft im Nachrichtenverlauf, ob ein bestimmtes Tool schon aufgerufen wurde |

### Selbstkorrektur

Ein Plan wird nur angenommen, wenn er mehrere Prüfungen besteht. Sonst wirft der Validator `ModelRetry` mit einer konkreten Begründung. PydanticAI schickt diese ans LLM zurück, das den Plan überarbeitet (bis zu `retries=3`):

- Vor jedem Plan muss **eine Recherche in der Wissensbasis** (`research_knowledge`) stattgefunden haben. Das garantiert, dass RAG tatsächlich genutzt wird. Eine Recherche aus der Vorab-Phase des Graphen zählt, denn sie steht als Tool-Aufruf im Verlauf. Ältere Abenteuer mit eigenen `search_knowledge`-Aufrufen des Planers gelten ebenfalls.
- Das **Startdatum** darf nicht in der Vergangenheit liegen.
- Keine Etappe darf länger sein als das Limit der Fortbewegungsart (`MAX_STAGE_KM` in `plan_schema.py`, z. B. Wandern 35 km, Rad 160 km).
- Eine mehrtägige **Hüttenwanderung** mit verschiedenem Start und Ziel muss über `plan_hut_tour` geplant sein (`check_hut_tour`).

### Modellwahl

Das Modell steht in der `.env` (`LLM_MODEL`, z. B. `openai:gpt-5-mini`). Da PydanticAI anbieter-agnostisch ist, lässt sich der Anbieter ohne Codeänderung wechseln, z. B. `groq:openai/gpt-oss-120b`. Voraussetzung ist, dass das Paket-Extra installiert und der API-Key gesetzt ist.

**Ratenlimit:** Der genutzte API-Key erlaubt nur 5 Anfragen pro Minute. Beim Streaming meldet OpenAI ein überschrittenes Limit nicht als HTTP 429, sondern als Fehler im schon geöffneten Stream; weder das SDK noch ein HTTP-Transport können die Anfrage dann wiederholen. Mit `LLM_REQUESTS_PER_MINUTE` wickelt `build_model()` (`agent/models.py`) das Modell deshalb in `RateLimitedModel` (`agent/rate_limit.py`): Es sendet höchstens so viele Anfragen in jedem 60-Sekunden-Fenster und wartet sonst vor dem Senden. OpenAI zählt das Limit je Modell; Agenten mit demselben Modell teilen sich deshalb ein Budget (`shared_budget()`), derzeit Intake und Relevanzprüfung (`gpt-4o-mini`). Der Recherche-Agent hat mit `gpt-4.1-mini` ein eigenes Budget, sodass die Bewertung seiner Suchen nicht auf das nächste Minutenfenster warten muss. Jede Modellanfrage hat außerdem ein Zeitlimit von 120 s (`REQUEST_TIMEOUT_SECONDS` in `agent/models.py`), damit eine hängende Verbindung den Durchlauf nicht blockiert. Kommt trotzdem ein 429 (z. B. weil eine Evaluation parallel dasselbe Limit nutzt), zeigt der Chat einen Hinweis, es in einer Minute erneut zu versuchen.

Weil jeder Modellaufruf zählt, bitten die Anweisungen das Modell, voneinander unabhängige Tools im selben Schritt aufzurufen; PydanticAI führt sie parallel aus. Das halbierte die Modellaufrufe einer dreitägigen Hüttenwanderung (Messwerte in der README unter „Planer messen“).

**Reasoning-Aufwand:** `LLM_REASONING_EFFORT` setzt bei Reasoning-Modellen `openai_reasoning_effort`. Mit `low` sinken die Ausgabe-Tokens auf etwa ein Viertel, der Planer fragte bei einer Radtour mit Zug-Anreise aber nach Uhrzeiten, statt sie selbst zu wählen. Daher bleibt es beim Standard des Modells.

### Der Recherche-Agent (Agent als Tool)

**Dateien:** [`backend/app/agent/research.py`](../backend/app/agent/research.py), [`backend/app/agent/tools/research.py`](../backend/app/agent/tools/research.py)

Die Suche in der Wissensbasis hat der Planer an einen zweiten Agenten abgegeben. Bei einer neuen Tour ruft der Graph das Tool `research_knowledge(region, transport, accommodation_type, extra_topic)` schon vorab auf (Knoten `prefetch_research`), sonst der Planer selbst. Darin läuft der Agent `knowledge-researcher` (`RESEARCH_MODEL`, Standard `gpt-4.1-mini`) mit eigener Schleife und eigenem Tool `search_knowledge`:

1. Er stellt kurze Suchen zu je einem Thema (Region bzw. Route, Ausrüstung und Sicherheit, Übernachtung, Zusatzthema), parallel in einer Runde, und formuliert eine Suche ohne Treffer höchstens einmal um. Nach 5 Suchen blendet `until_search_limit()` (`prepare` des Tools in `agent/research.py`) `search_knowledge` aus, sodass nur noch die Zusammenfassung bleibt: Ohne diese Grenze formulierte der Agent vereinzelt immer weiter um (ein Lauf mit rund 50 Anfragen und 20 Minuten, bis das Anfragelimit die Vorab-Recherche abbrach). Die Relevanzprüfung sammelt die gleichzeitig eintreffenden Treffer dieser Suchen kurz und bewertet sie in einem Aufruf (`grade_hits()` mit gemeinsamem `batch_key` in `rag/relevance.py`).
2. Er fasst die geprüften Fundstellen als `ResearchBrief` zusammen: `region_notes`, `rules`, `packing_tips`, `safety_notes`.
3. Das Tool ergänzt deterministisch aus dem Verlauf des Recherche-Agenten die gestellten Suchen (`queries`) und die Titel der gefundenen Dokumente (`documents`). Daraus entstehen die Quellenangaben (`rag/attribution.py`) und die Prüfung der Evaluation, nicht aus Angaben des LLM.

Der Planer sieht nur Auftrag und Zusammenfassung, nicht die einzelnen Fundstellen. Sein Kontext bleibt kleiner, denn die Fundstellen lagen vorher in jedem weiteren Modellaufruf des Planers erneut im Verlauf. Ruft der Planer das Tool selbst auf, zählt `usage=ctx.usage` die Anfragen des Recherche-Agenten zum Limit des Planer-Laufs (`USAGE_LIMITS`). Im Tracing hängt der Span `knowledge-researcher` dann unter dem Tool-Aufruf des Planers, bei der Vorab-Recherche direkt unter dem Trace der Nachricht.

**Warum nur die Recherche:** Sie ist eine abgeschlossene Teilaufgabe mit klarer Schnittstelle. Route, Etappenorte und Unterkünfte hängen dagegen eng zusammen (deshalb plant `plan_hut_tour` sie gemeinsam). Getrennte Agenten müssten sich ständig abstimmen. Bei 5 Anfragen pro Minute kostet jeder weitere Agent spürbar Zeit. Die Prüfungen bleiben deterministischer Code statt eines Kritiker-Agenten.

**Länge der Anweisungen:** Eine um ein Viertel gekürzte Fassung brachte kaum Ersparnis (die Anweisungen kommen fast immer aus dem Prompt-Cache), verschlechterte aber dieselbe Anfrage: Das Modell behandelte sie als reine Bahnauskunft. Die Abgrenzung zwischen Bahnauskunft und Tour ist für gpt-5-mini empfindlich, Änderungen daran sollten mit `python -m app.evaluation.planner` geprüft werden.

---

## Das Ergebnis-Schema

**Datei:** [`backend/app/agent/plan_schema.py`](../backend/app/agent/plan_schema.py)

- **`AdventurePlan`**: Titel, Zusammenfassung, Fortbewegungsart, Startdatum, Dauer, Start/Ziel, Übernachtungsart, Etappen (`Stage`), `route_ids`, Wetter, Packliste, Sicherheitshinweise, Quellen.
- **`ClarificationQuestion`**: Rückfrage und Liste der fehlenden Angaben.

Der `model_validator` in `AdventurePlan` erzwingt fachliche Regeln:

- Bei mehrtägigen Touren ist `accommodation_type` Pflicht.
- Es gibt genau eine Etappe pro Tag.

Verstößt das LLM dagegen, erhält es die Pydantic-Fehlermeldung und korrigiert sich. So entsteht immer ein **strukturierter, validierter** Plan statt Freitext, und Frontend, Karte, GPX- und PDF-Export können sich darauf verlassen.

---

## Die Tools

**Verzeichnis:** [`backend/app/agent/tools/`](../backend/app/agent/tools/)

Jedes Tool ist eine normale Python-Funktion. PydanticAI erzeugt aus **Docstring und Typ-Hinweisen** die Tool-Beschreibung (JSON-Schema), die das LLM sieht. Die Docstrings sind deshalb Teil der Agentenlogik: Sie bestimmen mit, wann und wie das LLM ein Tool wählt.

| Tool | Datei | Quelle | Wann typischerweise |
|---|---|---|---|
| `research_knowledge` | `research.py` | **Recherche-Agent** mit dem Tool `search_knowledge` (`knowledge.py`) auf der eigenen Wissensbasis (RAG): Vektorsuche, Mindestähnlichkeit, Relevanzprüfung durch ein kleines LLM (`rag/relevance.py`) | Regeln, Tipps, Ausrüstung, Sicherheit |
| `geocode` | `geocode.py` | Nominatim (OSM) | Start, Ziel, Etappenorte |
| `plan_route` | `routing.py`, `named_route.py` | OpenRouteService → BRouter → Schätzung; mit `route_name` Verlauf des Fernwegs aus OSM (Overpass) | Strecke, Höhenmeter, Werte je Etappe; beim Wandern als Dauer die Gehzeit nach DAV-Formel |
| `find_accommodation` | `accommodation.py` | Nominatim-Kategoriesuche → Overpass | Hütten, Camping, Hotels nahe Etappenenden |
| `plan_hut_tour` | `huts.py` | Nominatim-Kategoriesuche, dann `plan_route` | Hüttenwanderung von A nach B: Hütten auf dem Weg samt Route |
| `get_weather` | `weather.py` | Open-Meteo | Vorhersage (≤ 15 Tage) bzw. Klimamittel |
| `estimate_hiking_time` | `routing.py` | lokal (DAV-Formel) | Gehzeit für Etappen ohne berechnete Route (sonst liefert sie `plan_route`) |
| `sun_times` | `sun.py` | lokal (`astral`) | Tageslicht bei langen Etappen |
| `find_train_connections` | `transit.py` | Transitous (MOTIS-Fahrplanrouter) | An- und Abreise mit Bahn/ÖPNV |

### Gemeinsamer Kontext: `AgentDeps`

**Datei:** [`backend/app/agent/deps.py`](../backend/app/agent/deps.py)

Jedes Tool erhält über `ctx.deps` den HTTP-Client, eine DB-Session-Factory, die Einstellungen und das heutige Datum. Eine Factory statt einer einzelnen Session ist nötig, weil Tools parallel laufen können.

`AgentDeps.routes` sammelt die Routengeometrien. Sie gehen **nicht** ans LLM, das würde nur Tokens kosten. Das LLM bekommt eine `route_id` und trägt sie in `AdventurePlan.route_ids` ein. Der Server ordnet beim Speichern die Geometrie zu, für Karte, GPX- und PDF-Export.

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

`search_knowledge(query)` verbindet RAG und Agent. Entscheidend: **Den Suchtext formuliert das LLM**, passend zur aktuellen Planungssituation, z. B. „Wildcampen Österreich Biwak Regeln“ statt der Originalfrage des Nutzers. Das Tool gehört dem Recherche-Agenten. Er sucht mehrmals mit unterschiedlichen Anfragen und gibt dem Planer eine Zusammenfassung (siehe [Der Recherche-Agent](#der-recherche-agent-agent-als-tool)).

### Quellen der Wissensbasis

- `backend/app/seed/documents/*.md`: Beispieldokumente und importierte Wikivoyage-Artikel, eingelesen mit `python -m app.seed`.
- Upload über die Admin-Seite *Wissensbasis* (`api/documents.py`).
- Wikivoyage-Import: `python -m app.seed.wikivoyage "<Titel>"` (Code in `rag/wikivoyage.py` und `seed/wikivoyage.py`).

Details zum Erweitern stehen in der [README](../README.md#wissensbasis-erweitern).

---

## Streaming, Gedächtnis und Speicherung

### Event-Streaming: `agent/runner.py`

`run_planner()` startet den Graphen mit `astream(..., stream_mode=["custom", "values"])`. Der Planer-Knoten läuft mit `run_stream_events()` und übersetzt die PydanticAI-Events in einfache UI-Events (`ui_event()` in `graph.py`); die Vorab-Knoten senden für ihre Aufrufe dieselben `tool_call`- und `tool_result`-Events. Text und Tool-Aufrufe landen schon während des Streams im `RunOutcome`, damit sie auch bei einem Abbruch durch den Client erhalten bleiben:

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
- **Quellen:** Wie die Wikivoyage-Artikel (`knowledge_attributions`, `rag/attribution.py`) bestimmt das Backend auch die Datendienste deterministisch aus dem Verlauf (`data_sources()` in `agent/data_sources.py`). Jedes erfolgreiche Tool-Ergebnis zählt, auch aus früheren Nachrichten und aus den Vorab-Aufrufen. `geocode`, `find_accommodation` und `plan_hut_tour` stehen für Nominatim bzw. Overpass, `plan_route` für den Router aus `source` und für die Herkunft des Fernweg-Verlaufs aus `named_route.source` (Katalog, Waymarked Trails oder Overpass). `get_weather` steht für Open-Meteo, `find_train_connections` mit Verbindungen für Transitous. Je Dienst speichert der Plan unter `data_sources` Name, Link, Zweck, Urheber der Daten und Lizenz, z. B. „© OpenStreetMap-Mitwirkende“ unter ODbL oder Open-Meteo unter CC BY 4.0. Luftlinien-Schätzungen, Fehler und leere Ergebnisse zählen nicht. In `sources` trägt das LLM nur noch Titel von Wissensbasis-Dokumenten ein. Früher nannte es dort auch Datendienste, aber uneinheitlich („OpenRouteService (r-0665adad)“, „plan_hut_tour Ergebnis“) oder gar nicht. Ältere Pläne ergänzt `python -m app.seed`.

---

## Robustheit gegenüber externen Diensten

Öffentliche Dienste fallen zeitweise aus. Jedes Tool hat deshalb eine Rückfallebene:

| Tool | Strategie |
|---|---|
| `plan_route` | Kette ORS → BRouter → weitere ORS-Profile → Luftlinien-Schätzung; sporadische Timeouts werden wiederholt, hängende Profile 10 min übersprungen |
| `plan_route` mit `route_name` | Soll die Tour einem ausgeschilderten Fernweg folgen (z. B. Ostseeküsten-Radweg), kommt dessen Verlauf zuerst aus dem vorab importierten Katalog (Tabelle `long_distance_routes`, `python -m app.seed.long_distance_routes`): sofort verfügbar und mit weiteren Namen wie „Ostseeradweg“ oder „D2“. Kürzel müssen exakt passen, längere Namen auch als Anfang. Fehlt der Fernweg im Katalog, sucht Overpass live die OSM-Routenrelationen an Wegen im Umkreis von 3 km um die Wegpunkte und holt die übrigen Abschnitte derselben Superroute samt Wegen; das dauert auf den öffentlichen Servern oft ein bis zwei Minuten oder scheitert (dann wird normal geroutet). Dabei muss der Name am Anfang passen (ein Kürzel wie „D2“ davor darf fehlen), damit „Ostseeradweg“ nicht den Nord-Ostsee-Radweg trifft; ohne Treffer nennt der Fehler die Routen in der Nähe. Aus den ungeordneten Wegen entsteht ein Graph (Lücken bis 100 m überbrückt), zwischen den Wegpunkten wird der kürzeste Pfad darin gesucht und alle 2 km als Stützpunkt an den Router gegeben (BRouter zuerst, Teilanfragen mit je 40 Punkten). Wegpunkte weiter als 5 km vom Fernweg und Umwege über dem Sechsfachen der Luftlinie werden normal geroutet und im Ergebnis mit Grund genannt. Die Grenze ist bewusst großzügig: Küstenwege um Halbinseln und Buchten sind oft 3-4-mal so lang wie die Luftlinie (Nordseeküsten-Radweg Emden–Wilhelmshaven 201 km bei 62 km), und genau diesen Verlauf will der Nutzer; mit dem früheren Dreifachen wurde solch eine Tour direkt über Land geroutet. Weil der Agent den Verlauf nicht kennt und Etappenorte sonst rät (z. B. Jever, auf der Küstenroute 171 km von Emden), liefert das Ergebnis `checkpoints`: Punkte entlang der Route mit Kilometerstand (Rad alle 10 km, Wandern alle 4 km, höchstens 30). Daraus wählt der Agent den Etappenort bei der gewünschten Tagesdistanz und sucht dort Unterkünfte. Der Validator lässt keinen Plan mit einer älteren Route ohne Fernweg durch: Passt die Fernweg-Strecke nicht in die gewünschten Tage, soll der Agent nach mehr Tagen fragen statt abzukürzen. |
| `find_accommodation` | Nominatim; bei Fehler Overpass mit Circuit Breaker (2 min Pause nach Ausfall) |
| `plan_hut_tour` | Sucht der Agent Hütten rund um Talorte, liegen sie oft abseits: als Sackgasse über dem Tal, quer zur Laufrichtung oder als Alm ohne Übernachtung, und die Route verbindet sie durchs Tal. Das Tool sucht deshalb alle Hütten im Abstand von höchstens 8-10 km zur Luftlinie Start–Ziel, auch rund um Start und Ziel (Kreissuchen, weil Nominatim je Suche höchstens 40 Treffer liefert). Zwischen nahen Talorten liegen oft keine Hütten auf der Achse: Bei Garmisch–Mittenwald liegt die Meilerhütte 7 km seitlich, die Brunnsteinhütte oberhalb des Ziels und fasst Einträge unter 400 m Abstand zusammen. Eine dynamische Programmierung wählt je Nacht eine Hütte in Laufrichtung (nur zum Ziel darf es zurückgehen), sodass die geschätzten Etappen (Luftlinie × 1,4) möglichst gleich lang, mindestens etwa 10 km und höchstens 22 km lang sind. Hütten ohne Hinweis auf Übernachtung (Bettenzahl, Alpenverein als Betreiber oder Hüttenname, aber keine Alm, kein Winterraum) kosten einen hohen Aufschlag. Danach wird die Kette geroutet. Ist eine Etappe länger als 26 km, mehr als 2,2-mal so lang wie die Luftlinie oder steigt sie mehr als 800 m unter beide Enden ab (z. B. Wank-Haus → Meilerhütte über Garmisch), wird genau diese Verbindung gesperrt und die Kette neu gewählt; von bis zu 3 Versuchen gewinnt der mit den wenigsten solchen Etappen. `plan_route` meldet beim Wandern je Etappe den tiefsten Punkt (`lowest_m`); Etappen, die mehr als 500 m unter beide Hütten absteigen, nennt das Ergebnis als Talabstieg. |
| `geocode` | Drosselung auf 1 Anfrage/s (Nominatim-Richtlinie), Cache |
| `find_train_connections` | Fehler oder fehlende Verbindungen als Tool-Ergebnis; der Agent plant ohne konkrete An- und Abreise weiter. Transitous übernimmt den neuen Jahresfahrplan der DB erst um den Fahrplanwechsel (zweiter Sonntag im Dezember); bis dahin fehlen danach oft alle Züge, übrig bleiben Busketten zu ganz anderen Uhrzeiten. Liegt das Datum nach dem nächsten Fahrplanwechsel und keine Verbindung innerhalb von 3 Stunden um die Wunschzeit, gilt der Fahrplan als noch nicht verfügbar: keine Verbindungen, stattdessen ein vorausgefüllter Link zur Verbindungssuche auf bahn.de (Orte als Koordinaten, Datum, Ab- bzw. Ankunftszeit). Das Backend hängt ihn selbst an, unter `answer_user` bzw. im Plan als `timetable_links` (Planansicht und PDF unter „An- und Abreise“); das LLM muss keine URL abschreiben. Die Links stammen aus allen Bahnabfragen des Verlaufs mit Reisedatum vom Tag vor dem Tourstart bis zum Tag nach der letzten Etappe, denn oft fragt der Agent die Bahn in einer Nachricht ab und gibt den Plan erst später aus; nach einer reinen Antwort mit neuer Bahnabfrage aktualisiert das Backend die Links des bestehenden Plans. Verbindungen in Plan und Reisekarte prüft der Validator gegen die Ergebnisse von `find_train_connections`. Passende Verbindungen nach dem Fahrplanwechsel erscheinen mit Hinweis und demselben Link zum Prüfen. Eine DB-Schnittstelle gibt es dafür nicht: Die offizielle Timetables-API kennt nur Abfahrtstafeln, inoffizielle Zugänge (db-rest) sperrt die DB. |
| `research_knowledge` | Fehler des Recherche-Agenten oder keine Treffer als Tool-Ergebnis; der Planer plant ohne RAG-Inhalte weiter. `search_knowledge` meldet dem Recherche-Agenten Fehler (z. B. Embedding-Anbieter nicht erreichbar) ebenso als Ergebnis. Fällt die Relevanzprüfung aus, gehen die Treffer ungeprüft weiter |

---

## Tests ohne echtes LLM

**Dateien:** [`backend/tests/test_agent.py`](../backend/tests/test_agent.py), [`backend/tests/test_graph.py`](../backend/tests/test_graph.py)

Die Agenten-Tests ersetzen das LLM durch ein **geskriptetes `FunctionModel`**. Es ruft Tools in festgelegter Reihenfolge auf und gibt bewusst fehlerhafte Pläne aus. So lässt sich die Schleife deterministisch prüfen:

- Rückfrage bei fehlenden Angaben (`ask_user`)
- Selbstkorrektur: erst Ablehnung ohne Recherche, dann wegen zu langer Etappe, schließlich Annahme
- Der Recherche-Agent läuft dabei mit einem `TestModel` (`conftest.fake_research`); seine eigenen Tests stehen in `test_research.py`
- Folgenachrichten sehen den Verlauf der vorherigen
- Fremde Abenteuer sind nicht erreichbar

`test_graph.py` prüft den Graphen: Verzweigung nach Art des Ergebnisses, Vorab-Aufrufe im Verlauf des Planers (auch fehlgeschlagene), Rückfall auf den Planer allein, wenn das Intake scheitert, keine erneuten Vorab-Aufrufe für Orte aus dem Verlauf, den Startdatum-Hinweis nur in den Anweisungen und das Teilergebnis, wenn der Client die Verbindung trennt.

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
6. Liefert das Tool Daten eines externen Dienstes, diesen mit Lizenz in `PROVIDERS` und `tool_uses()` (`agent/data_sources.py`) eintragen, damit er unter den Quellen des Plans erscheint.
7. Test mit `respx`-Mocks in `backend/tests/test_tools.py` ergänzen.

### Weitere Regeln für den Plan

- Fachliche Pflichtregeln gehören in den `model_validator` von `AdventurePlan`.
- Plausibilitätsprüfungen, die den Kontext brauchen (Datum, Verlauf, genutzte Tools), gehören in `check_plausibility()`.
