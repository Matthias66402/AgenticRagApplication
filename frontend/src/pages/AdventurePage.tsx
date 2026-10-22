import { useCallback, useEffect, useRef, useState, type FormEvent, type KeyboardEvent } from "react";
import { Link, useParams } from "react-router-dom";
import { api, json, streamChat } from "../api/client";
import type { Adventure, ChatEvent, StoredPlan, ToolCall } from "../api/types";
import { LinkedText } from "../components/LinkedText";
import { PlanCard } from "../components/PlanCard";
import { ToolCallBadge } from "../components/ToolCallBadge";
import { TrainTripCard } from "../components/TrainTripCard";

interface LiveMessage {
  content: string;
  tool_calls: ToolCall[];
  error?: string;
}

const EXAMPLES = [
  "3-tägige Hüttenwanderung von Garmisch-Partenkirchen nach Innsbruck, Start nächsten Freitag",
  "Radtour am Wochenende von Ulm an den Bodensee mit Übernachtung auf dem Campingplatz",
  "Tageswanderung morgen rund um den Tegernsee",
];

export function AdventurePage() {
  const id = Number(useParams().id);
  const [adventure, setAdventure] = useState<Adventure | null>(null);
  const [plan, setPlan] = useState<StoredPlan | null>(null);
  const [input, setInput] = useState("");
  const [pendingUser, setPendingUser] = useState<string | null>(null);
  const [live, setLive] = useState<LiveMessage | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [editingTitle, setEditingTitle] = useState(false);
  const chatEnd = useRef<HTMLDivElement>(null);
  const abort = useRef<AbortController | null>(null);

  const load = useCallback(async () => {
    const data = await api<Adventure>(`/api/adventures/${id}`);
    setAdventure(data);
    setPlan(data.plan);
  }, [id]);

  useEffect(() => {
    load().catch((e) => setError(e.message));
    return () => abort.current?.abort();
  }, [load]);

  useEffect(() => {
    chatEnd.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [adventure?.messages.length, live, pendingUser]);

  const busy = live !== null;

  function handleEvent(event: ChatEvent) {
    setLive((current) => {
      const m = current ?? { content: "", tool_calls: [] };
      switch (event.type) {
        case "token":
          return { ...m, content: m.content + event.content };
        case "tool_call":
          return { ...m, tool_calls: [...m.tool_calls, { id: event.id, tool: event.tool, args: event.args }] };
        case "tool_result":
          return {
            ...m,
            tool_calls: m.tool_calls.map((c) =>
              c.id === event.id ? { ...c, result: event.result, failed: event.failed } : c,
            ),
          };
        case "question":
          return { ...m, content: event.question };
        case "answer":
          return { ...m, content: event.content };
        case "error":
          return { ...m, error: event.message };
        default:
          return m;
      }
    });
    if (event.type === "plan") setPlan(event.plan);
  }

  async function send(text: string) {
    const message = text.trim();
    if (!message || busy) return;
    setInput("");
    setError(null);
    setPendingUser(message);
    setLive({ content: "", tool_calls: [] });
    abort.current = new AbortController();
    try {
      await streamChat(id, message, handleEvent, abort.current.signal);
      await load();
    } catch (e) {
      if ((e as Error).name !== "AbortError") setError((e as Error).message);
    } finally {
      setPendingUser(null);
      setLive(null);
    }
  }

  function onSubmit(e: FormEvent) {
    e.preventDefault();
    send(input);
  }

  function onKeyDown(e: KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      send(input);
    }
  }

  async function saveTitle(title: string) {
    setEditingTitle(false);
    if (!adventure || !title.trim() || title === adventure.title) return;
    setAdventure(await api<Adventure>(`/api/adventures/${id}`, { method: "PATCH", body: json({ title }) }));
  }

  if (error && !adventure) return <div className="page error">{error}</div>;
  if (!adventure) return <div className="page muted">Lade …</div>;

  const messages = adventure.messages;
  const isEmpty = messages.length === 0 && !pendingUser;

  return (
    <div className="planner">
      <section className="chat-pane">
        <div className="chat-header">
          <Link to="/" className="muted small">
            ← Alle Abenteuer
          </Link>
          {editingTitle ? (
            <input
              autoFocus
              defaultValue={adventure.title}
              onBlur={(e) => saveTitle(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && saveTitle((e.target as HTMLInputElement).value)}
            />
          ) : (
            <h1 onClick={() => setEditingTitle(true)} title="Klicken zum Umbenennen">
              {adventure.title}
            </h1>
          )}
        </div>

        <div className="messages">
          {isEmpty && (
            <div className="intro">
              <p>
                Beschreibe Dein Abenteuer: <strong>Fortbewegungsart, Zeitpunkt, Dauer, Start und Ziel</strong> – bei
                mehrtägigen Touren auch, wie Du übernachten möchtest. Fehlt etwas, fragt der Agent nach.
              </p>
              <div className="examples">
                {EXAMPLES.map((example) => (
                  <button key={example} className="example" onClick={() => send(example)}>
                    {example}
                  </button>
                ))}
              </div>
            </div>
          )}
          {messages.map((m) =>
            m.role === "user" ? (
              <div key={m.id} className="msg user">
                <LinkedText text={m.content} />
              </div>
            ) : (
              <div key={m.id} className="msg assistant">
                {m.tool_calls.length > 0 && (
                  <details className="tool-summary">
                    <summary>{m.tool_calls.length} Tool-Aufrufe</summary>
                    {m.tool_calls.map((c) => (
                      <ToolCallBadge key={c.id} call={c} />
                    ))}
                  </details>
                )}
                <div className="msg-text">
                  <LinkedText text={m.content} />
                </div>
              </div>
            ),
          )}
          {pendingUser && (
            <div className="msg user">
              <LinkedText text={pendingUser} />
            </div>
          )}
          {live && (
            <div className="msg assistant live">
              {live.tool_calls.map((c) => (
                <ToolCallBadge key={c.id} call={c} />
              ))}
              {live.content && (
                <div className="msg-text">
                  <LinkedText text={live.content} />
                </div>
              )}
              {live.error && <p className="error">{live.error}</p>}
              {!live.content && !live.error && (
                <div className="muted small thinking">
                  <span className="spinner" /> Agent arbeitet …
                </div>
              )}
            </div>
          )}
          <div ref={chatEnd} />
        </div>

        {error && <p className="error">{error}</p>}
        <form className="composer" onSubmit={onSubmit}>
          <textarea
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={onKeyDown}
            placeholder={plan ? "Änderungswunsch, z. B. „Mach daraus 4 Tage“ …" : "Beschreibe Dein Abenteuer …"}
            rows={3}
            disabled={busy}
          />
          <button className="primary" disabled={busy || !input.trim()}>
            {busy ? "Plant …" : "Senden"}
          </button>
        </form>
      </section>

      <section className="plan-pane">
        {plan?.kind === "train_trip" ? (
          <TrainTripCard plan={plan} adventureId={id} />
        ) : plan ? (
          <PlanCard plan={plan} adventureId={id} />
        ) : (
          <div className="plan-placeholder muted">
            <div className="big-icon" aria-hidden>
              🗺️
            </div>
            <p>Hier erscheint Dein Plan mit Karte, Etappen, Wetter und Tipps.</p>
          </div>
        )}
      </section>
    </div>
  );
}
