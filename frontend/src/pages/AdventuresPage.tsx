import { useEffect, useMemo, useState } from "react";
import { Link, useNavigate, useSearchParams } from "react-router-dom";
import { api, json } from "../api/client";
import type { Adventure, AdventureListItem } from "../api/types";
import { ConfirmButton } from "../components/ConfirmButton";
import { formatDateTime } from "../lib/labels";

/** Kleinbuchstaben ohne Akzente, damit "zurich" auch "Zürich" findet. */
const normalize = (text: string) =>
  text
    .normalize("NFD")
    .replace(/\p{Diacritic}/gu, "")
    .toLowerCase();

export function AdventuresPage() {
  const navigate = useNavigate();
  const [adventures, setAdventures] = useState<AdventureListItem[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  // Suchbegriff in der URL: bleibt erhalten, wenn man aus einem Abenteuer zurückkommt
  const [searchParams, setSearchParams] = useSearchParams();
  const query = searchParams.get("q") ?? "";
  const filtered = useMemo(() => {
    const words = normalize(query).split(/\s+/).filter(Boolean);
    return (
      adventures?.filter((a) => {
        // Titel und Orte des Plans (Start, Ziel, Etappenorte, Unterkünfte), z. B. "Dangast" auch ohne Titelnennung
        const text = normalize([a.title, ...a.places].join(" "));
        return words.every((word) => text.includes(word));
      }) ?? null
    );
  }, [adventures, query]);

  function search(text: string) {
    setSearchParams(text ? { q: text } : {}, { replace: true });
  }

  useEffect(() => {
    api<AdventureListItem[]>("/api/adventures").then(setAdventures).catch((e) => setError(e.message));
  }, []);

  async function createAdventure() {
    try {
      const adventure = await api<Adventure>("/api/adventures", { method: "POST", body: json({}) });
      navigate(`/adventures/${adventure.id}`);
    } catch (e) {
      setError((e as Error).message);
    }
  }

  async function remove(id: number) {
    await api(`/api/adventures/${id}`, { method: "DELETE" });
    setAdventures((list) => list?.filter((a) => a.id !== id) ?? null);
  }

  return (
    <div className="page">
      <div className="page-header">
        <div>
          <h1>Deine Abenteuer</h1>
          <p className="muted">Beschreibe Deine Tour in eigenen Worten – der Agent plant Route, Etappen, Wetter und Unterkünfte.</p>
        </div>
        <button className="primary" onClick={createAdventure}>
          + Neues Abenteuer
        </button>
      </div>
      {error && <p className="error">{error}</p>}
      {adventures === null ? (
        <p className="muted">Lade …</p>
      ) : adventures.length === 0 ? (
        <div className="card empty">
          <p>Noch keine Abenteuer geplant.</p>
          <button className="primary" onClick={createAdventure}>
            Erstes Abenteuer planen
          </button>
        </div>
      ) : (
        <>
          <input
            type="search"
            className="search"
            placeholder="Titel oder Ort suchen, z. B. Wanderung oder Dangast"
            aria-label="Abenteuer durchsuchen"
            value={query}
            onChange={(e) => search(e.target.value)}
          />
          {filtered?.length === 0 && <p className="muted">Kein Abenteuer passt zu „{query}“.</p>}
          <div className="grid">
            {filtered?.map((a) => (
              <Link key={a.id} to={`/adventures/${a.id}`} className="card adventure-card">
                <div className="row between">
                  <span className={`badge ${a.status}`}>{a.status === "planned" ? "Geplant" : "Entwurf"}</span>
                  <ConfirmButton onConfirm={() => remove(a.id)} />
                </div>
                <h2>{a.title}</h2>
                <p className="muted small">Zuletzt bearbeitet {formatDateTime(a.updated_at)}</p>
              </Link>
            ))}
          </div>
        </>
      )}
    </div>
  );
}
