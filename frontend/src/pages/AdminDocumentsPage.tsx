import { useEffect, useRef, useState, type FormEvent } from "react";
import { api } from "../api/client";
import type { KnowledgeDocument } from "../api/types";
import { ConfirmButton } from "../components/ConfirmButton";
import { formatDateTime } from "../lib/labels";

export function AdminDocumentsPage() {
  const [documents, setDocuments] = useState<KnowledgeDocument[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [uploading, setUploading] = useState(false);
  const fileInput = useRef<HTMLInputElement>(null);

  useEffect(() => {
    api<KnowledgeDocument[]>("/api/documents").then(setDocuments).catch((e) => setError(e.message));
  }, []);

  async function upload(e: FormEvent) {
    e.preventDefault();
    const file = fileInput.current?.files?.[0];
    if (!file) return;
    const body = new FormData();
    body.append("file", file);
    setUploading(true);
    setError(null);
    try {
      const doc = await api<KnowledgeDocument>("/api/documents", { method: "POST", body });
      setDocuments((list) => [...list.filter((d) => d.id !== doc.id && d.source !== doc.source), doc]);
      fileInput.current!.value = "";
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setUploading(false);
    }
  }

  async function remove(id: number) {
    await api(`/api/documents/${id}`, { method: "DELETE" });
    setDocuments((list) => list.filter((d) => d.id !== id));
  }

  return (
    <div className="page">
      <h1>Wissensbasis</h1>
      <p className="muted">
        Dokumente werden in Abschnitte zerlegt, eingebettet und in pgvector gespeichert. Der Agent durchsucht sie mit dem
        Tool <code>search_knowledge</code>. Gleicher Dateiname ersetzt die vorhandene Version.
      </p>
      {error && <p className="error">{error}</p>}
      <form className="card row wrap" onSubmit={upload}>
        <input ref={fileInput} type="file" accept=".md,.txt,.pdf" required />
        <button className="primary" disabled={uploading}>
          {uploading ? "Wird eingelesen …" : "Hochladen & einlesen"}
        </button>
      </form>
      <div className="card table-wrap">
        <table>
          <thead>
            <tr>
              <th>Dokument</th>
              <th>Abschnitte</th>
              <th>Eingelesen</th>
              <th />
            </tr>
          </thead>
          <tbody>
            {documents.map((d) => (
              <tr key={d.id}>
                <td>
                  <strong>{d.title}</strong>
                  <div className="muted small">{d.source}</div>
                </td>
                <td>{d.chunk_count}</td>
                <td className="muted small">{formatDateTime(d.created_at)}</td>
                <td className="actions">
                  <ConfirmButton onConfirm={() => remove(d.id)} />
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
