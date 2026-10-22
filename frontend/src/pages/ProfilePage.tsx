import { useState, type FormEvent } from "react";
import { api, json } from "../api/client";
import type { DefaultTransport, User } from "../api/types";
import { useAuth } from "../auth/AuthContext";
import { TRANSPORT_LABELS } from "../lib/labels";

const DEFAULT_TRANSPORTS: DefaultTransport[] = ["hiking", "cycling"];

export function ProfilePage() {
  const { user, setUser } = useAuth();
  const [form, setForm] = useState({
    display_name: user!.display_name,
    email: user!.email,
    password: "",
    home_address: user!.home_address ?? "",
    home_city: user!.home_city ?? "",
    default_transport: user!.default_transport ?? "",
  });
  const [status, setStatus] = useState<{ ok: boolean; text: string } | null>(null);

  async function submit(e: FormEvent) {
    e.preventDefault();
    const changes: Record<string, string | null> = {};
    if (form.display_name !== user!.display_name) changes.display_name = form.display_name;
    if (form.email !== user!.email) changes.email = form.email;
    if (form.password) changes.password = form.password;
    // Leere Felder löschen die Angabe
    if (form.home_address !== (user!.home_address ?? "")) changes.home_address = form.home_address;
    if (form.home_city !== (user!.home_city ?? "")) changes.home_city = form.home_city;
    if (form.default_transport !== (user!.default_transport ?? "")) changes.default_transport = form.default_transport || null;
    if (Object.keys(changes).length === 0) return setStatus({ ok: true, text: "Keine Änderungen." });
    try {
      const updated = await api<User>("/api/users/me", { method: "PATCH", body: json(changes) });
      setUser(updated);
      setForm({
        ...form,
        password: "",
        home_address: updated.home_address ?? "",
        home_city: updated.home_city ?? "",
      });
      setStatus({ ok: true, text: "Gespeichert." });
    } catch (err) {
      setStatus({ ok: false, text: (err as Error).message });
    }
  }

  return (
    <div className="page narrow">
      <h1>Profil</h1>
      <form className="card stack" onSubmit={submit}>
        <label>
          Name
          <input required value={form.display_name} onChange={(e) => setForm({ ...form, display_name: e.target.value })} />
        </label>
        <label>
          E-Mail
          <input type="email" required value={form.email} onChange={(e) => setForm({ ...form, email: e.target.value })} />
        </label>
        <label>
          Neues Passwort <span className="muted">(leer lassen, um es beizubehalten)</span>
          <input
            type="password"
            minLength={8}
            autoComplete="new-password"
            value={form.password}
            onChange={(e) => setForm({ ...form, password: e.target.value })}
          />
        </label>

        <h2>Vorgaben für die Planung</h2>
        <p className="muted small">
          Nennst Du im Chat keinen anderen Startort bzw. keine andere Fortbewegungsart, plant der Assistent mit diesen
          Angaben. Die Adresse wird dafür an das Sprachmodell und an die Ortssuche (OpenStreetMap) übermittelt.
        </p>
        <label>
          Adresse <span className="muted">(Straße und Hausnummer)</span>
          <input
            maxLength={200}
            autoComplete="street-address"
            value={form.home_address}
            onChange={(e) => setForm({ ...form, home_address: e.target.value })}
          />
        </label>
        <label>
          Ort <span className="muted">(PLZ und Ort)</span>
          <input
            maxLength={100}
            autoComplete="address-level2"
            value={form.home_city}
            onChange={(e) => setForm({ ...form, home_city: e.target.value })}
          />
        </label>
        <label>
          Standard-Fortbewegung
          <select
            value={form.default_transport}
            onChange={(e) => setForm({ ...form, default_transport: e.target.value as DefaultTransport | "" })}
          >
            <option value="">Keine Vorgabe</option>
            {DEFAULT_TRANSPORTS.map((mode) => (
              <option key={mode} value={mode}>
                {TRANSPORT_LABELS[mode]}
              </option>
            ))}
          </select>
        </label>

        <p className="muted small">Rolle: {user!.role === "admin" ? "Administrator" : "Nutzer"}</p>
        {status && <p className={status.ok ? "success" : "error"}>{status.text}</p>}
        <button className="primary">Speichern</button>
      </form>
    </div>
  );
}
