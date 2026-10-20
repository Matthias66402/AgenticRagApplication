import { useEffect, useState, type FormEvent } from "react";
import { api, json } from "../api/client";
import type { Role, User } from "../api/types";
import { useAuth } from "../auth/AuthContext";
import { ConfirmButton } from "../components/ConfirmButton";
import { formatDateTime } from "../lib/labels";

const EMPTY = { email: "", display_name: "", password: "", role: "user" as Role };

function UserRow({ user, isSelf, onChange, onDelete }: {
  user: User;
  isSelf: boolean;
  onChange: (changes: Partial<User> & { password?: string }) => Promise<void>;
  onDelete: () => void;
}) {
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState({ display_name: user.display_name, email: user.email, password: "" });

  async function save(e: FormEvent) {
    e.preventDefault();
    await onChange({
      display_name: draft.display_name,
      email: draft.email,
      ...(draft.password ? { password: draft.password } : {}),
    });
    setEditing(false);
  }

  if (editing) {
    return (
      <tr>
        <td colSpan={5}>
          <form className="row wrap" onSubmit={save}>
            <input required value={draft.display_name} onChange={(e) => setDraft({ ...draft, display_name: e.target.value })} />
            <input type="email" required value={draft.email} onChange={(e) => setDraft({ ...draft, email: e.target.value })} />
            <input
              type="password"
              minLength={8}
              placeholder="Neues Passwort (optional)"
              value={draft.password}
              onChange={(e) => setDraft({ ...draft, password: e.target.value })}
            />
            <button className="primary">Speichern</button>
            <button type="button" className="ghost" onClick={() => setEditing(false)}>
              Abbrechen
            </button>
          </form>
        </td>
      </tr>
    );
  }

  return (
    <tr className={user.is_active ? "" : "inactive"}>
      <td>
        <strong>{user.display_name}</strong>
        <div className="muted small">{user.email}</div>
      </td>
      <td>
        <select value={user.role} disabled={isSelf} onChange={(e) => onChange({ role: e.target.value as Role })}>
          <option value="user">Nutzer</option>
          <option value="admin">Admin</option>
        </select>
      </td>
      <td>
        <label className="toggle">
          <input
            type="checkbox"
            checked={user.is_active}
            disabled={isSelf}
            onChange={(e) => onChange({ is_active: e.target.checked })}
          />
          {user.is_active ? "aktiv" : "deaktiviert"}
        </label>
      </td>
      <td className="muted small">{formatDateTime(user.created_at)}</td>
      <td className="actions">
        <button className="ghost" onClick={() => setEditing(true)}>
          Bearbeiten
        </button>
        {!isSelf && <ConfirmButton onConfirm={onDelete} />}
      </td>
    </tr>
  );
}

export function AdminUsersPage() {
  const { user: me } = useAuth();
  const [users, setUsers] = useState<User[]>([]);
  const [form, setForm] = useState(EMPTY);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api<User[]>("/api/users").then(setUsers).catch((e) => setError(e.message));
  }, []);

  async function run(action: () => Promise<void>) {
    setError(null);
    try {
      await action();
    } catch (e) {
      setError((e as Error).message);
    }
  }

  const create = (e: FormEvent) => {
    e.preventDefault();
    run(async () => {
      const created = await api<User>("/api/users", { method: "POST", body: json(form) });
      setUsers([...users, created]);
      setForm(EMPTY);
    });
  };

  const update = (id: number) => (changes: Partial<User> & { password?: string }) =>
    run(async () => {
      const updated = await api<User>(`/api/users/${id}`, { method: "PATCH", body: json(changes) });
      setUsers((list) => list.map((u) => (u.id === id ? updated : u)));
    });

  const remove = (id: number) =>
    run(async () => {
      await api(`/api/users/${id}`, { method: "DELETE" });
      setUsers((list) => list.filter((u) => u.id !== id));
    });

  return (
    <div className="page">
      <h1>Nutzerverwaltung</h1>
      {error && <p className="error">{error}</p>}

      <form className="card row wrap" onSubmit={create}>
        <input required placeholder="Name" value={form.display_name} onChange={(e) => setForm({ ...form, display_name: e.target.value })} />
        <input type="email" required placeholder="E-Mail" value={form.email} onChange={(e) => setForm({ ...form, email: e.target.value })} />
        <input
          type="password"
          required
          minLength={8}
          placeholder="Passwort"
          value={form.password}
          onChange={(e) => setForm({ ...form, password: e.target.value })}
        />
        <select value={form.role} onChange={(e) => setForm({ ...form, role: e.target.value as Role })}>
          <option value="user">Nutzer</option>
          <option value="admin">Admin</option>
        </select>
        <button className="primary">Nutzer anlegen</button>
      </form>

      <div className="card table-wrap">
        <table>
          <thead>
            <tr>
              <th>Nutzer</th>
              <th>Rolle</th>
              <th>Status</th>
              <th>Angelegt</th>
              <th />
            </tr>
          </thead>
          <tbody>
            {users.map((u) => (
              <UserRow key={u.id} user={u} isSelf={u.id === me?.id} onChange={update(u.id)} onDelete={() => remove(u.id)} />
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
