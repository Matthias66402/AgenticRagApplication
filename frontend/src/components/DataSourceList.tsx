import type { DataSource } from "../api/types";

/** Datendienste des Plans mit Urheber und Lizenz (z. B. OpenStreetMap-Daten unter ODbL). */
export function DataSourceList({ sources }: { sources: DataSource[] }) {
  if (sources.length === 0) return null;
  return (
    <div className="attributions small">
      <p className="muted">Daten von:</p>
      <ul>
        {sources.map((s) => (
          <li key={s.name}>
            <a href={s.url} target="_blank" rel="noreferrer">
              {s.name}
            </a>{" "}
            ({s.purpose})
            {s.data && <> · {s.data.startsWith("©") ? `Daten ${s.data}` : s.data}</>}
            {s.license && (
              <>
                {" "}
                · Lizenz:{" "}
                {s.license_url ? (
                  <a href={s.license_url} target="_blank" rel="noreferrer license">
                    {s.license}
                  </a>
                ) : (
                  s.license
                )}
              </>
            )}
          </li>
        ))}
      </ul>
    </div>
  );
}
