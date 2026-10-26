import { useMemo, useState } from "react";
import { downloadFile } from "../api/client";
import type { AdventurePlan } from "../api/types";
import { useAuth } from "../auth/AuthContext";
import { ACCOMMODATION_LABELS, TRANSPORT_LABELS, formatDate, shortPlace } from "../lib/labels";
import { ElevationProfile, buildProfile } from "./ElevationProfile";
import type { ProfilePoint } from "./ElevationProfile";
import { JourneyCard, TransitNote } from "./JourneyCard";
import { LinkedText } from "./LinkedText";
import { RouteMap } from "./RouteMap";

const num = (value: number | null | undefined, unit: string) =>
  value === null || value === undefined ? null : `${value.toLocaleString("de-DE", { maximumFractionDigits: 1 })} ${unit}`;

interface DownloadButtonProps {
  path: string;
  fallbackName: string;
  label: string;
  busyLabel: string;
  title: string;
}

export function DownloadButton({ path, fallbackName, label, busyLabel, title }: DownloadButtonProps) {
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function download() {
    setBusy(true);
    setError(null);
    try {
      await downloadFile(path, fallbackName);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="download">
      <button onClick={download} disabled={busy} title={title}>
        {busy ? busyLabel : label}
      </button>
      {error && <span className="error small">{error}</span>}
    </div>
  );
}

export function PlanCard({ plan, adventureId }: { plan: AdventurePlan; adventureId: number }) {
  const totalKm = plan.stages.reduce((sum, s) => sum + s.distance_km, 0);
  const totalAscent = plan.stages.reduce((sum, s) => sum + (s.ascent_m ?? 0), 0);
  const hasTrack = plan.route_ids.some((id) => (plan.routes?.[id]?.length ?? 0) > 0);
  const profile = useMemo(() => buildProfile(plan), [plan]);
  const [hovered, setHovered] = useState<ProfilePoint | null>(null);
  const attributions = plan.attributions ?? [];
  const timetableLinks = plan.timetable_links ?? [];
  const multiDay = plan.stages.length > 1;
  // Mehrtagestouren als Gesamt-GPX oder eine GPX je Tagesetappe; Vorauswahl aus dem Profil, hier umschaltbar
  const [perDay, setPerDay] = useState(useAuth().user?.gpx_mode === "per_day");
  const gpxPerDay = multiDay && perDay;

  return (
    <article className="plan">
      <header>
        <div className="plan-title">
          <h2>{plan.title}</h2>
          <div className="downloads">
            <DownloadButton
              path={`/api/adventures/${adventureId}/pdf`}
              fallbackName="abenteuer.pdf"
              label="⬇ PDF"
              busyLabel="Erstelle PDF …"
              title="Druckversion mit Routenskizze, Höhenprofil, Etappen und Packliste"
            />
            {multiDay && (
              <select
                className="gpx-mode"
                value={perDay ? "per_day" : "combined"}
                onChange={(e) => setPerDay(e.target.value === "per_day")}
                aria-label="GPX-Download für Mehrtagestouren"
                title="Gesamte Tour als eine GPX oder je Tag eine eigene GPX herunterladen"
              >
                <option value="combined">Gesamt-GPX</option>
                <option value="per_day">GPX je Tag</option>
              </select>
            )}
            {gpxPerDay ? (
              plan.stages.map((stage) => (
                <DownloadButton
                  key={stage.day}
                  path={`/api/adventures/${adventureId}/gpx?day=${stage.day}`}
                  fallbackName={`abenteuer_${stage.day}.gpx`}
                  label={`⬇ GPX Tag ${stage.day}`}
                  busyLabel="Erstelle GPX …"
                  title={hasTrack ? `Track und Wegpunkte für Tag ${stage.day}` : `Wegpunkte für Tag ${stage.day} (keine berechnete Route)`}
                />
              ))
            ) : (
              <DownloadButton
                path={`/api/adventures/${adventureId}/gpx`}
                fallbackName="abenteuer.gpx"
                label={hasTrack ? "⬇ GPX" : "⬇ GPX (nur Wegpunkte)"}
                busyLabel="Erstelle GPX …"
                title={hasTrack ? "Track mit Höhenprofil und Wegpunkten" : "Keine berechnete Route – enthält nur Wegpunkte"}
              />
            )}
          </div>
        </div>
        <div className="chips">
          <span className="chip">{TRANSPORT_LABELS[plan.transport]}</span>
          <span className="chip">📅 {formatDate(plan.start_date)}</span>
          <span className="chip">
            {plan.duration_days} {plan.duration_days === 1 ? "Tag" : "Tage"}
          </span>
          {plan.accommodation_type && <span className="chip">🛏️ {ACCOMMODATION_LABELS[plan.accommodation_type]}</span>}
          <span className="chip">↔ {num(totalKm, "km")}</span>
          {totalAscent > 0 && <span className="chip">↗ {num(totalAscent, "Hm")}</span>}
        </div>
        <p>
          <LinkedText text={plan.summary} />
        </p>
      </header>

      <RouteMap plan={plan} highlight={hovered && [hovered.lat, hovered.lon]} />

      {profile && (
        <section>
          <h3>Höhenprofil</h3>
          <ElevationProfile plan={plan} points={profile} onHover={setHovered} />
        </section>
      )}

      <section>
        <h3>Etappen</h3>
        <ol className="stages">
          {plan.stages.map((s) => (
            <li key={s.day} className="stage">
              <div className="stage-day">
                <strong>Tag {s.day}</strong>
                <span className="muted small">{formatDate(s.date)}</span>
              </div>
              <div className="stage-body">
                <div className="stage-route">
                  {shortPlace(s.start.name)} → {shortPlace(s.end.name)}
                </div>
                <div className="muted small">
                  {[
                    num(s.distance_km, "km"),
                    s.ascent_m ? `↗ ${num(s.ascent_m, "Hm")}` : null,
                    s.descent_m ? `↘ ${num(s.descent_m, "Hm")}` : null,
                    s.duration_hours ? `⏱ ${num(s.duration_hours, "h")}` : null,
                  ]
                    .filter(Boolean)
                    .join(" · ")}
                </div>
                {s.accommodation && (
                  <div className="small">
                    🛏️{" "}
                    {s.accommodation.website ? (
                      <a href={s.accommodation.website} target="_blank" rel="noreferrer">
                        {s.accommodation.name}
                      </a>
                    ) : (
                      s.accommodation.name
                    )}
                  </div>
                )}
                {s.notes && (
                  <p className="small">
                    <LinkedText text={s.notes} />
                  </p>
                )}
              </div>
            </li>
          ))}
        </ol>
      </section>

      {(plan.outbound_journey || plan.return_journey || timetableLinks.length > 0) && (
        <section>
          <h3>An- und Abreise</h3>
          {(plan.outbound_journey || plan.return_journey) && (
            <div className="journeys">
              {plan.outbound_journey && <JourneyCard label="🚆 Anreise" journey={plan.outbound_journey} />}
              {plan.return_journey && <JourneyCard label="🚆 Rückreise" journey={plan.return_journey} />}
            </div>
          )}
          {timetableLinks.length > 0 && (
            <ul className="timetable-links">
              {timetableLinks.map((link) => (
                <li key={link.url}>
                  🚆 {link.label}:{" "}
                  {link.missing ? "noch kein Fahrplan verfügbar" : "Fahrplan nach dem Fahrplanwechsel evtl. unvollständig"}{" "}
                  –{" "}
                  <a href={link.url} target="_blank" rel="noreferrer">
                    Verbindungen bei bahn.de
                  </a>
                </li>
              ))}
            </ul>
          )}
          {(plan.outbound_journey || plan.return_journey) && <TransitNote />}
        </section>
      )}

      {plan.weather_summary && (
        <section>
          <h3>Wetter</h3>
          <p>
            <LinkedText text={plan.weather_summary} />
          </p>
        </section>
      )}
      {plan.packing_tips.length > 0 && (
        <section>
          <h3>Packliste & Tipps</h3>
          <ul>
            {plan.packing_tips.map((t, i) => (
              <li key={i}>
                <LinkedText text={t} />
              </li>
            ))}
          </ul>
        </section>
      )}
      {plan.safety_notes.length > 0 && (
        <section>
          <h3>Sicherheit & Regeln</h3>
          <ul>
            {plan.safety_notes.map((t, i) => (
              <li key={i}>
                <LinkedText text={t} />
              </li>
            ))}
          </ul>
        </section>
      )}
      {(plan.sources.length > 0 || attributions.length > 0) && (
        <section>
          <h3>Quellen</h3>
          <ul className="muted small">
            {plan.sources.map((t, i) => (
              <li key={i}>
                <LinkedText text={t} />
              </li>
            ))}
          </ul>
          {attributions.length > 0 && (
            <div className="attributions small">
              <p className="muted">
                Für diesen Plan wurden Inhalte der folgenden Artikel zusammengefasst und bearbeitet. Sie stehen unter
                der jeweils genannten Lizenz.
              </p>
              <ul>
                {attributions.map((a) => (
                  <li key={a.url}>
                    <a href={a.url} target="_blank" rel="noreferrer">
                      „{a.title}“
                    </a>{" "}
                    – {a.publisher}, Version vom {formatDate(a.revision_date)} ·{" "}
                    <a href={a.history_url} target="_blank" rel="noreferrer">
                      Autoren
                    </a>{" "}
                    · Lizenz:{" "}
                    <a href={a.license_url} target="_blank" rel="noreferrer license">
                      {a.license}
                    </a>
                  </li>
                ))}
              </ul>
            </div>
          )}
        </section>
      )}
    </article>
  );
}
