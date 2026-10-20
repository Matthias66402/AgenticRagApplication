import type { AdventurePlan } from "../api/types";
import { ACCOMMODATION_LABELS, TRANSPORT_LABELS, formatDate, shortPlace } from "../lib/labels";
import { RouteMap } from "./RouteMap";

const num = (value: number | null | undefined, unit: string) =>
  value === null || value === undefined ? null : `${value.toLocaleString("de-DE", { maximumFractionDigits: 1 })} ${unit}`;

export function PlanCard({ plan }: { plan: AdventurePlan }) {
  const totalKm = plan.stages.reduce((sum, s) => sum + s.distance_km, 0);
  const totalAscent = plan.stages.reduce((sum, s) => sum + (s.ascent_m ?? 0), 0);

  return (
    <article className="plan">
      <header>
        <h2>{plan.title}</h2>
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
        <p>{plan.summary}</p>
      </header>

      <RouteMap plan={plan} />

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
                {s.notes && <p className="small">{s.notes}</p>}
              </div>
            </li>
          ))}
        </ol>
      </section>

      {plan.weather_summary && (
        <section>
          <h3>Wetter</h3>
          <p>{plan.weather_summary}</p>
        </section>
      )}
      {plan.packing_tips.length > 0 && (
        <section>
          <h3>Packliste & Tipps</h3>
          <ul>
            {plan.packing_tips.map((t, i) => (
              <li key={i}>{t}</li>
            ))}
          </ul>
        </section>
      )}
      {plan.safety_notes.length > 0 && (
        <section>
          <h3>Sicherheit & Regeln</h3>
          <ul>
            {plan.safety_notes.map((t, i) => (
              <li key={i}>{t}</li>
            ))}
          </ul>
        </section>
      )}
      {plan.sources.length > 0 && (
        <section>
          <h3>Quellen</h3>
          <ul className="muted small">
            {plan.sources.map((t, i) => (
              <li key={i}>{t}</li>
            ))}
          </ul>
        </section>
      )}
    </article>
  );
}
