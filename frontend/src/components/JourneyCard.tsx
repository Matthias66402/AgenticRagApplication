import type { Journey } from "../api/types";
import { formatDate } from "../lib/labels";
import { LinkedText } from "./LinkedText";

// Zeiten kommen als Ortszeit ohne Zeitzone ("2026-10-24T07:36") und werden nicht umgerechnet
const time = (iso: string) => iso.slice(11, 16);
const day = (iso: string) => iso.slice(0, 10);

const formatDuration = (minutes: number) => `${Math.floor(minutes / 60)}:${String(minutes % 60).padStart(2, "0")} h`;

function DayOffset({ from, to }: { from: string; to: string }) {
  const days = Math.round((Date.parse(day(to)) - Date.parse(day(from))) / 86_400_000);
  return days > 0 ? <sup className="muted"> +{days}</sup> : null;
}

export function JourneyCard({ label, journey }: { label: string; journey: Journey }) {
  const first = journey.legs[0];
  const last = journey.legs[journey.legs.length - 1];
  return (
    <div className="journey">
      <div className="journey-head">
        <strong>{label}</strong>
        <span className="muted small">{formatDate(day(journey.departure))}</span>
      </div>
      <div className="journey-summary">
        {time(journey.departure)} {first.from_stop} → {time(journey.arrival)}
        <DayOffset from={journey.departure} to={journey.arrival} /> {last.to_stop}
      </div>
      <div className="muted small">
        {formatDuration(journey.duration_minutes)} ·{" "}
        {journey.transfers === 0 ? "ohne Umstieg" : `${journey.transfers} ${journey.transfers === 1 ? "Umstieg" : "Umstiege"}`}
      </div>
      <ol className="journey-legs small">
        {journey.legs.map((leg, i) => {
          const walk = leg.line.startsWith("Fußweg");
          return (
            <li key={i} className={walk ? "walk muted" : undefined}>
              <span className="journey-time">
                {time(leg.departure)}–{time(leg.arrival)}
              </span>
              {walk ? (
                <span>
                  🚶 {leg.line} → {leg.to_stop}
                </span>
              ) : (
                <>
                  <span className="journey-line">{leg.line}</span>
                  <span>
                    {leg.from_stop}
                    {leg.track && <span className="muted"> (Gl. {leg.track})</span>} → {leg.to_stop}
                    {leg.direction && <span className="muted"> · Richtung {leg.direction}</span>}
                  </span>
                </>
              )}
            </li>
          );
        })}
      </ol>
      {journey.notes && (
        <p className="small">
          <LinkedText text={journey.notes} />
        </p>
      )}
    </div>
  );
}
