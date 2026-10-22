import { useMemo, useState } from "react";
import type { TrainTripPlan } from "../api/types";
import { formatDate, shortPlace } from "../lib/labels";
import { JourneyCard, TransitNote } from "./JourneyCard";
import { LinkedText } from "./LinkedText";
import { ElevationProfile, buildProfile, type ProfilePoint } from "./ElevationProfile";
import { DownloadButton } from "./PlanCard";
import { RouteMap, type MapPlan } from "./RouteMap";

const ACCESS_LABELS = { cycling: "🚲 Mit dem Rad zum Bahnhof", hiking: "🚶 Zu Fuß zum Bahnhof" };

const formatMinutes = (hours: number) => `${Math.round(hours * 60)} min`;

/** Reisekarte einer Bahnfahrt: ggf. Weg zum Bahnhof mit Karte, dann die gewählte(n) Verbindung(en). */
export function TrainTripCard({ plan, adventureId }: { plan: TrainTripPlan; adventureId: number }) {
  const outbound = plan.outbound_journey;
  const access = plan.access;
  const map = useMemo<MapPlan | null>(
    () =>
      access
        ? {
            route_ids: [access.route_id],
            routes: plan.routes ?? {},
            start: access.start,
            stages: [{ day: 1, end: access.end, label: `🚆 ${shortPlace(access.end.name)}` }],
          }
        : null,
    [access, plan.routes],
  );
  const profile = useMemo(() => (map ? buildProfile(map) : null), [map]);
  const [hovered, setHovered] = useState<ProfilePoint | null>(null);

  return (
    <article className="plan">
      <header>
        <div className="plan-title">
          <h2>{plan.title}</h2>
          <div className="downloads">
            <DownloadButton
              path={`/api/adventures/${adventureId}/pdf`}
              fallbackName="bahnfahrt.pdf"
              label="⬇ PDF"
              busyLabel="Erstelle PDF …"
              title={access ? "Druckversion mit Weg zum Bahnhof, Höhenprofil und Verbindung" : "Druckversion der Verbindung"}
            />
            {access && (
              <DownloadButton
                path={`/api/adventures/${adventureId}/gpx`}
                fallbackName="weg-zum-bahnhof.gpx"
                label="⬇ GPX"
                busyLabel="Erstelle GPX …"
                title="Weg zum Bahnhof als Track mit Höhen"
              />
            )}
          </div>
        </div>
        <div className="chips">
          <span className="chip">{access ? (access.transport === "cycling" ? "🚲 + 🚆" : "🚶 + 🚆") : "🚆 Bahnfahrt"}</span>
          <span className="chip">📅 {formatDate(outbound.departure.slice(0, 10))}</span>
          {plan.return_journey && <span className="chip">↔ mit Rückfahrt</span>}
        </div>
        <p>
          <LinkedText text={plan.summary} />
        </p>
      </header>

      {access && map && (
        <section>
          <h3>{ACCESS_LABELS[access.transport]}</h3>
          <RouteMap plan={map} highlight={hovered && [hovered.lat, hovered.lon]} />
          {profile && <ElevationProfile plan={map} points={profile} onHover={setHovered} />}
          <div className="journey">
            <div className="journey-summary">
              {access.departure.slice(11, 16)} {shortPlace(access.start.name)} → {shortPlace(access.end.name)}
            </div>
            <div className="muted small">
              {access.distance_km.toLocaleString("de-DE", { maximumFractionDigits: 1 })} km ·{" "}
              {formatMinutes(access.duration_hours)} · Zug ab {outbound.departure.slice(11, 16)}
            </div>
          </div>
        </section>
      )}

      <section>
        <h3>Verbindung</h3>
        <div className="journeys">
          <JourneyCard label="🚆 Hinfahrt" journey={outbound} />
          {plan.return_journey && <JourneyCard label="🚆 Rückfahrt" journey={plan.return_journey} />}
        </div>
        <TransitNote />
      </section>
    </article>
  );
}
