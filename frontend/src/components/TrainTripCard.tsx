import { useMemo, useState } from "react";
import type { AccessRoute, TrainTripPlan } from "../api/types";
import { formatDate, shortPlace } from "../lib/labels";
import { JourneyCard, TransitNote } from "./JourneyCard";
import { LinkedText } from "./LinkedText";
import { ElevationProfile, buildProfile, type ProfilePoint } from "./ElevationProfile";
import { DataSourceList } from "./DataSourceList";
import { DownloadButton } from "./PlanCard";
import { RouteMap, type MapPlan } from "./RouteMap";

const ACCESS_LABELS = { cycling: "🚲 Mit dem Rad zum Bahnhof", hiking: "🚶 Zu Fuß zum Bahnhof" };
const EGRESS_LABELS = { cycling: "🚲 Mit dem Rad vom Bahnhof", hiking: "🚶 Zu Fuß vom Bahnhof" };

const formatMinutes = (hours: number) => `${Math.round(hours * 60)} min`;

/** Weg zum Abfahrtsbahnhof bzw. vom Zielbahnhof: Karte, Höhenprofil und Eckdaten. */
function StationRoute({
  leg,
  routes,
  heading,
  endLabel,
  timing,
}: {
  leg: AccessRoute;
  routes: TrainTripPlan["routes"];
  heading: string;
  endLabel: string;
  timing: string;
}) {
  const map = useMemo<MapPlan>(
    () => ({
      route_ids: [leg.route_id],
      routes: routes ?? {},
      start: leg.start,
      stages: [{ day: 1, end: leg.end, label: endLabel }],
    }),
    [leg, routes, endLabel],
  );
  const profile = useMemo(() => buildProfile(map), [map]);
  const [hovered, setHovered] = useState<ProfilePoint | null>(null);
  return (
    <section>
      <h3>{heading}</h3>
      <RouteMap plan={map} highlight={hovered && [hovered.lat, hovered.lon]} />
      {profile && <ElevationProfile plan={map} points={profile} onHover={setHovered} />}
      <div className="journey">
        <div className="journey-summary">
          {leg.departure.slice(11, 16)} {shortPlace(leg.start.name)} → {shortPlace(leg.end.name)}
        </div>
        <div className="muted small">
          {leg.distance_km.toLocaleString("de-DE", { maximumFractionDigits: 1 })} km · {formatMinutes(leg.duration_hours)}{" "}
          · {timing}
        </div>
      </div>
    </section>
  );
}

/** Reisekarte einer Bahnfahrt: ggf. Weg zum Bahnhof, die gewählte(n) Verbindung(en), ggf. Weg vom Zielbahnhof. */
export function TrainTripCard({ plan, adventureId }: { plan: TrainTripPlan; adventureId: number }) {
  const outbound = plan.outbound_journey;
  const { access, egress } = plan;
  // PDF und GPX enthalten einen Weg: den am Zielort, sonst den zum Bahnhof (siehe app.export.trip)
  const exported = egress ?? access;
  const transport = exported?.transport;

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
              title={
                egress
                  ? "Druckversion mit Weg vom Bahnhof, Höhenprofil und Verbindung"
                  : access
                    ? "Druckversion mit Weg zum Bahnhof, Höhenprofil und Verbindung"
                    : "Druckversion der Verbindung"
              }
            />
            {exported && (
              <DownloadButton
                path={`/api/adventures/${adventureId}/gpx`}
                fallbackName={egress ? "weg-vom-bahnhof.gpx" : "weg-zum-bahnhof.gpx"}
                label="⬇ GPX"
                busyLabel="Erstelle GPX …"
                title={egress ? "Weg vom Bahnhof als Track mit Höhen" : "Weg zum Bahnhof als Track mit Höhen"}
              />
            )}
          </div>
        </div>
        <div className="chips">
          <span className="chip">{transport ? (transport === "cycling" ? "🚲 + 🚆" : "🚶 + 🚆") : "🚆 Bahnfahrt"}</span>
          <span className="chip">📅 {formatDate(outbound.departure.slice(0, 10))}</span>
          {plan.return_journey && <span className="chip">↔ mit Rückfahrt</span>}
        </div>
        <p>
          <LinkedText text={plan.summary} />
        </p>
      </header>

      {access && (
        <StationRoute
          leg={access}
          routes={plan.routes}
          heading={ACCESS_LABELS[access.transport]}
          endLabel={`🚆 ${shortPlace(access.end.name)}`}
          timing={`Zug ab ${outbound.departure.slice(11, 16)}`}
        />
      )}

      <section>
        <h3>Verbindung</h3>
        <div className="journeys">
          <JourneyCard label="🚆 Hinfahrt" journey={outbound} />
          {plan.return_journey && <JourneyCard label="🚆 Rückfahrt" journey={plan.return_journey} />}
        </div>
        <TransitNote />
      </section>

      {egress && (
        <StationRoute
          leg={egress}
          routes={plan.routes}
          heading={EGRESS_LABELS[egress.transport]}
          endLabel={shortPlace(egress.end.name)}
          timing={`Zug an ${outbound.arrival.slice(11, 16)}`}
        />
      )}

      {(plan.data_sources ?? []).length > 0 && (
        <section>
          <DataSourceList sources={plan.data_sources ?? []} />
        </section>
      )}
    </article>
  );
}
