import { useEffect, useMemo } from "react";
import { CircleMarker, MapContainer, Polyline, TileLayer, Tooltip, useMap } from "react-leaflet";
import type { LatLngTuple } from "leaflet";
import type { AdventurePlan } from "../api/types";
import { shortPlace } from "../lib/labels";

function FitBounds({ points }: { points: LatLngTuple[] }) {
  const map = useMap();
  useEffect(() => {
    if (points.length === 1) map.setView(points[0], 12);
    else if (points.length > 1) map.fitBounds(points, { padding: [30, 30] });
  }, [map, points]);
  return null;
}

export function RouteMap({ plan }: { plan: AdventurePlan }) {
  const routes = useMemo(
    () => plan.route_ids.map((id) => plan.routes?.[id]).filter((r): r is [number, number][] => Boolean(r)),
    [plan],
  );
  // Ohne berechnete Route (z. B. Schätzung ohne ORS-Key): Etappenpunkte gestrichelt verbinden
  const stagePoints = useMemo<LatLngTuple[]>(
    () => [
      [plan.start.lat, plan.start.lon],
      ...plan.stages.map((s): LatLngTuple => [s.end.lat, s.end.lon]),
    ],
    [plan],
  );
  const allPoints = useMemo<LatLngTuple[]>(
    () => (routes.length ? routes.flat().map(([lat, lon]): LatLngTuple => [lat, lon]) : stagePoints),
    [routes, stagePoints],
  );

  return (
    <MapContainer className="map" center={stagePoints[0]} zoom={9} scrollWheelZoom={false}>
      <TileLayer
        attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>-Mitwirkende'
        url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
      />
      <FitBounds points={allPoints} />
      {routes.length > 0 ? (
        routes.map((route, i) => <Polyline key={i} positions={route} pathOptions={{ color: "#2563eb", weight: 4 }} />)
      ) : (
        <Polyline positions={stagePoints} pathOptions={{ color: "#2563eb", weight: 3, dashArray: "6 8" }} />
      )}
      {plan.stages.map((stage) => (
        <CircleMarker
          key={stage.day}
          center={[stage.end.lat, stage.end.lon]}
          radius={7}
          pathOptions={{ color: "#fff", weight: 2, fillColor: "#2563eb", fillOpacity: 1 }}
        >
          <Tooltip>
            Tag {stage.day}: {shortPlace(stage.end.name)}
          </Tooltip>
        </CircleMarker>
      ))}
      {plan.stages
        .filter((s) => s.accommodation)
        .map((s) => (
          <CircleMarker
            key={`acc-${s.day}`}
            center={[s.accommodation!.lat, s.accommodation!.lon]}
            radius={6}
            pathOptions={{ color: "#fff", weight: 2, fillColor: "#16a34a", fillOpacity: 1 }}
          >
            <Tooltip>🛏️ {s.accommodation!.name}</Tooltip>
          </CircleMarker>
        ))}
      <CircleMarker
        center={[plan.start.lat, plan.start.lon]}
        radius={8}
        pathOptions={{ color: "#fff", weight: 2, fillColor: "#ea580c", fillOpacity: 1 }}
      >
        <Tooltip>Start: {shortPlace(plan.start.name)}</Tooltip>
      </CircleMarker>
    </MapContainer>
  );
}
