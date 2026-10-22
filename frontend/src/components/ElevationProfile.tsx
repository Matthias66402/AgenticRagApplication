import { useEffect, useMemo, useRef, useState } from "react";
import type { KeyboardEvent, PointerEvent } from "react";
import type { MapPlan } from "./RouteMap";

export interface ProfilePoint {
  lat: number;
  lon: number;
  ele: number;
  km: number;
}

const HEIGHT = 170;
const PAD = { top: 12, right: 12, bottom: 24, left: 48 };
const MIN_SPAN_M = 50; // flache Touren nicht zu Gebirgen aufblasen

const fmt = (value: number, digits = 0) => value.toLocaleString("de-DE", { maximumFractionDigits: digits });

function distanceKm(a: ProfilePoint | { lat: number; lon: number }, b: { lat: number; lon: number }) {
  const rad = Math.PI / 180;
  const dLat = (b.lat - a.lat) * rad;
  const dLon = (b.lon - a.lon) * rad;
  const h = Math.sin(dLat / 2) ** 2 + Math.cos(a.lat * rad) * Math.cos(b.lat * rad) * Math.sin(dLon / 2) ** 2;
  return 2 * 6371 * Math.asin(Math.sqrt(h));
}

/** Track mit Höhe und kumulierter Distanz; null, wenn die Route keine Höhendaten hat. */
export function buildProfile(plan: MapPlan): ProfilePoint[] | null {
  const track = plan.route_ids.flatMap((id) => plan.routes?.[id] ?? []);
  if (track.length < 2 || track.some((p) => p.length < 3)) return null;
  const points: ProfilePoint[] = [];
  for (const [lat, lon, ele] of track as [number, number, number][]) {
    const km = points.length ? points[points.length - 1].km + distanceKm(points[points.length - 1], { lat, lon }) : 0;
    points.push({ lat, lon, ele, km });
  }
  return points;
}

/** Runde Achsenwerte (1, 2, 5 × 10^n), etwa `count` Stück im Bereich. */
function niceTicks(min: number, max: number, count: number): number[] {
  const raw = (max - min) / count;
  const magnitude = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 5, 10].map((f) => f * magnitude).find((s) => s >= raw) ?? raw;
  const ticks = [];
  for (let t = Math.ceil(min / step) * step; t <= max + 1e-9; t += step) ticks.push(t);
  return ticks;
}

function useWidth<T extends HTMLElement>() {
  const ref = useRef<T>(null);
  const [width, setWidth] = useState(0);
  useEffect(() => {
    if (!ref.current) return;
    const observer = new ResizeObserver(([entry]) => setWidth(entry.contentRect.width));
    observer.observe(ref.current);
    return () => observer.disconnect();
  }, []);
  return [ref, width] as const;
}

interface Props {
  plan: MapPlan;
  points: ProfilePoint[];
  /** Position unter dem Fadenkreuz, z. B. für einen Marker auf der Karte */
  onHover: (point: ProfilePoint | null) => void;
}

export function ElevationProfile({ plan, points, onHover }: Props) {
  const [ref, width] = useWidth<HTMLDivElement>();
  const [active, setActive] = useState<number | null>(null);

  const stats = useMemo(() => {
    const elevations = points.map((p) => p.ele);
    const low = Math.min(...elevations);
    const high = Math.max(...elevations);
    const span = Math.max(high - low, MIN_SPAN_M);
    const axisLow = (low + high - span) / 2;
    return { low, high, axisLow, axisHigh: axisLow + span, total: points[points.length - 1].km };
  }, [points]);

  // Etappenwechsel: nächster Trackpunkt zum Etappenziel, fortlaufend gesucht (Rundtouren!)
  const stageMarks = useMemo(() => {
    const marks: { day: number; km: number }[] = [];
    let from = 0;
    for (const stage of plan.stages.slice(0, -1)) {
      let best = from;
      for (let i = from; i < points.length; i++) {
        if (distanceKm(points[i], stage.end) < distanceKm(points[best], stage.end)) best = i;
      }
      marks.push({ day: stage.day, km: points[best].km });
      from = best;
    }
    return marks;
  }, [plan.stages, points]);

  const innerWidth = Math.max(width - PAD.left - PAD.right, 1);
  const innerHeight = HEIGHT - PAD.top - PAD.bottom;
  const x = (km: number) => PAD.left + (km / stats.total) * innerWidth;
  const y = (ele: number) => PAD.top + (1 - (ele - stats.axisLow) / (stats.axisHigh - stats.axisLow)) * innerHeight;

  const line = useMemo(
    () => points.map((p, i) => `${i ? "L" : "M"}${x(p.km).toFixed(1)},${y(p.ele).toFixed(1)}`).join(""),
    [points, width, stats],
  );
  const baseline = PAD.top + innerHeight;
  const area = `${line}L${x(stats.total).toFixed(1)},${baseline}L${PAD.left},${baseline}Z`;

  function select(index: number | null) {
    setActive(index);
    onHover(index === null ? null : points[index]);
  }

  function handlePointer(event: PointerEvent<SVGSVGElement>) {
    const rect = event.currentTarget.getBoundingClientRect();
    const km = ((event.clientX - rect.left - PAD.left) / innerWidth) * stats.total;
    // Binärsuche nach dem nächsten Punkt (Distanzen sind aufsteigend)
    let lo = 0;
    let hi = points.length - 1;
    while (hi - lo > 1) {
      const mid = (lo + hi) >> 1;
      if (points[mid].km < km) lo = mid;
      else hi = mid;
    }
    select(Math.abs(points[lo].km - km) <= Math.abs(points[hi].km - km) ? lo : hi);
  }

  function handleKey(event: KeyboardEvent<SVGSVGElement>) {
    const step = Math.max(1, Math.round(points.length / 50));
    const current = active ?? 0;
    if (event.key === "ArrowRight") select(Math.min(points.length - 1, current + step));
    else if (event.key === "ArrowLeft") select(Math.max(0, current - step));
    else if (event.key === "Escape") select(null);
    else return;
    event.preventDefault();
  }

  const yTicks = niceTicks(stats.axisLow, stats.axisHigh, 3);
  const xTicks = niceTicks(0, stats.total, Math.max(2, Math.floor(innerWidth / 90)));
  const point = active === null ? null : points[active];
  const summary =
    `Höhenprofil über ${fmt(stats.total, 1)} km, ` + `tiefster Punkt ${fmt(stats.low)} m, höchster Punkt ${fmt(stats.high)} m`;

  return (
    <div className="profile">
      <div className="profile-summary muted small">
        <span>
          Höchster Punkt <strong>{fmt(stats.high)} m</strong>
        </span>
        <span>
          Tiefster Punkt <strong>{fmt(stats.low)} m</strong>
        </span>
        <span>
          Strecke <strong>{fmt(stats.total, 1)} km</strong>
        </span>
      </div>
      <div className="profile-chart" ref={ref}>
      {width > 0 && (
        <svg
          width={width}
          height={HEIGHT}
          role="img"
          aria-label={summary}
          tabIndex={0}
          onPointerMove={handlePointer}
          onPointerDown={handlePointer}
          onPointerLeave={() => select(null)}
          onBlur={() => select(null)}
          onKeyDown={handleKey}
        >
          {yTicks.map((t) => (
            <g key={t} className="profile-grid">
              <line x1={PAD.left} x2={PAD.left + innerWidth} y1={y(t)} y2={y(t)} />
              <text x={PAD.left - 6} y={y(t)} dy="0.32em" textAnchor="end">
                {fmt(t)} m
              </text>
            </g>
          ))}
          {xTicks.map((t) => (
            <text key={t} className="profile-axis" x={x(t)} y={baseline + 16} textAnchor="middle">
              {fmt(t)} km
            </text>
          ))}
          {stageMarks.map((m) => (
            <g key={m.day} className="profile-stage">
              <line x1={x(m.km)} x2={x(m.km)} y1={PAD.top} y2={baseline} />
              <text x={x(m.km) + 4} y={PAD.top + 10}>
                Ende Tag {m.day}
              </text>
            </g>
          ))}
          <path className="profile-area" d={area} />
          <path className="profile-line" d={line} />
          <line className="profile-baseline" x1={PAD.left} x2={PAD.left + innerWidth} y1={baseline} y2={baseline} />
          {point && (
            <g className="profile-cursor">
              <line x1={x(point.km)} x2={x(point.km)} y1={PAD.top} y2={baseline} />
              <circle cx={x(point.km)} cy={y(point.ele)} r={4.5} />
            </g>
          )}
        </svg>
      )}
      {point && (
        <div
          className="profile-tooltip"
          style={{
            left: Math.min(Math.max(x(point.km), 60), width - 60),
            top: Math.max(y(point.ele) - 52, 0),
          }}
        >
          <strong>{fmt(point.ele)} m</strong>
          <span className="muted">bei km {fmt(point.km, 1)}</span>
        </div>
      )}
      </div>
    </div>
  );
}
