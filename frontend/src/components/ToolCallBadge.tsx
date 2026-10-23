import { useState } from "react";
import type { ToolCall } from "../api/types";
import { TOOL_LABELS } from "../lib/labels";

function describeArgs(call: ToolCall): string {
  const a = call.args;
  switch (call.tool) {
    case "geocode":
      return String(a.place_name ?? "");
    case "search_knowledge":
      return `„${a.query}“`;
    case "find_accommodation":
      return `${a.accommodation_type}, ${a.radius_km ?? 5} km`;
    case "plan_route":
      return (
        `${Array.isArray(a.waypoints) ? a.waypoints.length : "?"} Wegpunkte, ${a.transport}` +
        (a.route_name ? `, entlang „${a.route_name}“` : "")
      );
    case "get_weather":
      return `ab ${a.start_date}, ${a.days ?? 1} Tag(e)`;
    case "estimate_hiking_time":
      return `${a.distance_km} km, +${a.ascent_m ?? 0} Hm`;
    case "sun_times":
      return String(a.date ?? "");
    case "find_train_connections":
      return `${a.from_name} → ${a.to_name}, ${a.date} ${a.arrive_by ? "an" : "ab"} ${String(a.time ?? "08:00").slice(0, 5)}`;
    default:
      return "";
  }
}

export function ToolCallBadge({ call }: { call: ToolCall }) {
  const [open, setOpen] = useState(false);
  const meta = TOOL_LABELS[call.tool] ?? { icon: "🔧", label: call.tool };
  const state = call.result === undefined ? "running" : call.failed ? "failed" : "ok";

  return (
    <div className={`tool-call ${state}`}>
      <button type="button" className="tool-call-head" onClick={() => setOpen(!open)} aria-expanded={open}>
        <span aria-hidden>{meta.icon}</span>
        <strong>{meta.label}</strong>
        <span className="tool-args">{describeArgs(call)}</span>
        <span className="tool-state" aria-label={state}>
          {state === "running" ? <span className="spinner" /> : state === "failed" ? "⚠️" : "✓"}
        </span>
      </button>
      {open && (
        <div className="tool-detail">
          <div className="muted small">Argumente</div>
          <pre>{JSON.stringify(call.args, null, 2)}</pre>
          {call.result !== undefined && (
            <>
              <div className="muted small">Ergebnis (gekürzt)</div>
              <pre>{call.result}</pre>
            </>
          )}
        </div>
      )}
    </div>
  );
}
