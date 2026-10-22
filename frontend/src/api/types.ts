export type Role = "user" | "admin";

export interface User {
  id: number;
  email: string;
  display_name: string;
  role: Role;
  is_active: boolean;
  created_at: string;
  home_address: string | null;
  home_city: string | null;
  default_transport: DefaultTransport | null;
}

/** Im Profil wählbare Standard-Fortbewegungsarten */
export type DefaultTransport = Extract<TransportMode, "hiking" | "cycling">;

export interface Place {
  name: string;
  lat: number;
  lon: number;
}

export type TransportMode = "hiking" | "cycling" | "mtb" | "car" | "canoe" | "public_transport";
export type AccommodationType = "hut" | "camping" | "hotel" | "hostel" | "bivouac";

export interface AccommodationStop {
  name: string;
  type: AccommodationType;
  lat: number;
  lon: number;
  website?: string | null;
}

export interface Stage {
  day: number;
  date: string;
  start: Place;
  end: Place;
  distance_km: number;
  ascent_m?: number | null;
  descent_m?: number | null;
  duration_hours?: number | null;
  accommodation?: AccommodationStop | null;
  notes?: string | null;
}

export interface JourneyLeg {
  line: string;
  from_stop: string;
  to_stop: string;
  /** Ortszeit ohne Zeitzone, z. B. 2026-10-24T07:36 */
  departure: string;
  arrival: string;
  direction?: string | null;
  track?: string | null;
}

export interface Journey {
  departure: string;
  arrival: string;
  duration_minutes: number;
  transfers: number;
  legs: JourneyLeg[];
  notes?: string | null;
}

export interface Attribution {
  publisher: string;
  title: string;
  url: string;
  revision_date: string;
  history_url: string;
  license: string;
  license_url: string;
}

export interface AdventurePlan {
  /** Fehlt bei älteren Plänen */
  kind?: "tour";
  title: string;
  summary: string;
  transport: TransportMode;
  start_date: string;
  duration_days: number;
  start: Place;
  destination: Place;
  accommodation_type: AccommodationType | null;
  stages: Stage[];
  route_ids: string[];
  routes: Record<string, ([number, number] | [number, number, number])[]>;
  /** An- und Rückreise mit Bahn/ÖPNV; fehlen bei alten Plänen */
  outbound_journey?: Journey | null;
  return_journey?: Journey | null;
  weather_summary?: string | null;
  packing_tips: string[];
  safety_notes: string[];
  sources: string[];
  /** Namensnennung für genutzte Wissensbasis-Inhalte (z. B. Wikivoyage, CC BY-SA); fehlt bei alten Plänen */
  attributions?: Attribution[];
}

/** Reisekarte für eine reine Bahnfahrt ohne Tour (keine Etappen, kein PDF/GPX) */
export interface TrainTripPlan {
  kind: "train_trip";
  title: string;
  summary: string;
  outbound_journey: Journey;
  return_journey?: Journey | null;
}

export type StoredPlan = AdventurePlan | TrainTripPlan;

export interface ToolCall {
  id: string;
  tool: string;
  args: Record<string, unknown>;
  result?: string;
  failed?: boolean;
}

export interface ChatMessage {
  id: number;
  role: "user" | "assistant";
  content: string;
  tool_calls: ToolCall[];
  created_at: string;
}

export interface AdventureSummary {
  id: number;
  title: string;
  status: "draft" | "planned";
  created_at: string;
  updated_at: string;
}

export interface Adventure extends AdventureSummary {
  plan: StoredPlan | null;
  messages: ChatMessage[];
}

export interface KnowledgeDocument {
  id: number;
  title: string;
  source: string;
  chunk_count: number;
  created_at: string;
}

export type ChatEvent =
  | { type: "token"; content: string }
  | ({ type: "tool_call" } & ToolCall)
  | { type: "tool_result"; id: string; result: string; failed: boolean }
  | { type: "question"; question: string; missing_fields: string[] }
  | { type: "answer"; content: string }
  | { type: "plan"; plan: StoredPlan }
  | { type: "error"; message: string }
  | { type: "done" };
