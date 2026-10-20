export type Role = "user" | "admin";

export interface User {
  id: number;
  email: string;
  display_name: string;
  role: Role;
  is_active: boolean;
  created_at: string;
}

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

export interface AdventurePlan {
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
  routes: Record<string, [number, number][]>;
  weather_summary?: string | null;
  packing_tips: string[];
  safety_notes: string[];
  sources: string[];
}

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
  plan: AdventurePlan | null;
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
  | { type: "plan"; plan: AdventurePlan }
  | { type: "error"; message: string }
  | { type: "done" };
