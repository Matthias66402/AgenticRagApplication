import type { AccommodationType, TransportMode } from "../api/types";

export const TRANSPORT_LABELS: Record<TransportMode, string> = {
  hiking: "🥾 Wandern",
  cycling: "🚲 Radtour",
  mtb: "🚵 Mountainbike",
  car: "🚗 Auto",
  canoe: "🛶 Kanu",
  public_transport: "🚆 Öffentlich",
};

export const ACCOMMODATION_LABELS: Record<AccommodationType, string> = {
  hut: "Hütte",
  camping: "Camping",
  hotel: "Hotel",
  hostel: "Hostel",
  bivouac: "Biwak",
};

export const TOOL_LABELS: Record<string, { icon: string; label: string }> = {
  geocode: { icon: "📍", label: "Ort suchen" },
  plan_route: { icon: "🧭", label: "Route berechnen" },
  estimate_hiking_time: { icon: "⏱️", label: "Gehzeit" },
  get_weather: { icon: "🌦️", label: "Wetter" },
  find_accommodation: { icon: "🏕️", label: "Unterkünfte" },
  search_knowledge: { icon: "📚", label: "Wissensbasis" },
  sun_times: { icon: "🌅", label: "Sonnenzeiten" },
  find_train_connections: { icon: "🚆", label: "Bahnverbindung" },
};

const dateFormat = new Intl.DateTimeFormat("de-DE", { weekday: "short", day: "2-digit", month: "2-digit", year: "numeric" });
const dateTimeFormat = new Intl.DateTimeFormat("de-DE", { dateStyle: "medium", timeStyle: "short" });

export const formatDate = (iso: string) => dateFormat.format(new Date(`${iso}T12:00:00`));
export const formatDateTime = (iso: string) => dateTimeFormat.format(new Date(iso));

/** Kurzer Ortsname: "Ulm, Baden-Württemberg, Deutschland" → "Ulm" */
export const shortPlace = (name: string) => name.split(",")[0].trim();
