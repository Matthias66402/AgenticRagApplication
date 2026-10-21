"""Export eines gespeicherten Abenteuerplans als druckbares PDF (A4)."""

import datetime as dt
import math
from pathlib import Path
from typing import Any

from fpdf import FPDF
from fpdf.enums import XPos, YPos
from PIL import Image

from app.export.basemap import ATTRIBUTION, Viewport, fit_viewport
from app.export.gpx import ACCOMMODATION_LABELS

# Wird im Dockerfile installiert (fonts-dejavu-core); die PDF-Standardschriften können nur Latin-1
FONT_DIR = Path("/usr/share/fonts/truetype/dejavu")
FONT = "DejaVu"

TRANSPORT_LABELS = {
    "hiking": "Wandern", "cycling": "Radtour", "mtb": "Mountainbike",
    "car": "Auto", "canoe": "Kanu", "public_transport": "Öffentliche Verkehrsmittel",
}  # fmt: skip
WEEKDAYS = ["Mo.", "Di.", "Mi.", "Do.", "Fr.", "Sa.", "So."]

# Farben wie in der Weboberfläche (styles.css, RouteMap.tsx)
TEXT = (28, 33, 40)
MUTED = (102, 112, 133)
BORDER = (226, 229, 234)
SURFACE = (246, 247, 249)
PRIMARY = (37, 99, 235)
PRIMARY_SOFT = (232, 239, 255)
START_COLOR = (234, 88, 12)
ACCOMMODATION_COLOR = (22, 163, 74)

MARGIN = 18.0
MAP_WIDTH = 210.0 - 2 * MARGIN  # A4, volle Textbreite
MAP_HEIGHT = 80.0
PROFILE_HEIGHT = 38.0
DAY_COLUMN = 30.0
EARTH_RADIUS_KM = 6371.0


def _num(value: float, unit: str) -> str:
    """Deutsche Zahlendarstellung mit höchstens einer Nachkommastelle, wie num() im Frontend."""
    text = f"{value:,.1f}".rstrip("0").rstrip(".")
    return text.replace(",", " ").replace(".", ",").replace(" ", ".") + f" {unit}"


def _date(iso: str) -> str:
    day = dt.date.fromisoformat(iso)
    return f"{WEEKDAYS[day.weekday()]}, {day:%d.%m.%Y}"


def _short(name: str) -> str:
    return name.split(",")[0].strip()


def _distance_km(a: list[float], b: list[float]) -> float:
    lat1, lon1, lat2, lon2 = map(math.radians, (a[0], a[1], b[0], b[1]))
    h = math.sin((lat2 - lat1) / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2) ** 2
    return 2 * EARTH_RADIUS_KM * math.asin(math.sqrt(h))


def _track(plan: dict[str, Any]) -> list[list[float]]:
    routes = plan.get("routes", {})
    return [point for rid in plan.get("route_ids", []) for point in routes.get(rid, [])]


def _stage_points(plan: dict[str, Any]) -> list[list[float]]:
    return [[plan["start"]["lat"], plan["start"]["lon"]]] + [[s["end"]["lat"], s["end"]["lon"]] for s in plan["stages"]]


def _accommodations(plan: dict[str, Any]) -> list[dict[str, Any]]:
    return [s["accommodation"] for s in plan["stages"] if s.get("accommodation")]


def map_viewport(plan: dict[str, Any]) -> Viewport:
    """Kartenausschnitt mit Track, Etappenzielen und Unterkünften; gemeinsam für Hintergrundkarte und Zeichnung."""
    points = _track(plan) + _stage_points(plan) + [[a["lat"], a["lon"]] for a in _accommodations(plan)]
    return fit_viewport([(p[0], p[1]) for p in points], MAP_WIDTH / MAP_HEIGHT)


def _nice_scale_km(max_km: float) -> float:
    candidates = [0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]
    return max((c for c in candidates if c <= max_km), default=candidates[0])


class PlanPdf(FPDF):
    def __init__(self, plan: dict[str, Any], basemap: Image.Image | None) -> None:
        super().__init__(format="A4")
        self.plan = plan
        self.basemap = basemap
        self.set_margins(MARGIN, MARGIN, MARGIN)
        self.set_auto_page_break(True, margin=MARGIN)
        self.add_font(FONT, "", FONT_DIR / "DejaVuSans.ttf")
        self.add_font(FONT, "B", FONT_DIR / "DejaVuSans-Bold.ttf")
        self.set_title(plan["title"])
        self.set_creator("AgenticRAG-Abenteuerplaner")

    def footer(self) -> None:
        self.set_y(-12)
        self.style(7.5, color=MUTED)
        page_label = f"Seite {self.page_no()}/{{nb}}"
        text = f"{self.plan['title']} · Erstellt mit dem AgenticRAG-Abenteuerplaner"
        max_width = self.epw - self.get_string_width(page_label) - 6
        while self.get_string_width(text) > max_width:
            text = text[:-2] + "…"
        self.cell(0, 5, text)
        self.set_x(self.l_margin)
        self.cell(0, 5, page_label, align="R")

    def style(self, size: float, bold: bool = False, color: tuple[int, int, int] = TEXT) -> None:
        self.set_font(FONT, "B" if bold else "", size)
        self.set_text_color(*color)

    def ensure_space(self, height: float) -> None:
        if self.get_y() + height > self.page_break_trigger:
            self.add_page()

    def paragraph(self, text: str, size: float = 10, color: tuple[int, int, int] = TEXT, **kwargs: Any) -> None:
        self.style(size, color=color)
        self.multi_cell(0, size * 0.5, text, align="L", new_x=XPos.LMARGIN, new_y=YPos.NEXT, **kwargs)

    def heading(self, text: str) -> None:
        self.ensure_space(20)  # Überschrift nicht allein am Seitenende
        self.ln(4)
        self.style(13, bold=True, color=PRIMARY)
        self.cell(0, 8, text, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_draw_color(*BORDER)
        self.set_line_width(0.3)
        self.line(self.l_margin, self.get_y(), self.w - self.r_margin, self.get_y())
        self.ln(3)

    def bullets(self, items: list[str], size: float = 10, color: tuple[int, int, int] = TEXT) -> None:
        for item in items:
            self.style(size, color=color)
            self.cell(5, size * 0.5, "•")
            self.multi_cell(0, size * 0.5, item, align="L", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            self.ln(1)

    # --- Inhalte -------------------------------------------------------------------------------------------------

    def title_block(self) -> None:
        plan = self.plan
        stages = plan["stages"]
        total_km = sum(s["distance_km"] for s in stages)
        total_ascent = sum(s.get("ascent_m") or 0 for s in stages)
        days = plan["duration_days"]
        facts = [
            TRANSPORT_LABELS.get(plan["transport"], plan["transport"]),
            _date(plan["start_date"]) + (f" – {_date(stages[-1]['date'])}" if days > 1 else ""),
            f"{days} {'Tag' if days == 1 else 'Tage'}",
            ACCOMMODATION_LABELS.get(plan.get("accommodation_type") or "", ""),
            _num(total_km, "km"),
            f"↗ {_num(total_ascent, 'Hm')}" if total_ascent else "",
        ]
        self.style(20, bold=True)
        self.multi_cell(0, 9, plan["title"], align="L", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.ln(1)
        self.paragraph(" · ".join(f for f in facts if f), size=9.5, color=MUTED)
        self.paragraph(f"{_short(plan['start']['name'])} → {_short(plan['destination']['name'])}", size=9.5, color=MUTED)
        self.ln(3)
        self.paragraph(plan.get("summary", ""))

    def route_map(self) -> None:
        """Übersichtskarte: OSM-Hintergrund (falls geladen), darüber Track bzw. Luftlinie und Marker als Vektoren."""
        plan = self.plan
        track = _track(plan)
        accommodations = _accommodations(plan)
        line = track or _stage_points(plan)
        viewport = map_viewport(plan)

        self.heading("Routenübersicht")
        self.ensure_space(MAP_HEIGHT + 8)
        x0, y0, width, height = self.l_margin, self.get_y(), MAP_WIDTH, MAP_HEIGHT
        if self.basemap:
            self.image(self.basemap, x0, y0, width, height)
        else:
            self.set_fill_color(*SURFACE)
            self.rect(x0, y0, width, height, style="F")

        def project(lat: float, lon: float) -> tuple[float, float]:
            fx, fy = viewport.project(lat, lon)
            return x0 + fx * width, y0 + fy * height

        self.set_draw_color(*PRIMARY)
        self.set_line_width(0.7)
        if not track:
            self.set_dash_pattern(dash=1.5, gap=1.5)
        self.polyline([project(p[0], p[1]) for p in line])
        self.set_dash_pattern()

        self.set_draw_color(255, 255, 255)
        self.set_line_width(0.4)
        for acc in accommodations:
            self.marker(*project(acc["lat"], acc["lon"]), 1.3, ACCOMMODATION_COLOR)
        for stage in plan["stages"]:
            x, y = project(stage["end"]["lat"], stage["end"]["lon"])
            self.marker(x, y, 1.6, PRIMARY)
            self.map_label(x + 2.2, y - 3.6, str(stage["day"]), bold=True, color=PRIMARY)
        self.marker(*project(plan["start"]["lat"], plan["start"]["lon"]), 1.9, START_COLOR)

        self.set_draw_color(*BORDER)
        self.set_line_width(0.3)
        self.rect(x0, y0, width, height)

        self.set_y(y0 + height + 2)
        legend = [(START_COLOR, "Start"), (PRIMARY, "Etappenziel (Tag)")]
        if accommodations:
            legend.append((ACCOMMODATION_COLOR, "Unterkunft"))
        x = x0
        self.style(7.5, color=MUTED)
        for color, label in legend:
            self.set_draw_color(*color)
            self.marker(x + 1.2, self.get_y() + 2.5, 1.2, color)
            self.set_xy(x + 3, self.get_y())
            self.cell(self.get_string_width(label) + 6, 5, label)
            x = self.get_x()
        self.scale_and_attribution(viewport, y0 + height + 2)
        self.ln(5)
        if not track:
            self.paragraph("Keine berechnete Route – gestrichelt ist die Luftlinie zwischen den Etappenzielen.", size=7.5, color=MUTED)

    def scale_and_attribution(self, viewport: Viewport, y: float) -> None:
        """Rechts unter der Karte, damit nichts die Route verdeckt: Maßstabsleiste und OSM-Hinweis (ODbL)."""
        km_per_mm = viewport.meters_per_pixel() * viewport.width / MAP_WIDTH / 1000
        bar_km = _nice_scale_km(MAP_WIDTH / 5 * km_per_mm)
        bar_mm = bar_km / km_per_mm
        self.style(7.5, color=MUTED)
        parts = [_num(bar_km, "km")] + ([ATTRIBUTION] if self.basemap else [])
        label = " · ".join(parts)
        right = self.l_margin + MAP_WIDTH
        text_x = right - self.get_string_width(label)
        self.text(text_x, y + 3.6, label)
        bar_x, bar_y = text_x - 2 - bar_mm, y + 3.2
        self.set_draw_color(*TEXT)
        self.set_line_width(0.4)
        self.line(bar_x, bar_y, bar_x + bar_mm, bar_y)
        self.line(bar_x, bar_y - 1, bar_x, bar_y)
        self.line(bar_x + bar_mm, bar_y - 1, bar_x + bar_mm, bar_y)

    def map_label(
        self, x: float, y: float, text: str, size: float = 7, bold: bool = False, color: tuple[int, int, int] = TEXT
    ) -> None:
        """Kurzer Text mit weißem Hintergrund, lesbar auf der Karte; (x, y) ist die linke obere Ecke."""
        self.style(size, bold=bold, color=color)
        self.set_fill_color(255, 255, 255)
        self.set_xy(x, y)
        self.cell(self.get_string_width(text) + 1, size * 0.45, text, fill=True, align="C")

    def marker(self, x: float, y: float, radius: float, color: tuple[int, int, int]) -> None:
        self.set_fill_color(*color)
        self.circle(x, y, radius, style="DF")

    def elevation_profile(self) -> None:
        track = _track(self.plan)
        if len(track) < 2 or any(len(p) < 3 for p in track):
            return
        distances = [0.0]
        for a, b in zip(track, track[1:]):
            distances.append(distances[-1] + _distance_km(a, b))
        elevations = [p[2] for p in track]
        low, high = min(elevations), max(elevations)
        span = max(high - low, 50.0)
        axis_low = (low + high - span) / 2  # flache Profile mittig statt am unteren Rand
        total = max(distances[-1], 1e-3)

        self.heading("Höhenprofil")
        self.ensure_space(PROFILE_HEIGHT + 8)
        label_width = 16.0
        x0, y0 = self.l_margin + label_width, self.get_y()
        width, height = self.epw - label_width, PROFILE_HEIGHT

        def project(km: float, ele: float) -> tuple[float, float]:
            return x0 + km / total * width, y0 + height - (ele - axis_low) / span * height

        outline = [project(d, e) for d, e in zip(distances, elevations)]
        self.set_fill_color(*PRIMARY_SOFT)
        self.polygon([(x0, y0 + height), *outline, (x0 + width, y0 + height)], style="F")
        self.set_draw_color(*PRIMARY)
        self.set_line_width(0.5)
        self.polyline(outline)
        self.set_draw_color(*BORDER)
        self.set_line_width(0.3)
        self.line(x0, y0 + height, x0 + width, y0 + height)
        self.line(x0, y0, x0, y0 + height)

        self.style(7, color=MUTED)
        for ele in (high, low):
            y = project(0, ele)[1]
            self.set_xy(self.l_margin, y - 2)
            self.cell(label_width - 2, 4, _num(round(ele), "m"), align="R")
        self.set_xy(x0, y0 + height + 0.5)
        self.cell(20, 4, "0 km")
        self.set_xy(x0 + width - 30, y0 + height + 0.5)
        self.cell(30, 4, _num(total, "km"), align="R")
        self.set_y(y0 + height + 6)

    def stages(self) -> None:
        self.heading("Etappen")
        right = self.l_margin + DAY_COLUMN
        for index, stage in enumerate(self.plan["stages"]):
            self.ensure_space(22)
            if index:
                self.set_draw_color(*BORDER)
                self.set_line_width(0.2)
                self.line(self.l_margin, self.get_y(), self.w - self.r_margin, self.get_y())
                self.ln(2.5)
            y0, page = self.get_y(), self.page
            self.style(10.5, bold=True)
            self.cell(DAY_COLUMN, 5.5, f"Tag {stage['day']}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            self.style(8.5, color=MUTED)
            self.cell(DAY_COLUMN, 4.5, _date(stage["date"]))
            left_bottom = self.get_y() + 4.5

            details = [
                _num(stage["distance_km"], "km"),
                f"↗ {_num(stage['ascent_m'], 'Hm')}" if stage.get("ascent_m") else "",
                f"↘ {_num(stage['descent_m'], 'Hm')}" if stage.get("descent_m") else "",
                _num(stage["duration_hours"], "h") if stage.get("duration_hours") else "",
            ]
            width = self.epw - DAY_COLUMN
            self.set_xy(right, y0)
            self.style(10.5, bold=True)
            self.multi_cell(width, 5.5, f"{_short(stage['start']['name'])} → {_short(stage['end']['name'])}", align="L", new_x=XPos.LEFT, new_y=YPos.NEXT)
            self.style(8.5, color=MUTED)
            self.multi_cell(width, 4.5, " · ".join(d for d in details if d), align="L", new_x=XPos.LEFT, new_y=YPos.NEXT)
            if acc := stage.get("accommodation"):
                self.set_xy(right, self.get_y() + 1)  # ln() würde an den linken Rand springen
                label = ACCOMMODATION_LABELS.get(acc.get("type", ""), "Unterkunft")
                self.style(9.5)
                self.multi_cell(width, 5, f"Übernachtung: {acc['name']} ({label})", align="L", new_x=XPos.LEFT, new_y=YPos.NEXT)
                if acc.get("website"):
                    self.style(8.5, color=PRIMARY)
                    self.multi_cell(width, 4.5, acc["website"], link=acc["website"], align="L", new_x=XPos.LEFT, new_y=YPos.NEXT)
            if stage.get("notes"):
                self.set_xy(right, self.get_y() + 1)
                self.style(9.5)
                self.multi_cell(width, 5, stage["notes"], align="L", new_x=XPos.LEFT, new_y=YPos.NEXT)
            # Bei einem Seitenumbruch in der rechten Spalte gilt left_bottom nur für die vorige Seite
            bottom = max(self.get_y(), left_bottom) if self.page == page else self.get_y()
            self.set_xy(self.l_margin, bottom + 3)

    def lists(self) -> None:
        plan = self.plan
        if plan.get("weather_summary"):
            self.heading("Wetter")
            self.paragraph(plan["weather_summary"])
        if plan.get("packing_tips"):
            self.heading("Packliste & Tipps")
            self.bullets(plan["packing_tips"])
        if plan.get("safety_notes"):
            self.heading("Sicherheit & Regeln")
            self.bullets(plan["safety_notes"])
        if plan.get("sources"):
            self.heading("Quellen")
            self.bullets(plan["sources"], size=8.5, color=MUTED)


def build_pdf(plan: dict[str, Any], basemap: Image.Image | None = None) -> bytes:
    """Druckversion des Plans: Überblick, Karte, Höhenprofil, Etappen, Wetter, Packliste, Sicherheit, Quellen.

    `basemap` ist der Kartenhintergrund für map_viewport(plan), siehe basemap.render_basemap.
    """
    pdf = PlanPdf(plan, basemap)
    pdf.add_page()
    pdf.title_block()
    pdf.route_map()
    pdf.elevation_profile()
    pdf.stages()
    pdf.lists()
    return bytes(pdf.output())
