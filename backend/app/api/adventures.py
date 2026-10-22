from typing import Any

from fastapi import APIRouter, HTTPException, Response, status
from sqlalchemy import select
from sqlalchemy.orm import selectinload

from app.auth.deps import CurrentUser, SessionDep
from app.export.filename import export_filename
from app.export.basemap import render_basemap
from app.export.gpx import build_gpx, has_track
from app.export.pdf import build_pdf, map_viewport
from app.export.trip import exportable_plan
from app.models import Adventure
from app.schemas.adventure import AdventureCreate, AdventureRead, AdventureSummary, AdventureUpdate

router = APIRouter(prefix="/api/adventures", tags=["adventures"])

GPX_MEDIA_TYPE = "application/gpx+xml"


async def get_owned_adventure(session: SessionDep, adventure_id: int, user_id: int) -> Adventure:
    adventure = await session.scalar(
        select(Adventure).where(Adventure.id == adventure_id).options(selectinload(Adventure.messages))
    )
    # 404 statt 403, um die Existenz fremder Abenteuer nicht preiszugeben
    if adventure is None or adventure.owner_id != user_id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Abenteuer nicht gefunden")
    return adventure


async def get_export_plan(session: SessionDep, adventure_id: int, user_id: int) -> tuple[Adventure, dict[str, Any]]:
    """Abenteuer und Plan in Tourform für die Exporte (Reisekarten siehe app.export.trip)."""
    adventure = await get_owned_adventure(session, adventure_id, user_id)
    if not adventure.plan:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Für dieses Abenteuer gibt es noch keinen Plan")
    return adventure, exportable_plan(adventure.plan)


@router.get("", response_model=list[AdventureSummary])
async def list_adventures(user: CurrentUser, session: SessionDep) -> list[Adventure]:
    query = select(Adventure).where(Adventure.owner_id == user.id).order_by(Adventure.updated_at.desc())
    return list(await session.scalars(query))


@router.post("", response_model=AdventureRead, status_code=status.HTTP_201_CREATED)
async def create_adventure(data: AdventureCreate, user: CurrentUser, session: SessionDep) -> Adventure:
    adventure = Adventure(owner_id=user.id, title=data.title, agent_history=[])
    session.add(adventure)
    await session.commit()
    return await get_owned_adventure(session, adventure.id, user.id)


@router.get("/{adventure_id}", response_model=AdventureRead)
async def read_adventure(adventure_id: int, user: CurrentUser, session: SessionDep) -> Adventure:
    return await get_owned_adventure(session, adventure_id, user.id)


@router.get("/{adventure_id}/gpx", response_class=Response, responses={200: {"content": {GPX_MEDIA_TYPE: {}}}})
async def download_gpx(adventure_id: int, user: CurrentUser, session: SessionDep) -> Response:
    """GPX-Datei mit Track (falls eine Route berechnet wurde) und Wegpunkten für Start, Etappen und Unterkünfte."""
    adventure, plan = await get_export_plan(session, adventure_id, user.id)
    if not plan["stages"]:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="GPX gibt es für Bahnfahrten nur mit Weg zum Bahnhof; Verbindungen haben keine Koordinaten",
        )
    return Response(
        content=build_gpx(plan),
        media_type=GPX_MEDIA_TYPE,
        headers={
            "Content-Disposition": f'attachment; filename="{export_filename(adventure.title, "gpx")}"',
            "X-Gpx-Has-Track": "true" if has_track(plan) else "false",
        },
    )


@router.get("/{adventure_id}/pdf", response_class=Response, responses={200: {"content": {"application/pdf": {}}}})
async def download_pdf(adventure_id: int, user: CurrentUser, session: SessionDep) -> Response:
    """Druckbarer Plan als PDF mit Karte, Höhenprofil, Etappen, Packliste und Sicherheitshinweisen."""
    adventure, plan = await get_export_plan(session, adventure_id, user.id)
    # Ohne erreichbaren Tile-Server enthält das PDF Route und Marker ohne Kartenhintergrund; ohne Etappen keine Karte
    basemap = await render_basemap(map_viewport(plan)) if plan["stages"] else None
    return Response(
        content=build_pdf(plan, basemap),
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{export_filename(adventure.title, "pdf")}"'},
    )


@router.patch("/{adventure_id}", response_model=AdventureRead)
async def update_adventure(
    adventure_id: int, data: AdventureUpdate, user: CurrentUser, session: SessionDep
) -> Adventure:
    adventure = await get_owned_adventure(session, adventure_id, user.id)
    if data.title is not None:
        adventure.title = data.title
    await session.commit()
    return await get_owned_adventure(session, adventure_id, user.id)


@router.delete("/{adventure_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_adventure(adventure_id: int, user: CurrentUser, session: SessionDep) -> None:
    adventure = await get_owned_adventure(session, adventure_id, user.id)
    await session.delete(adventure)
    await session.commit()
