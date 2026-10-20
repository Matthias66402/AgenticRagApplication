from fastapi import APIRouter, HTTPException, Response, status
from sqlalchemy import select
from sqlalchemy.orm import selectinload

from app.auth.deps import CurrentUser, SessionDep
from app.export.gpx import build_gpx, gpx_filename, has_track
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
    adventure = await get_owned_adventure(session, adventure_id, user.id)
    if not adventure.plan:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Für dieses Abenteuer gibt es noch keinen Plan")
    return Response(
        content=build_gpx(adventure.plan),
        media_type=GPX_MEDIA_TYPE,
        headers={
            "Content-Disposition": f'attachment; filename="{gpx_filename(adventure.title)}"',
            "X-Gpx-Has-Track": "true" if has_track(adventure.plan) else "false",
        },
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
