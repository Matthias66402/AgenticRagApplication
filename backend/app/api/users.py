from fastapi import APIRouter, HTTPException, status
from sqlalchemy import select

from app.api.auth import ensure_email_free
from app.auth.deps import AdminUser, CurrentUser, SessionDep
from app.auth.security import hash_password
from app.models import User, UserRole
from app.schemas.user import CLEARABLE_FIELDS, UserAdminUpdate, UserCreate, UserRead, UserSelfUpdate

router = APIRouter(prefix="/api/users", tags=["users"])


async def apply_update(session: SessionDep, user: User, data: UserSelfUpdate) -> User:
    changes = {
        field: value
        for field, value in data.model_dump(exclude_unset=True).items()
        if value is not None or field in CLEARABLE_FIELDS
    }
    if "email" in changes:
        await ensure_email_free(session, changes["email"], exclude_user_id=user.id)
        changes["email"] = changes["email"].lower()
    if "password" in changes:
        user.password_hash = hash_password(changes.pop("password"))
    for field, value in changes.items():
        setattr(user, field, value)
    await session.commit()
    await session.refresh(user)
    return user


@router.get("/me", response_model=UserRead)
async def read_me(user: CurrentUser) -> User:
    return user


@router.patch("/me", response_model=UserRead)
async def update_me(data: UserSelfUpdate, user: CurrentUser, session: SessionDep) -> User:
    return await apply_update(session, user, data)


@router.get("", response_model=list[UserRead])
async def list_users(_: AdminUser, session: SessionDep) -> list[User]:
    return list(await session.scalars(select(User).order_by(User.id)))


@router.post("", response_model=UserRead, status_code=status.HTTP_201_CREATED)
async def create_user(data: UserCreate, _: AdminUser, session: SessionDep) -> User:
    await ensure_email_free(session, data.email)
    user = User(
        email=data.email.lower(),
        display_name=data.display_name,
        password_hash=hash_password(data.password),
        role=data.role,
        is_active=data.is_active,
    )
    session.add(user)
    await session.commit()
    await session.refresh(user)
    return user


async def get_user_or_404(session: SessionDep, user_id: int) -> User:
    user = await session.get(User, user_id)
    if user is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Nutzer nicht gefunden")
    return user


@router.get("/{user_id}", response_model=UserRead)
async def read_user(user_id: int, _: AdminUser, session: SessionDep) -> User:
    return await get_user_or_404(session, user_id)


@router.patch("/{user_id}", response_model=UserRead)
async def update_user(user_id: int, data: UserAdminUpdate, admin: AdminUser, session: SessionDep) -> User:
    user = await get_user_or_404(session, user_id)
    # Schutz vor Selbstaussperrung
    if user.id == admin.id and (data.role == UserRole.user or data.is_active is False):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Du kannst Dir selbst weder die Admin-Rolle entziehen noch Dich deaktivieren",
        )
    return await apply_update(session, user, data)


@router.delete("/{user_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_user(user_id: int, admin: AdminUser, session: SessionDep) -> None:
    user = await get_user_or_404(session, user_id)
    if user.id == admin.id:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Du kannst Dich nicht selbst löschen")
    await session.delete(user)
    await session.commit()
