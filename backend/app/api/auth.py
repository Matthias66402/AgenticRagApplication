from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy import select

from app.auth.deps import SessionDep
from app.auth.security import create_access_token, hash_password, verify_password
from app.models import User, UserRole
from app.schemas.user import Token, UserRead, UserRegister

router = APIRouter(prefix="/api/auth", tags=["auth"])


async def ensure_email_free(session: SessionDep, email: str, exclude_user_id: int | None = None) -> None:
    query = select(User.id).where(User.email == email.lower())
    if exclude_user_id is not None:
        query = query.where(User.id != exclude_user_id)
    if await session.scalar(query) is not None:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="E-Mail-Adresse ist bereits vergeben")


@router.post("/register", response_model=UserRead, status_code=status.HTTP_201_CREATED)
async def register(data: UserRegister, session: SessionDep) -> User:
    await ensure_email_free(session, data.email)
    user = User(
        email=data.email.lower(),
        display_name=data.display_name,
        password_hash=hash_password(data.password),
        role=UserRole.user,
    )
    session.add(user)
    await session.commit()
    await session.refresh(user)
    return user


@router.post("/login", response_model=Token)
async def login(form: Annotated[OAuth2PasswordRequestForm, Depends()], session: SessionDep) -> Token:
    """OAuth2-Passwort-Flow: `username` ist die E-Mail-Adresse."""
    user = await session.scalar(select(User).where(User.email == form.username.lower()))
    if user is None or not verify_password(form.password, user.password_hash):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="E-Mail oder Passwort falsch")
    if not user.is_active:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Konto ist deaktiviert")
    return Token(access_token=create_access_token(user.id))
