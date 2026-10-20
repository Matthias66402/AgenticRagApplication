from datetime import datetime

from pydantic import BaseModel, ConfigDict, EmailStr, Field

from app.models import UserRole


class UserBase(BaseModel):
    email: EmailStr
    display_name: str = Field(min_length=1, max_length=100)


class UserRegister(UserBase):
    password: str = Field(min_length=8, max_length=128)


class UserCreate(UserRegister):
    """Admin legt einen Nutzer an."""

    role: UserRole = UserRole.user
    is_active: bool = True


class UserSelfUpdate(BaseModel):
    email: EmailStr | None = None
    display_name: str | None = Field(default=None, min_length=1, max_length=100)
    password: str | None = Field(default=None, min_length=8, max_length=128)


class UserAdminUpdate(UserSelfUpdate):
    role: UserRole | None = None
    is_active: bool | None = None


class UserRead(UserBase):
    model_config = ConfigDict(from_attributes=True)

    id: int
    role: UserRole
    is_active: bool
    created_at: datetime


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"
