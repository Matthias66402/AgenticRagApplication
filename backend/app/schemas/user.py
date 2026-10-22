from datetime import datetime
from typing import Annotated, Literal

from pydantic import BaseModel, BeforeValidator, ConfigDict, EmailStr, Field

from app.models import UserRole

# Fortbewegungsarten, die als Standard im Profil wählbar sind (Werte wie app.agent.plan_schema.TransportMode)
DefaultTransport = Literal["hiking", "cycling"]
# Leere Eingabe löscht die Angabe
OptionalText = Annotated[str | None, BeforeValidator(lambda v: (v.strip() or None) if isinstance(v, str) else v)]
# Felder, die sich über PATCH mit null bzw. "" wieder leeren lassen
CLEARABLE_FIELDS = {"home_address", "home_city", "default_transport"}


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
    home_address: OptionalText = Field(default=None, max_length=200)
    home_city: OptionalText = Field(default=None, max_length=100)
    default_transport: DefaultTransport | None = None


class UserAdminUpdate(UserSelfUpdate):
    role: UserRole | None = None
    is_active: bool | None = None


class UserRead(UserBase):
    model_config = ConfigDict(from_attributes=True)

    id: int
    role: UserRole
    is_active: bool
    created_at: datetime
    home_address: str | None = None
    home_city: str | None = None
    default_transport: DefaultTransport | None = None


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"
