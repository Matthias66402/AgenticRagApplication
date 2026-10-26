import httpx

from tests.conftest import PASSWORD, auth_headers, create_user


async def test_register_login_and_me(client: httpx.AsyncClient) -> None:
    response = await client.post(
        "/api/auth/register", json={"email": "Neu@Example.com", "display_name": "Neu", "password": PASSWORD}
    )
    assert response.status_code == 201
    assert response.json()["email"] == "neu@example.com"
    assert response.json()["role"] == "user"

    headers = await auth_headers(client, "neu@example.com")
    me = await client.get("/api/users/me", headers=headers)
    assert me.json()["display_name"] == "Neu"


async def test_register_duplicate_email(client: httpx.AsyncClient) -> None:
    await create_user("dup@example.com")
    response = await client.post(
        "/api/auth/register", json={"email": "dup@example.com", "display_name": "X", "password": PASSWORD}
    )
    assert response.status_code == 409


async def test_login_wrong_password(client: httpx.AsyncClient) -> None:
    await create_user("a@example.com")
    response = await client.post("/api/auth/login", data={"username": "a@example.com", "password": "falsch-falsch"})
    assert response.status_code == 401


async def test_self_update_cannot_change_role(client: httpx.AsyncClient, user_headers: dict[str, str]) -> None:
    response = await client.patch(
        "/api/users/me", headers=user_headers, json={"display_name": "Ute B.", "role": "admin"}
    )
    assert response.status_code == 200
    assert response.json()["display_name"] == "Ute B."
    assert response.json()["role"] == "user"


async def test_profile_home_and_default_transport(client: httpx.AsyncClient, user_headers: dict[str, str]) -> None:
    profile = {"home_address": " Nelkenstraße 10 ", "home_city": "26121 Oldenburg", "default_transport": "cycling"}
    response = await client.patch("/api/users/me", headers=user_headers, json=profile)
    assert response.status_code == 200
    me = response.json()
    assert (me["home_address"], me["home_city"], me["default_transport"]) == (
        "Nelkenstraße 10", "26121 Oldenburg", "cycling",
    )  # fmt: skip

    # Andere Änderungen lassen die Angaben stehen; "" bzw. null leert sie
    response = await client.patch("/api/users/me", headers=user_headers, json={"display_name": "Ute"})
    assert response.json()["home_city"] == "26121 Oldenburg"
    response = await client.patch(
        "/api/users/me", headers=user_headers, json={"home_address": "", "default_transport": None}
    )
    assert (response.json()["home_address"], response.json()["default_transport"]) == (None, None)
    assert response.json()["home_city"] == "26121 Oldenburg"

    # Nur Wandern und Radtour sind als Standard wählbar; Name und E-Mail lassen sich nicht leeren
    for invalid in ({"default_transport": "car"}, {"display_name": ""}):
        assert (await client.patch("/api/users/me", headers=user_headers, json=invalid)).status_code == 422
    response = await client.patch("/api/users/me", headers=user_headers, json={"display_name": None})
    assert response.json()["display_name"] == "Ute"


async def test_profile_gpx_mode(client: httpx.AsyncClient, user_headers: dict[str, str]) -> None:
    assert (await client.get("/api/users/me", headers=user_headers)).json()["gpx_mode"] == "combined"
    response = await client.patch("/api/users/me", headers=user_headers, json={"gpx_mode": "per_day"})
    assert response.json()["gpx_mode"] == "per_day"
    # null lässt die Einstellung stehen, unbekannte Werte werden abgelehnt
    response = await client.patch("/api/users/me", headers=user_headers, json={"gpx_mode": None})
    assert response.json()["gpx_mode"] == "per_day"
    assert (await client.patch("/api/users/me", headers=user_headers, json={"gpx_mode": "zip"})).status_code == 422


async def test_requires_login(client: httpx.AsyncClient) -> None:
    assert (await client.get("/api/users/me")).status_code == 401


async def test_user_cannot_manage_users(client: httpx.AsyncClient, user_headers: dict[str, str]) -> None:
    assert (await client.get("/api/users", headers=user_headers)).status_code == 403


async def test_admin_crud(client: httpx.AsyncClient, admin_headers: dict[str, str]) -> None:
    created = await client.post(
        "/api/users",
        headers=admin_headers,
        json={"email": "b@example.com", "display_name": "B", "password": PASSWORD, "role": "user"},
    )
    assert created.status_code == 201
    user_id = created.json()["id"]

    updated = await client.patch(
        f"/api/users/{user_id}", headers=admin_headers, json={"is_active": False, "role": "admin"}
    )
    assert updated.json()["is_active"] is False
    assert updated.json()["role"] == "admin"

    # Deaktivierte Nutzer können sich nicht anmelden
    login = await client.post("/api/auth/login", data={"username": "b@example.com", "password": PASSWORD})
    assert login.status_code == 403

    assert len((await client.get("/api/users", headers=admin_headers)).json()) == 2
    assert (await client.delete(f"/api/users/{user_id}", headers=admin_headers)).status_code == 204


async def test_admin_cannot_lock_out_self(client: httpx.AsyncClient, admin_headers: dict[str, str]) -> None:
    me = (await client.get("/api/users/me", headers=admin_headers)).json()
    response = await client.patch(f"/api/users/{me['id']}", headers=admin_headers, json={"role": "user"})
    assert response.status_code == 400
    assert (await client.delete(f"/api/users/{me['id']}", headers=admin_headers)).status_code == 400
