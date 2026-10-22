from types import SimpleNamespace

import pytest

from app import observability
from app.observability import mask_home_addresses, mask_text, register_home_address


@pytest.fixture
def tracing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Maskierung wie bei aktivem Tracing, ohne Langfuse-Client und mit leerer Adressliste."""
    for name, value in {
        "_enabled": True, "_addresses": set(), "_streets": set(), "_address_pattern": None, "_street_pattern": None,
    }.items():  # fmt: skip
        monkeypatch.setattr(observability, name, value)


def test_mask_text_replaces_address_and_street(tracing: None) -> None:
    register_home_address(" Nelkenstraße 10a ")
    # Im Agent-Kontext, im Geocoding-Aufruf und im Geocoding-Ergebnis (andere Reihenfolge, andere Schreibweise)
    assert mask_text("Wohnadresse: Nelkenstraße 10a, 26121 Oldenburg.") == "Wohnadresse: [Adresse], 26121 Oldenburg."
    assert mask_text('{"place_name": "nelkenstraße 10a, 26121 Oldenburg"}') == '{"place_name": "[Adresse], 26121 Oldenburg"}'
    assert mask_text("10a, Nelkenstraße, Ziegelhof, Oldenburg") == "10a, [Straße], Ziegelhof, Oldenburg"
    assert mask_text("Radtour ab Oldenburg") == "Radtour ab Oldenburg"


def test_mask_hook_patches_only_changed_text_attributes(tracing: None) -> None:
    register_home_address("Nelkenstraße 10")
    spans = {
        "span-1": SimpleNamespace(attributes={
            "gen_ai.input.messages": '[{"content": "Start: Nelkenstraße 10"}]',
            "gen_ai.usage.input_tokens": 120,
            "tags": ("chat", "Nelkenstraße"),
        }),
        "span-2": SimpleNamespace(attributes={"gen_ai.output.messages": "Plan fertig"}),
    }  # fmt: skip
    result = mask_home_addresses(params=SimpleNamespace(spans=spans))

    assert list(result.span_patches) == ["span-1"]
    assert result.span_patches["span-1"].set_attributes == {
        "gen_ai.input.messages": '[{"content": "Start: [Adresse]"}]',
        "tags": ["chat", "[Straße]"],
    }


def test_nothing_registered_without_tracing() -> None:
    register_home_address("Nelkenstraße 10")
    assert mask_text("Nelkenstraße 10") == "Nelkenstraße 10"
    assert mask_home_addresses(params=SimpleNamespace(spans={})) is None
