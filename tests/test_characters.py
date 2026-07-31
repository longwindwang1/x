from pathlib import Path

import pytest
from pydantic import ValidationError

from parley.characters import CharacterCard, CharacterRegistry, build_system_prompt

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_bundled_characters_load():
    registry = CharacterRegistry(REPO_ROOT / "characters")
    ids = {card.id for card in registry.all()}
    assert {"garrick", "nyx", "liriel"} <= ids


def test_system_prompt_carries_the_card():
    registry = CharacterRegistry(REPO_ROOT / "characters")
    card = registry.get("garrick")
    prompt = build_system_prompt(card)
    assert "Garrick Thorne" in prompt
    assert "Emberhollow" in prompt
    # Knowledge boundaries drive the world-leakage eval: they must appear.
    for banned in card.knowledge.must_not_know:
        assert banned in prompt
    assert "no stage directions" in prompt.lower() or "stage directions" in prompt


def test_invalid_card_rejected():
    with pytest.raises(ValidationError):
        CharacterCard.model_validate({"id": "x", "name": "No World Given"})


def test_public_info_is_minimal():
    registry = CharacterRegistry(REPO_ROOT / "characters")
    info = registry.get("nyx").public_info()
    assert set(info) == {"id", "name", "role", "world"}
