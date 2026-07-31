"""Character cards: schema, registry, and system-prompt construction.

A character is a single YAML file (see characters/*.yaml). Cards are the
contract shared by the runtime (prompt mode), the Forge fine-tuning pipeline
(training-data synthesis), and the evals (consistency / world-leakage sets) —
which is why knowledge boundaries and taboos are structured fields rather
than free text.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field


class SpeechStyle(BaseModel):
    tone: str = ""
    vocabulary: str = ""
    quirks: list[str] = Field(default_factory=list)
    example_lines: list[str] = Field(default_factory=list)


class KnowledgeBoundaries(BaseModel):
    knows: list[str] = Field(default_factory=list)
    must_not_know: list[str] = Field(default_factory=list)


class VoiceSpec(BaseModel):
    ref_audio: str | None = None      # reference clip for voice-cloning TTS
    tts_voice: str | None = None      # backend-specific voice id (e.g. kokoro "am_michael")
    tts_instruct: str | None = None   # natural-language style hint for instruct TTS


class ModelSpec(BaseModel):
    mode: Literal["prompt", "lora"] = "prompt"
    lora_adapter: str | None = None   # adapter name registered with the LLM server


class CharacterCard(BaseModel):
    id: str
    name: str
    world: str
    role: str
    personality: str
    backstory: str = ""
    speech_style: SpeechStyle = Field(default_factory=SpeechStyle)
    knowledge: KnowledgeBoundaries = Field(default_factory=KnowledgeBoundaries)
    taboos: list[str] = Field(default_factory=list)
    greeting: str | None = None
    voice: VoiceSpec = Field(default_factory=VoiceSpec)
    model: ModelSpec = Field(default_factory=ModelSpec)

    def public_info(self) -> dict:
        """The subset of the card exposed to clients in session.ready."""
        return {"id": self.id, "name": self.name, "role": self.role, "world": self.world}


def load_character(path: str | Path) -> CharacterCard:
    with open(path, encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    return CharacterCard.model_validate(data)


class CharacterRegistry:
    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)
        self._cards: dict[str, CharacterCard] = {}
        self.reload()

    def reload(self) -> None:
        cards: dict[str, CharacterCard] = {}
        for path in sorted(self.directory.glob("*.yaml")):
            card = load_character(path)
            if card.id in cards:
                raise ValueError(f"Duplicate character id '{card.id}' in {path}")
            cards[card.id] = card
        self._cards = cards

    def get(self, character_id: str) -> CharacterCard | None:
        return self._cards.get(character_id)

    def all(self) -> list[CharacterCard]:
        return list(self._cards.values())


def build_system_prompt(card: CharacterCard) -> str:
    """Render a card into the system prompt used in prompt mode.

    LoRA-mode characters get the same prompt; the adapter reinforces style so
    the prompt-vs-LoRA eval comparison holds everything else constant.
    """
    lines: list[str] = []
    lines.append(
        f"You are {card.name}, {card.role}, a character living in {card.world}. "
        "You are speaking with a traveler (the player) face to face, out loud."
    )
    lines.append(f"\nPersonality: {card.personality}")
    if card.backstory:
        lines.append(f"\nBackstory: {card.backstory}")

    style = card.speech_style
    if style.tone or style.vocabulary or style.quirks:
        lines.append("\nHow you speak:")
        if style.tone:
            lines.append(f"- Tone: {style.tone}")
        if style.vocabulary:
            lines.append(f"- Vocabulary: {style.vocabulary}")
        for quirk in style.quirks:
            lines.append(f"- {quirk}")
    if style.example_lines:
        lines.append("\nLines that sound like you:")
        for ex in style.example_lines:
            lines.append(f'- "{ex}"')

    kb = card.knowledge
    if kb.knows:
        lines.append("\nYou know about: " + "; ".join(kb.knows) + ".")
    if kb.must_not_know:
        lines.append(
            "\nYou have NO knowledge of the following, and no words for them: "
            + "; ".join(kb.must_not_know)
            + ". If the player mentions such things, react with genuine confusion, "
            "in character, and steer back to your world. Never explain, define, or "
            "acknowledge them."
        )
    if card.taboos:
        lines.append("\nYou never: " + "; ".join(card.taboos) + ".")

    lines.append(
        "\nRules of the conversation:\n"
        "- Your replies are SPOKEN aloud: keep them short (one to three sentences), "
        "natural, and conversational.\n"
        "- Plain speech only: no stage directions, no asterisks, no emoji, no lists.\n"
        "- Never mention being an AI, a language model, or a game character. "
        "You are simply " + card.name + ".\n"
        "- Stay in character no matter what the player says."
    )
    return "\n".join(lines)
