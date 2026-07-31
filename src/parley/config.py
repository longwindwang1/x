"""Server configuration and backend factories.

Backends (VAD/ASR/LLM/TTS) are chosen by name in server.yaml. Heavy backends
import lazily so the core install stays light; a missing optional dependency
produces an actionable error instead of an import-time crash.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field


class BackendCfg(BaseModel):
    backend: str
    options: dict = Field(default_factory=dict)


class ServerConfig(BaseModel):
    host: str = "127.0.0.1"
    port: int = 8000
    characters_dir: str = "characters"
    log_dir: str | None = "logs"
    web_dir: str | None = "web"
    history_max_turns: int = 12
    speak_greeting: bool = True
    vad: BackendCfg = Field(default_factory=lambda: BackendCfg(backend="energy"))
    asr: BackendCfg = Field(default_factory=lambda: BackendCfg(backend="mock"))
    llm: BackendCfg = Field(default_factory=lambda: BackendCfg(backend="mock"))
    tts: BackendCfg = Field(default_factory=lambda: BackendCfg(backend="mock"))


def load_config(path: str | Path | None) -> ServerConfig:
    if path is None:
        return ServerConfig()
    with open(path, encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    return ServerConfig.model_validate(data)


def _missing(backend: str, extra: str) -> ImportError:
    return ImportError(
        f"Backend '{backend}' needs optional dependencies. "
        f'Install them with: pip install "parley[{extra}]"'
    )


def build_vad(cfg: BackendCfg):
    if cfg.backend == "energy":
        from .vad.energy import EnergyVAD

        return EnergyVAD(**cfg.options)
    if cfg.backend == "silero":
        try:
            from .vad.silero import SileroVAD
        except ImportError as exc:
            raise _missing("silero", "vad") from exc
        return SileroVAD(**cfg.options)
    raise ValueError(f"Unknown VAD backend: {cfg.backend}")


def build_asr(cfg: BackendCfg):
    if cfg.backend == "mock":
        from .asr.mock import MockASR

        return MockASR(**cfg.options)
    if cfg.backend == "faster_whisper":
        try:
            from .asr.faster_whisper import FasterWhisperASR
        except ImportError as exc:
            raise _missing("faster_whisper", "asr") from exc
        return FasterWhisperASR(**cfg.options)
    raise ValueError(f"Unknown ASR backend: {cfg.backend}")


def build_llm(cfg: BackendCfg):
    if cfg.backend == "mock":
        from .llm.mock import MockLLM

        return MockLLM(**cfg.options)
    if cfg.backend == "openai_compat":
        from .llm.openai_compat import OpenAICompatLLM

        return OpenAICompatLLM(**cfg.options)
    raise ValueError(f"Unknown LLM backend: {cfg.backend}")


def build_tts(cfg: BackendCfg):
    if cfg.backend == "mock":
        from .tts.mock import MockTTS

        return MockTTS(**cfg.options)
    if cfg.backend == "kokoro":
        try:
            from .tts.kokoro import KokoroTTS
        except ImportError as exc:
            raise _missing("kokoro", "tts") from exc
        return KokoroTTS(**cfg.options)
    raise ValueError(f"Unknown TTS backend: {cfg.backend}")
