"""FastAPI application: WebSocket endpoint, character listing, dev web client."""

from __future__ import annotations

import logging
from pathlib import Path

from fastapi import FastAPI, WebSocket
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

from . import __version__
from .characters import CharacterRegistry
from .config import ServerConfig, build_asr, build_llm, build_tts, build_vad
from .metrics import MetricsLog
from .session import Session

logger = logging.getLogger("parley.server")


def create_app(cfg: ServerConfig | None = None) -> FastAPI:
    cfg = cfg or ServerConfig()
    registry = CharacterRegistry(cfg.characters_dir)
    # ASR/LLM/TTS are shared across sessions (stateless per call, models load
    # once). VAD is stateful per audio stream, so each session gets its own.
    asr = build_asr(cfg.asr)
    llm = build_llm(cfg.llm)
    tts = build_tts(cfg.tts)
    metrics_log = MetricsLog(cfg.log_dir)

    app = FastAPI(title="Parley", version=__version__)

    @app.get("/api/characters")
    def list_characters() -> list[dict]:
        return [card.public_info() for card in registry.all()]

    @app.get("/api/health")
    def health() -> dict:
        return {"status": "ok", "version": __version__}

    @app.websocket("/ws")
    async def ws_endpoint(websocket: WebSocket) -> None:
        await websocket.accept()
        session = Session(
            websocket,
            registry,
            cfg,
            vad=build_vad(cfg.vad),
            asr=asr,
            llm=llm,
            tts=tts,
            metrics_log=metrics_log,
        )
        logger.info("Session %s connected", session.session_id)
        try:
            await session.run()
        finally:
            logger.info("Session %s closed", session.session_id)

    web_dir = Path(cfg.web_dir) if cfg.web_dir else None
    if web_dir and web_dir.is_dir():
        app.mount("/web", StaticFiles(directory=str(web_dir), html=True), name="web")

        @app.get("/")
        def index() -> RedirectResponse:
            return RedirectResponse(url="/web/")

    return app
