"""Per-connection session: the heart of the runtime.

One Session owns one WebSocket, one character, and the dialogue history. The
pipeline per turn is:

    utterance (VAD) -> ASR -> LLM stream -> sentence chunker -> TTS stream

LLM generation and TTS synthesis run concurrently: completed sentences go
onto a queue consumed by a TTS worker, so the first sentence is speaking
while later sentences are still being generated — the key to low
time-to-first-audio.

Barge-in: mic audio keeps flowing while the NPC speaks. If the VAD detects
new player speech (or the client sends an explicit ``barge_in``), the active
turn task is cancelled, ``turn.cancelled`` is sent, and whatever the player
already heard stays in history.
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid

from fastapi import WebSocket

from . import protocol
from .characters import CharacterCard, CharacterRegistry, build_system_prompt
from .chunker import SentenceChunker
from .config import ServerConfig
from .metrics import MetricsLog, TurnTimeline
from .vad import BaseVAD, VADEventType

logger = logging.getLogger("parley.session")


class Session:
    def __init__(
        self,
        ws: WebSocket,
        registry: CharacterRegistry,
        cfg: ServerConfig,
        vad: BaseVAD,
        asr,
        llm,
        tts,
        metrics_log: MetricsLog,
    ) -> None:
        self.ws = ws
        self.registry = registry
        self.cfg = cfg
        self.vad = vad
        self.asr = asr
        self.llm = llm
        self.tts = tts
        self.metrics_log = metrics_log

        self.session_id = uuid.uuid4().hex[:12]
        self.character: CharacterCard | None = None
        self.system_prompt = ""
        self.history: list[dict[str, str]] = []
        self._turn_counter = 0
        self._turn_task: asyncio.Task | None = None
        self._turn_id_active: int | None = None
        self._partial_reply = ""
        self._send_lock = asyncio.Lock()
        self._closed = False

    # -- main loop -----------------------------------------------------------

    async def run(self) -> None:
        try:
            while True:
                message = await self.ws.receive()
                if message["type"] == "websocket.disconnect":
                    break
                if message.get("text") is not None:
                    try:
                        data = json.loads(message["text"])
                    except json.JSONDecodeError:
                        await self._send_json(protocol.error("Invalid JSON"))
                        continue
                    if not await self._handle_control(data):
                        break
                elif message.get("bytes") is not None:
                    await self._handle_audio(message["bytes"])
        finally:
            self._closed = True
            await self._cancel_turn("disconnect", notify=False)

    # -- control & audio -----------------------------------------------------

    async def _handle_control(self, msg: dict) -> bool:
        """Returns False when the session should close."""
        mtype = msg.get("type")

        if mtype == protocol.C_SESSION_START:
            card = self.registry.get(msg.get("character_id", ""))
            if card is None:
                await self._send_json(
                    protocol.error(f"Unknown character_id: {msg.get('character_id')!r}")
                )
                return True
            self.character = card
            self.system_prompt = build_system_prompt(card)
            self.history = []
            await self._send_json(
                protocol.session_ready(
                    self.session_id, card.public_info(), card.greeting, self.tts.sample_rate
                )
            )
            if card.greeting and self.cfg.speak_greeting:
                self._start_turn(source="greeting", text=card.greeting)
            return True

        if self.character is None:
            await self._send_json(protocol.error("Send session.start first"))
            return True

        if mtype == protocol.C_INPUT_TEXT:
            text = (msg.get("text") or "").strip()
            if text:
                await self._cancel_turn("superseded")
                self._start_turn(source="text", text=text)
        elif mtype == protocol.C_INPUT_END:
            for event in self.vad.flush():
                if event.type is VADEventType.SPEECH_END and event.utterance:
                    await self._cancel_turn("superseded")
                    self._start_turn(source="voice", utterance=event.utterance)
        elif mtype == protocol.C_BARGE_IN:
            await self._cancel_turn("barge_in")
        elif mtype == protocol.C_SESSION_END:
            await self._cancel_turn("session_end", notify=False)
            return False
        else:
            await self._send_json(protocol.error(f"Unknown message type: {mtype!r}"))
        return True

    async def _handle_audio(self, frame: bytes) -> None:
        if self.character is None:
            return
        for event in self.vad.process(frame):
            if event.type is VADEventType.SPEECH_START:
                await self._send_json(protocol.vad_event("speech_start"))
                # The player started talking over the NPC: interrupt.
                await self._cancel_turn("barge_in")
            elif event.type is VADEventType.SPEECH_END and event.utterance:
                await self._send_json(protocol.vad_event("speech_end"))
                await self._cancel_turn("superseded")
                self._start_turn(source="voice", utterance=event.utterance)

    # -- turn pipeline ---------------------------------------------------------

    def _start_turn(
        self, source: str, text: str | None = None, utterance: bytes | None = None
    ) -> None:
        self._turn_counter += 1
        turn_id = self._turn_counter
        timeline = TurnTimeline(
            turn_id=turn_id,
            source=source,
            character_id=self.character.id if self.character else "",
        )
        self._turn_id_active = turn_id
        self._partial_reply = ""
        self._turn_task = asyncio.create_task(
            self._run_turn(turn_id, timeline, source, text, utterance)
        )

    async def _run_turn(
        self,
        turn_id: int,
        timeline: TurnTimeline,
        source: str,
        text: str | None,
        utterance: bytes | None,
    ) -> None:
        card = self.character
        assert card is not None
        sentence_q: asyncio.Queue[str | None] = asyncio.Queue()
        tts_task = asyncio.create_task(self._tts_worker(turn_id, timeline, sentence_q))
        try:
            await self._send_json(protocol.turn_start(turn_id, source))

            if source == "voice":
                assert utterance is not None
                text = await self.asr.transcribe(utterance)
                timeline.mark("asr_done")
                await self._send_json(protocol.asr_final(turn_id, text))
                if not text.strip():
                    await sentence_q.put(None)
                    await tts_task
                    await self._send_json(protocol.turn_end(turn_id))
                    return

            assert text is not None
            timeline.user_chars = len(text)

            if source == "greeting":
                # Fixed line, no LLM: straight to TTS.
                reply = text
                self._partial_reply = reply
                timeline.mark("llm_first_token")
                timeline.mark("llm_done")
                await sentence_q.put(reply)
            else:
                messages = (
                    [{"role": "system", "content": self.system_prompt}]
                    + self.history
                    + [{"role": "user", "content": text}]
                )
                lora = card.model.lora_adapter if card.model.mode == "lora" else None
                chunker = SentenceChunker()
                async for delta in self.llm.stream_reply(messages, lora=lora):
                    timeline.mark("llm_first_token")
                    self._partial_reply += delta
                    await self._send_json(protocol.reply_delta(turn_id, delta))
                    for sentence in chunker.push(delta):
                        await sentence_q.put(sentence)
                timeline.mark("llm_done")
                tail = chunker.flush()
                if tail:
                    await sentence_q.put(tail)
                reply = self._partial_reply.strip()

            await sentence_q.put(None)
            await tts_task
            timeline.reply_chars = len(reply)

            self._append_history(source, text, reply)
            await self._send_json(protocol.turn_metrics(turn_id, timeline.breakdown()))
            await self._send_json(protocol.turn_end(turn_id))
            self.metrics_log.write(timeline)
        except asyncio.CancelledError:
            timeline.cancelled = True
            timeline.reply_chars = len(self._partial_reply)
            # Keep what the player actually heard (approximated by what was
            # generated) so the interrupted thought stays in context.
            self._append_history(source, text or "", self._partial_reply.strip())
            self.metrics_log.write(timeline)
            raise
        except Exception:
            logger.exception("Turn %s failed", turn_id)
            await self._send_json(protocol.error(f"Turn {turn_id} failed internally"))
            await self._send_json(protocol.turn_end(turn_id))
        finally:
            if not tts_task.done():
                tts_task.cancel()
                try:
                    await tts_task
                except asyncio.CancelledError:
                    pass
            if self._turn_id_active == turn_id:
                self._turn_id_active = None

    async def _tts_worker(
        self, turn_id: int, timeline: TurnTimeline, queue: asyncio.Queue[str | None]
    ) -> None:
        card = self.character
        assert card is not None
        audio_started = False
        index = 0
        while True:
            sentence = await queue.get()
            if sentence is None:
                break
            await self._send_json(protocol.reply_sentence(turn_id, index, sentence))
            async for chunk in self.tts.synth(sentence, card.voice):
                timeline.mark("tts_first_chunk")
                if not audio_started:
                    await self._send_json(protocol.audio_start(turn_id, self.tts.sample_rate))
                    audio_started = True
                await self._send_bytes(chunk)
                timeline.mark("first_audio_sent")
            index += 1
        if audio_started:
            await self._send_json(protocol.audio_end(turn_id))
        timeline.mark("audio_done")

    async def _cancel_turn(self, reason: str, notify: bool = True) -> None:
        task = self._turn_task
        if task is None or task.done():
            return
        turn_id = self._turn_id_active
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        self._turn_task = None
        if notify and turn_id is not None:
            await self._send_json(protocol.turn_cancelled(turn_id, reason))

    def _append_history(self, source: str, user_text: str, reply: str) -> None:
        if source != "greeting" and user_text:
            self.history.append({"role": "user", "content": user_text})
        if reply:
            self.history.append({"role": "assistant", "content": reply})
        max_msgs = self.cfg.history_max_turns * 2
        if len(self.history) > max_msgs:
            self.history = self.history[-max_msgs:]

    # -- socket helpers --------------------------------------------------------

    async def _send_json(self, payload: dict) -> None:
        if self._closed:
            return
        async with self._send_lock:
            try:
                await self.ws.send_text(json.dumps(payload))
            except Exception:
                self._closed = True

    async def _send_bytes(self, data: bytes) -> None:
        if self._closed:
            return
        async with self._send_lock:
            try:
                await self.ws.send_bytes(data)
            except Exception:
                self._closed = True
