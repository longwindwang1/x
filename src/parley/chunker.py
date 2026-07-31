"""Split a streaming LLM token feed into TTS-sized sentence chunks.

Latency rationale: time-to-first-audio is dominated by how soon we hand the
FIRST chunk to TTS, so the first chunk is allowed to be shorter
(``first_min_chars``) than subsequent ones (``min_chars``). Later chunks are
batched a little larger because very short TTS calls waste synthesis overhead
and produce choppy prosody.
"""

from __future__ import annotations

_BOUNDARY_CHARS = ".!?…。！？"
# Trailing characters allowed to ride along after the boundary punctuation.
_TRAILERS = "\"'”’)»]"
# Tokens that end with '.' but do not end a sentence.
_ABBREVIATIONS = {
    "mr.", "mrs.", "ms.", "dr.", "st.", "sr.", "jr.", "prof.", "vs.", "etc.",
    "e.g.", "i.e.", "ft.",
}


class SentenceChunker:
    def __init__(
        self,
        min_chars: int = 24,
        first_min_chars: int = 12,
        max_chars: int = 280,
    ) -> None:
        self.min_chars = min_chars
        self.first_min_chars = first_min_chars
        self.max_chars = max_chars
        self._buf = ""
        self._emitted = 0

    def push(self, delta: str) -> list[str]:
        """Feed a token/delta; return zero or more completed chunks."""
        self._buf += delta
        out: list[str] = []
        while True:
            chunk = self._next_chunk()
            if chunk is None:
                break
            out.append(chunk)
        return out

    def flush(self) -> str | None:
        """Return whatever remains (stream ended mid-sentence)."""
        rest = self._buf.strip()
        self._buf = ""
        if rest:
            self._emitted += 1
            return rest
        return None

    def reset(self) -> None:
        self._buf = ""
        self._emitted = 0

    # -- internals ---------------------------------------------------------

    def _threshold(self) -> int:
        return self.first_min_chars if self._emitted == 0 else self.min_chars

    def _next_chunk(self) -> str | None:
        cut = self._find_boundary()
        if cut is not None:
            chunk = self._buf[:cut].strip()
            self._buf = self._buf[cut:]
            if chunk:
                self._emitted += 1
                return chunk
            return None
        # No sentence boundary yet: hard-split overly long buffers on
        # whitespace so a single run-on sentence cannot stall TTS.
        if len(self._buf) > self.max_chars:
            split = self._buf.rfind(" ", 0, self.max_chars)
            if split <= 0:
                split = self.max_chars
            chunk = self._buf[:split].strip()
            self._buf = self._buf[split:]
            if chunk:
                self._emitted += 1
                return chunk
        return None

    def _find_boundary(self) -> int | None:
        threshold = self._threshold()
        for i, ch in enumerate(self._buf):
            if ch not in _BOUNDARY_CHARS:
                continue
            end = i + 1
            while end < len(self._buf) and self._buf[end] in _TRAILERS:
                end += 1
            # Boundary must be followed by whitespace (or more text pending
            # means we can check); end-of-buffer is NOT a boundary because the
            # next delta may continue the token (e.g. "3." then "5 miles").
            if end >= len(self._buf):
                return None
            if not self._buf[end].isspace():
                continue
            if ch == "." and self._is_non_terminal_period(i):
                continue
            if end < threshold:
                continue
            return end
        return None

    def _is_non_terminal_period(self, i: int) -> bool:
        """True for abbreviations ('Dr.'), initials ('J.'), decimals ('3.5')."""
        if i + 1 < len(self._buf) and self._buf[i + 1].isdigit():
            return True
        start = i
        while start > 0 and not self._buf[start - 1].isspace():
            start -= 1
        word = self._buf[start : i + 1].lower()
        if word in _ABBREVIATIONS:
            return True
        # "No." is a sentence ("No.") unless a number follows ("No. 5").
        if word == "no.":
            j = i + 1
            while j < len(self._buf) and self._buf[j] == " ":
                j += 1
            return j < len(self._buf) and self._buf[j].isdigit()
        # Single-letter initial like "J."
        if len(word) == 2 and word[0].isalpha():
            return True
        return False
