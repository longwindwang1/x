from parley.chunker import SentenceChunker


def push_all(chunker: SentenceChunker, text: str, step: int = 4) -> list[str]:
    """Feed text in small deltas the way an LLM streams it."""
    out = []
    for i in range(0, len(text), step):
        out.extend(chunker.push(text[i : i + step]))
    tail = chunker.flush()
    if tail:
        out.append(tail)
    return out


def test_splits_on_sentence_boundaries():
    chunker = SentenceChunker(min_chars=4, first_min_chars=4)
    chunks = push_all(chunker, "Steel is honest, traveler. Folk are not. Aye.")
    assert chunks == ["Steel is honest, traveler.", "Folk are not.", "Aye."]


def test_first_chunk_may_be_short():
    chunker = SentenceChunker(min_chars=24, first_min_chars=5)
    chunks = push_all(chunker, "Hmph. That is a longer second sentence for you.")
    assert chunks == ["Hmph.", "That is a longer second sentence for you."]


def test_min_chars_batches_later_short_sentences():
    chunker = SentenceChunker(min_chars=24, first_min_chars=4)
    chunks = push_all(chunker, "Aye. No. Maybe so. That is my honest answer. Go now.")
    # After the eager first chunk, short sentences ride along until the
    # accumulated chunk reaches min_chars.
    assert chunks[0] == "Aye."
    assert chunks[1] == "No. Maybe so. That is my honest answer."
    assert chunks[2] == "Go now."


def test_does_not_split_abbreviations_and_decimals():
    chunks = push_all(SentenceChunker(), "Dr. Aldwin owes me 3.5 crowns. He pays tomorrow.")
    assert chunks == ["Dr. Aldwin owes me 3.5 crowns.", "He pays tomorrow."]


def test_hard_split_of_runaway_sentence():
    text = "so " * 200  # 600 chars with no boundary
    chunks = push_all(SentenceChunker(max_chars=100), text.strip())
    assert len(chunks) > 1
    assert all(len(c) <= 100 for c in chunks)


def test_flush_returns_trailing_fragment():
    chunker = SentenceChunker()
    assert chunker.push("An unfinished thou") == []
    assert chunker.flush() == "An unfinished thou"


def test_trailing_quote_stays_with_sentence():
    chunks = push_all(SentenceChunker(first_min_chars=4), '"Begone!" he said. And so I went away.')
    assert chunks[0] == '"Begone!"'
