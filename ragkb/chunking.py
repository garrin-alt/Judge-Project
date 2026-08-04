"""Split text into overlapping chunks for embedding."""

_SENTENCE_ENDS = (". ", "! ", "? ", ".\n", "!\n", "?\n")


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 100) -> list[str]:
    """Split text into overlapping chunks, preferring sentence boundaries.

    Cut points snap to the last sentence end inside the window when one
    exists past the halfway point; otherwise to the nearest preceding
    whitespace, so words are never split. The next chunk starts `overlap`
    characters before the previous cut, so no text is skipped regardless
    of where the cut landed.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be >= 0 and < chunk_size")

    text = text.strip()
    if not text:
        return []

    chunks = []
    start = 0
    length = len(text)

    while start < length:
        end = min(start + chunk_size, length)
        if end < length:
            window = text[start:end]
            # prefer the last sentence end in the second half of the window
            sentence_cut = -1
            for sep in _SENTENCE_ENDS:
                pos = window.rfind(sep)
                if pos != -1:
                    sentence_cut = max(sentence_cut, pos + 1)  # keep the punctuation
            if sentence_cut >= chunk_size // 2:
                end = start + sentence_cut
            else:
                snap = text.rfind(" ", start, end)
                if snap != -1 and snap > start:
                    end = snap
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= length:
            break
        start = max(end - overlap, start + 1)

    return chunks
