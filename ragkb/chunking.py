"""Split text into overlapping chunks for embedding."""


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 100) -> list[str]:
    """Split text into overlapping chunks, breaking on whitespace boundaries.

    Falls back to paragraph-aware splitting: whole thing is treated as one
    stream of characters, but cut points are snapped to the nearest
    preceding whitespace so words aren't split in half.
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
    step = chunk_size - overlap

    while start < length:
        end = min(start + chunk_size, length)
        if end < length:
            snap = text.rfind(" ", start, end)
            if snap != -1 and snap > start:
                end = snap
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= length:
            break
        start += step

    return chunks
