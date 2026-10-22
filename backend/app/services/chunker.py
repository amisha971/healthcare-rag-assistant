"""Split extracted pages into overlapping word windows."""

import re

def chunk_pages(
    pages: list[dict],
    filename: str,
    chunk_size: int = 200,
    overlap: int = 40,
) -> list[dict]:
    """Keep each chunk within one page; sizes are words, not model tokens."""
    if chunk_size <= 0 or not 0 <= overlap < chunk_size:
        raise ValueError("Require chunk_size > 0 and 0 <= overlap < chunk_size.")

    chunks = []
    step = chunk_size - overlap
    for page in pages:
        text = page["text"]
        words = list(re.finditer(r"\S+", text))
        for start in range(0, len(words), step):
            end = min(start + chunk_size, len(words))
            chunks.append({
                "chunk_index": len(chunks),
                "filename": filename,
                "page_number": page["page_number"],
                # Slice the original text so heading/paragraph breaks survive.
                "text": text[words[start].start():words[end - 1].end()],
                "word_count": end - start,
            })
            # Stop when the page is covered; do not emit overlap-only tails.
            if end == len(words):
                break
    return chunks
