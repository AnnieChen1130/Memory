import re
from typing import List, Optional


def split_text(
    text: str,
    chunk_size: int = 512,
    overlap: int = 50,
    separators: Optional[List[str]] = None,
) -> List[str]:
    """
    Split text into chunks of specified size using a recursive character-based strategy.

    Args:
        text (str): The original text to be split.
        chunk_size (int): Maximum length of each chunk (in characters).
        chunk_overlap (int): Number of overlapping characters between adjacent chunks.
        separators (List[str]): List of separators to split the text, ordered by priority.
                                Defaults to common semantic boundaries.

    Returns:
        List[str]: List of text chunks after splitting.
    """
    if overlap >= chunk_size:
        raise ValueError(
            f"Overlap ({overlap}) cannot be greater than or equal to chunk size ({chunk_size})!"
        )

    if separators is None:
        separators = ["\n\n", "\n", "。 ", ". ", "！", "!", "？", "?", "，", ",", " ", ""]

    final_splits = [text]

    for sep in separators:
        current_splits = []
        for s in final_splits:
            if len(s) <= chunk_size:
                current_splits.append(s)
                continue
            if sep:
                parts = re.split(f"(?<={re.escape(sep)})", s)
                current_splits.extend(parts)
            else:
                current_splits.extend([s[i : i + chunk_size] for i in range(0, len(s), chunk_size)])
        final_splits = current_splits

    chunks = []
    current_chunk_parts = []
    current_length = 0

    for part in final_splits:
        part_len = len(part)

        if current_length + part_len > chunk_size and current_chunk_parts:
            chunk = "".join(current_chunk_parts)
            chunks.append(chunk)

            overlap_parts = []
            overlap_len = 0
            for i in range(len(current_chunk_parts) - 1, -1, -1):
                p = current_chunk_parts[i]
                if overlap_len + len(p) > overlap:
                    break
                overlap_len += len(p)
                overlap_parts.insert(0, p)

            current_chunk_parts = overlap_parts + [part]
            current_length = sum(len(p) for p in current_chunk_parts)
        else:
            current_chunk_parts.append(part)
            current_length += part_len

    if current_chunk_parts:
        chunk = "".join(current_chunk_parts)
        chunks.append(chunk)

    return [c.strip() for c in chunks if c.strip()]
