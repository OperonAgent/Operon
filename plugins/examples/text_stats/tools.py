"""text_stats plugin — basic text analytics. Dependency-free."""


def text_stats(text: str = "", wpm: int = 200) -> dict:
    """
    Return word/character/line counts and an estimated reading time.

    Args:
        text: the text to analyse.
        wpm:  reading speed in words per minute (default 200).
    """
    if not isinstance(text, str):
        text = str(text)
    words = text.split()
    lines = text.splitlines() or ([""] if text else [])
    n_words = len(words)
    minutes = round(n_words / max(1, wpm), 2)
    return {
        "success": True,
        "characters": len(text),
        "characters_no_spaces": len(text.replace(" ", "").replace("\t", "")),
        "words": n_words,
        "lines": len(lines),
        "sentences": text.count(".") + text.count("!") + text.count("?"),
        "reading_time_min": minutes,
        "avg_word_length": round(sum(len(w) for w in words) / n_words, 1) if n_words else 0,
    }
