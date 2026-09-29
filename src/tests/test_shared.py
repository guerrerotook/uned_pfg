import unicodedata

def normalize(text: str) -> str:
    if text is None:
        return ""
    return unicodedata.normalize("NFKC", text).strip().casefold()