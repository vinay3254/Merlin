import re
import unicodedata

ABBREVIATION_MAP = {
    "corp": "corporation",
    "co": "company",
    "inc": "incorporated",
    "ltd": "limited",
    "pvt": "private",
    "llc": "limited liability company",
    "llp": "limited liability partnership",
    "&": "and",
    "rd": "road",
    "st": "street",
    "ave": "avenue",
    "blvd": "boulevard",
    "dr": "drive",
    "ln": "lane",
    "apt": "apartment",
    "bldg": "building",
    "no": "number",
}

_PUNCT_RE = re.compile(r"[^\w\s&-]")
_WS_RE = re.compile(r"\s+")


def normalize_text(text) -> str:
    if not text:
        return ""
    text = unicodedata.normalize("NFKD", str(text))
    text = text.encode("ascii", "ignore").decode("ascii")
    text = text.lower()
    text = _PUNCT_RE.sub(" ", text)
    tokens = text.split()
    expanded = [ABBREVIATION_MAP.get(tok, tok) for tok in tokens]
    return _WS_RE.sub(" ", " ".join(expanded)).strip()


def tokenize(normalized_text: str) -> list:
    if not normalized_text:
        return []
    return [t for t in normalized_text.split(" ") if t]
