"""
Second-generation normalization (used by blocking_v3 / features_v3).

Differences from src/normalize.py, each driven by real misses of model_v2
on held-out true pairs:
- Non-Latin scripts (Devanagari, Gujarati, Tamil, Bengali, ...) are kept as
  tokens instead of being deleted by an ASCII-only encode. Accented Latin
  (French) is still folded to ASCII.
- Numeric tokens lose leading zeros ("03415" == "3415", "A-011" == "A-11").
- Address abbreviations cover US, Indian and French forms, and US state
  names map to their 2-letter code ("Texas" == "TX").
- Legal-form words (Inc, LLC, Pvt Ltd, SARL, SAS, EURL, ...) are split off
  the business name into a separate set, so name similarity is computed on
  the distinctive core only.
"""
import re
import unicodedata

US_STATES = {
    "alabama": "al", "alaska": "ak", "arizona": "az", "arkansas": "ar", "california": "ca",
    "colorado": "co", "connecticut": "ct", "delaware": "de", "florida": "fl", "georgia": "ga",
    "hawaii": "hi", "idaho": "id", "illinois": "il", "indiana": "in", "iowa": "ia",
    "kansas": "ks", "kentucky": "ky", "louisiana": "la", "maine": "me", "maryland": "md",
    "massachusetts": "ma", "michigan": "mi", "minnesota": "mn", "mississippi": "ms",
    "missouri": "mo", "montana": "mt", "nebraska": "ne", "nevada": "nv", "ohio": "oh",
    "oklahoma": "ok", "oregon": "or", "pennsylvania": "pa", "tennessee": "tn", "texas": "tx",
    "utah": "ut", "vermont": "vt", "virginia": "va", "washington": "wa", "wisconsin": "wi",
    "wyoming": "wy",
}
# Two-word state names, matched on the joined text before tokenizing.
US_STATES_MULTI = {
    "new hampshire": "nh", "new jersey": "nj", "new mexico": "nm", "new york": "ny",
    "north carolina": "nc", "north dakota": "nd", "rhode island": "ri",
    "south carolina": "sc", "south dakota": "sd", "west virginia": "wv",
    "district of columbia": "dc",
}

ADDRESS_ABBREVIATIONS = {
    # US
    "st": "street", "str": "street", "rd": "road", "ave": "avenue", "av": "avenue",
    "blvd": "boulevard", "bd": "boulevard", "bld": "boulevard", "dr": "drive", "ln": "lane",
    "ct": "court", "cir": "circle", "pl": "place", "pkwy": "parkway", "hwy": "highway",
    "trl": "trail", "ter": "terrace", "sq": "square", "mt": "mount", "ft": "fort",
    "twp": "township", "apt": "apartment", "ste": "suite", "fl": "floor", "flr": "floor",
    "bldg": "building", "n": "north", "s": "south", "e": "east", "w": "west",
    "ne": "northeast", "nw": "northwest", "se": "southeast", "sw": "southwest",
    "pt": "point", "hts": "heights", "jct": "junction", "cres": "crescent",
    # India
    "rd.": "road", "marg": "road", "nr": "near", "opp": "opposite", "bh": "behind",
    "co": "care", "c/o": "care", "s/o": "son", "d/o": "daughter", "w/o": "wife",
    "hno": "house", "h": "house", "no": "number", "nagr": "nagar",
    # France
    "r": "rue", "all": "allee", "imp": "impasse", "ch": "chemin", "rte": "route",
    "pl.": "place", "sq.": "square", "fg": "faubourg", "fbg": "faubourg", "qu": "quai",
    "crs": "cours", "lot": "lotissement", "res": "residence", "zi": "zone", "za": "zone",
    "st.": "saint", "ste.": "sainte",
}

ADDRESS_STOPWORDS = {"number", "de", "du", "des", "la", "le", "les", "l", "d", "of", "the", "n/a", "na", "null"}

LEGAL_FORMS = {
    "inc", "incorporated", "corp", "corporation", "co", "company", "ltd", "limited", "llc",
    "llp", "lp", "plc", "pllc", "pc", "pa", "pvt", "private", "pte", "opc", "gmbh",
    "sarl", "sas", "sasu", "sa", "eurl", "sci", "snc", "scop", "scp", "selarl", "gie", "ei",
    # Common Indic-script spellings of limited / private / LLP.
    "लिमिटेड", "प्राइवेट", "प्रा", "लि", "एलएलपी", "લિમિટેડ", "પ્રાઇવેટ", "লিমিটেড", "প্রাইভেট",
    "லிமிடெட்", "பிரைவேட்", "లిమిటెడ్", "ప్రైవేట్", "ಲಿಮಿಟೆಡ್", "ಪ್ರೈವೇಟ್",
    "and", "the", "et", "of", "de", "du", "des", "la", "le", "les", "l", "d", "mr", "mrs", "smt", "shri", "sri",
}

_WS_RE = re.compile(r"\s+")
# Keep letters/digits of any script; everything else is a separator. \w in
# Python's re is Unicode-aware, so Devanagari etc. survive; "_" is dropped.
# Indic combining vowel signs (U+0900-U+0DFF) are not \w, so the Indic
# blocks are added explicitly or every word would split at each vowel sign.
_SEP_RE = re.compile(r"[^\w\u0900-\u0dff]+|_")
_NUMBER_SIGN_RE = re.compile(r"\b[nN]\s*[°º]|\b[nN]/[aA]\b|<null>|\bnull\b", re.IGNORECASE)
_DIGITS_RE = re.compile(r"^\d+$")
_DOTTED_ACRONYM_RE = re.compile(r"\b(?:[A-Za-z]\.){2,}")


def _fold(text: str) -> str:
    """Lowercase, strip Latin accents, keep non-Latin letters intact."""
    decomposed = unicodedata.normalize("NFKD", text)
    out = []
    for ch in decomposed:
        if unicodedata.combining(ch) and ord(ch) < 0x0900:
            # Only drop combining marks in the Latin range; Indic vowel signs
            # (U+0900 and up) are combining too but carry meaning.
            continue
        out.append(ch)
    return unicodedata.normalize("NFC", "".join(out)).lower()


def _strip_zeros(tok: str) -> str:
    if _DIGITS_RE.match(tok):
        return tok.lstrip("0") or "0"
    return tok


def base_tokens(text) -> list:
    if text is None or text == "" or (isinstance(text, float) and text != text):
        return []
    text = str(text)
    # "S.A.R.L." / "L.L.C." -> "sarl" / "llc" before punctuation splitting.
    text = _DOTTED_ACRONYM_RE.sub(lambda m: m.group(0).replace(".", ""), text)
    text = _NUMBER_SIGN_RE.sub(" ", text)
    text = text.replace("&", " and ").replace("+", " and ")
    text = _fold(text)
    text = _SEP_RE.sub(" ", text)
    return [_strip_zeros(t) for t in text.split() if t]


def name_parts(text):
    """Return (core_tokens, legal_tokens) for a business name."""
    tokens = base_tokens(text)
    core = [t for t in tokens if t not in LEGAL_FORMS]
    legal = [t for t in tokens if t in LEGAL_FORMS]
    return core, legal


def address_tokens(text) -> list:
    tokens = base_tokens(text)
    if not tokens:
        return []
    joined = " " + " ".join(tokens) + " "
    for full, code in US_STATES_MULTI.items():
        if " " + full + " " in joined:
            joined = joined.replace(" " + full + " ", " " + code + " ")
    out = []
    for tok in joined.split():
        tok = US_STATES.get(tok, tok)
        tok = ADDRESS_ABBREVIATIONS.get(tok, tok)
        if tok not in ADDRESS_STOPWORDS:
            out.append(tok)
    return out


def is_non_latin(text) -> bool:
    if not text:
        return False
    for ch in str(text):
        if ch.isalpha():
            return ord(ch) >= 0x0370
    return False


# --- Phonetic consonant skeletons (script-independent name matching) -------
#
# Indic names ("आनंद हेल्थकेयर") and their English originals ("Anand
# Healthcare") share no characters. Reducing both to a consonant skeleton
# ("nnd hltkr") makes them comparable: Indic letters are mapped through
# their Unicode character names ("DEVANAGARI LETTER THA" -> "th" -> "t"),
# Latin words by dropping vowels and folding similar-sounding consonants.

_CONSONANT_FOLD = [
    ("ph", "f"), ("bh", "b"), ("dh", "d"), ("th", "t"), ("kh", "k"), ("gh", "g"),
    ("ch", "c"), ("jh", "j"), ("sh", "s"), ("ck", "k"), ("q", "k"), ("x", "ks"),
    ("c", "k"), ("z", "j"), ("w", "v"), ("y", ""), ("h", ""),
]
_VOWELS = set("aeiou")
_INDIC_SKELETON_CACHE = {}


_SOFT_C_RE = re.compile(r"c(?=[eiy])")


def _fold_consonants(latin: str) -> str:
    latin = _SOFT_C_RE.sub("s", latin)
    for src, dst in _CONSONANT_FOLD:
        latin = latin.replace(src, dst)
    out = []
    for ch in latin:
        if ch in _VOWELS or not ch.isalnum():
            continue
        if out and out[-1] == ch:
            continue
        out.append(ch)
    return "".join(out)


def _indic_char_latin(ch: str) -> str:
    if ch in _INDIC_SKELETON_CACHE:
        return _INDIC_SKELETON_CACHE[ch]
    try:
        name = unicodedata.name(ch)
    except ValueError:
        name = ""
    latin = ""
    if " LETTER " in name and "VOWEL" not in name:
        # "DEVANAGARI LETTER KHA" -> "kha"; "TAMIL LETTER NNA" -> "nna"
        parts = name.split(" LETTER ", 1)[1].split()
        # "MALAYALAM LETTER CHILLU L" -> "l"
        latin = (parts[-1] if parts[0] == "CHILLU" else parts[0]).lower()
    elif "SIGN ANUSVARA" in name or "SIGN CANDRABINDU" in name:
        latin = "n"
    _INDIC_SKELETON_CACHE[ch] = latin
    return latin


def skeleton(token: str) -> str:
    """Consonant skeleton of one token, for Latin or Indic script."""
    if not token:
        return ""
    if is_non_latin(token):
        latin = "".join(_indic_char_latin(ch) for ch in token)
    else:
        latin = token
    return _fold_consonants(latin)


def name_skeletons(core_tokens: list) -> list:
    return [s for s in (skeleton(t) for t in core_tokens) if len(s) >= 2]
