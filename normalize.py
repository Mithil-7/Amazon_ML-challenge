"""
Normalization utilities for business_name and business_address fields.

Design notes:
- We do NOT try to be clever with locale-specific NLP libraries (keeps the
  pipeline dependency-light and license-clean).
- Legal-suffix / abbreviation normalization is handled with simple lookup
  tables built from the noise patterns called out in the problem statement
  (Corp/Corporation, Pvt/Private, Ltd/Limited, Rd/Road, St/Street, ...).
- Every function is pure text -> text so it is trivially unit-testable and
  safe to run identically on train and test data (no fitting involved).
"""
import re
import unicodedata

# --- legal-entity suffix normalization (business_name) -----------------
_NAME_SUFFIX_MAP = {
    "corporation": "corp", "corp": "corp", "co": "co", "company": "co",
    "incorporated": "inc", "inc": "inc",
    "limited": "ltd", "ltd": "ltd",
    "private": "pvt", "pvt": "pvt",
    "llc": "llc", "llp": "llp", "plc": "plc",
    "enterprises": "ent", "enterprise": "ent",
    "industries": "ind", "industry": "ind",
    "international": "intl", "intl": "intl",
    "associates": "assoc", "assoc": "assoc",
    "brothers": "bros", "bros": "bros",
    "and": "&",
}

# --- address token normalization (business_address) --------------------
_ADDR_ABBR_MAP = {
    "road": "rd", "rd": "rd",
    "street": "st", "st": "st",
    "avenue": "ave", "ave": "ave",
    "boulevard": "blvd", "blvd": "blvd",
    "lane": "ln", "ln": "ln",
    "drive": "dr", "dr": "dr",
    "circle": "cir", "cir": "cir",
    "court": "ct", "ct": "ct",
    "place": "pl", "pl": "pl",
    "square": "sq", "sq": "sq",
    "floor": "fl", "fl": "fl",
    "building": "bldg", "bldg": "bldg",
    "apartment": "apt", "apt": "apt",
    "suite": "ste", "ste": "ste",
    "north": "n", "south": "s", "east": "e", "west": "w",
    "saint": "st",
    "near": "near", "opposite": "opp", "opp": "opp",
    "post": "post",
}

_STOPWORDS_ADDR = {"near", "opp", "opposite", "behind", "next", "to", "the"}

_WS_RE = re.compile(r"\s+")
_PUNCT_RE = re.compile(r"[^\w\s&]")


def _strip_accents(text: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFKD", text)
        if not unicodedata.combining(c)
    )


def _basic_clean(text) -> str:
    if text is None:
        return ""
    text = str(text)
    if text.strip().lower() in ("nan", "none", ""):
        return ""
    text = _strip_accents(text)
    text = text.lower()
    text = text.replace("&", " and ")
    text = _PUNCT_RE.sub(" ", text)
    text = _WS_RE.sub(" ", text).strip()
    return text


def normalize_name(raw_name) -> str:
    """Lowercase, strip punctuation/accents, canonicalize legal suffixes."""
    text = _basic_clean(raw_name)
    if not text:
        return ""
    tokens = text.split(" ")
    norm_tokens = [_NAME_SUFFIX_MAP.get(t, t) for t in tokens]
    return " ".join(t for t in norm_tokens if t)


def normalize_address(raw_addr) -> str:
    """Lowercase, strip punctuation/accents, canonicalize street abbreviations,
    and drop landmark stop-phrases like 'near'/'opposite' that add noise
    without identifying information."""
    text = _basic_clean(raw_addr)
    if not text:
        return ""
    tokens = text.split(" ")
    norm_tokens = []
    for t in tokens:
        t = _ADDR_ABBR_MAP.get(t, t)
        if t in _STOPWORDS_ADDR:
            continue
        norm_tokens.append(t)
    return " ".join(t for t in norm_tokens if t)


def normalize_country(raw_country) -> str:
    text = _basic_clean(raw_country)
    aliases = {
        "us": "united states", "usa": "united states", "u s a": "united states",
        "united states of america": "united states",
        "india": "india", "in": "india",
        "france": "france", "fr": "france",
    }
    return aliases.get(text, text)


def token_set(text: str) -> set:
    return set(t for t in text.split(" ") if t)


_LEADING_NUM_RE = re.compile(r"^\D*(\d+)")
_ALL_NUMS_RE = re.compile(r"\d+")


def extract_house_number(raw_addr) -> str:
    """The first number in the address, if any (a proxy for a house/unit
    number when there's no parsed address field to read one from)."""
    text = _basic_clean(raw_addr)
    m = _LEADING_NUM_RE.search(text)
    return m.group(1) if m else ""


def extract_postcode_like(raw_addr) -> str:
    """The longest digit run of length >= 4 anywhere in the address (a
    proxy for a postal code). Falls back to the last digit run of any
    length if nothing >= 4 digits is found."""
    text = _basic_clean(raw_addr)
    nums = _ALL_NUMS_RE.findall(text)
    long_nums = [n for n in nums if len(n) >= 4]
    if long_nums:
        return max(long_nums, key=len)
    return nums[-1] if nums else ""


_SOUNDEX_CODES = {
    **{c: "1" for c in "bfpv"}, **{c: "2" for c in "cgjkqsxz"},
    **{c: "3" for c in "dt"}, "l": "4", **{c: "5" for c in "mn"}, "r": "6",
}


def soundex_lite(word: str) -> str:
    """A small, dependency-free Soundex variant: robust to typos and many
    transliteration differences in a name's leading token, without needing
    a phonetics library. Not a full Soundex implementation, just the
    standard collapse-adjacent-codes / keep-first-letter idea."""
    word = re.sub(r"[^a-z]", "", (word or "").lower())
    if not word:
        return ""
    first = word[0]
    codes = [_SOUNDEX_CODES.get(c, "") for c in word[1:]]
    collapsed = []
    prev = _SOUNDEX_CODES.get(word[0], "")
    for c in codes:
        if c and c != prev:
            collapsed.append(c)
        prev = c if c else prev
    return (first + "".join(collapsed))[:5]
