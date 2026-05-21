import re
import unicodedata
from typing import Dict


MOJIBAKE_REPLACEMENTS: Dict[str, str] = {
    "â€™": "'",
    "â€˜": "'",
    "â€œ": '"',
    "â€�": '"',
    "â€": '"',
    "â€“": "-",
    "â€”": "-",
    "â€•": "-",
    "â€¦": "...",
    "â€": '"',
    "Â ": " ",
    "Â": "",
    "Ã©": "é",
    "Ã¨": "è",
    "Ã¡": "á",
    "Ã ": "à",
    "Ã¢": "â",
    "Ãª": "ê",
    "Ã®": "î",
    "Ã´": "ô",
    "Ã¶": "ö",
    "Ã¼": "ü",
    "Ã±": "ñ",
}

QUOTE_REPLACEMENTS: Dict[str, str] = {
    "\u2018": "'",
    "\u2019": "'",
    "\u201a": "'",
    "\u201b": "'",
    "\u2032": "'",
    "\u2035": "'",
    "\u201c": '"',
    "\u201d": '"',
    "\u201e": '"',
    "\u201f": '"',
    "\u2033": '"',
    "\u2036": '"',
}

DASH_REPLACEMENTS: Dict[str, str] = {
    "\u2010": "-",
    "\u2011": "-",
    "\u2012": "-",
    "\u2013": "-",
    "\u2014": "-",
    "\u2015": "-",
    "\u2212": "-",
}

SPACE_REPLACEMENTS: Dict[str, str] = {
    "\u00a0": " ",
    "\u1680": " ",
    "\u2000": " ",
    "\u2001": " ",
    "\u2002": " ",
    "\u2003": " ",
    "\u2004": " ",
    "\u2005": " ",
    "\u2006": " ",
    "\u2007": " ",
    "\u2008": " ",
    "\u2009": " ",
    "\u200a": " ",
    "\u202f": " ",
    "\u205f": " ",
    "\u3000": " ",
}

INVISIBLE_PATTERN = re.compile("[\u200b\u200c\u200d\ufeff]")


def _replace_many(text: str, replacements: Dict[str, str]) -> str:
    for src, dst in replacements.items():
        text = text.replace(src, dst)
    return text


def repair_with_ftfy_if_available(text: str) -> str:
    try:
        import ftfy
    except ImportError:
        return text

    return ftfy.fix_text(text)


def normalize_text(
    text: str,
    *,
    repair_mojibake: bool = True,
    use_ftfy: bool = False,
    unicode_form: str = "NFKC",
    normalize_quotes: bool = True,
    normalize_dashes: bool = True,
    normalize_spaces: bool = True,
    preserve_linebreaks: bool = True,
) -> str:
    """
    Conservatively normalize detector input text.

    This is intentionally not a heavy linguistic normalizer. It repairs common
    encoding artifacts and Unicode variants while preserving stylistic content,
    especially poetry line breaks.
    """
    if not isinstance(text, str):
        text = str(text or "")

    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = INVISIBLE_PATTERN.sub("", text)

    if repair_mojibake:
        if use_ftfy:
            text = repair_with_ftfy_if_available(text)
        text = _replace_many(text, MOJIBAKE_REPLACEMENTS)

    if unicode_form:
        text = unicodedata.normalize(unicode_form, text)

    if normalize_quotes:
        text = _replace_many(text, QUOTE_REPLACEMENTS)

    if normalize_dashes:
        text = _replace_many(text, DASH_REPLACEMENTS)

    if normalize_spaces:
        text = _replace_many(text, SPACE_REPLACEMENTS)
        if preserve_linebreaks:
            text = re.sub(r"[ \t\f\v]+", " ", text)
            text = re.sub(r" *\n *", "\n", text)
            text = re.sub(r"\n{4,}", "\n\n\n", text)
        else:
            text = re.sub(r"\s+", " ", text)

    return text.strip()


def count_changed_texts(texts) -> int:
    return sum(1 for text in texts if normalize_text(text) != text)
