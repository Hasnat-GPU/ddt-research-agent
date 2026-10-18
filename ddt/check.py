"""The quote check: a claim counts only if its quoted words are found in its source page."""
import re

from . import research

_PLAIN = str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"',
                        "–": "-", "—": "-", "−": "-", " ": " "})


def normalize(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").lower().translate(_PLAIN)).strip()


def check_claim(claim: dict, page_text: str) -> bool:
    quote = normalize(claim.get("quote", ""))
    return len(quote.split()) >= 8 and quote in normalize(page_text)


def page_text_for(url: str, results: list[dict]) -> str:
    for r in results:
        if r["url"] == url and r.get("raw_content"):
            return r["raw_content"]
    return research.extract(url)
