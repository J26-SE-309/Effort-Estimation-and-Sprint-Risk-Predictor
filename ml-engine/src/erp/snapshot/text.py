"""Story text as the encoders will read it."""

import re

import pandas as pd

from erp import tawos

# Jira wiki markup for pasted code and logs: {code}, {code:java}, {noformat} ... up to the closing tag
# (or the end, when someone forgot it). TAWOS's Description_Text removes the same blocks, but only for
# today's description, so the snapshot does it itself for the description as it was at commitment.
_CODE_BLOCK = re.compile(r"\{(code|noformat)(?::[^}]*)?\}.*?(?:\{\1\}|\Z)", re.S | re.I)
_SPACE = re.compile(r"\s+")


def has_code(text: str | None) -> bool:
    return isinstance(text, str) and _CODE_BLOCK.search(text) is not None


def clean_text(text: str | None, quoted: bool = False) -> str:
    """Text without code blocks and with whitespace collapsed; quoted=True first undoes TAWOS's CSV quoting."""
    if text is None or pd.isna(text):
        return ""
    if quoted:
        text = tawos.unquote(text)
    return _SPACE.sub(" ", _CODE_BLOCK.sub(" ", text)).strip()
