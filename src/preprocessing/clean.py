"""Text cleaning: lowercasing and regex-based removal of URLs, @mentions,
HTML entities, control characters, and emoji.

Implements the EARS criterion in specs/001-hate-speech-detection/spec.md
(section 2, Preprocessing): "WHEN raw comment text is passed in, THE
SYSTEM SHALL return cleaned text with no URLs, HTML entities, or control
characters."
"""
import html
import re

URL_RE = re.compile(r"https?://\S+|www\.\S+")
MENTION_RE = re.compile(r"@\w+")
CONTROL_CHAR_RE = re.compile(r"[\x00-\x1f\x7f]")
EMOJI_RE = re.compile(
    "["
    "\U0001f300-\U0001faff"  # symbols, pictographs, emoticons, transport
    "\U00002600-\U000027bf"  # misc symbols, dingbats
    "\U0001f1e6-\U0001f1ff"  # regional indicators (flags)
    "]+",
    flags=re.UNICODE,
)
WHITESPACE_RE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """Lowercase and strip URLs, @mentions, HTML entities, control
    characters, and emoji from a comment. Collapses repeated whitespace."""
    text = html.unescape(text)
    text = text.lower()
    text = URL_RE.sub(" ", text)
    text = MENTION_RE.sub(" ", text)
    text = EMOJI_RE.sub(" ", text)
    text = CONTROL_CHAR_RE.sub(" ", text)
    return WHITESPACE_RE.sub(" ", text).strip()
