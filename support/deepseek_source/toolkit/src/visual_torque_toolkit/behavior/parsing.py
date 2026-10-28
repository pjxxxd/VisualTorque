"""Strict parsing for one-letter multiple-choice generations."""

from __future__ import annotations

import json
import re
from collections.abc import Sequence


def parse_choice_label(response: str | None, legal_labels: Sequence[str]) -> str | None:
    """Parse a legal label without guessing from arbitrary answer prose."""

    if response is None:
        return None
    legal = {label.upper() for label in legal_labels}
    text = response.strip()
    if not text:
        return None

    # Accept the standardized JSON format if a model emits it despite the
    # one-letter instruction.
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        value = None
    if isinstance(value, dict) and isinstance(value.get("answer"), str):
        label = value["answer"].strip().upper()
        return label if label in legal else None
    if isinstance(value, str):
        label = value.strip().upper()
        return label if label in legal else None

    cleaned = re.sub(r"<[^>]+>", " ", text).strip()
    direct = re.fullmatch(r"[\s\[\](){}.:*`_-]*([A-Za-z])[\s\[\](){}.:*`_-]*", cleaned)
    if direct:
        label = direct.group(1).upper()
        return label if label in legal else None

    explicit = re.search(
        r"(?:answer|choice|option)\s*(?:is|:|=)?\s*[\[\](){}]*([A-Za-z])\b",
        cleaned,
        flags=re.IGNORECASE,
    )
    if explicit:
        label = explicit.group(1).upper()
        return label if label in legal else None
    return None
