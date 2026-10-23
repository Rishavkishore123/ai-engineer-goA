"""Conservative request classifier for concierge messages."""

import re

REQUEST_PATTERNS = [
    r"\b(book|booking|reserve|reservation)\b",
    r"\b(cancel|cancellation|reschedule)\b",
    r"\b(table|restaurant|dinner|lunch)\b",
    r"\b(find us|arrange|organize|get us)\b",
    r"\b(kar do|kardo|karna hai|book kar)\b",
    r"\b(vendor|availability|available)\b",
]


def predict(texts: list[str]) -> list[str]:
    """Return 'request' or 'not_request' for each input message.

    Favor recall over precision: potentially actionable messages should
    be passed to the concierge agent for its safety checks.
    """
    predictions = []

    for text in texts:
        normalized = text.lower()
        is_request = any(
            re.search(pattern, normalized)
            for pattern in REQUEST_PATTERNS
        )
        predictions.append("request" if is_request else "not_request")

    return predictions