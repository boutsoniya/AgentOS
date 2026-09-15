"""Input safety helpers. Retrieved/uploaded text is always treated as data."""

PROMPT_INJECTION_MARKERS = ("ignore previous instructions", "system message", "developer message", "reveal your prompt")


def sanitize_untrusted_text(text: str) -> str:
    # We do not execute instructions found inside documents. This lightweight
    # marker is only telemetry; semantic filtering belongs in the retrieval layer.
    return text.replace("\x00", "")[:12000]
