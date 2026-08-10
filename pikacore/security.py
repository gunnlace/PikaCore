"""Environment filtering and recursive secret redaction."""

import re
from collections.abc import Mapping
from typing import Any

_ALLOWED_ENV_NAMES = {
    "PATH",
    "HOME",
    "USER",
    "SHELL",
    "LANG",
    "TERM",
    "TMPDIR",
    "VIRTUAL_ENV",
    "PYTHONPATH",
}
_SECRET_NAME_RE = re.compile(r"KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL", re.IGNORECASE)
_BEARER_RE = re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]+")
_API_KEY_RE = re.compile(r"\b(?:sk|pk)-[A-Za-z0-9_-]{8,}\b")
_CREDENTIAL_URL_RE = re.compile(r"(://)[^\s/@:]+:[^\s/@]+@")
_MAX_STRING_LENGTH = 4000


def sanitize_environment(source: Mapping[str, str]) -> dict[str, str]:
    """Return only safe process variables from the explicit allowlist."""
    sanitized = {}
    for name, value in source.items():
        upper_name = name.upper()
        if _SECRET_NAME_RE.search(upper_name):
            continue
        if upper_name in _ALLOWED_ENV_NAMES or upper_name.startswith("LC_"):
            sanitized[name] = value
    return sanitized


def redact(value: Any) -> Any:
    """Recursively redact common credentials and limit retained strings."""
    if isinstance(value, dict):
        return {
            key: "[REDACTED]" if _SECRET_NAME_RE.search(str(key)) else redact(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [redact(item) for item in value]
    if isinstance(value, tuple):
        return tuple(redact(item) for item in value)
    if isinstance(value, str):
        redacted = _BEARER_RE.sub("Bearer [REDACTED]", value)
        redacted = _API_KEY_RE.sub("[REDACTED]", redacted)
        redacted = _CREDENTIAL_URL_RE.sub(r"\1[REDACTED]@", redacted)
        if len(redacted) > _MAX_STRING_LENGTH:
            redacted = redacted[:_MAX_STRING_LENGTH] + "... [truncated]"
        return redacted
    return value
