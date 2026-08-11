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
    # Windows CreateProcess/cmd.exe and common child runtimes depend on these.
    "COMSPEC",
    "SYSTEMROOT",
    "SYSTEMDRIVE",
    "WINDIR",
    "PATHEXT",
    "TEMP",
    "TMP",
    "USERPROFILE",
    "USERNAME",
    "HOMEDRIVE",
    "HOMEPATH",
}
_SECRET_NAME_RE = re.compile(r"KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL", re.IGNORECASE)
_TOKEN_COUNTER_NAME_RE = re.compile(r"(?:^|_)TOKENS(?:_|$)", re.IGNORECASE)
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


def redact(value: Any, *, max_string_length: int | None = _MAX_STRING_LENGTH) -> Any:
    """Recursively redact common credentials and optionally limit strings."""
    if isinstance(value, dict):
        return {
            key: (
                "[REDACTED]"
                if _SECRET_NAME_RE.search(str(key))
                and not (
                    _TOKEN_COUNTER_NAME_RE.search(str(key))
                    and isinstance(item, (int, float))
                    and not isinstance(item, bool)
                )
                else redact(item, max_string_length=max_string_length)
            )
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [redact(item, max_string_length=max_string_length) for item in value]
    if isinstance(value, tuple):
        return tuple(
            redact(item, max_string_length=max_string_length) for item in value
        )
    if isinstance(value, str):
        redacted = _BEARER_RE.sub("Bearer [REDACTED]", value)
        redacted = _API_KEY_RE.sub("[REDACTED]", redacted)
        redacted = _CREDENTIAL_URL_RE.sub(r"\1[REDACTED]@", redacted)
        if max_string_length is not None and len(redacted) > max_string_length:
            redacted = redacted[:max_string_length] + "... [truncated]"
        return redacted
    return value
