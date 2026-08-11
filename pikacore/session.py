"""Session persistence - save and resume conversations.

Claude Code maintains session state via QueryEngine (1295 lines).
PikaCore distills this to: JSON dump of messages + model config.
"""

import re
import time
import uuid
from pathlib import Path

from .security import redact
from .state import SchemaMismatchError, SessionState, utc_now
from .store import atomic_write_json, read_json


def _find_repo_root(start: Path | None = None) -> Path:
    """Find the nearest Git repository root, falling back to the start directory."""
    start = (start or Path.cwd()).resolve()
    for candidate in (start, *start.parents):
        if (candidate / ".git").exists():
            return candidate
    return start


SESSIONS_DIR = _find_repo_root() / ".pikacore" / "sessions"
_SAFE_SESSION_RE = re.compile(r"[^A-Za-z0-9._-]+")
_MAX_SESSION_ID_LEN = 100  # keep filenames comfortably under the OS limit


def _normalize_session_id(session_id: str | None) -> str:
    if not session_id:
        return _new_session_id()

    name = session_id.strip().replace("\\", "/").split("/")[-1]
    name = _SAFE_SESSION_RE.sub("-", name).strip(".-_")
    if len(name) > _MAX_SESSION_ID_LEN:
        name = name[:_MAX_SESSION_ID_LEN].strip(".-_")
    return name or _new_session_id()


def _new_session_id() -> str:
    return f"session_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"


def _session_path(session_id: str) -> Path:
    path = (SESSIONS_DIR / f"{_normalize_session_id(session_id)}.json").resolve()
    root = SESSIONS_DIR.resolve()
    if root != path.parent:
        raise ValueError("Invalid session id")
    return path


def save_session(messages: list[dict], model: str, session_id: str | None = None) -> str:
    """Save conversation to disk. Returns the session ID."""
    SESSIONS_DIR.mkdir(parents=True, exist_ok=True)

    session_id = _normalize_session_id(session_id)

    timestamp = utc_now()
    state = SessionState(
        session_id=session_id,
        created_at=timestamp,
        updated_at=timestamp,
        repo_root=str(_find_repo_root()),
        model=model,
        messages=messages,
    )

    path = _session_path(session_id)
    atomic_write_json(path, redact(state.to_dict(), max_string_length=None))
    return session_id


def load_session(session_id: str) -> SessionState | None:
    """Load the complete saved session, upgrading the legacy shape if needed."""
    path = _session_path(session_id)
    if not path.exists():
        return None

    try:
        data = read_json(path)
        if "schema_version" not in data:
            timestamp = data.get("saved_at") or utc_now()
            return SessionState(
                session_id=_normalize_session_id(
                    data.get("session_id", data.get("id", session_id))
                ),
                created_at=timestamp,
                updated_at=timestamp,
                repo_root=str(_find_repo_root()),
                model=data["model"],
                messages=data["messages"],
            )
        return SessionState.from_dict(data)
    except SchemaMismatchError:
        raise
    except (TypeError, ValueError, KeyError, OSError):
        # a corrupt or truncated session file shouldn't crash resume
        return None


def list_sessions() -> list[dict]:
    """List available sessions, newest first."""
    if not SESSIONS_DIR.exists():
        return []

    sessions = []
    for f in sorted(SESSIONS_DIR.glob("*.json"), reverse=True):
        try:
            data = read_json(f)
            # grab first user message as preview
            preview = ""
            for m in data.get("messages", []):
                if m.get("role") == "user" and m.get("content"):
                    preview = m["content"][:80]
                    break
            sessions.append({
                "id": data.get("session_id", data.get("id", f.stem)),
                "model": data.get("model", "?"),
                "saved_at": data.get("updated_at", data.get("saved_at", "?")),
                "preview": preview,
            })
        except (ValueError, KeyError, OSError):
            continue

    return sessions[:20]  # cap at 20
