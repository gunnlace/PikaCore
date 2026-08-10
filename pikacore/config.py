"""Configuration - env vars and defaults."""

import os
from dataclasses import dataclass
from pathlib import Path


def _getenv(primary: str, legacy: str, default: str) -> str:
    """Read a PikaCore variable, falling back to its legacy CoreCoder name."""
    return os.getenv(primary) or os.getenv(legacy) or default


def _load_dotenv():
    """Load .env from cwd, walking up to home dir. No-op if python-dotenv missing."""
    try:
        from dotenv import load_dotenv
        # search cwd first, then parent dirs up to ~
        env_path = Path(".env")
        if not env_path.exists():
            cur = Path.cwd()
            home = Path.home()
            while cur != home and cur != cur.parent:
                candidate = cur / ".env"
                if candidate.exists():
                    env_path = candidate
                    break
                cur = cur.parent
        load_dotenv(env_path, override=False)
    except ImportError:
        pass  # python-dotenv not installed, silently skip


@dataclass
class Config:
    model: str = "gpt-5.5"
    api_key: str = ""
    base_url: str | None = None
    max_tokens: int = 4096
    temperature: float = 0.0
    max_context_tokens: int = 128_000
    provider: str = "openai"

    @classmethod
    def from_env(cls) -> "Config":
        # load .env if present (won't override existing env vars)
        _load_dotenv()
        # pick up common env vars automatically
        api_key = (
            os.getenv("PIKACORE_API_KEY")
            or os.getenv("OPENAI_API_KEY")
            or os.getenv("DEEPSEEK_API_KEY")
            or os.getenv("CORECODER_API_KEY")
            or ""
        )
        return cls(
            model=_getenv("PIKACORE_MODEL", "CORECODER_MODEL", "gpt-5.5"),
            api_key=api_key,
            base_url=(
                os.getenv("PIKACORE_BASE_URL")
                or os.getenv("OPENAI_BASE_URL")
                or os.getenv("CORECODER_BASE_URL")
            ),
            max_tokens=int(_getenv("PIKACORE_MAX_TOKENS", "CORECODER_MAX_TOKENS", "4096")),
            temperature=float(_getenv("PIKACORE_TEMPERATURE", "CORECODER_TEMPERATURE", "0")),
            max_context_tokens=int(_getenv("PIKACORE_MAX_CONTEXT", "CORECODER_MAX_CONTEXT", "128000")),
            provider=_getenv("PIKACORE_PROVIDER", "CORECODER_PROVIDER", "openai"),
        )
