import os
from pathlib import Path
from typing import Dict


def load_env(env_path: str | None = None, override: bool = False) -> None:
    """
    Minimal .env loader without external dependencies.
    - env_path: optional path to .env; defaults to repo root/.env
    - override: if True, overwrite existing os.environ values
    """
    path = Path(env_path) if env_path else Path(__file__).resolve().parents[2] / ".env"
    if not path.exists():
        return
    try:
        with path.open("r", encoding="utf-8") as f:
            for raw in f:
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" not in line:
                    continue
                k, v = line.split("=", 1)
                key = k.strip()
                val = v.strip().strip('"').strip("'")
                if key and (override or key not in os.environ):
                    os.environ[key] = val
    except Exception:
        # Fail quietly; the app can still rely on existing environment
        pass


# Singleton settings dict
_settings: Dict[str, str] | None = None


def get_settings() -> Dict[str, str]:
    """Return application settings from environment with sane defaults.

    Returns a singleton dict that can be modified at runtime for testing.
    """
    global _settings
    if _settings is None:
        _settings = {
            "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY", ""),
            "OPENAI_EMBEDDING_MODEL": os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
            "OPENAI_CHAT_MODEL": os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini"),
            "CHROMA_DB_DIR": os.getenv("CHROMA_DB_DIR", ".chroma"),
            "HR_AGGREGATE_MODE": os.getenv("HR_AGGREGATE_MODE", "disabled"),
            "HR_MASKED_ROWS_MODE": os.getenv("HR_MASKED_ROWS_MODE", "disabled"),
            "AUDIT_LOG_DIR": os.getenv("AUDIT_LOG_DIR", os.path.join("logs", "audit")),
            "HOST": os.getenv("HOST", "127.0.0.1"),
            "PORT": os.getenv("PORT", "8000"),
        }
    return _settings

