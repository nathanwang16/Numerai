"""Numerai credentials loader.

Resolution order (first match wins):

1. Explicit ``public_id`` / ``secret_key`` arguments to ``get_credentials``.
2. Environment variables ``NUMERAI_PUBLIC_ID`` + ``NUMERAI_SECRET_KEY``.
3. A KEY=VALUE / ``KEY: VALUE`` file at ``$NUMERAI_CREDENTIALS_FILE`` if set.
4. ``credentials.txt`` in the repo root.
5. ``.env`` in the repo root.

The repo-root files use a tolerant key/value format -- both ``KEY=VALUE`` and
``KEY: VALUE`` lines are accepted, blank lines and ``#`` comments are skipped.
The following key aliases all map to the same logical pair:

    NUMERAI_PUBLIC_ID  | public_id | NUMERAI_PUBLIC_KEY
    NUMERAI_SECRET_KEY | secret    | secret_key

This file MUST stay out of version control. The repo's ``.gitignore`` covers
``credentials.txt`` and ``.env*``.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

_PUBLIC_ALIASES = ("NUMERAI_PUBLIC_ID", "PUBLIC_ID", "NUMERAI_PUBLIC_KEY", "public_id")
_SECRET_ALIASES = ("NUMERAI_SECRET_KEY", "SECRET_KEY", "NUMERAI_SECRET", "secret_key", "secret")


@dataclass(frozen=True)
class NumeraiCredentials:
    public_id: str
    secret_key: str
    source: str

    def as_env(self) -> dict[str, str]:
        return {"NUMERAI_PUBLIC_ID": self.public_id, "NUMERAI_SECRET_KEY": self.secret_key}


def _repo_root() -> Path:
    # numerai_stack/compute/credentials.py -> repo root is parents[2]
    return Path(__file__).resolve().parents[2]


def _parse_kv_file(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" in line:
            k, _, v = line.partition("=")
        elif ":" in line:
            k, _, v = line.partition(":")
        else:
            continue
        k = k.strip()
        v = v.strip().strip('"').strip("'")
        if k:
            out[k] = v
    return out


def _pick(d: dict[str, str], aliases: tuple[str, ...]) -> str | None:
    for key in aliases:
        if key in d and d[key]:
            return d[key]
    return None


def _from_file(path: Path) -> tuple[str | None, str | None]:
    if not path.exists():
        return None, None
    parsed = _parse_kv_file(path)
    return _pick(parsed, _PUBLIC_ALIASES), _pick(parsed, _SECRET_ALIASES)


def get_credentials(
    public_id: str | None = None,
    secret_key: str | None = None,
    *,
    require: bool = True,
) -> NumeraiCredentials | None:
    """Resolve Numerai credentials. See module docstring for ordering.

    Returns ``None`` (and never raises) when ``require=False`` and no creds are
    found. With ``require=True`` (default) raises ``RuntimeError`` instead.
    """
    if public_id and secret_key:
        return NumeraiCredentials(public_id, secret_key, source="explicit")

    env_pid = os.environ.get("NUMERAI_PUBLIC_ID")
    env_sk = os.environ.get("NUMERAI_SECRET_KEY")
    if env_pid and env_sk:
        return NumeraiCredentials(env_pid, env_sk, source="env")

    candidates: list[Path] = []
    explicit = os.environ.get("NUMERAI_CREDENTIALS_FILE")
    if explicit:
        candidates.append(Path(explicit).expanduser())
    root = _repo_root()
    candidates.append(root / "credentials.txt")
    candidates.append(root / ".env")

    for path in candidates:
        pid, sk = _from_file(path)
        if pid and sk:
            return NumeraiCredentials(pid, sk, source=str(path))

    if require:
        searched = ", ".join(str(p) for p in candidates)
        raise RuntimeError(
            "Numerai credentials missing. Provide them via:\n"
            "  - NUMERAI_PUBLIC_ID + NUMERAI_SECRET_KEY env vars, or\n"
            f"  - one of: {searched}\n"
            "  - or pass public_id/secret_key explicitly."
        )
    return None


def load_into_env(*, override: bool = False) -> NumeraiCredentials | None:
    """Resolve creds and stamp them into ``os.environ`` so subprocesses see them.

    Idempotent: if env vars are already populated and ``override`` is False,
    nothing changes. Returns the resolved credentials, or ``None`` if none were
    found (never raises).
    """
    if not override and os.environ.get("NUMERAI_PUBLIC_ID") and os.environ.get("NUMERAI_SECRET_KEY"):
        return NumeraiCredentials(
            os.environ["NUMERAI_PUBLIC_ID"], os.environ["NUMERAI_SECRET_KEY"], source="env"
        )
    creds = get_credentials(require=False)
    if creds is None:
        return None
    os.environ["NUMERAI_PUBLIC_ID"] = creds.public_id
    os.environ["NUMERAI_SECRET_KEY"] = creds.secret_key
    return creds


__all__ = ["NumeraiCredentials", "get_credentials", "load_into_env"]
