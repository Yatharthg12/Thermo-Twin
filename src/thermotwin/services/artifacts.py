"""Atomic JSON writes, source fingerprints, manifests, and scoped artifact catalogs."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import os
from pathlib import Path
import platform
from typing import Any

from thermotwin.config import PROJECT_ROOT, config_fingerprint


EXCLUDED = {
    "artifacts",
    "reports",
    "docs",
    "prompt",
    ".git",
    ".venv",
    "__pycache__",
    ".pytest_cache",
}


def _package_versions() -> dict[str, str]:
    """Capture installed distribution versions without importing heavyweight libraries."""
    result = {"python": platform.python_version()}
    for key, distribution in (("flask", "Flask"), ("numpy", "numpy"), ("scipy", "scipy"), ("pandas", "pandas"),
                              ("scikit_learn", "scikit-learn"), ("matplotlib", "matplotlib"), ("pyyaml", "PyYAML"), ("torch", "torch")):
        try:
            result[key] = version(distribution)
        except PackageNotFoundError:
            result[key] = "unavailable"
    return result


def atomic_json(path: str | Path, data: Any) -> Path:
    """Write strict finite JSON and atomically replace the destination."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_suffix(target.suffix + ".tmp")
    temp.write_text(json.dumps(data, indent=2, allow_nan=False, default=str), encoding="utf-8")
    os.replace(temp, target)
    return target


def source_fingerprint(root: Path = PROJECT_ROOT) -> str:
    """Hash source/config/frontend files while excluding generated outputs and caches."""
    digest = hashlib.sha256()
    included: list[Path] = []
    for directory, names, filenames in os.walk(root):
        names[:] = sorted(name for name in names if name not in EXCLUDED and not name.endswith(".egg-info"))
        base = Path(directory)
        for filename in filenames:
            path = base / filename
            if path.suffix.lower() in {".py", ".yaml", ".toml", ".html", ".css", ".js", ".svg", ".txt"}:
                included.append(path)
    for path in sorted(included):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()[:16]


def create_manifest(run_id: str, config: dict, status: str = "running") -> dict[str, Any]:
    """Create reproducibility metadata without relying on Git being installed."""
    return {
        "run_id": run_id, "created_utc": datetime.now(timezone.utc).isoformat(),
        "profile": config["project"]["profile"], "status": status,
        "resolved_config": config, "seeds": {"root": config["project"]["seed"]},
        "interpreter": platform.python_version(), "platform": platform.platform(),
        "packages": _package_versions(),
        "source_fingerprint": source_fingerprint(), "config_fingerprint": config_fingerprint(config),
        "git_revision": None, "git_note": "Git executable was unavailable; source fingerprint recorded instead",
        "artifacts": [],
    }


def artifact_catalog(run_dir: str | Path) -> list[dict[str, Any]]:
    """Enumerate only files beneath one known run directory using opaque relative IDs."""
    root = Path(run_dir).resolve()
    allowed = {".json", ".csv", ".html", ".png", ".svg", ".yaml", ".txt"}
    return [{"artifact_id": path.relative_to(root).as_posix(), "name": path.name, "size_bytes": path.stat().st_size}
            for path in sorted(root.rglob("*")) if path.is_file() and path.suffix.lower() in allowed]


def resolve_artifact(run_dir: str | Path, artifact_id: str) -> Path:
    """Resolve a catalogued artifact and reject traversal or unlisted file types."""
    root = Path(run_dir).resolve()
    target = (root / artifact_id).resolve()
    try:
        target.relative_to(root)
    except ValueError as error:
        raise ValueError("artifact path traversal rejected") from error
    ids = {item["artifact_id"] for item in artifact_catalog(root)}
    if artifact_id not in ids or not target.is_file():
        raise FileNotFoundError(artifact_id)
    return target
