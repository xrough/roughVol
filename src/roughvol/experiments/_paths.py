from __future__ import annotations

from pathlib import Path


def project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def output_dir(purpose: str) -> Path:
    path = project_root() / "output" / purpose
    path.mkdir(parents=True, exist_ok=True)
    return path


def output_path(purpose: str, filename: str) -> str:
    return str(output_dir(purpose) / filename)
