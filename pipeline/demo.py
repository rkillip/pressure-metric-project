# pipeline/demo.py
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import requests


# SkillCorner OpenData repo
OWNER = "SkillCorner"
REPO = "opendata"
BRANCH = "master"


@dataclass(frozen=True)
class MatchFiles:
    """
    Raw payloads for one match. `process_match()` expects these exact attribute names:
      - match_json
      - tracking
      - dynamic_events
      - phases_of_play
    """
    match_id: str
    match_json: bytes
    tracking: bytes
    dynamic_events: bytes
    phases_of_play: bytes


def load_curated_demo_list(path: Path) -> list[dict[str, Any]]:
    """Read data/demo_matches.json (a list of {match_id, label?})."""
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def _raw_url(rel_path: str) -> str:
    # Often works for non-LFS files
    return f"https://raw.githubusercontent.com/{OWNER}/{REPO}/{BRANCH}/{rel_path}"


def _media_url(rel_path: str) -> str:
    # Works well for large/LFS-tracked files
    return f"https://media.githubusercontent.com/media/{OWNER}/{REPO}/{BRANCH}/{rel_path}"


def _download_bytes(rel_path: str, *, timeout: int = 90) -> bytes:
    """
    Download file bytes from GitHub. Tries raw.githubusercontent.com first, then
    media.githubusercontent.com (handles large/LFS cases).

    Raises a ValueError with a useful message if we get HTML or an LFS pointer.
    """
    urls = [_raw_url(rel_path), _media_url(rel_path)]
    last_err: Optional[Exception] = None

    for url in urls:
        try:
            r = requests.get(url, timeout=timeout)
            r.raise_for_status()
            content = r.content

            head = content[:200].decode("utf-8", errors="replace").lower()
            if "<html" in head or "<!doctype html" in head:
                raise ValueError(f"Got HTML instead of file bytes from {url}")

            # Git LFS pointer file (text) instead of the actual binary content
            if head.startswith("version https://git-lfs.github.com/spec"):
                raise ValueError(f"Got a Git LFS pointer (not the file content) from {url}")

            return content
        except Exception as e:
            last_err = e

    raise ValueError(f"Failed to download {rel_path}: {last_err}")


def load_demo_match(match_id: str, *, raw_cache_dir: Path, force_download: bool = False) -> MatchFiles:
    """
    Download (or load from cache) the 4 raw match files SkillCorner provides:
      - {id}_match.json
      - {id}_tracking_extrapolated.jsonl
      - {id}_dynamic_events.csv
      - {id}_phases_of_play.csv

    Caches them to data/raw/ so Streamlit doesn't re-download every rerun.
    """
    raw_cache_dir.mkdir(parents=True, exist_ok=True)

    # cache paths
    p_match = raw_cache_dir / f"{match_id}_match.json"
    p_track = raw_cache_dir / f"{match_id}_tracking_extrapolated.jsonl"
    p_dyn = raw_cache_dir / f"{match_id}_dynamic_events.csv"
    p_pop = raw_cache_dir / f"{match_id}_phases_of_play.csv"

    if (
        not force_download
        and p_match.exists()
        and p_track.exists()
        and p_dyn.exists()
        and p_pop.exists()
    ):
        return MatchFiles(
            match_id=match_id,
            match_json=p_match.read_bytes(),
            tracking=p_track.read_bytes(),
            dynamic_events=p_dyn.read_bytes(),
            phases_of_play=p_pop.read_bytes(),
        )

    base = f"data/matches/{match_id}"

    # NOTE: if SkillCorner changes extensions to .jsonl.gz for tracking,
    # adjust this filename accordingly and keep your gzip-decompress logic
    # in pipeline/process.py.
    match_json = _download_bytes(f"{base}/{match_id}_match.json")
    tracking = _download_bytes(f"{base}/{match_id}_tracking_extrapolated.jsonl")
    dynamic_events = _download_bytes(f"{base}/{match_id}_dynamic_events.csv")
    phases_of_play = _download_bytes(f"{base}/{match_id}_phases_of_play.csv")

    # write cache
    p_match.write_bytes(match_json)
    p_track.write_bytes(tracking)
    p_dyn.write_bytes(dynamic_events)
    p_pop.write_bytes(phases_of_play)

    return MatchFiles(
        match_id=match_id,
        match_json=match_json,
        tracking=tracking,
        dynamic_events=dynamic_events,
        phases_of_play=phases_of_play,
    )
