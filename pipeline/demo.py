from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests

from .io import MatchFiles, load_match_from_folder

# SkillCorner Open Data repo layout:
#   data/matches.json
#   data/matches/<match_id>/{match_id}_match.json, ... tracking_extrapolated.jsonl, dynamic_events.csv, phases_of_play.csv
# Documented in the repo README. :contentReference[oaicite:1]{index=1}


@dataclass(frozen=True)
class RepoSpec:
    owner: str = "SkillCorner"
    repo: str = "opendata"
    branch: str = "master"

    def raw_base(self) -> str:
        # raw file host for GitHub repos
        return f"https://raw.githubusercontent.com/{self.owner}/{self.repo}/{self.branch}"


DEFAULT_REPO = RepoSpec()


REQUIRED_SUFFIXES = (
    "match.json",
    "tracking_extrapolated.jsonl",
    "dynamic_events.csv",
    "phases_of_play.csv",  # optional for your current app, but we fetch it for completeness
)


def _http_get_bytes(url: str, *, timeout_s: float = 30.0) -> bytes:
    r = requests.get(url, timeout=timeout_s)
    r.raise_for_status()
    return r.content


def list_available_match_ids(repo: RepoSpec = DEFAULT_REPO) -> List[str]:
    """
    Reads data/matches.json from the SkillCorner Open Data repo and returns match ids as strings.
    """
    url = f"{repo.raw_base()}/data/matches.json"
    raw = _http_get_bytes(url)
    data = json.loads(raw.decode("utf-8"))

    # matches.json is a list of match objects containing an "id"
    ids: List[str] = []
    for row in data:
        mid = row.get("id")
        if mid is not None:
            ids.append(str(mid))
    return sorted(set(ids))


def download_raw_match_files(
    match_id: str,
    *,
    repo: RepoSpec = DEFAULT_REPO,
    raw_cache_dir: Path = Path("data/raw"),
    force: bool = False,
) -> Path:
    """
    Downloads the 4 raw files for a match into:
      data/raw/<match_id>/

    Returns the folder path.

    This is designed for Streamlit Community Cloud:
    - cached on the container filesystem (ephemeral, but fast)
    - NEVER committed to GitHub (gitignore data/raw/)
    """
    match_id = str(match_id)
    out_dir = raw_cache_dir / match_id
    out_dir.mkdir(parents=True, exist_ok=True)

    # Repo path: data/matches/<match_id>/{match_id}_<suffix>
    base = f"{repo.raw_base()}/data/matches/{match_id}"

    for suffix in REQUIRED_SUFFIXES:
        fname = f"{match_id}_{suffix}"
        url = f"{base}/{fname}"
        out_path = out_dir / fname

        if out_path.exists() and not force:
            continue

        blob = _http_get_bytes(url)
        out_path.write_bytes(blob)

    return out_dir


def load_demo_match(
    match_id: str,
    *,
    repo: RepoSpec = DEFAULT_REPO,
    raw_cache_dir: Path = Path("data/raw"),
    force_download: bool = False,
) -> MatchFiles:
    """
    Ensures raw files exist in cache, then loads them into MatchFiles for processing.
    """
    folder = download_raw_match_files(
        match_id,
        repo=repo,
        raw_cache_dir=raw_cache_dir,
        force=force_download,
    )
    return load_match_from_folder(str(match_id), folder)


def load_curated_demo_list(path: Path = Path("data/demo_matches.json")) -> List[Dict[str, str]]:
    """
    Optional: if you want a stable 'top 10 matches' list with labels,
    keep a tiny json file in the repo.
    """
    if not path.exists():
        return []
    data = json.loads(path.read_text())
    out: List[Dict[str, str]] = []
    for row in data:
        mid = str(row.get("match_id", "")).strip()
        label = str(row.get("label", "")).strip()
        if mid:
            out.append({"match_id": mid, "label": label or mid})
    return out
