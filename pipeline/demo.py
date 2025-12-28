from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests


SKILLCORNER_REPO = ("SkillCorner", "opendata")
SKILLCORNER_BRANCH = "master"


@dataclass(frozen=True)
class DemoMatchFiles:
    match_id: str
    match_json: bytes
    tracking_jsonl: bytes
    dynamic_events_csv: bytes
    phases_of_play_csv: bytes


def load_curated_demo_list(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return json.loads(path.read_text())


def _raw_url(owner: str, repo: str, branch: str, rel_path: str) -> str:
    # IMPORTANT: raw.githubusercontent.com serves the actual file bytes.
    return f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{rel_path}"


def _download_bytes(url: str) -> bytes:
    r = requests.get(url, timeout=60)
    r.raise_for_status()

    head = r.content[:200].decode("utf-8", errors="replace").lower()
    if "<html" in head or "<!doctype html" in head:
        raise ValueError(f"Got HTML instead of a file. URL likely wrong: {url}")

    return r.content


def load_demo_match(match_id: str, *, raw_cache_dir: Path, force_download: bool = False) -> DemoMatchFiles:
    raw_cache_dir.mkdir(parents=True, exist_ok=True)

    # Cache paths
    p_match = raw_cache_dir / f"{match_id}_match.json"
    p_track = raw_cache_dir / f"{match_id}_tracking_extrapolated.jsonl"
    p_de = raw_cache_dir / f"{match_id}_dynamic_events.csv"
    p_pop = raw_cache_dir / f"{match_id}_phases_of_play.csv"

    have_all = p_match.exists() and p_track.exists() and p_de.exists() and p_pop.exists()
    if have_all and not force_download:
        return DemoMatchFiles(
            match_id=match_id,
            match_json=p_match.read_bytes(),
            tracking_jsonl=p_track.read_bytes(),
            dynamic_events_csv=p_de.read_bytes(),
            phases_of_play_csv=p_pop.read_bytes(),
        )

    owner, repo = SKILLCORNER_REPO
    base = f"data/matches/{match_id}"

    urls = {
        "match": _raw_url(owner, repo, SKILLCORNER_BRANCH, f"{base}/{match_id}_match.json"),
        "tracking": _raw_url(owner, repo, SKILLCORNER_BRANCH, f"{base}/{match_id}_tracking_extrapolated.jsonl"),
        "dynamic_events": _raw_url(owner, repo, SKILLCORNER_BRANCH, f"{base}/{match_id}_dynamic_events.csv"),
        "phases": _raw_url(owner, repo, SKILLCORNER_BRANCH, f"{base}/{match_id}_phases_of_play.csv"),
    }

    match_json = _download_bytes(urls["match"])
    tracking_jsonl = _download_bytes(urls["tracking"])
    dynamic_events_csv = _download_bytes(urls["dynamic_events"])
    phases_of_play_csv = _download_bytes(urls["phases"])

    # Write cache
    p_match.write_bytes(match_json)
    p_track.write_bytes(tracking_jsonl)
    p_de.write_bytes(dynamic_events_csv)
    p_pop.write_bytes(phases_of_play_csv)

    return DemoMatchFiles(
        match_id=match_id,
        match_json=match_json,
        tracking_jsonl=tracking_jsonl,
        dynamic_events_csv=dynamic_events_csv,
        phases_of_play_csv=phases_of_play_csv,
    )

