from __future__ import annotations

import io
import json
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd


@dataclass(frozen=True)
class MatchInputs:
    match_json: dict
    df_tracking: pd.DataFrame
    df_events: pd.DataFrame
    df_phases: pd.DataFrame


def _read_text_from_zip(zf: zipfile.ZipFile, name: str) -> str:
    with zf.open(name) as f:
        return f.read().decode("utf-8")


def _read_json_from_zip(zf: zipfile.ZipFile, name: str) -> dict:
    return json.loads(_read_text_from_zip(zf, name))


def _read_csv_from_zip(zf: zipfile.ZipFile, name: str) -> pd.DataFrame:
    with zf.open(name) as f:
        return pd.read_csv(f)


def _read_jsonl_tracking_from_zip(zf: zipfile.ZipFile, jsonl_name: str) -> pd.DataFrame:
    # pandas can read jsonl from a file-like object; handle both .jsonl and .jsonl.gz
    with zf.open(jsonl_name) as f:
        if jsonl_name.endswith(".gz"):
            import gzip
            with gzip.GzipFile(fileobj=f, mode="rb") as gf:
                return pd.read_json(gf, lines=True)
        return pd.read_json(f, lines=True)


def load_match_from_demo_zip(match_id: str, demo_zip_dir: Path) -> MatchInputs:
    zip_path = demo_zip_dir / f"{match_id}.zip"
    if not zip_path.exists():
        raise FileNotFoundError(f"Demo zip not found: {zip_path}")

    match_json_name = f"{match_id}_match.json"
    events_name = f"{match_id}_dynamic_events.csv"
    phases_name = f"{match_id}_phases_of_play.csv"
    tracking_jsonl = f"{match_id}_tracking_extrapolated.jsonl"
    tracking_jsonl_gz = f"{tracking_jsonl}.gz"

    with zipfile.ZipFile(zip_path, "r") as zf:
        names = set(zf.namelist())

        missing = [n for n in [match_json_name, events_name, phases_name] if n not in names]
        if missing:
            raise ValueError(f"Zip {zip_path.name} missing required files: {missing}")

        tracking_name: Optional[str] = None
        if tracking_jsonl_gz in names:
            tracking_name = tracking_jsonl_gz
        elif tracking_jsonl in names:
            tracking_name = tracking_jsonl
        else:
            raise ValueError(
                f"Zip {zip_path.name} missing tracking file: "
                f"{tracking_jsonl} or {tracking_jsonl_gz}"
            )

        match_json = _read_json_from_zip(zf, match_json_name)
        df_events = _read_csv_from_zip(zf, events_name)
        df_phases = _read_csv_from_zip(zf, phases_name)
        df_tracking = _read_jsonl_tracking_from_zip(zf, tracking_name)

    return MatchInputs(
        match_json=match_json,
        df_tracking=df_tracking,
        df_events=df_events,
        df_phases=df_phases,
    )

