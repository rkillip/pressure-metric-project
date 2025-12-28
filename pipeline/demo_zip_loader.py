from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import pandas as pd


DATA_DIR = Path("data")


def _zip_members(zf: zipfile.ZipFile) -> set[str]:
    # ignore folders + mac metadata
    return {
        n for n in zf.namelist()
        if not n.endswith("/")
        and not n.startswith("__MACOSX/")
        and "/._" not in n
    }


def load_match_from_repo_zip(match_id: str, zip_dir: Path = DATA_DIR):
    """
    Expects: data/{match_id}.zip

    Inside zip:
      {match_id}/{match_id}_match.json
      {match_id}/{match_id}_tracking_extrapolated.jsonl OR .jsonl.gz
      {match_id}/{match_id}_dynamic_events.csv
      {match_id}/{match_id}_phases_of_play.csv
    """
    zip_path = zip_dir / f"{match_id}.zip"
    if not zip_path.exists():
        raise FileNotFoundError(f"Demo zip not found: {zip_path}")

    base = f"{match_id}/{match_id}"
    match_name = f"{base}_match.json"
    events_name = f"{base}_dynamic_events.csv"
    phases_name = f"{base}_phases_of_play.csv"
    tracking_jsonl = f"{base}_tracking_extrapolated.jsonl"
    tracking_gz = f"{tracking_jsonl}.gz"

    with zipfile.ZipFile(zip_path, "r") as zf:
        members = _zip_members(zf)

        missing = [n for n in [match_name, events_name, phases_name] if n not in members]
        if missing:
            raise ValueError(f"{zip_path.name} missing required files: {missing}")

        if tracking_gz in members:
            tracking_name = tracking_gz
            with zf.open(tracking_name) as f:
                # pandas handles gz via compression='gzip' when reading file-like
                df_tracking = pd.read_json(f, lines=True, compression="gzip")
        elif tracking_jsonl in members:
            tracking_name = tracking_jsonl
            with zf.open(tracking_name) as f:
                df_tracking = pd.read_json(f, lines=True)
        else:
            raise ValueError(
                f"{zip_path.name} missing tracking file: {tracking_jsonl} or {tracking_gz}"
            )

        with zf.open(match_name) as f:
            match_json = json.loads(f.read().decode("utf-8"))

        with zf.open(events_name) as f:
            df_events = pd.read_csv(f)

        with zf.open(phases_name) as f:
            df_phases = pd.read_csv(f)

    return match_json, df_tracking, df_events, df_phases
