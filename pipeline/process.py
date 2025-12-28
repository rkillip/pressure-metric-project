from __future__ import annotations

import io
import json
import pandas as pd
from pathlib import Path
from typing import Any, Dict, Tuple


from .io import MatchFiles
from .schemas import ProcessedMatch
from .validate import require_columns, require_nonempty


def _read_tracking(match) -> pd.DataFrame:
    raw = match.tracking  # bytes (expected)

    if raw is None or len(raw) == 0:
        raise ValueError("Tracking payload is empty.")

    # If this is a gzip file (.gz), decompress first
    if isinstance(raw, (bytes, bytearray)) and len(raw) >= 2 and raw[0] == 0x1F and raw[1] == 0x8B:
        raw = gzip.decompress(raw)

    # Quick guardrail: if this looks like HTML, it's probably a bad download (404/rate limit)
    head = raw[:200].decode("utf-8", errors="replace") if isinstance(raw, (bytes, bytearray)) else str(raw)[:200]
    if "<html" in head.lower() or "<!doctype html" in head.lower():
        raise ValueError(
            "Tracking download returned HTML instead of JSONL. "
            "This usually means the GitHub URL is wrong (404) or rate-limited."
        )

    try:
        return pd.read_json(io.BytesIO(raw), lines=True)
    except ValueError as e:
        preview = head.replace("\n", "\\n")
        raise ValueError(f"Failed to parse tracking JSONL. First bytes: {preview}") from e



def _read_events(match: MatchFiles) -> pd.DataFrame:
    mid = match.match_id
    name = f"{mid}_dynamic_events.csv"
    raw = match.get(name)
    return pd.read_csv(io.BytesIO(raw))


def _read_match_json(match: MatchFiles) -> Dict[str, Any]:
    mid = match.match_id
    name = f"{mid}_match.json"
    raw = match.get(name)
    return json.loads(raw.decode("utf-8"))


def _build_players_and_ball(df_tracking: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    require_columns(df_tracking, ["frame", "timestamp", "period", "player_data", "ball_data"], name="tracking")

    valid = df_tracking[df_tracking["player_data"].apply(lambda x: isinstance(x, list) and len(x) > 0)].copy()
    require_nonempty(valid, name="tracking(valid frames)")

    # players
    base = valid[["frame", "timestamp", "period", "player_data"]].explode("player_data", ignore_index=True)
    p = pd.json_normalize(base["player_data"])
    players = pd.concat([base.drop(columns=["player_data"]).reset_index(drop=True), p.reset_index(drop=True)], axis=1)

    # ball
    b = pd.json_normalize(valid["ball_data"])
    ball = pd.concat([valid[["frame", "timestamp", "period"]].reset_index(drop=True), b.reset_index(drop=True)], axis=1)

    players = _finalize_players(players)
    ball = _finalize_ball(ball)
    return players, ball


def _finalize_players(df: pd.DataFrame) -> pd.DataFrame:
    """
    Match the processed contract your Streamlit app expects:
      frame,timestamp,period,x,y,player_id,is_detected,team_id
    """
    keep = ["frame", "timestamp", "period", "x", "y", "player_id", "is_detected", "team_id"]
    for c in keep:
        if c not in df.columns:
            df[c] = pd.NA

    out = df[keep].copy()

    for c in ["frame", "player_id", "team_id"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    for c in ["x", "y"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")

    out = out.dropna(subset=["frame", "player_id", "team_id", "x", "y"]).copy()
    out["frame"] = out["frame"].astype(int)
    out["player_id"] = out["player_id"].astype(int)
    out["team_id"] = out["team_id"].astype(int)
    out["x"] = out["x"].astype(float)
    out["y"] = out["y"].astype(float)

    return out


def _finalize_ball(df: pd.DataFrame) -> pd.DataFrame:
    """
    Match the processed contract your Streamlit app expects:
      frame,timestamp,period,x,y,z,is_detected
    """
    keep = ["frame", "timestamp", "period", "x", "y", "z", "is_detected"]
    for c in keep:
        if c not in df.columns:
            df[c] = pd.NA

    out = df[keep].copy()

    out["frame"] = pd.to_numeric(out["frame"], errors="coerce")
    out = out.dropna(subset=["frame"]).copy()
    out["frame"] = out["frame"].astype(int)

    for c in ["x", "y", "z"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")

    return out


def _build_events_poss(df_events: pd.DataFrame) -> pd.DataFrame:
    """
    Filter to possession events, but KEEP ALL COLUMNS (your app / future work may rely on them).
    """
    require_columns(df_events, ["event_type", "frame_start", "player_id", "team_id"], name="events")

    poss = df_events.loc[
        (df_events["event_type"] == "player_possession")
        & df_events["frame_start"].notna()
        & df_events["player_id"].notna()
        & df_events["team_id"].notna()
    ].copy()

    for c in ["frame_start", "player_id", "team_id"]:
        poss[c] = pd.to_numeric(poss[c], errors="coerce")

    poss = poss.dropna(subset=["frame_start", "player_id", "team_id"]).copy()
    poss["frame_start"] = poss["frame_start"].astype(int)
    poss["player_id"] = poss["player_id"].astype(int)
    poss["team_id"] = poss["team_id"].astype(int)

    require_nonempty(poss, name="events_poss")
    return poss


def _build_meta(match_id: str, mj: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build meta.json fields used by your app for:
      - pitch size
      - match label (home/away names)
      - team id mapping
      - kit colors (primary)
    """
    home = mj.get("home_team", {}) or {}
    away = mj.get("away_team", {}) or {}
    home_kit = mj.get("home_team_kit", {}) or {}
    away_kit = mj.get("away_team_kit", {}) or {}

    meta: Dict[str, Any] = {
        "match_id": match_id,
        "pitch_length": mj.get("pitch_length"),
        "pitch_width": mj.get("pitch_width"),
        "home_team_side": mj.get("home_team_side"),
        "home_team_id": home.get("id"),
        "away_team_id": away.get("id"),
        "home_team_name": home.get("name"),
        "away_team_name": away.get("name"),
        "home_colors": {"primary": home_kit.get("jersey_color")},
        "away_colors": {"primary": away_kit.get("jersey_color")},
    }
    return meta


def process_match(match: MatchFiles) -> ProcessedMatch:
    df_tracking = _read_tracking(match)
    df_events = _read_events(match)
    mj = _read_match_json(match)

    players, ball = _build_players_and_ball(df_tracking)
    events_poss = _build_events_poss(df_events)
    meta = _build_meta(match.match_id, mj)

    return ProcessedMatch(
        match_id=match.match_id,
        players=players,
        ball=ball,
        events_poss=events_poss,
        meta=meta,
    )


def save_processed(pm: ProcessedMatch, out_root: Path) -> Path:
    out_dir = Path(out_root) / pm.match_id
    out_dir.mkdir(parents=True, exist_ok=True)

    pm.players.to_csv(out_dir / "players.csv.gz", index=False, compression="gzip")
    pm.ball.to_csv(out_dir / "ball.csv.gz", index=False, compression="gzip")
    pm.events_poss.to_csv(out_dir / "events_poss.csv.gz", index=False, compression="gzip")
    (out_dir / "meta.json").write_text(json.dumps(pm.meta, indent=2))

    return out_dir
