from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from mplsoccer import Pitch


@dataclass(frozen=True)
class ModelConfig:
    fps: float = 10.0

    r_core: float = 6.0
    r_max: float = 8.0
    v_gate: float = 1.5

    v0: float = 2.0
    tau: float = 0.8
    alpha: float = 0.35
    k: float = 0.7

    outer_tail: float = 0.35
    ttc_scale: float = 1.0
    beta: float = 0.7


CFG = ModelConfig()

# -----------------------------------------------------------------------------
# Page config + header (UPDATED)
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Football Pressure Metric", layout="wide")
st.title("Football Pressure Metric")
st.caption("Pick an event, scrub frames, and inspect the defensive influence shaping decisions.")

try:
    pd.set_option("mode.dtype_backend", "numpy")
except Exception:
    pass


# -----------------------------------------------------------------------------
# Paths
#
# Repo stores processed artifacts under:
#   data/<match_id>/players.csv.gz
#   data/<match_id>/ball.csv.gz
#   data/<match_id>/events_poss.csv.gz
#   data/<match_id>/meta.json
# -----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data"

_REQUIRED_FILES = ("players.csv.gz", "ball.csv.gz", "events_poss.csv.gz", "meta.json")


def _safe_int(x) -> Optional[int]:
    try:
        return int(x)
    except Exception:
        return None


def _pick_color(value: Optional[str], default: str) -> str:
    if value is None:
        return default
    s = str(value).strip()
    return s if s else default


def _w_v(v_close: np.ndarray | float, v0: float) -> np.ndarray | float:
    return np.minimum(1.0, np.maximum(0.0, v_close) / max(v0, 1e-6))


def _ball_player_dist(frame: int, players: pd.DataFrame, ball: pd.DataFrame, pid: int) -> Optional[float]:
    if ball.empty or ("x" not in ball.columns) or ("y" not in ball.columns):
        return None
    b = ball[ball["frame"] == frame].dropna(subset=["x", "y"])
    if b.empty:
        return None
    p = players[(players["frame"] == frame) & (players["player_id"] == pid)]
    if p.empty:
        return None
    bx, by = float(b.iloc[0]["x"]), float(b.iloc[0]["y"])
    px, py = float(p.iloc[0]["x"]), float(p.iloc[0]["y"])
    return float(np.hypot(bx - px, by - py))


def _ball_state_label(dist_m: Optional[float]) -> str:
    if dist_m is None or not np.isfinite(dist_m):
        return "Unknown"
    if dist_m <= 1.5:
        return "On feet"
    if dist_m <= 4.0:
        return "Receiving"
    return "Ball away"


def _nearest_defender(frame: int, pid: int, tid: int, players: pd.DataFrame) -> tuple[Optional[int], float]:
    snap = players[players["frame"] == frame]
    me = snap[snap["player_id"] == pid]
    if snap.empty or me.empty:
        return None, np.nan

    x0, y0 = float(me.iloc[0]["x"]), float(me.iloc[0]["y"])
    opp = snap[snap["team_id"] != tid]
    if opp.empty:
        return None, np.nan

    dx = opp["x"].to_numpy(float) - x0
    dy = opp["y"].to_numpy(float) - y0
    d = np.sqrt(dx * dx + dy * dy)
    j = int(np.argmin(d))
    return int(opp.iloc[j]["player_id"]), float(d[j])


def _lane_denial(
    passer_xy: np.ndarray,
    ball_xy: np.ndarray,
    opp_xy: np.ndarray,
    opp_d_to_passer: np.ndarray,
    lane_width_m: float = 1.5,
    passer_radius_m: float = 6.0,
) -> float:
    lane_vec = ball_xy - passer_xy
    lane_len = float(np.linalg.norm(lane_vec))
    if lane_len < 1e-6:
        return 0.0

    lane_hat = lane_vec / lane_len
    rel = opp_xy - passer_xy[None, :]
    proj = rel @ lane_hat
    proj = np.clip(proj, 0.0, lane_len)

    closest = passer_xy[None, :] + proj[:, None] * lane_hat[None, :]
    d_lane = np.linalg.norm(opp_xy - closest, axis=1)

    mask = (d_lane <= lane_width_m) & (opp_d_to_passer <= passer_radius_m)
    if not np.any(mask):
        return 0.0

    contrib = (1.0 - (d_lane[mask] / lane_width_m)).clip(0.0, 1.0)
    return float(contrib.sum())


def _is_match_dir(p: Path) -> bool:
    if not p.is_dir():
        return False
    for f in _REQUIRED_FILES:
        if not (p / f).exists():
            return False
    return True


@st.cache_data(show_spinner=False)
def list_matches() -> list[str]:
    if not PROCESSED_DIR.exists():
        return []

    out: list[str] = []
    for p in PROCESSED_DIR.iterdir():
        if _is_match_dir(p):
            out.append(p.name)

    return sorted(out)


@st.cache_data(show_spinner=False)
def load_meta(match_id: str) -> dict:
    base = PROCESSED_DIR / match_id
    path = base / "meta.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


@st.cache_data(show_spinner=False)
def match_label(match_id: str) -> str:
    return match_id


@st.cache_data(show_spinner=False)
def load_match(match_id: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    base = PROCESSED_DIR / match_id

    players = pd.read_csv(base / "players.csv.gz")
    ball = pd.read_csv(base / "ball.csv.gz")
    events = pd.read_csv(base / "events_poss.csv.gz")
    meta = load_meta(match_id)

    # players clean
    for c in ["frame", "player_id", "team_id"]:
        players[c] = pd.to_numeric(players[c], errors="coerce")
    for c in ["x", "y"]:
        players[c] = pd.to_numeric(players[c], errors="coerce")
    players = players.dropna(subset=["frame", "player_id", "team_id", "x", "y"]).copy()
    players["frame"] = players["frame"].astype(int)
    players["player_id"] = players["player_id"].astype(int)
    players["team_id"] = players["team_id"].astype(int)
    players["x"] = players["x"].astype(float)
    players["y"] = players["y"].astype(float)

    # ball clean
    if "frame" in ball.columns:
        ball["frame"] = pd.to_numeric(ball["frame"], errors="coerce")
        ball = ball.dropna(subset=["frame"]).copy()
        ball["frame"] = ball["frame"].astype(int)

    # events clean
    for c in ["frame_start", "player_id", "team_id"]:
        if c in events.columns:
            events[c] = pd.to_numeric(events[c], errors="coerce")
    events = events.dropna(subset=["frame_start", "player_id", "team_id"]).copy()
    events["frame_start"] = events["frame_start"].astype(int)
    events["player_id"] = events["player_id"].astype(int)
    events["team_id"] = events["team_id"].astype(int)

    return players, ball, events, meta


def quick_clip_quality(players: pd.DataFrame, ball: pd.DataFrame, frame0: int, W: int) -> dict:
    frames = list(range(frame0 - W, frame0 + W + 1))
    clip_players = players[players["frame"].isin(frames)]
    frames_sorted = sorted(clip_players["frame"].unique().tolist())

    expected = 2 * W + 1
    frame_coverage = len(frames_sorted) / max(expected, 1)

    players_per_frame = clip_players.groupby("frame")["player_id"].nunique()
    median_players = float(players_per_frame.median()) if len(players_per_frame) else 0.0

    ball_cov = 0.0
    if "frame" in ball.columns and ("x" in ball.columns) and ("y" in ball.columns):
        clip_ball = ball[ball["frame"].isin(frames)].dropna(subset=["x", "y"])
        ball_cov = (clip_ball["frame"].nunique() / len(frames_sorted)) if len(frames_sorted) else 0.0

    return {
        "frame_coverage": float(frame_coverage),
        "median_players": float(median_players),
        "ball_coverage": float(ball_cov),
        "is_good": (frame_coverage >= 0.70) and (median_players >= 18) and (ball_cov >= 0.50),
    }


def build_clip(
    players: pd.DataFrame,
    ball: pd.DataFrame,
    ev: pd.Series,
    W: int,
    *,
    cfg: ModelConfig,
    k_: float,
    tau_: float,
    v0_: float,
    alpha_: float,
    ttc_scale_: float,
    beta_: float,
):
    frame0 = int(ev["frame_start"])
    pid = int(ev["player_id"])
    tid = int(ev["team_id"])

    frames = list(range(frame0 - W, frame0 + W + 1))
    clip_players = players[players["frame"].isin(frames)].copy()
    clip_ball = ball[ball["frame"].isin(frames)].copy() if "frame" in ball.columns else ball.iloc[0:0].copy()

    frames_sorted = sorted(clip_players["frame"].unique().tolist())
    if not frames_sorted:
        return [], clip_players, clip_ball, {}, pd.DataFrame(), {}

    snap_by_f = {f: clip_players[clip_players["frame"] == f] for f in frames_sorted}

    # ball lookup (centered coords)
    ball_xy_by_f: dict[int, np.ndarray | None] = {}
    if not clip_ball.empty and ("x" in clip_ball.columns) and ("y" in clip_ball.columns):
        b = clip_ball.dropna(subset=["x", "y"])
        for f in frames_sorted:
            bf = b[b["frame"] == f]
            if bf.empty:
                ball_xy_by_f[f] = None
            else:
                ball_xy_by_f[f] = np.array([float(bf.iloc[0]["x"]), float(bf.iloc[0]["y"])], dtype=float)
    else:
        for f in frames_sorted:
            ball_xy_by_f[f] = None

    # passer position at release
    passer_xy_release = None
    snap0 = snap_by_f.get(frame0)
    if snap0 is not None:
        me0 = snap0[snap0["player_id"] == pid]
        if not me0.empty:
            passer_xy_release = np.array([float(me0.iloc[0]["x"]), float(me0.iloc[0]["y"])], dtype=float)

    rows: list[dict] = []
    nearest_map: dict[int, tuple[Optional[int], float]] = {}

    for f in frames_sorted:
        nid, nd = _nearest_defender(f, pid, tid, clip_players)
        nearest_map[f] = (nid, nd)

        fprev = max(frames_sorted[0], f - 5)
        _, nd_prev = _nearest_defender(fprev, pid, tid, clip_players)
        dt = max((f - fprev) / cfg.fps, 1e-6)
        closing_nd = (nd_prev - nd) / dt if np.isfinite(nd_prev) and np.isfinite(nd) else 0.0

        rows.append({"frame": f, "nearest_dist": nd, "closing_mps": closing_nd})

    met = pd.DataFrame(rows).set_index("frame")

    P_release = None

    for f in frames_sorted:
        snap = snap_by_f[f]
        me = snap[snap["player_id"] == pid]
        if me.empty:
            met.loc[f, "P_raw"] = 0.0
            continue

        me_xy = np.array([float(me.iloc[0]["x"]), float(me.iloc[0]["y"])], dtype=float)

        opp = snap[snap["team_id"] != tid]
        if opp.empty:
            met.loc[f, "P_raw"] = 0.0
            continue

        opp_xy = opp[["x", "y"]].to_numpy(float)
        rel = me_xy[None, :] - opp_xy
        d = np.linalg.norm(rel, axis=1)

        d_ok = (d <= cfg.r_max) & (d > 1e-6)
        if not np.any(d_ok):
            P_raw = 0.0
        else:
            opp_ids = opp["player_id"].to_numpy(int)

            fprev = max(frames_sorted[0], f - 5)
            snap_prev = snap_by_f[fprev]
            dt = max((f - fprev) / cfg.fps, 1e-6)
            prev_lookup = snap_prev.set_index("player_id")[["x", "y"]] if not snap_prev.empty else pd.DataFrame()

            v = np.zeros_like(opp_xy)
            for idx, oid in enumerate(opp_ids):
                if (not prev_lookup.empty) and (oid in prev_lookup.index):
                    x_prev = float(prev_lookup.loc[oid, "x"])
                    y_prev = float(prev_lookup.loc[oid, "y"])
                    v[idx, 0] = (opp_xy[idx, 0] - x_prev) / dt
                    v[idx, 1] = (opp_xy[idx, 1] - y_prev) / dt

            r_hat = rel / d[:, None]
            v_close = np.maximum(0.0, np.sum(v * r_hat, axis=1))

            ttc = d / np.maximum(v_close, 0.1)
            w_ttc = np.exp(-ttc / max(ttc_scale_, 1e-6))
            w_intent = np.maximum(_w_v(v_close, v0_), w_ttc)

            outer_band = (d > cfg.r_core) & d_ok
            mask = d_ok & ((d <= cfg.r_core) | (outer_band & (v_close >= cfg.v_gate)))

            if not np.any(mask):
                P_raw = 0.0
            else:
                w_dist = np.where(
                    d <= cfg.r_core,
                    np.maximum(0.0, 1.0 - (d / max(cfg.r_core, 1e-6))),
                    np.maximum(0.0, 1.0 - ((d - cfg.r_core) / (cfg.r_max - cfg.r_core))) * cfg.outer_tail,
                )

                inner = d <= cfg.r_core
                blend = np.where(inner, (beta_ + (1.0 - beta_) * w_intent), w_intent)
                contrib = w_dist * blend
                P_raw = float(contrib[mask].sum())

        met.loc[f, "P_raw"] = P_raw
        if f == frame0:
            P_release = P_raw

    if P_release is None:
        if frame0 in met.index:
            P_release = float(met.loc[frame0, "P_raw"])
        else:
            nearest_idx = int(met.index[np.argmin(np.abs(met.index.to_numpy() - frame0))])
            P_release = float(met.loc[nearest_idx, "P_raw"])

    for f in frames_sorted:
        if f <= frame0:
            P_eff = float(met.loc[f, "P_raw"])
        else:
            dt = (f - frame0) / cfg.fps
            P_decay = float(P_release) * float(np.exp(-dt / max(tau_, 1e-6)))

            P_lane = 0.0
            ball_xy = ball_xy_by_f.get(f)
            if (passer_xy_release is not None) and (ball_xy is not None):
                snap = snap_by_f[f]
                opp = snap[snap["team_id"] != tid]
                if not opp.empty:
                    opp_xy = opp[["x", "y"]].to_numpy(float)
                    opp_d_to_passer = np.linalg.norm(opp_xy - passer_xy_release[None, :], axis=1)
                    P_lane = _lane_denial(passer_xy_release, ball_xy, opp_xy, opp_d_to_passer)

            P_eff = P_decay + alpha_ * P_lane

        met.loc[f, "P_eff"] = P_eff
        met.loc[f, "pressure_live_0_100"] = 100.0 * (1.0 - float(np.exp(-k_ * max(0.0, float(met.loc[f, "P_raw"])))))
        met.loc[f, "pressure_decision_0_100"] = 100.0 * (1.0 - float(np.exp(-k_ * max(0.0, P_eff))))

    quality = quick_clip_quality(players, ball, frame0, W)
    return frames_sorted, clip_players, clip_ball, nearest_map, met, quality


def plot_timeline(met: pd.DataFrame, frame0: int, frame_cur: int):
    xs = met.index.to_numpy(int)
    live = met["pressure_live_0_100"].to_numpy(float)
    deci = met["pressure_decision_0_100"].to_numpy(float)

    fig, ax = plt.subplots(figsize=(10, 2.8))
    ax.plot(xs, live, label="Live")
    ax.plot(xs, deci, label="Decision")
    ax.axvline(frame0, linestyle="--")
    ax.axvline(frame_cur, linestyle=":")
    ax.set_ylim(0, 100)
    ax.set_ylabel("0–100")
    ax.set_xlabel("Frame")
    ax.legend(loc="upper right", fontsize=9)
    ax.set_title("Pressure over clip")
    fig.tight_layout()
    return fig


def plot_snapshot(
    frame: int,
    players: pd.DataFrame,
    ball: pd.DataFrame,
    ev: pd.Series,
    nearest_map: dict[int, tuple[Optional[int], float]],
    meta: dict,
    *,
    flip_x: bool,
    show_ball_trail: bool,
    ball_trail_len: int,
):
    pid = int(ev["player_id"])
    tid = int(ev["team_id"])

    snap = players[players["frame"] == frame].copy()
    me = snap[snap["player_id"] == pid]
    team = snap[snap["team_id"] == tid]
    opp = snap[snap["team_id"] != tid]
    nid, nd = nearest_map.get(frame, (None, np.nan))

    L = float(meta.get("pitch_length", 105.0))
    Wp = float(meta.get("pitch_width", 68.0))

    def to_pitch_xy(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        x = df["x"].to_numpy(float) + (L / 2.0)
        y = df["y"].to_numpy(float) + (Wp / 2.0)
        if flip_x:
            x = L - x
            y = Wp - y
        return x, y

    home_c = _pick_color(meta.get("home_colors", {}).get("primary"), "#1f77b4")
    away_c = _pick_color(meta.get("away_colors", {}).get("primary"), "#d62728")
    gk_c = "#FFD700"

    home_team_id = _safe_int(meta.get("home_team_id"))
    team_color = home_c if (home_team_id is not None and home_team_id == tid) else away_c
    opp_color = away_c if team_color == home_c else home_c

    selected_pos = str(ev.get("player_position", "")).upper()
    selected_is_gk = "GK" in selected_pos

    pitch = Pitch(pitch_type="custom", pitch_length=L, pitch_width=Wp, line_zorder=2)
    fig, ax = pitch.draw(figsize=(10, 6.5))

    x_opp, y_opp = to_pitch_xy(opp)
    x_team, y_team = to_pitch_xy(team)
    ax.scatter(x_opp, y_opp, s=55, c=opp_color, alpha=0.55, zorder=3)
    ax.scatter(x_team, y_team, s=55, c=team_color, alpha=0.55, zorder=3)

    if not me.empty:
        x_me, y_me = to_pitch_xy(me)
        ax.scatter(x_me, y_me, s=240, c="black", zorder=6)
        ax.scatter(x_me, y_me, s=170, c="white", zorder=7)
        ax.scatter(x_me, y_me, s=120, c=(gk_c if selected_is_gk else "red"), zorder=8)

    if nid is not None and not me.empty:
        nd_row = snap[snap["player_id"] == nid]
        if not nd_row.empty:
            x_nd, y_nd = to_pitch_xy(nd_row)
            x_me, y_me = to_pitch_xy(me)
            ax.scatter(x_nd, y_nd, s=210, c="black", zorder=9)
            ax.plot(
                [float(x_me[0]), float(x_nd[0])],
                [float(y_me[0]), float(y_nd[0])],
                linestyle="--",
                linewidth=2,
                c="black",
                zorder=5,
            )
            ax.text(
                float(x_nd[0]),
                float(y_nd[0]),
                "ND",
                fontsize=10,
                ha="center",
                va="center",
                color="white",
                zorder=10,
                bbox=dict(boxstyle="circle,pad=0.2", fc="black", ec="none"),
            )

    if not ball.empty and "x" in ball.columns and "y" in ball.columns:
        b = ball.dropna(subset=["x", "y"])
        bf = b[b["frame"] == frame]
        if not bf.empty:
            bx0 = float(bf.iloc[0]["x"]) + (L / 2.0)
            by0 = float(bf.iloc[0]["y"]) + (Wp / 2.0)
            if flip_x:
                bx0 = L - bx0
                by0 = Wp - by0
            ax.scatter(bx0, by0, s=70, c="orange", zorder=11)

            if show_ball_trail and ball_trail_len > 0:
                fmin_local = frame - int(ball_trail_len)
                trail = b[(b["frame"] >= fmin_local) & (b["frame"] <= frame)].sort_values("frame")
                if not trail.empty:
                    tx = trail["x"].to_numpy(float) + (L / 2.0)
                    ty = trail["y"].to_numpy(float) + (Wp / 2.0)
                    if flip_x:
                        tx = L - tx
                        ty = Wp - ty
                    ax.plot(tx, ty, linestyle="-", linewidth=2, c="orange", alpha=0.45, zorder=10)

    # UPDATED: remove the plot title (info already shown above the pitch)
    return fig


# -----------------------------------------------------------------------------
# UI
# -----------------------------------------------------------------------------
matches = list_matches()
if not matches:
    st.error(
        "No processed matches found.\n\n"
        "Expected folders like: data/<match_id>/\n"
        "containing: players.csv.gz, ball.csv.gz, events_poss.csv.gz, meta.json"
    )
    st.stop()

with st.sidebar:
    st.subheader("Review")
    match_id = st.selectbox("Match", matches, format_func=match_label)

    # UPDATED label
    only_good = st.checkbox("Only high quality clips", value=True)

    show_details = st.checkbox("Show details", value=False)

    # UPDATED: clip window default to 25, and only show when show_details is enabled.
    W = 25
    if show_details:
        W = st.slider("Clip window (± frames)", 10, 40, 25, 1)

players_df, ball_df, events_df, meta = load_match(match_id)

# Home/Away IDs: prefer meta if present, otherwise infer from events.
home_id_meta = _safe_int(meta.get("home_team_id", None))
away_id_meta = _safe_int(meta.get("away_team_id", None))
event_team_ids = sorted(events_df["team_id"].dropna().astype(int).unique().tolist())

home_id = home_id_meta if (home_id_meta is not None and home_id_meta in event_team_ids) else None
away_id = away_id_meta if (away_id_meta is not None and away_id_meta in event_team_ids) else None
if (home_id is None or away_id is None) and len(event_team_ids) >= 2:
    home_id = event_team_ids[0] if home_id is None else home_id
    away_id = event_team_ids[1] if away_id is None else away_id

with st.sidebar:
    team_choice = st.radio("Team", ["Home", "Away"], horizontal=True)

team_id_selected = home_id if team_choice == "Home" else away_id

cols = ["player_id", "team_id"]
if "player_name" in events_df.columns:
    cols.append("player_name")
if "player_position" in events_df.columns:
    cols.append("player_position")

player_cards = events_df[cols].drop_duplicates()

tmp = events_df.groupby(["team_id", "player_id"]).size().rename("n").reset_index()
top = tmp.sort_values(["team_id", "n"], ascending=[True, False]).groupby("team_id").head(11)
starter_ids = set(top["player_id"].astype(int).tolist())
player_cards["starter_flag"] = player_cards["player_id"].astype(int).isin(starter_ids)

active_roster = (
    player_cards[player_cards["team_id"].astype(int) == int(team_id_selected)]
    if team_id_selected is not None
    else player_cards.iloc[0:0].copy()
)

if active_roster.empty:
    st.error("No players found for this team in events.")
    st.stop()

sort_cols = ["starter_flag", "player_name"] if "player_name" in active_roster.columns else ["starter_flag", "player_id"]
active_roster = active_roster.sort_values(sort_cols, ascending=[False, True])


def fmt_player(pid: int) -> str:
    row = active_roster[active_roster["player_id"].astype(int) == int(pid)].iloc[0]
    name = row["player_name"] if "player_name" in active_roster.columns else f"Player {pid}"
    pos = row["player_position"] if "player_position" in active_roster.columns else ""
    star = "★ " if bool(row.get("starter_flag", False)) else ""
    return f"{star}{name} — {pos}".strip(" —")


with st.sidebar:
    player_id = st.selectbox(
        "Player",
        active_roster["player_id"].astype(int).tolist(),
        format_func=fmt_player,
    )

sub = (
    events_df[
        (events_df["player_id"].astype(int) == int(player_id))
        & (events_df["team_id"].astype(int) == int(team_id_selected))
    ]
    .copy()
    .sort_values("frame_start")
)

if only_good:
    keep = []
    for _, row in sub.iterrows():
        q = quick_clip_quality(players_df, ball_df, int(row["frame_start"]), W=W)
        if q.get("is_good", False):
            keep.append(row)
    sub = pd.DataFrame(keep) if keep else sub.iloc[0:0].copy()

if sub.empty:
    st.warning("No clips found for this player with current settings.")
    st.stop()


def _event_label(r: pd.Series) -> str:
    et = str(r.get("event_type", "")).replace("_", " ").title()
    mn = r.get("minute_start", None)
    sc = r.get("second_start", None)
    t = ""
    if pd.notna(mn) and pd.notna(sc):
        t = f"{int(mn):02d}:{int(sc):02d} "
    return f"{t}{et} (frame {int(r['frame_start'])})".strip()


sub["label"] = sub.apply(_event_label, axis=1)

with st.sidebar:
    event_key = "event_id" if "event_id" in sub.columns else None
    if event_key:
        event_options = sub[[event_key, "label"]].drop_duplicates()
        event_id = st.selectbox(
            "Event",
            event_options[event_key].tolist(),
            format_func=lambda eid: event_options.loc[event_options[event_key] == eid, "label"].iloc[0],
        )
        ev = sub.loc[sub[event_key] == event_id].iloc[0]
    else:
        idx = st.selectbox("Event", sub.index.tolist(), format_func=lambda i: sub.loc[i, "label"])
        ev = sub.loc[idx]

k_ui = float(CFG.k)
tau_ui = float(CFG.tau)
show_ball_trail = True
ball_trail_len = 10
flip_x = False

if show_details:
    with st.sidebar:
        with st.expander("Advanced", expanded=False):
            k_ui = st.slider("Sensitivity", 0.3, 3.0, float(CFG.k), 0.05)
            tau_ui = st.slider("Decision decay (s)", 0.3, 3.0, float(CFG.tau), 0.05)
            show_ball_trail = st.checkbox("Ball trail", value=True)
            ball_trail_len = st.slider("Trail length", 0, 30, 10, 1)
            flip_x = st.checkbox("Flip pitch", value=False)

frames_sorted, clip_players, clip_ball, nearest_map, met, quality = build_clip(
    players_df,
    ball_df,
    ev,
    W=W,
    cfg=CFG,
    k_=float(k_ui),
    tau_=float(tau_ui),
    v0_=float(CFG.v0),
    alpha_=float(CFG.alpha),
    ttc_scale_=float(CFG.ttc_scale),
    beta_=float(CFG.beta),
)

if not frames_sorted or met.empty:
    st.warning("No tracking frames in this window.")
    st.stop()

frame0 = int(ev["frame_start"])
fmin, fmax = int(min(frames_sorted)), int(max(frames_sorted))

qc = quality
if show_details or (not qc.get("is_good", False)):
    badge = "✅" if qc["is_good"] else "⚠️"
    st.caption(
        f"{badge} clip quality — frames: {qc['frame_coverage']:.0%} | "
        f"players/frame (median): {qc['median_players']:.0f} | "
        f"ball coverage: {qc['ball_coverage']:.0%}"
    )

st.caption("Pressure is scaled 0–100. Higher means more immediate defensive influence on the ball-carrier’s decision.")

# UPDATED: move frame slider to the sidebar (so pitch + metrics stay visible)
with st.sidebar:
    st.subheader("Frame")
    frame = st.slider(
        "Frame",
        min_value=fmin,
        max_value=fmax,
        value=frame0 if fmin <= frame0 <= fmax else fmin,
        step=1,
        label_visibility="collapsed",
    )

r = met.loc[frame] if frame in met.index else met.iloc[0]
pid = int(ev["player_id"])

ball_d = _ball_player_dist(frame, clip_players, clip_ball, pid)
ball_state = _ball_state_label(ball_d)

live_val = float(r.get("pressure_live_0_100", np.nan))
dec_val = float(r.get("pressure_decision_0_100", np.nan))
nearest_d = float(r.get("nearest_dist", np.nan))
closing = float(r.get("closing_mps", np.nan))

dpi = live_val

if dpi >= 80:
    label = "High"
    color = "#8B0000"
elif dpi >= 60:
    label = "Med"
    color = "#B22222"
elif dpi >= 40:
    label = "Moderate"
    color = "#AA5500"
else:
    label = "Low"
    color = "#222222"

context_bits = []
if np.isfinite(nearest_d):
    context_bits.append(f"Nearest defender: {nearest_d:.1f}m")
if np.isfinite(closing):
    context_bits.append(f"closing {closing:.1f} m/s")
if ball_d is not None and np.isfinite(ball_d):
    context_bits.append(f"Ball: {ball_state.lower()} ({ball_d:.1f}m)")
else:
    context_bits.append(f"Ball: {ball_state.lower()}")
context_line = " — ".join(context_bits)

c_main, c_a, c_b, c_c = st.columns([2.2, 1, 1, 1])

with c_main:
    st.markdown(
        f"""
        <div style="padding: 14px; border-radius: 14px; border: 1px solid #eee;">
          <div style="font-size: 13px; color: #666;">Live pressure</div>
          <div style="font-size: 40px; font-weight: 900; color: {color}; line-height: 1;">
            {dpi:.1f}
          </div>
          <div style="font-size: 12px; font-weight: 800; color: {color}; margin-top: 6px;">
            {label} pressure
          </div>
          <div style="font-size: 12px; color: #666; margin-top: 8px;">
            {context_line}
          </div>
          <div style="font-size: 12px; color: #666; margin-top: 6px;">
            Decision pressure (decayed): {dec_val:.1f}
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

c_a.metric("Nearest defender", f"{nearest_d:.2f} m")
c_b.metric("Closing speed", f"{closing:.2f} m/s")
c_c.metric("Event frame", f"{frame0}")

fig = plot_snapshot(
    frame,
    clip_players,
    clip_ball,
    ev,
    nearest_map,
    meta,
    flip_x=bool(flip_x),
    show_ball_trail=bool(show_ball_trail),
    ball_trail_len=int(ball_trail_len),
)
st.pyplot(fig, use_container_width=True)

if show_details:
    st.subheader("Timeline")
    st.pyplot(plot_timeline(met, frame0=frame0, frame_cur=frame), use_container_width=True)

    with st.expander("Debug", expanded=False):
        st.write(
            {
                "frame": int(frame),
                "P_raw": float(r.get("P_raw", np.nan)),
                "P_eff": float(r.get("P_eff", np.nan)),
                "live_0_100": float(r.get("pressure_live_0_100", np.nan)),
                "decision_0_100": float(r.get("pressure_decision_0_100", np.nan)),
                "paths": {
                    "PROJECT_ROOT": str(PROJECT_ROOT),
                    "PROCESSED_DIR": str(PROCESSED_DIR),
                    "match_dir": str(PROCESSED_DIR / match_id),
                },
            }
        )
        st.json(meta)
