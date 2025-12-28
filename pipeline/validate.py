from __future__ import annotations

from collections.abc import Iterable

import pandas as pd


def require_columns(df: pd.DataFrame, cols: Iterable[str], *, name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{name}: missing required columns: {missing}")


def require_nonempty(df: pd.DataFrame, *, name: str) -> None:
    if df.empty:
        raise ValueError(f"{name}: produced an empty dataframe (unexpected)")


def coerce_int_series(s: pd.Series, *, name: str) -> pd.Series:
    out = pd.to_numeric(s, errors="coerce")
    if out.isna().all():
        raise ValueError(f"{name}: could not coerce any values to int")
    return out
