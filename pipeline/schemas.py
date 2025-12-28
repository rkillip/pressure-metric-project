from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import pandas as pd


@dataclass(frozen=True)
class ProcessedMatch:
    match_id: str
    players: pd.DataFrame
    ball: pd.DataFrame
    events_poss: pd.DataFrame
    meta: Dict[str, Any]
