from __future__ import annotations

import io
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Union

BytesLike = Union[bytes, bytearray, memoryview]


@dataclass
class MatchFiles:
    """
    In-memory representation of the raw files for a single match.

    files maps basename -> bytes, e.g.
      "1886347_tracking_extrapolated.jsonl" -> b"..."
    """

    match_id: str
    files: Dict[str, bytes]

    def has(self, name: str) -> bool:
        return name in self.files

    def get(self, name: str) -> bytes:
        try:
            return self.files[name]
        except KeyError as e:
            raise FileNotFoundError(f"Missing required file: {name}") from e


def load_match_from_folder(match_id: str, folder: Union[str, Path]) -> MatchFiles:
    folder = Path(folder)
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder}")

    files: Dict[str, bytes] = {}
    for p in folder.glob(f"{match_id}_*"):
        if p.is_file():
            files[p.name] = p.read_bytes()

    return MatchFiles(match_id=match_id, files=files)


def load_match_from_zip(match_id: str, zip_bytes: BytesLike, *, root_prefix: Optional[str] = None) -> MatchFiles:
    """
    Load match files from a zip.

    - match_id: expected file prefix, e.g. "1886347"
    - root_prefix: optional path prefix inside zip (e.g. "raw/1886347/")

    We match basenames like '{match_id}_tracking_extrapolated.jsonl' regardless of zip nesting.
    """
    files: Dict[str, bytes] = {}

    with zipfile.ZipFile(io.BytesIO(zip_bytes), "r") as z:
        for info in z.infolist():
            if info.is_dir():
                continue

            path = info.filename
            if root_prefix and not path.startswith(root_prefix):
                continue

            name = Path(path).name
            if name.startswith(f"{match_id}_"):
                files[name] = z.read(info.filename)

    return MatchFiles(match_id=match_id, files=files)
