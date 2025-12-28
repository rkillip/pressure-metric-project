from __future__ import annotations

import argparse
from pathlib import Path

from pipeline.io import load_match_from_folder, load_match_from_zip
from pipeline.process import process_match, save_processed


def main() -> None:
    ap = argparse.ArgumentParser(description="Smoke test: raw -> processed pipeline.")
    ap.add_argument("--match-id", required=True)
    ap.add_argument("--raw-folder", default=None, help="Folder containing raw match files (preferred for local runs).")
    ap.add_argument("--raw-zip", default=None, help="Path to a raw match zip file.")
    ap.add_argument("--out", default="data/processed", help="Output root for processed match folder.")
    args = ap.parse_args()

    if bool(args.raw_folder) == bool(args.raw_zip):
        raise SystemExit("Provide exactly one of --raw-folder or --raw-zip")

    if args.raw_folder:
        mf = load_match_from_folder(args.match_id, args.raw_folder)
    else:
        zip_bytes = Path(args.raw_zip).read_bytes()
        mf = load_match_from_zip(args.match_id, zip_bytes)

    pm = process_match(mf)
    out_dir = save_processed(pm, Path(args.out))

    print(f"✅ Saved processed outputs to: {out_dir}")
    print(f"players rows: {len(pm.players):,}")
    print(f"ball rows:    {len(pm.ball):,}")
    print(f"events_poss:  {len(pm.events_poss):,}")
    print(f"meta keys:    {sorted(pm.meta.keys())}")


if __name__ == "__main__":
    main()
