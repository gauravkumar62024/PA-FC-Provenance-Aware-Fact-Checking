#!/usr/bin/env bash
# Unpack the released PA-FC datasets in data/ .
# Keeps the .gz files so the working tree still matches what is committed.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$HERE"

shopt -s nullglob
found=0
for gz in data/*/*.json.gz data/*/*.tsv.gz; do
  found=1
  out="${gz%.gz}"
  if [[ -f "$out" && "$out" -nt "$gz" ]]; then
    echo "skip   $out (already current)"
    continue
  fi
  echo "unpack $gz"
  gunzip -kf "$gz"
done

if [[ "$found" -eq 0 ]]; then
  echo "No .json.gz or .tsv.gz files under data/. Nothing to do." >&2
  exit 1
fi

echo
echo "Record counts:"
python3 - <<'PY'
import glob, json, os
import csv
csv.field_size_limit(10**9)
for p in sorted(glob.glob("data/*/*.json") + glob.glob("data/*/*.tsv")):
    try:
        if p.endswith(".tsv"):
            with open(p, newline="") as fh:
                n = sum(1 for _ in csv.reader(fh, delimiter="\t")) - 1
        else:
            with open(p) as fh:
                n = len(json.load(fh))
        print(f"  {os.path.relpath(p):<34} {n:>6,} records")
    except Exception as exc:
        print(f"  {os.path.relpath(p):<34} UNREADABLE: {exc}")
PY
