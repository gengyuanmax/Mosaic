#!/usr/bin/env python3
"""MMIU, tar-resident: canonical JSONs from all.parquet + one images.tar
repacked straight from the 7 category zips (nothing is ever extracted — MMIU's
~77k loose images can blow a cluster file-count quota, so the runner reads them
from the tar via the .tar_direct marker).

  python scripts/data_prep/prepare_mmiu_tar.py --snapshot <hf snapshot dir>

Steps: copy all.parquet -> data/mmiu/, run prepare_mmiu_canonical on it,
stream every zip member into data/mmiu/images.tar, touch .tar_direct, then
verify every record's image paths against the tar index.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import tarfile
import zipfile


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", required=True, help="HF snapshot dir of FanqingM/MMIU-Benchmark")
    ap.add_argument("--out", default="data/mmiu")
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    shutil.copy(os.path.join(a.snapshot, "all.parquet"), os.path.join(a.out, "all.parquet"))

    print("== canonical JSONs from all.parquet")
    subprocess.check_call([sys.executable, "scripts/data_prep/prepare_mmiu_canonical.py",
                           "--mmiu-root", a.out])

    tar_path = os.path.join(a.out, "images.tar")
    seen = set()
    with tarfile.open(tar_path, "w") as tar:
        for zpath in sorted(glob.glob(os.path.join(a.snapshot, "*.zip"))):
            # zip members carry no category dir, but the parquet paths do
            # ("3D-spatial/<task>/<img>.png") — prepend the zip's stem.
            prefix = os.path.basename(zpath)[:-4]
            zf = zipfile.ZipFile(zpath)
            n = 0
            for zi in zf.infolist():
                name = prefix + "/" + zi.filename.lstrip("./")
                if zi.is_dir() or name in seen:
                    continue
                info = tarfile.TarInfo(name)
                info.size = zi.file_size
                with zf.open(zi) as fh:
                    tar.addfile(info, fh)
                seen.add(name)
                n += 1
            print("  %-32s %6d members" % (os.path.basename(zpath), n))
    print("images.tar: %d members, %.1f GB" % (len(seen), os.path.getsize(tar_path) / 2**30))

    open(os.path.join(a.out, ".tar_direct"), "w").write(
        "images are read directly from images.tar; do NOT extract (quota)\n")

    print("== verify record image paths against tar index")
    missing = total = recs = 0
    for f in glob.glob(os.path.join(a.out, "datasets", "*.json")):
        if f.endswith("manifest.json"):
            continue
        for r in json.load(open(f)):
            recs += 1
            for im in r["images"]:
                total += 1
                if im["uri_or_path"] not in seen:
                    if missing < 5:
                        print("  MISSING:", im["uri_or_path"])
                    missing += 1
    print("records=%d image_refs=%d missing=%d" % (recs, total, missing))
    if missing:
        print("!! %d refs not found in tar — inspect path prefixes" % missing)


if __name__ == "__main__":
    main()
