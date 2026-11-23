"""Module 2 command-line interface for the group pipeline.

Reads rectified plate crops produced by Module 1 (any folder of .png/.jpg),
applies one illumination-correction method and writes the result for Module 3
(binarization) with the SAME file names and image size. Never overwrites.

Example:
  .venv/bin/python src/module2_cli.py --input-dir ../module1_output/crops \
      --method clahe --out-dir ../module2_output/clahe
"""
import argparse
import csv
import os
import sys

import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import module2 as m2  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input-dir", required=True, help="Module 1 rectified crops")
    ap.add_argument("--method", default="clahe", choices=list(m2.METHOD_FUNCS))
    ap.add_argument("--out-dir", required=True, help="new folder for Module 3 input")
    a = ap.parse_args()
    names = sorted(f for f in os.listdir(a.input_dir) if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")))
    if not names:
        sys.exit(f"no images in {a.input_dir}")
    os.makedirs(a.out_dir, exist_ok=True)
    log = os.path.join(a.out_dir, "module2_log.csv")
    if os.path.exists(log):
        sys.exit(f"{log} exists - use a new --out-dir")
    rows = []
    for n in names:
        img = cv2.imread(os.path.join(a.input_dir, n), cv2.IMREAD_COLOR)
        if img is None:
            rows.append({"file": n, "status": "unreadable"})
            continue
        out, diag = m2.correct_illumination(img, a.method)
        dst = os.path.join(a.out_dir, os.path.splitext(n)[0] + ".png")   # lossless for Module 3
        if os.path.exists(dst):
            sys.exit(f"refusing to overwrite {dst}")
        cv2.imwrite(dst, out)
        rows.append({"file": n, "status": "ok", "method": a.method, "h": out.shape[0], "w": out.shape[1],
                     **{k: v for k, v in diag.items()}})
    with open(log, "x", newline="", encoding="utf-8") as fh:
        keys = sorted({k for r in rows for k in r}, key=lambda k: (k != "file", k))
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader(); w.writerows(rows)
    print(f"{sum(r['status'] == 'ok' for r in rows)}/{len(rows)} images -> {a.out_dir} ({a.method})")


if __name__ == "__main__":
    main()
