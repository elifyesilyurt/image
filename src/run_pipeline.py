"""Module 2 pipeline runner.

Stages
  roi      : selection (30 candidates + seeded extension), ROI validation,
             temporary perspective rectification, contact sheet
  enhance  : baseline / CLAHE / CLAHE+TopHat-BlackHat / Homomorphic on the same crops
  ocr      : Tesseract on every variant, CER vs CCPD ground truth
  metrics  : auxiliary image measures + summary tables
  figures  : comparison panels, parameter sweeps

Usage:  .venv/bin/python src/run_pipeline.py --out results [--stages roi,enhance,ocr,metrics,figures]
Existing output files are never overwritten (the run aborts instead).
"""
from __future__ import annotations

import argparse
import csv
import os
import random
import sys

import cv2
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import module2 as m2  # noqa: E402

METHODS = ["original", "clahe", "clahe_tophat", "homomorphic"]
METHOD_LABEL = {"original": "Original (baseline)", "clahe": "CLAHE (L channel)",
                "clahe_tophat": "CLAHE + Top/Black-Hat", "homomorphic": "Homomorphic (L channel)"}


def safe_write_image(path, img):
    if os.path.exists(path):
        raise FileExistsError(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ok = cv2.imwrite(path, img)
    if not ok:
        raise IOError(path)


def safe_to_csv(df, path):
    if os.path.exists(path):
        raise FileExistsError(path)
    df.to_csv(path, index=False, encoding="utf-8")


def short_id(i):
    return f"S{i:02d}"


# ----------------------------------------------------------------------------
def stage_roi(out):
    cfg = m2.CONFIG
    sel_dir = cfg["paths"]["selection_dir"]
    db = cfg["paths"]["ccpd_db"]
    rows = list(csv.DictReader(open(os.path.join(sel_dir, "manifest.csv"), encoding="utf-8")))
    records = []
    for r in rows:
        records.append({"source_list": "elif_ccpd_db_selection (seed 42)", "group": r["group"],
                        "filename": r["filename"], "selection_seed": r["selection_seed"]})

    # ---- seeded extension to reach the group plan of 50 difficult-lighting images
    ex = cfg["extension"]
    taken = {r["filename"] for r in records}
    pool_dark, pool_bright = [], []
    for fn in sorted(os.listdir(db)):
        if not fn.endswith(".jpg") or fn in taken:
            continue
        rec = m2.parse_ccpd_name(fn)
        st, _, pw, _ = m2.validate_roi(rec, 720, 1160)
        if st != "ok" or pw < ex["min_plate_width_px"]:
            continue
        if rec.brightness <= ex["dark_max"]:
            pool_dark.append(fn)
        elif rec.brightness >= ex["bright_min"]:
            pool_bright.append(fn)
    rng = random.Random(ex["seed"])
    ext_dark = rng.sample(pool_dark, ex["n_dark"])
    ext_bright = rng.sample(pool_bright, ex["n_bright"])
    rule = (f"random.Random({ex['seed']}).sample over sorted ccpd_db filenames not in the 30-candidate list, "
            f"ROI valid and plate width >= {ex['min_plate_width_px']}px; dark: brightness <= {ex['dark_max']} "
            f"(pool {len(pool_dark)}), bright: brightness >= {ex['bright_min']} (pool {len(pool_bright)})")
    for fn in ext_dark:
        records.append({"source_list": "extension (seed %d)" % ex["seed"], "group": "dark", "filename": fn,
                        "selection_seed": ex["seed"]})
    for fn in ext_bright:
        records.append({"source_list": "extension (seed %d)" % ex["seed"], "group": "bright", "filename": fn,
                        "selection_seed": ex["seed"]})

    man = []
    for i, r in enumerate(records, 1):
        fn = r["filename"]
        img = cv2.imread(os.path.join(db, fn), cv2.IMREAD_COLOR)
        if img is None:
            man.append({**r, "id": short_id(i), "roi_status": "invalid_roi", "roi_reason": "unreadable image"})
            continue
        h, w = img.shape[:2]
        rec = m2.parse_ccpd_name(fn)
        st, reason, pw, ph = m2.validate_roi(rec, w, h)
        row = {"id": short_id(i), **r, "ccpd_subset": "ccpd_db", "img_w": w, "img_h": h,
               "ccpd_brightness": rec.brightness, "ccpd_blurriness": rec.blurriness,
               "ccpd_area": rec.area, "tilt_h": rec.tilt_h, "tilt_v": rec.tilt_v,
               "vertices_RB_LB_LT_RT": ";".join(f"{int(x)}&{int(y)}" for x, y in rec.vertices),
               "plate_w_px": round(pw, 1), "plate_h_px": round(ph, 1),
               "gt_plate": rec.plate_text, "roi_status": st, "roi_reason": reason,
               "crop_file": ""}
        if st != "invalid_roi":
            crop = m2.rectify_plate(img, rec)
            cf = f"crops/original/{short_id(i)}.png"
            safe_write_image(os.path.join(out, cf), crop)
            row["crop_file"] = cf
        man.append(row)

    df = pd.DataFrame(man)
    safe_to_csv(df, os.path.join(out, "manifest_roi.csv"))
    with open(os.path.join(out, "selection_rule.txt"), "x", encoding="utf-8") as fh:
        fh.write("Original 30 candidates: elif_ccpd_db_selection/manifest.csv (seed 42; 10 dark/10 mid/10 bright "
                 "by CCPD filename brightness metadata).\n")
        fh.write("Extension (20): " + rule + "\n")
        fh.write("Group labels use CCPD filename brightness metadata (plate-region brightness as "
                 "annotated by CCPD), not a local light measurement.\n")
    make_contact_sheet(df, out)
    print(df.groupby(["group", "roi_status"]).size())
    return df


def make_contact_sheet(df, out):
    db = m2.CONFIG["paths"]["ccpd_db"]
    cell_w, scene_w = 520, 150
    scene_h = int(scene_w * 1160 / 720)
    crop_w = cell_w - scene_w - 20
    crop_h = int(crop_w * 140 / 440)
    cell_h = scene_h + 10
    cols = 5
    n = len(df)
    rows = (n + cols - 1) // cols
    sheet = np.full((rows * cell_h + 10, cols * cell_w + 10, 3), 255, np.uint8)
    colors = {"ok": (40, 140, 40), "review_required": (0, 140, 230), "invalid_roi": (30, 30, 220)}
    for k, r in enumerate(df.itertuples()):
        y0 = (k // cols) * cell_h + 10
        x0 = (k % cols) * cell_w + 10
        img = cv2.imread(os.path.join(db, r.filename))
        v = np.array([[int(c) for c in p.split("&")] for p in r.vertices_RB_LB_LT_RT.split(";")], np.int32)
        cv2.polylines(img, [v.reshape(-1, 1, 2)], True, (0, 255, 255), 6)
        sheet[y0:y0 + scene_h, x0:x0 + scene_w] = cv2.resize(img, (scene_w, scene_h), interpolation=cv2.INTER_AREA)
        if r.crop_file:
            c = cv2.imread(os.path.join(out, r.crop_file))
            sheet[y0:y0 + crop_h, x0 + scene_w + 10:x0 + scene_w + 10 + crop_w] = cv2.resize(c, (crop_w, crop_h), interpolation=cv2.INTER_AREA)
        ty = y0 + crop_h + 28
        tx = x0 + scene_w + 10
        cv2.putText(sheet, f"{r.id} {r.group}  B={r.ccpd_brightness} blur={r.ccpd_blurriness}", (tx, ty),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(sheet, f"plate {r.plate_w_px:.0f}x{r.plate_h_px:.0f}px  tilt {r.tilt_h}/{r.tilt_v}", (tx, ty + 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (60, 60, 60), 1, cv2.LINE_AA)
        cv2.putText(sheet, r.roi_status, (tx, ty + 52), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                    colors.get(r.roi_status, (0, 0, 0)), 2, cv2.LINE_AA)
        src = "ext" if "extension" in r.source_list else "orig30"
        cv2.putText(sheet, src, (tx, ty + 76), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120, 120, 120), 1, cv2.LINE_AA)
    safe_write_image(os.path.join(out, "figures/contact_sheet_roi.jpg"), sheet)


# ----------------------------------------------------------------------------
def load_manifest(out):
    return pd.read_csv(os.path.join(out, "manifest_roi.csv"), keep_default_na=False)


def stage_enhance(out):
    df = load_manifest(out)
    rows = []
    for r in df.itertuples():
        if not r.crop_file:
            continue
        src = cv2.imread(os.path.join(out, r.crop_file))
        res = {"original": (src, {})}
        img, d = m2.apply_clahe(src), {}
        res["clahe"] = (img, d)
        res["clahe_tophat"] = m2.apply_clahe_tophat(src)
        res["homomorphic"] = m2.apply_homomorphic(src)
        for meth, (img, diag) in res.items():
            assert img.shape == src.shape and img.dtype == np.uint8, (r.id, meth)
            assert np.isfinite(img).all()
            path = f"crops/{meth}/{r.id}.png"
            if meth != "original":
                safe_write_image(os.path.join(out, path), img)
            row = {"id": r.id, "group": r.group, "method": meth, "file": path}
            row.update({f"diag_{k}": v for k, v in diag.items()})
            row.update(m2.aux_metrics(img, src))
            rows.append(row)
    safe_to_csv(pd.DataFrame(rows), os.path.join(out, "metrics_aux_per_image.csv"))
    m2.save_config(os.path.join(out, "config_used.json"))
    print("enhanced", len(rows) // len(METHODS), "crops x", len(METHODS), "variants")


def stage_sweep(out):
    """Parameter sensitivity figures (visual only, 3 example crops)."""
    import copy
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    df = load_manifest(out)
    ids = [i for i in m2.CONFIG.get("sweep_ids", ["S06", "S17", "S23"]) if i in set(df.id)]
    crops = {i: cv2.imread(os.path.join(out, f"crops/original/{i}.png")) for i in ids}

    def grid(title, settings, fn, fname):
        fig, axes = plt.subplots(len(ids), len(settings) + 1, figsize=(3.2 * (len(settings) + 1), 1.35 * len(ids) + 0.8))
        for a, i in enumerate(ids):
            axes[a, 0].imshow(cv2.cvtColor(crops[i], cv2.COLOR_BGR2RGB)); axes[a, 0].set_title(f"{i} original", fontsize=8)
            for b, (lab, cfg) in enumerate(settings, 1):
                img = fn(crops[i], cfg)
                img = img[0] if isinstance(img, tuple) else img
                axes[a, b].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)); axes[a, b].set_title(lab, fontsize=8)
        for ax in axes.ravel():
            ax.axis("off")
        fig.suptitle(title, fontsize=10)
        fig.tight_layout()
        p = os.path.join(out, "figures", fname)
        if os.path.exists(p):
            raise FileExistsError(p)
        fig.savefig(p, dpi=150); plt.close(fig)

    cl = []
    for clip in (1.0, 2.0, 4.0):
        for tg in ((8, 2), (8, 8)):
            c = copy.deepcopy(m2.CONFIG); c["clahe"]["clip_limit"] = clip; c["clahe"]["tile_grid"] = list(tg)
            cl.append((f"clip={clip} tiles={tg[0]}x{tg[1]}", c))
    grid("CLAHE parameter sweep (L channel)", cl, m2.apply_clahe, "sweep_clahe.png")
    hm = []
    for d0 in (5.0, 10.0, 30.0):
        for gl, gh in ((0.5, 1.5), (0.3, 2.0)):
            c = copy.deepcopy(m2.CONFIG); c["homomorphic"].update(d0=d0, gamma_low=gl, gamma_high=gh)
            hm.append((f"D0={d0:g} gL={gl} gH={gh}", c))
    grid("Homomorphic parameter sweep (L channel)", hm, m2.apply_homomorphic, "sweep_homomorphic.png")


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--stages", default="roi")
    a = ap.parse_args()
    out = a.out if os.path.isabs(a.out) else os.path.join(PROJECT, a.out)
    os.makedirs(out, exist_ok=True)
    stages = a.stages.split(",")
    if "roi" in stages:
        stage_roi(out)
    for s in stages:
        if s != "roi":
            globals()[f"stage_{s}"](out)


if __name__ == "__main__":
    main()
