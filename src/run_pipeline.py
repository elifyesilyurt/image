"""Module 2 pipeline runner.

Stages
  roi      : selection (30 candidates + seeded extension), ROI validation,
             temporary perspective rectification, contact sheet
  enhance  : baseline / global HE / CLAHE / CLAHE+TopHat-BlackHat / Homomorphic on the same crops
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

METHODS = ["original", "global_he", "clahe", "clahe_tophat", "homomorphic"]
METHOD_LABEL = {"original": "Original (baseline)", "global_he": "Global HE (L channel)",
                "clahe": "CLAHE (L channel)", "clahe_tophat": "CLAHE + Top/Black-Hat",
                "homomorphic": "Homomorphic (L channel)"}
METHOD_COLOR = {"original": "#7f7f7f", "global_he": "#e1a43b", "clahe": "#2a78b5",
                "clahe_tophat": "#6aa84f", "homomorphic": "#c0504d"}
# 'mid' is kept as the data label (traceable to the selection manifest) but CCPD-DB
# brightness metadata is bimodal, so 'mid' is really the upper end of the dark mode.
GROUP_LABEL = {"dark": "dark (CCPD B<=21)", "mid": "dark-upper ('mid', B 22-39)", "bright": "bright (B>=204)"}


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
        res = {meth: m2.correct_illumination(src, meth) for meth in METHODS}
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
def stage_calibrate(out):
    """Re-runs the OCR-config grid on the separate calibration set and saves it."""
    import itertools
    import pytesseract
    cfg = m2.CONFIG
    db = cfg["paths"]["ccpd_db"]
    used = set(load_manifest(out)["filename"])
    pool = [f for f in sorted(os.listdir(db)) if f.endswith(".jpg") and f not in used]
    cal = random.Random(cfg["ocr"]["calibration_seed"]).sample(pool, cfg["ocr"]["calibration_n"])
    crops = []
    for f in cal:
        r = m2.parse_ccpd_name(f)
        crops.append((f, r.plate_text, r.brightness, m2.rectify_plate(cv2.imread(os.path.join(db, f)), r)))
    td = os.path.join(PROJECT, cfg["ocr"]["tessdata_dir"])
    wl = "".join(m2.PROVINCES[:-1]) + "ABCDEFGHJKLMNPQRSTUVWXYZ0123456789"
    rows = []
    for h, gray, pad, psm, use_wl in itertools.product([140, 70, 48], [False, True], [0, 12], [7, 8, 13], [True, False]):
        conf = f'--tessdata-dir "{td}" --oem 1 --psm {psm}' + (f" -c tessedit_char_whitelist={wl}" if use_wl else "")
        E = N = empty = 0
        for _, gt, _, c in crops:
            x = cv2.resize(c, (int(440 * h / 140), h), interpolation=cv2.INTER_AREA if h < 140 else cv2.INTER_CUBIC)
            if gray:
                x = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)
            if pad:
                x = cv2.copyMakeBorder(x, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
            p = m2.normalize_plate(pytesseract.image_to_string(x, lang="chi_sim", config=conf))
            E += m2.levenshtein(p, gt); N += len(gt); empty += (p == "")
        rows.append({"height": h, "grayscale": gray, "pad": pad, "psm": psm, "whitelist": use_wl,
                     "cer": round(E / N, 4), "empty_outputs": empty})
    safe_to_csv(pd.DataFrame(rows).sort_values("cer"), os.path.join(out, "ocr_calibration_grid.csv"))
    safe_to_csv(pd.DataFrame([{"filename": f, "gt_plate": g, "ccpd_brightness": b} for f, g, b, _ in crops]),
                os.path.join(out, "ocr_calibration_set.csv"))


def stage_ocr(out):
    import pytesseract
    df = load_manifest(out)
    rows = []
    for r in df.itertuples():
        if not r.crop_file:
            continue
        gt = m2.normalize_plate(r.gt_plate)
        for meth in METHODS:
            img = cv2.imread(os.path.join(out, f"crops/{meth}/{r.id}.png"))
            raw, conf = m2.run_ocr(img, PROJECT)
            pred = m2.normalize_plate(raw)
            pred_alnum = "".join(c for c in pred if not ("一" <= c <= "鿿"))
            rows.append({"id": r.id, "group": r.group, "roi_status": r.roi_status, "method": meth,
                         "gt": gt, "ocr_raw": raw.replace("\n", " "), "ocr_norm": pred,
                         "edit_dist": m2.levenshtein(pred, gt), "gt_len": len(gt),
                         "edit_dist_alnum": m2.levenshtein(pred_alnum, gt[1:]), "gt_len_alnum": len(gt) - 1,
                         "exact_match": pred == gt, "empty": pred == ""})
    safe_to_csv(pd.DataFrame(rows), os.path.join(out, "ocr_results_per_image.csv"))
    with open(os.path.join(out, "ocr_settings.txt"), "x", encoding="utf-8") as fh:
        fh.write(f"tesseract {pytesseract.get_tesseract_version()}; pytesseract {pytesseract.__version__}\n")
        fh.write(f"lang={m2.CONFIG['ocr']['lang']}; config: {conf.replace(PROJECT + '/', '')}\n")
        fh.write("Input: 440x140 crop of each method, converted BGR->GRAY (cv2), no other preprocessing.\n")


def _cer(g, col="edit_dist", n="gt_len"):
    return g[col].sum() / g[n].sum()


def _paired_bootstrap(per_img, meth, n_boot=5000, seed=0):
    """95% CI of CER(method) - CER(original), resampling images (paired)."""
    piv_e = per_img.pivot(index="id", columns="method", values="edit_dist")
    piv_n = per_img.pivot(index="id", columns="method", values="gt_len")
    ids = piv_e.index.values
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_boot):
        s = rng.choice(len(ids), len(ids), replace=True)
        n = piv_n["original"].values[s].sum()
        diffs.append((piv_e[meth].values[s].sum() - piv_e["original"].values[s].sum()) / n)
    return np.percentile(diffs, [2.5, 97.5])


def stage_metrics(out):
    ocr = pd.read_csv(os.path.join(out, "ocr_results_per_image.csv"), keep_default_na=False)
    aux = pd.read_csv(os.path.join(out, "metrics_aux_per_image.csv"), keep_default_na=False)
    rows = []
    # Exploratory illumination strata, measured on the ORIGINAL crop (defined after v1 results were seen):
    #   uneven      : illum_cv (std/mean of the low-pass L field) above the median of the 50 crops
    #   partial_glare: 5-40 % of plate pixels saturated (L >= 250)
    o = aux[aux.method == "original"].set_index("id")
    cv_med = float(o.illum_cv.median())
    uneven = set(o.index[o.illum_cv > cv_med])
    glare = set(o.index[(o.sat_hi_frac > 0.05) & (o.sat_hi_frac < 0.40)])
    subsets = [("all", ocr)] + [(f"group={g}", ocr[ocr.group == g]) for g in ["dark", "mid", "bright"]] + \
              [("roi_status=ok", ocr[ocr.roi_status == "ok"]), ("roi_status=review_required", ocr[ocr.roi_status == "review_required"])] + \
              [(f"exploratory:uneven(illum_cv>{cv_med:.3f})", ocr[ocr.id.isin(uneven)]),
               (f"exploratory:uniform(illum_cv<={cv_med:.3f})", ocr[~ocr.id.isin(uneven)]),
               ("exploratory:partial_glare(5-40% saturated)", ocr[ocr.id.isin(glare)]),
               ("exploratory:no_partial_glare", ocr[~ocr.id.isin(glare)])]
    with open(os.path.join(out, "illumination_strata.txt"), "x", encoding="utf-8") as fh:
        fh.write(f"median illum_cv of original crops = {cv_med:.4f}\n")
        fh.write("uneven ids: " + ",".join(sorted(uneven)) + "\n")
        fh.write("partial_glare ids: " + ",".join(sorted(glare)) + "\n")
        fh.write("Strata are EXPLORATORY: defined after the first run's results were seen.\n")
    for name, sub in subsets:
        if sub.empty:
            continue
        base = sub[sub.method == "original"].set_index("id")
        for meth in METHODS:
            g = sub[sub.method == meth]
            cur = g.set_index("id")
            d = (cur["edit_dist"] - base["edit_dist"])
            row = {"subset": name, "method": meth, "n_images": len(g),
                   "CER_full": round(_cer(g), 4),
                   "CER_alnum_6": round(_cer(g, "edit_dist_alnum", "gt_len_alnum"), 4),
                   "exact_match": int(g["exact_match"].astype(str).eq("True").sum()),
                   "empty_outputs": int(g["empty"].astype(str).eq("True").sum()),
                   "imgs_better_than_orig": int((d < 0).sum()) if meth != "original" else "",
                   "imgs_same_as_orig": int((d == 0).sum()) if meth != "original" else "",
                   "imgs_worse_than_orig": int((d > 0).sum()) if meth != "original" else ""}
            if meth != "original":
                lo, hi = _paired_bootstrap(sub, meth)
                row["dCER_vs_orig_95CI"] = f"[{lo:+.3f}, {hi:+.3f}]"
            rows.append(row)
    summ = pd.DataFrame(rows)
    safe_to_csv(summ, os.path.join(out, "metrics_summary_ocr.csv"))
    auxs = aux.groupby(["group", "method"])[["L_mean", "L_std", "local_contrast", "grad_mean", "illum_nonuniformity",
                                             "illum_cv", "sat_hi_frac", "sat_lo_frac", "chroma_shift"]].mean().round(3).reset_index()
    safe_to_csv(auxs, os.path.join(out, "metrics_summary_aux.csv"))
    print(summ.to_string())


# ----------------------------------------------------------------------------
def _setup_mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = ["Arial Unicode MS", "Hiragino Sans GB", "DejaVu Sans"]
    return plt


def _rgb(path):
    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)


def panel(out, ids, ocr, man, fname, title=None, notes=None):
    plt = _setup_mpl()
    row_h = 2.55 if notes else 2.25
    top_pad = 0.75 if title else 0.15
    H = row_h * len(ids) + top_pad
    fig, axes = plt.subplots(len(ids), len(METHODS), figsize=(4.2 * len(METHODS), H), squeeze=False)
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=1 - top_pad / H, wspace=0.05,
                        hspace=(0.85 if notes else 1.0))
    for a, i in enumerate(ids):
        mr = man.set_index("id").loc[i]
        for b, meth in enumerate(METHODS):
            ax = axes[a, b]
            ax.imshow(_rgb(os.path.join(out, f"crops/{meth}/{i}.png")))
            o = ocr[(ocr.id == i) & (ocr.method == meth)].iloc[0]
            pc = o.edit_dist / o.gt_len
            ax.set_title(f"{METHOD_LABEL[meth]}\nOCR: {o.ocr_norm or '∅'}  (CER {pc:.2f})", fontsize=9)
            ax.axis("off")
        lab = f"{i} | {GROUP_LABEL.get(mr.group, mr.group)} | CCPD brightness={mr.ccpd_brightness} blur={mr.ccpd_blurriness} | GT {mr.gt_plate} | ROI {mr.roi_status}"
        if notes and i in notes:
            lab += f"\n{notes[i]}"
        axes[a, 0].annotate(lab, xy=(0, 1.0), xycoords="axes fraction", xytext=(0, 30), textcoords="offset points",
                            fontsize=9.5, fontweight="bold", va="bottom", ha="left", color="#1f3b5a")
    if title:
        fig.suptitle(title, fontsize=12, fontweight="bold", y=1 - 0.15 / H)
    p = os.path.join(out, "figures", fname)
    if os.path.exists(p):
        raise FileExistsError(p)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    fig.savefig(p, dpi=170); plt.close(fig)


def stage_figures(out):
    plt = _setup_mpl()
    man = load_manifest(out)
    ocr = pd.read_csv(os.path.join(out, "ocr_results_per_image.csv"), keep_default_na=False)
    ids = list(man[man.crop_file != ""].id)
    # every image: one panel file per image
    for i in ids:
        panel(out, [i], ocr, man, f"panels/{i}.png")
    # overview per group
    for g in ["dark", "mid", "bright"]:
        gi = list(man[(man.group == g) & (man.crop_file != "")].id)
        for k in range(0, len(gi), 10):
            panel(out, gi[k:k + 10], ocr, man, f"overview_{g}_{k // 10 + 1}.png",
                  title=f"Group {GROUP_LABEL[g]} - original vs. illumination correction")
    # CER bar charts
    summ = pd.read_csv(os.path.join(out, "metrics_summary_ocr.csv"), keep_default_na=False)
    colors = [METHOD_COLOR[m] for m in METHODS]

    def bars(subsets, xlabels, title, fname):
        sub = summ[summ.subset.isin(subsets)]
        fig, ax = plt.subplots(figsize=(10.5, 4.4))
        xs = np.arange(len(subsets))
        w = 0.8 / len(METHODS)
        for k, meth in enumerate(METHODS):
            vals = [sub[(sub.subset == s) & (sub.method == meth)].CER_full.values[0] for s in subsets]
            bb = ax.bar(xs + (k - (len(METHODS) - 1) / 2) * w, vals, w, label=METHOD_LABEL[meth], color=METHOD_COLOR[meth])
            for b_, v in zip(bb, vals):
                ax.text(b_.get_x() + b_.get_width() / 2, v + 0.01, f"{v:.2f}", ha="center", fontsize=6.5)
        ns = [int(sub[(sub.subset == s) & (sub.method == "original")].n_images.values[0]) for s in subsets]
        ax.set_xticks(xs, [f"{l}\n(n={n})" for l, n in zip(xlabels, ns)])
        ax.set_ylabel("CER (lower is better)")
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=7.5, ncol=3, loc="upper left")
        ax.set_ylim(0, 1.15)
        fig.tight_layout()
        p = os.path.join(out, "figures", fname)
        if os.path.exists(p):
            raise FileExistsError(p)
        fig.savefig(p, dpi=170); plt.close(fig)

    bars(["all", "group=dark", "group=mid", "group=bright"],
         ["all", GROUP_LABEL["dark"], GROUP_LABEL["mid"], GROUP_LABEL["bright"]],
         "Tesseract 5.5.4 (chi_sim, psm 8) CER on the same 440x140 crops", "cer_by_group.png")
    ex_sub = [s for s in summ.subset.unique() if s.startswith("exploratory:")]
    bars(ex_sub, [s.replace("exploratory:", "").replace("(", "\n(") for s in ex_sub],
         "EXPLORATORY strata measured on the original crop (defined after first results) - CER",
         "cer_by_illumination_exploratory.png")
    # representative success / unchanged / failure examples (ids + notes from review/visual_review.csv)
    rv_path = os.path.join(PROJECT, "review", "visual_review.csv")
    if os.path.exists(rv_path):
        rv = pd.read_csv(rv_path)
        rv = rv[rv.id.isin(ids)]
        for cat in ["success", "unchanged", "failure"]:
            sub = rv[rv.category == cat]
            if len(sub):
                notes = {r.id: f"[{cat.upper()}] {r.selection_reason}" for r in sub.itertuples()}
                panel(out, list(sub.id), ocr, man, f"representative_{cat}.png", notes=notes,
                      title=f"Representative {cat} examples (Module 2, same crops, same OCR settings)")
    # luminance histograms for 3 examples
    ex = [i for i in ["S06", "S17", "S23"] if i in ids]
    fig, axes = plt.subplots(1, len(ex), figsize=(5 * len(ex), 3.2))
    for ax, i in zip(np.atleast_1d(axes), ex):
        for meth, c in zip(METHODS, colors):
            L = m2.luminance(cv2.imread(os.path.join(out, f"crops/{meth}/{i}.png")))
            ax.hist(L.ravel(), bins=64, range=(0, 255), histtype="step", color=c, label=METHOD_LABEL[meth], density=True)
        ax.set_title(f"{i} - L channel histogram", fontsize=9); ax.set_xlabel("L (0-255)")
    np.atleast_1d(axes)[0].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(out, "figures/luminance_histograms.png"), dpi=170); plt.close(fig)


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
