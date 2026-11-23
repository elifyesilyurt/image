"""Module 2 - Illumination Correction (Elif Yesilyurt).

Group project: Image Enhancement in Licence Plate Recognition Systems.
All tunable parameters live in CONFIG. Every method receives the SAME
rectified plate crop and returns an image of the SAME size/orientation;
methods differ only in illumination processing.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass

import cv2
import numpy as np

# ----------------------------------------------------------------------------
# Central configuration
# ----------------------------------------------------------------------------
CONFIG = {
    # Override with environment variables CCPD_DB_DIR / ELIF_SELECTION_DIR on another machine.
    "paths": {
        "ccpd_db": os.environ.get("CCPD_DB_DIR", "/Users/elif/Desktop/image/CCPD2019/ccpd_db"),
        "selection_dir": os.environ.get("ELIF_SELECTION_DIR",
                                        "/Users/elif/Documents/Codex/2026-11-23/bu-x20/outputs/elif_ccpd_db_selection"),
    },
    # Temporary ROI rectification (NOT the final Module 1 output).
    "crop": {
        "width": 440,            # 440x140 = aspect of the 440 mm x 140 mm Chinese plate
        "height": 140,
        "interpolation": "INTER_CUBIC",
        "min_plate_width_px": 150,   # narrower plates are flagged review_required
        "min_plate_height_px": 45,
    },
    # Extra candidates to reach the group plan of 50 difficult-lighting images.
    "extension": {
        "seed": 2026,
        "n_dark": 10,            # CCPD brightness <= dark_max
        "n_bright": 10,          # CCPD brightness >= bright_min
        "dark_max": 21,
        "bright_min": 204,
        "min_plate_width_px": 150,
    },
    "clahe": {
        "color_space": "LAB",     # only L is processed; a, b untouched
        "clip_limit": 2.0,
        "tile_grid": [8, 2],      # (cols, rows) -> 55x70 px tiles on a 440x140 crop
    },
    "tophat": {
        # SE size = se_factor * estimated stroke width (forced odd), ellipse
        "se_factor": 2.0,
        "stroke_min": 3,
        "stroke_max": 25,
        "stroke_fallback_ratio": 10 / 140,  # used if the estimate fails
        "weight": 1.0,            # L' = L + w*(TopHat - BlackHat)
    },
    "homomorphic": {
        "color_space": "LAB",     # filter L only -> no hue shift by construction
        "gamma_low": 0.5,         # gain for low frequencies (illumination)
        "gamma_high": 1.5,        # gain for high frequencies (reflectance / edges)
        "c": 1.0,                 # slope of the Gaussian transition
        "d0": 10.0,               # cut-off radius in frequency samples (padded grid)
        "pad": "reflect",
        "norm_percentiles": [0.5, 99.5],
    },
    # OCR settings were chosen on a SEPARATE 20-image calibration set
    # (ccpd_db, random.Random(7), disjoint from the 50 test images), using the
    # unprocessed crops only -> no tuning toward any enhancement method.
    # Grid: height {140,70,48} x gray {y,n} x pad {0,12} x psm {7,8,13} x whitelist {y,n}.
    # Best: native 140 px, grayscale, no pad, psm 8, whitelist (calibration CER 0.507).
    "ocr": {
        "engine": "tesseract",
        "lang": "chi_sim",
        "oem": 1,                 # LSTM
        "psm": 8,                 # single word
        "grayscale": True,        # BGR -> gray (cv2) before OCR, identical for all methods
        "whitelist": True,
        "tessdata_dir": "ocr/tessdata",
        "calibration_seed": 7,
        "calibration_n": 20,
    },
}

# ----------------------------------------------------------------------------
# CCPD filename parsing (official README format)
# ----------------------------------------------------------------------------
PROVINCES = ["皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "京", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤", "桂", "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新", "警", "学", "O"]
ALPHABETS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'O']
ADS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'O']


@dataclass
class CCPDRecord:
    filename: str
    area: str
    tilt_h: int
    tilt_v: int
    bbox: np.ndarray        # [[x1,y1],[x2,y2]] left-up, right-bottom
    vertices: np.ndarray    # 4x2, order RB, LB, LT, RT (README: starts from right-bottom)
    plate_idx: list
    plate_text: str
    brightness: int
    blurriness: int


def parse_ccpd_name(filename: str) -> CCPDRecord:
    stem = os.path.splitext(os.path.basename(filename))[0]
    f = stem.split("-")
    if len(f) != 7:
        raise ValueError(f"expected 7 fields, got {len(f)}: {filename}")
    th, tv = (int(x) for x in f[1].split("_"))
    bbox = np.array([[int(c) for c in p.split("&")] for p in f[2].split("_")], dtype=np.float32)
    verts = np.array([[int(c) for c in p.split("&")] for p in f[3].split("_")], dtype=np.float32)
    idx = [int(i) for i in f[4].split("_")]
    text = PROVINCES[idx[0]] + ALPHABETS[idx[1]] + "".join(ADS[i] for i in idx[2:])
    return CCPDRecord(stem + ".jpg", f[0], th, tv, bbox, verts, idx, text, int(f[5]), int(f[6]))


def validate_roi(rec: CCPDRecord, img_w: int, img_h: int, cfg=CONFIG):
    """Return (status, reason, plate_w, plate_h)."""
    v = rec.vertices
    rb, lb, lt, rt = v
    reasons = []
    if (v[:, 0] < 0).any() or (v[:, 1] < 0).any() or (v[:, 0] >= img_w).any() or (v[:, 1] >= img_h).any():
        reasons.append("vertex outside image")
    if not (rb[0] > lb[0] and rt[0] > lt[0] and rb[1] > rt[1] and lb[1] > lt[1]):
        reasons.append("vertex order not RB,LB,LT,RT")
    if not cv2.isContourConvex(v.reshape(-1, 1, 2).astype(np.int32)):
        reasons.append("quadrilateral not convex")
    pw = float((np.linalg.norm(rb - lb) + np.linalg.norm(rt - lt)) / 2)
    ph = float((np.linalg.norm(rt - rb) + np.linalg.norm(lt - lb)) / 2)
    if reasons:
        return "invalid_roi", "; ".join(reasons), pw, ph
    c = cfg["crop"]
    if pw < c["min_plate_width_px"] or ph < c["min_plate_height_px"]:
        return "review_required", f"small plate ({pw:.0f}x{ph:.0f}px) upsampled to {c['width']}x{c['height']}", pw, ph
    return "ok", "", pw, ph


def rectify_plate(img_bgr: np.ndarray, rec: CCPDRecord, cfg=CONFIG) -> np.ndarray:
    """Temporary perspective rectification from CCPD vertex annotation."""
    W, H = cfg["crop"]["width"], cfg["crop"]["height"]
    dst = np.array([[W - 1, H - 1], [0, H - 1], [0, 0], [W - 1, 0]], dtype=np.float32)  # RB, LB, LT, RT
    M = cv2.getPerspectiveTransform(rec.vertices.astype(np.float32), dst)
    return cv2.warpPerspective(img_bgr, M, (W, H), flags=getattr(cv2, cfg["crop"]["interpolation"]),
                               borderMode=cv2.BORDER_REPLICATE)


# ----------------------------------------------------------------------------
# Illumination correction methods (input/output: uint8 BGR, same size)
# ----------------------------------------------------------------------------
def _split_lab(img_bgr):
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    return cv2.split(lab)


def _merge_lab(L, a, b):
    return cv2.cvtColor(cv2.merge([L, a, b]), cv2.COLOR_LAB2BGR)


def apply_global_he(img_bgr, cfg=CONFIG):
    """Global histogram equalisation of LAB-L (one mapping for the whole crop).
    Global counterpart of CLAHE, for the report's global-vs-adaptive comparison."""
    L, a, b = _split_lab(img_bgr)
    return _merge_lab(cv2.equalizeHist(L), a, b)


def apply_clahe(img_bgr, cfg=CONFIG):
    p = cfg["clahe"]
    L, a, b = _split_lab(img_bgr)
    clahe = cv2.createCLAHE(clipLimit=p["clip_limit"], tileGridSize=tuple(p["tile_grid"]))
    return _merge_lab(clahe.apply(L), a, b)


def estimate_stroke_width(L: np.ndarray, cfg=CONFIG):
    """Estimate character stroke width (px) on a luminance plate crop.

    Otsu split; the minority class is taken as character foreground
    (characters cover less area than plate background). Stroke width =
    2 * median of distance-transform ridge values inside the foreground.
    Returns (stroke_px, method).
    """
    p = cfg["tophat"]
    H = L.shape[0]
    blur = cv2.GaussianBlur(L, (3, 3), 0)
    _, bw = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if (bw > 0).mean() > 0.5:
        bw = 255 - bw
    # ignore a 6% border (plate frame / rivets)
    m = int(0.06 * H)
    inner = np.zeros_like(bw)
    inner[m:-m, m:-m] = bw[m:-m, m:-m]
    dt = cv2.distanceTransform(inner, cv2.DIST_L2, 3)
    ridge = (dt >= cv2.dilate(dt, np.ones((3, 3), np.uint8))) & (dt > 1.0)
    vals = dt[ridge]
    if vals.size < 30:
        return float(np.clip(round(p["stroke_fallback_ratio"] * H), p["stroke_min"], p["stroke_max"])), "fallback_ratio"
    s = 2.0 * float(np.median(vals))
    return float(np.clip(s, p["stroke_min"], p["stroke_max"])), "distance_transform"


def apply_tophat_blackhat(img_bgr, cfg=CONFIG):
    """L' = L + w*(TopHat(L) - BlackHat(L)); a,b untouched. Returns (img, info)."""
    p = cfg["tophat"]
    L, a, b = _split_lab(img_bgr)
    stroke, how = estimate_stroke_width(L, cfg)
    k = int(round(p["se_factor"] * stroke)) | 1
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    th = cv2.morphologyEx(L, cv2.MORPH_TOPHAT, se).astype(np.int16)
    bh = cv2.morphologyEx(L, cv2.MORPH_BLACKHAT, se).astype(np.int16)
    out = L.astype(np.int16) + np.int16(p["weight"]) * (th - bh)
    clip_frac = float(((out < 0) | (out > 255)).mean())
    out = np.clip(out, 0, 255).astype(np.uint8)
    return _merge_lab(out, a, b), {"stroke_px": round(stroke, 2), "stroke_method": how, "se_size": k,
                                   "tophat_clip_frac": round(clip_frac, 4)}


def apply_clahe_tophat(img_bgr, cfg=CONFIG):
    """Report pipeline: CLAHE on L, then Top-Hat/Black-Hat on L."""
    return apply_tophat_blackhat(apply_clahe(img_bgr, cfg), cfg)


def homomorphic_filter_channel(ch: np.ndarray, cfg=CONFIG):
    """Homomorphic filtering of one uint8 channel. Returns (uint8, diagnostics)."""
    p = cfg["homomorphic"]
    f = ch.astype(np.float64) / 255.0
    logf = np.log1p(f)                                     # log(1+f): avoids log(0)
    h, w = logf.shape
    ph, pw = h // 2, w // 2
    padded = np.pad(logf, ((ph, ph), (pw, pw)), mode=p["pad"])
    P, Q = padded.shape
    F = np.fft.fftshift(np.fft.fft2(padded))
    u = np.arange(P) - P / 2
    v = np.arange(Q) - Q / 2
    D2 = u[:, None] ** 2 + v[None, :] ** 2
    Hf = (p["gamma_high"] - p["gamma_low"]) * (1 - np.exp(-p["c"] * D2 / (p["d0"] ** 2))) + p["gamma_low"]
    g = np.real(np.fft.ifft2(np.fft.ifftshift(Hf * F)))[ph:ph + h, pw:pw + w]
    out = np.expm1(g)
    diag = {"nan_or_inf": int((~np.isfinite(out)).sum())}
    out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    lo, hi = np.percentile(out, p["norm_percentiles"])
    diag["clip_frac"] = round(float(((out < lo) | (out > hi)).mean()), 4)
    if hi - lo < 1e-9:
        diag["degenerate"] = True
        return ch.copy(), diag
    out = np.clip((out - lo) / (hi - lo), 0, 1)
    return (out * 255.0 + 0.5).astype(np.uint8), diag


def apply_homomorphic(img_bgr, cfg=CONFIG):
    L, a, b = _split_lab(img_bgr)
    L2, diag = homomorphic_filter_channel(L, cfg)
    return _merge_lab(L2, a, b), diag


# ----------------------------------------------------------------------------
# Public API for the group pipeline (Module 1 crop in -> Module 3 input out)
# ----------------------------------------------------------------------------
METHOD_FUNCS = {
    "original": lambda img, cfg=CONFIG: (img.copy(), {}),
    "global_he": lambda img, cfg=CONFIG: (apply_global_he(img, cfg), {}),
    "clahe": lambda img, cfg=CONFIG: (apply_clahe(img, cfg), {}),
    "clahe_tophat": apply_clahe_tophat,
    "homomorphic": apply_homomorphic,
}


def correct_illumination(img_bgr: np.ndarray, method: str = "clahe", cfg=CONFIG):
    """Module 2 entry point.

    img_bgr : uint8 BGR (or single-channel gray) rectified plate crop from Module 1.
    method  : one of METHOD_FUNCS ("original", "global_he", "clahe", "clahe_tophat", "homomorphic").
    Returns (corrected uint8 BGR image of identical size, diagnostics dict).
    """
    if method not in METHOD_FUNCS:
        raise ValueError(f"unknown method {method!r}; choose from {list(METHOD_FUNCS)}")
    if img_bgr is None or img_bgr.dtype != np.uint8:
        raise TypeError("expected a uint8 image")
    if img_bgr.ndim == 2:
        img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2BGR)
    out, diag = METHOD_FUNCS[method](img_bgr, cfg)
    assert out.shape == img_bgr.shape and out.dtype == np.uint8
    return out, diag


# ----------------------------------------------------------------------------
# Auxiliary (non-OCR) image measures - NOT evidence of OCR performance
# ----------------------------------------------------------------------------
def luminance(img_bgr):
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)[:, :, 0]


def aux_metrics(img_bgr, ref_bgr=None):
    L = luminance(img_bgr).astype(np.float64)
    gx = cv2.Sobel(L, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(L, cv2.CV_64F, 0, 1, ksize=3)
    # local contrast: mean of local std in 15x15 windows
    mu = cv2.blur(L, (15, 15))
    mu2 = cv2.blur(L * L, (15, 15))
    lstd = np.sqrt(np.maximum(mu2 - mu * mu, 0))
    # illumination non-uniformity: std of a heavily blurred L (low-frequency field)
    lowfreq = cv2.GaussianBlur(L, (0, 0), sigmaX=20)
    m = {
        "L_mean": L.mean(),
        "L_std": L.std(),
        "local_contrast": lstd.mean(),
        "grad_mean": np.sqrt(gx ** 2 + gy ** 2).mean(),
        "illum_nonuniformity": lowfreq.std(),
        "illum_cv": lowfreq.std() / max(lowfreq.mean(), 1.0),   # brightness-independent
        "sat_hi_frac": float((L >= 250).mean()),
        "sat_lo_frac": float((L <= 5).mean()),
    }
    if ref_bgr is not None:
        a0 = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2LAB)[:, :, 1:].astype(np.float64)
        a1 = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)[:, :, 1:].astype(np.float64)
        m["chroma_shift"] = float(np.abs(a1 - a0).mean())   # mean |delta a|,|delta b| (LAB, 8-bit units)
    return {k: round(float(v), 4) for k, v in m.items()}


# ----------------------------------------------------------------------------
# OCR + CER
# ----------------------------------------------------------------------------
def normalize_plate(s: str) -> str:
    """Upper-case, drop whitespace and separators (·, -, ., _, etc.).
    Keep only CJK ideographs, A-Z and 0-9."""
    s = s.upper()
    return "".join(ch for ch in s if ("A" <= ch <= "Z") or ("0" <= ch <= "9") or ("一" <= ch <= "鿿"))


def levenshtein(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def run_ocr(img_bgr, project_dir, cfg=CONFIG):
    import pytesseract
    o = cfg["ocr"]
    tessdir = os.path.join(project_dir, o["tessdata_dir"])
    whitelist = "".join(PROVINCES[:-1]) + "ABCDEFGHJKLMNPQRSTUVWXYZ0123456789"
    config = f'--tessdata-dir "{tessdir}" --oem {o["oem"]} --psm {o["psm"]}'
    if o["whitelist"]:
        config += f' -c tessedit_char_whitelist={whitelist}'
    x = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY) if o["grayscale"] else cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return pytesseract.image_to_string(x, lang=o["lang"], config=config).strip(), config


def save_config(path, cfg=CONFIG):
    with open(path, "x", encoding="utf-8") as fh:   # "x": never overwrite
        json.dump(cfg, fh, ensure_ascii=False, indent=2)
