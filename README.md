# Module 2 – Illumination Correction (Elif Yeşilyurt)

Part of the group project **"Image Enhancement in Licence Plate Recognition Systems"** (Image Processing course, guideline topic 4).
Group: Gül Berra Aslan (M1 Perspective Correction) · **Elif Yeşilyurt (M2 Illumination Correction)** · Beyza Şahin (M3 Binarization & Noise Removal) · Hakan Aksoy (M4 Character Segmentation).

This repository contains **only Module 2**: CLAHE and Homomorphic Filtering (plus the Top-Hat/Black-Hat step written in our method report), applied to the same licence-plate crops and compared with OCR Character Error Rate.

Guideline compliance checklist (TR): [`YONERGE_UYUMU.md`](YONERGE_UYUMU.md)

## Problem, input, output, evaluation
| | |
|---|---|
| Problem | Uneven plate illumination (night darkness, flash/sun glare, shadows) lowers character/background contrast before binarization and OCR. |
| Input | 50 rectified plate crops, 440×140 px, from CCPD-DB (20 dark, 10 "mid", 20 bright by CCPD brightness metadata). |
| Output | Illumination-corrected crop of identical size, ROI and orientation. |
| Methods | (1) original baseline, (2) CLAHE on LAB-L, (3) CLAHE + Top-Hat/Black-Hat on L, (4) Homomorphic filtering on LAB-L. |
| Primary metric | OCR CER = Σ Levenshtein(OCR, GT) / Σ GT characters (Tesseract 5.5.4, same settings for all methods). |
| Auxiliary | luminance mean/std, local contrast, illumination non-uniformity (CV of low-pass L), saturation fractions, chroma shift. Descriptive only. |

## Data source, licence, citation
* **CCPD – Chinese City Parking Dataset**: https://github.com/detectRecog/CCPD. Licence: **MIT License** (Copyright (c) 2017 CCPD).
* Subset `ccpd_db` from `CCPD2019.tar.xz` (10,132 images locally). The dataset is **not** included here. Only the 50 rectified plate crops and figures derived from it are included.
* Citation:
  > Xu, Z., Yang, W., Meng, A., Lu, N., & Huang, H. (2018). Towards End-to-End License Plate Detection and Recognition: A Large Dataset and Baseline. *ECCV*, 255–271.
* Filename annotations are parsed exactly as described in the official README: `area-tiltH_tiltV-bbox-4 vertices (starting from right-bottom: RB, LB, LT, RT)-plate indices-brightness-blurriness`. Plate text is decoded with the official `provinces`, `alphabets` and `ads` tables.

## Image selection
* 30 candidates: `elif_ccpd_db_selection/manifest.csv` (seed 42, 10 dark / 10 mid / 10 bright by CCPD brightness metadata).
* +20 extension to match the group plan of 50 difficult-lighting images: `random.Random(2026)` over sorted `ccpd_db` names not already selected, valid ROI, plate width ≥ 150 px; 10 with brightness ≤ 21 and 10 with brightness ≥ 204.
* **Note:** CCPD-DB brightness metadata is bimodal (5–60 or 180–242; 0 images in between). "mid" (22–39) is therefore the upper part of the dark mode, not a true medium level. Groups follow the CCPD metadata, not a local light measurement.
* ROI status: 48 `ok`, 2 `review_required` (S02, S04: plates of about 131×46 px upsampled ≈3.3×), 0 `invalid_roi`. **No image was excluded.**
* **Temporary rectification:** crops are warped from the CCPD corner annotation with `cv2.getPerspectiveTransform` (INTER_CUBIC). This is **not** Module 1's final output. If Module 1 crops become available, they should replace `results/crops/original/`, and the pipeline should be rerun.

## Parameters (central config: `src/module2.py → CONFIG`, saved as `results/config_used.json`)
| Method | Parameters | Why |
|---|---|---|
| CLAHE | LAB, L only; clip 2.0; tiles 8×2 (≈55×70 px) | Tiles are larger than one character, so a tile does not equalise a single stroke. Clip 4 visibly amplified JPEG blocks in the sweep. |
| Top/Black-Hat | L' = L + TH − BH; ellipse SE = 2 × stroke width (estimated per crop by distance transform, mean 8.7 px; fallback 10/140·H) | Report: "structuring element proportional to the estimated character thickness". |
| Homomorphic | L only; log1p → reflect pad → FFT → Gaussian high-emphasis H = (γH−γL)(1−e^{−cD²/D0²})+γL, γL 0.5, γH 1.5, c 1, D0 10 → expm1 → 0.5–99.5 % stretch | Moderate setting from the sweep; γL < 1 suppresses illumination, γH > 1 boosts edges. |
| OCR | Tesseract 5.5.4, `chi_sim` (tessdata_fast), `--oem 1 --psm 8`, whitelist, grayscale | Chosen on a **separate** 20-image calibration set (seed 7) using only unprocessed crops (`results/ocr_calibration_grid.csv`). |

Image parameters were fixed by visual inspection of `figures/sweep_*.png` (3 crops) **before** any OCR was run. Safety checks: NaN/Inf count (0), output shape/dtype assertions, clipping fraction, chroma shift (a/b channels are never modified).

## Results (n = 50, `results/metrics_summary_ocr.csv`)
| Subset | Original | CLAHE | CLAHE+Top/Black-Hat | Homomorphic |
|---|---|---|---|---|
| all (50) | 0.620 | **0.606** | 0.663 | 0.614 |
| dark (20) | 0.557 | 0.543 | 0.586 | **0.500** |
| mid (10) | 0.457 | **0.414** | 0.457 | 0.457 |
| bright (20) | **0.764** | **0.764** | 0.843 | 0.807 |
| CER without province char, all | 0.563 | **0.547** | 0.610 | 0.567 |
| ΔCER vs original, 95 % CI (paired bootstrap) | – | [−0.049, +0.017] | [−0.003, +0.091] | [−0.054, +0.046] |
| images better / same / worse | – | 9 / 33 / 8 | 10 / 23 / 17 | 14 / 25 / 11 |

* Exact plate matches are 0/50 for every method. Tesseract almost never reads the Chinese province character.
* None of the differences is statistically clear on 50 images. CLAHE + Top/Black-Hat is consistently the worst.
* Auxiliary: Homomorphic lowers illumination non-uniformity the most (illum_cv dark 0.374 → 0.146, bright 0.101 → 0.093). CLAHE does not (0.341 / 0.149).
* Representative examples: `results/figures/representative_{success,unchanged,failure}.png`, with notes in `review/visual_review.csv`.

### Failure causes (from visual evidence)
1. **Saturation/glare** (S45, S49, S28): clipped pixels carry no information. No illumination method recovers them.
2. **JPEG 8×8 block amplification** in very dark crops (S10, S14): contrast enhancement makes block edges as strong as strokes.
3. **Top/Black-Hat halos** (S26): double outlines around characters and the plate frame. About 18 % of pixels clip.
4. **Homomorphic haze** on bright plates (S26): suppressing the low-frequency illumination term flattens a uniformly bright plate to grey.
5. **Low resolution** (S02, S04): plates about 131 px wide, upsampled.

## Limitations
* Temporary rectification from annotations, not the group's Module 1 output.
* Small sample (50), one subset (CCPD-DB), groups based on CCPD metadata.
* Tesseract is a general document OCR, not a plate recogniser. CER is high for all methods and is partly limited by the OCR engine.
* Image parameters were chosen visually on 3 crops that are part of the test set. OCR parameters were chosen on a disjoint calibration set.
* Character segmentation accuracy (CSAcc/IoU) belongs to Module 4 and is not measured here. No deep-learning model was used.

## Environment and how to run
Tested on macOS (Apple M4), Python 3.9.6, OpenCV 5.1.0, NumPy 2.0.2, matplotlib 3.9.4, Tesseract 5.5.4 (Homebrew), pytesseract 0.3.13. The exact versions are in `requirements-lock.txt`.
```bash
# 1. environment (project-local)
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
brew install tesseract                     # OCR engine
bash ocr/setup_tessdata.sh                 # chi_sim model -> ocr/tessdata (checksum in ocr/tessdata.sha256)

# 2. point CONFIG["paths"] in src/module2.py at your ccpd_db folder and the 30-candidate selection

# 3. full pipeline (~2.5 min); never overwrites, so use a new folder
.venv/bin/python src/run_pipeline.py --out results_rerun --stages roi,enhance,sweep,ocr,metrics,figures,calibrate

# 4. notebook with all results
.venv/bin/jupyter notebook elif_module2_illumination.ipynb
```

## Repository layout
```
elif_module2_illumination.ipynb   main deliverable (executed, shows all results)
src/module2.py                    config, CCPD parser, methods, metrics, OCR/CER
src/run_pipeline.py               stages: roi, enhance, sweep, ocr, metrics, figures, calibrate
src/build_notebook.py             generates the notebook
review/visual_review.csv          representative examples + visual observations
presentation/                     Elif's talk script (TR), Q&A, contribution row
results/
  manifest_roi.csv                id, group, CCPD metadata, vertices, GT, ROI status/reason
  selection_rule.txt  config_used.json  ocr_settings.txt
  crops/{original,clahe,clahe_tophat,homomorphic}/Sxx.png
  ocr_results_per_image.csv  metrics_summary_ocr.csv
  metrics_aux_per_image.csv  metrics_summary_aux.csv
  ocr_calibration_grid.csv  ocr_calibration_set.csv
  figures/  contact_sheet_roi.jpg, panels/Sxx.png, overview_*.png, representative_*.png,
            cer_by_group.png, luminance_histograms.png, sweep_*.png
```
