# Module 2 – Illumination Correction (Elif Yeşilyurt)

Part of the group project **"Image Enhancement in Licence Plate Recognition Systems"** (Image Processing course, guideline topic 4).
Group: Gül Berra Aslan (M1 Perspective Correction) · **Elif Yeşilyurt (M2 Illumination Correction)** · Beyza Şahin (M3 Binarization & Noise Removal) · Hakan Aksoy (M4 Character Segmentation).

This repository contains **only Module 2**. It applies CLAHE and Homomorphic Filtering, plus the Top-Hat/Black-Hat step from our method report and a global histogram-equalisation baseline, to the same licence-plate crops. The methods are compared with OCR Character Error Rate.

* Guideline compliance checklist (TR): [`YONERGE_UYUMU.md`](YONERGE_UYUMU.md)
* Group integration (Module 1 → 2 → 3): [`INTEGRATION.md`](INTEGRATION.md)
* Presentation material (EN): [`presentation/elif_presentation_EN.md`](presentation/elif_presentation_EN.md)

## Problem, input, output, evaluation
| | |
|---|---|
| Problem | Uneven plate illumination (night darkness, flash/sun glare, shadows) lowers character/background contrast before binarization and OCR. |
| Input | 50 rectified plate crops, 440×140 px, from CCPD-DB: 20 dark, 10 dark-upper ("mid") and 20 bright, by CCPD brightness metadata. |
| Output | Illumination-corrected crop of identical size, ROI and orientation. |
| Methods | (1) original baseline, (2) **global HE** on LAB-L, (3) **CLAHE** on LAB-L, (4) CLAHE + **Top-Hat/Black-Hat** on L, (5) **Homomorphic filtering** on LAB-L. |
| Primary metric | OCR CER = Σ Levenshtein(OCR, GT) / Σ GT characters (Tesseract 5.5.4, same settings for all methods), with a paired-bootstrap 95 % CI of ΔCER. |
| Auxiliary | luminance mean/std, local contrast, illumination non-uniformity (`illum_cv` = std/mean of the low-pass L field), saturation fractions, chroma shift. Descriptive only. |

## Data source, licence, citation
* **CCPD – Chinese City Parking Dataset**: https://github.com/detectRecog/CCPD. Licence: **MIT License** (Copyright (c) 2017 CCPD).
* Subset `ccpd_db` from `CCPD2019.tar.xz` (10,132 images locally). The dataset is **not** included here. Only the 50 rectified plate crops and figures derived from it are included.
* Xu, Z., Yang, W., Meng, A., Lu, N., & Huang, H. (2018). Towards End-to-End License Plate Detection and Recognition: A Large Dataset and Baseline. *ECCV*, 255–271.
* Filename annotations are parsed as in the official README: `area-tiltH_tiltV-bbox-4 vertices (RB, LB, LT, RT)-plate indices-brightness-blurriness`. Plate text is decoded with the official `provinces`, `alphabets` and `ads` tables.

## Image selection
* 30 candidates (seed 42; 10 dark / 10 mid / 10 bright by CCPD brightness metadata), plus a 20-image extension to reach the group plan of **50 difficult-lighting images**: `random.Random(2026)`, valid ROI, plate width ≥ 150 px, 10 with brightness ≤ 21 and 10 with brightness ≥ 204.
* **"mid" is not a medium level.** CCPD-DB brightness metadata is bimodal (5–60 or 180–242, with no images in between). "mid" (22–39) is the upper end of the dark mode and is labelled **dark-upper** in all figures.
* ROI status: 48 `ok`, 2 `review_required` (S02, S04: plates of about 131×46 px upsampled ≈3.3×), 0 `invalid_roi`. **No image was excluded.**
* **Temporary rectification**: crops are warped from the CCPD corner annotation (`cv2.getPerspectiveTransform`, INTER_CUBIC). This is **not** Module 1's final output; see [`INTEGRATION.md`](INTEGRATION.md).

## Methods and parameters (central config `src/module2.py → CONFIG`, saved in `results_v2/config_used.json`)
| Method | Parameters | Rationale |
|---|---|---|
| Global HE | `cv2.equalizeHist` on LAB-L | Global counterpart of CLAHE, for the report's global-vs-adaptive comparison. |
| CLAHE [1, 2] | LAB-L only; clip 2.0; tiles 8×2 (≈55×70 px) | Tiles are larger than one character. Clip 4 amplified JPEG blocks in the sweep. |
| Top/Black-Hat [3] | L' = L + TH − BH after CLAHE; ellipse SE = 2 × stroke width (estimated per crop with a distance transform, mean 8.7 px) | Report: "structuring element proportional to the estimated character thickness". |
| Homomorphic [4, 5] | L only; log1p → reflect pad → FFT → Gaussian high-emphasis H = (γH−γL)(1−e^{−cD²/D0²})+γL, γL 0.5, γH 1.5, c 1, D0 10 → expm1 → 0.5–99.5 % stretch | f = i·r ⇒ ln f = ln i + ln r. γL < 1 attenuates illumination, γH > 1 boosts reflectance/edges. |
| OCR [6] | Tesseract 5.5.4, `chi_sim` (tessdata_fast), `--oem 1 --psm 8`, whitelist, grayscale | Chosen on a **separate** 20-image calibration set (seed 7) using only unprocessed crops. |

Image parameters were fixed by visual inspection of `figures/sweep_*.png` (3 crops) **before** any OCR was run. Only the LAB-L channel is modified, so a/b (hue) are never changed. Checks: NaN/Inf count (0), output shape/dtype asserts, clipping fraction, chroma shift. CER uses Levenshtein distance [7]; confidence intervals come from a paired bootstrap over images with 5,000 resamples [8].

## Results (`results_v2/metrics_summary_ocr.csv`, n = 50)
| Subset | Original | Global HE | CLAHE | CLAHE+Top/Black-Hat | Homomorphic |
|---|---|---|---|---|---|
| all (50) | 0.620 | 0.754 | **0.606** | 0.663 | 0.614 |
| ΔCER vs original, 95 % CI | – | **[+0.071, +0.197]** | [−0.049, +0.017] | [−0.003, +0.091] | [−0.054, +0.046] |
| dark (20) | 0.557 | 0.771 | 0.543 | 0.586 | **0.500** |
| dark-upper "mid" (10) | 0.457 | 0.600 | **0.414** | 0.457 | 0.457 |
| bright (20) | **0.764** | 0.814 | **0.764** | 0.843 | 0.807 |
| images better / same / worse | – | 6 / 19 / 25 | 9 / 33 / 8 | 10 / 23 / 17 | 14 / 25 / 11 |

**Exploratory strata.** These were measured on the original crop but defined *after* the first results were seen (`results_v2/illumination_strata.txt`, `figures/cer_by_illumination_exploratory.png`):
| Stratum | Original | Global HE | CLAHE | Top/Black-Hat | Homomorphic |
|---|---|---|---|---|---|
| uneven (illum_cv > 0.149, n=25) | 0.526 | 0.703 | 0.497 | 0.531 | **0.463** (Δ CI [−0.126, −0.006]) |
| uniform (n=25) | **0.714** | 0.806 | **0.714** | 0.794 | 0.766 |
| partial glare (5–40 % saturated, n=15) | 0.790 | 0.790 | **0.752** | 0.819 | 0.762 |

All 25 "uneven" crops come from the dark/dark-upper groups and all 15 "partial glare" crops from the bright group. Unevenness therefore cannot be separated from darkness in this sample.

**Findings**
1. **Adaptive beats global.** Global HE is the only method that is significantly worse than doing nothing. CLAHE, its local and contrast-limited version, avoids that degradation (0.606).
2. **No method gives a statistically clear improvement over the original overall.** CLAHE is the safest choice: best overall and never clearly worse.
3. **Homomorphic helps where illumination is uneven or dark** (0.463 vs 0.526 in the exploratory uneven stratum), which is exactly the case it is designed for. It hurts bright, uniformly lit plates.
4. **Top/Black-Hat** (from our report) consistently makes OCR worse. Its halos and about 18 % clipping are not useful before OCR.
5. 0/50 exact plate matches: Tesseract almost never reads the Chinese province character.

### Failure causes (visual evidence, `review/visual_review.csv`)
1. **Saturation/glare** (S45, S49, S28): clipped pixels carry no information, and no illumination method recovers them.
2. **JPEG 8×8 block amplification** in very dark crops (S10, S14). Global HE (S06, S39) is the most extreme case.
3. **Top/Black-Hat halos** (S26).
4. **Homomorphic haze** on bright, uniform plates (S26).
5. **Low resolution** (S02, S04).

## Limitations
* Temporary rectification from annotations, not the group's Module 1 output.
* Small sample (50), one subset (CCPD-DB), groups based on CCPD metadata, and illumination strata confounded with brightness.
* Tesseract is a general document OCR, not a plate recogniser. CER is high for every method.
* Image parameters were chosen visually on 3 crops that belong to the test set. OCR parameters were chosen on a disjoint calibration set. The exploratory strata are post hoc.
* Character segmentation accuracy (CSAcc/IoU) belongs to Module 4. No deep-learning model was used.

## Environment and how to run
Tested on macOS (Apple M4), Python 3.9.6, OpenCV 5.1.0, NumPy 2.0.2, matplotlib 3.9.4, Tesseract 5.5.4 (Homebrew), pytesseract 0.3.13 (`requirements-lock.txt`).
```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
brew install tesseract && bash ocr/setup_tessdata.sh
export CCPD_DB_DIR=/path/to/CCPD2019/ccpd_db
export ELIF_SELECTION_DIR=/path/to/elif_ccpd_db_selection      # contains manifest.csv
.venv/bin/python src/run_pipeline.py --out results_rerun --stages roi,enhance,sweep,ocr,metrics,figures,calibrate   # ~2.5 min, never overwrites
.venv/bin/jupyter notebook elif_module2_illumination.ipynb
```

## Repository layout
```
elif_module2_illumination.ipynb   main deliverable (executed, reads results_v2/)
src/module2.py                    config, CCPD parser, methods, correct_illumination() API, metrics, OCR/CER
src/run_pipeline.py               stages: roi, enhance, sweep, ocr, metrics, figures, calibrate
src/module2_cli.py                folder -> folder interface for the group pipeline
src/build_notebook.py             generates the notebook
review/visual_review.csv          representative examples + visual observations
presentation/                     English slides/script/Q&A (+ earlier Turkish draft)
results_v2/                       FINAL run: 5 methods, exploratory strata, all figures and CSVs
results/                          first run (4 methods) kept for traceability; identical to results_v2 for the shared methods (verified pixel- and OCR-identical)
```

## References
1. Pizer, S. M., Amburn, E. P., Austin, J. D., et al. (1987). Adaptive histogram equalization and its variations. *Computer Vision, Graphics, and Image Processing*, 39(3), 355–368.
2. Zuiderveld, K. (1994). Contrast limited adaptive histogram equalization. In P. S. Heckbert (Ed.), *Graphics Gems IV* (pp. 474–485). Academic Press.
3. Serra, J. (1982). *Image Analysis and Mathematical Morphology*. Academic Press.
4. Oppenheim, A. V., Schafer, R. W., & Stockham, T. G. (1968). Nonlinear filtering of multiplied and convolved signals. *Proceedings of the IEEE*, 56(8), 1264–1291.
5. Gonzalez, R. C., & Woods, R. E. (2018). *Digital Image Processing* (4th ed.). Pearson. (Histogram processing, homomorphic filtering.)
6. Smith, R. (2007). An overview of the Tesseract OCR engine. *Proc. ICDAR 2007*, 629–633.
7. Levenshtein, V. I. (1966). Binary codes capable of correcting deletions, insertions, and reversals. *Soviet Physics Doklady*, 10(8), 707–710.
8. Efron, B., & Tibshirani, R. J. (1993). *An Introduction to the Bootstrap*. Chapman & Hall.
