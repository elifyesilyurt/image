"""Builds elif_module2_illumination.ipynb (run once; then execute with nbconvert)."""
import os
import sys

import nbformat as nbf

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
out = os.path.join(PROJECT, "elif_module2_illumination.ipynb")
if os.path.exists(out):
    sys.exit(f"refusing to overwrite {out}")

nb = nbf.v4.new_notebook()
C = []
md = lambda s: C.append(nbf.v4.new_markdown_cell(s))  # noqa: E731
code = lambda s: C.append(nbf.v4.new_code_cell(s))    # noqa: E731

md("""# Module 2 – Illumination Correction (Elif Yeşilyurt, 240610008)

**Group project:** Image Enhancement in Licence Plate Recognition Systems (guideline topic 4).
**My module:** Correct uneven lighting (flash glare, shadows, darkness) on licence-plate crops before binarization (Module 3).

| Item | Content |
|---|---|
| Input | 50 perspective-rectified plate crops (440×140 px) from CCPD-DB |
| Methods compared (same crops) | Original (baseline) · CLAHE on LAB-L · CLAHE + Top-Hat/Black-Hat (report pipeline) · Homomorphic filtering on LAB-L |
| Expected output | Illumination-corrected crop of identical size/orientation |
| Primary metric | OCR Character Error Rate (Tesseract 5.5.4) vs. CCPD ground-truth plate text |
| Auxiliary measures | luminance mean/std, local contrast, illumination non-uniformity, saturation, chroma shift (not OCR evidence) |

**Data source:** CCPD (Chinese City Parking Dataset), Xu et al., ECCV 2018, pp. 255–271, https://github.com/detectRecog/CCPD — **MIT License** (Copyright (c) 2017 CCPD). Subset: `ccpd_db` (from `CCPD2019.tar.xz`).

> The crops are produced here from CCPD corner annotations as a **temporary** rectification. They are *not* Module 1's final output (Gül Berra Aslan). If Module 1 crops become available they should replace `results/crops/original/`.

This notebook **reads the results produced by `src/run_pipeline.py`** and re-demonstrates the methods on one crop. The README gives the command to regenerate everything.""")

code("""import os, sys, json
import cv2, numpy as np, pandas as pd
import matplotlib.pyplot as plt
from IPython.display import Image, display
PROJECT = os.path.abspath('.')
sys.path.insert(0, os.path.join(PROJECT, 'src'))
import module2 as m2
R = os.path.join(PROJECT, 'results')
plt.rcParams['font.family'] = ['Arial Unicode MS', 'Hiragino Sans GB', 'DejaVu Sans']
pd.set_option('display.width', 200)
print('OpenCV', cv2.__version__, '| numpy', np.__version__)""")

md("## 1. Central configuration (exact values used in the reported run)")
code("""cfg = json.load(open(os.path.join(R, 'config_used.json')))
for k in ['crop', 'clahe', 'tophat', 'homomorphic', 'ocr']:
    print(k, '->', cfg[k])""")

md("""## 2. Data: selection and ROI validation
* 30 candidates from `elif_ccpd_db_selection` (seed 42; 10 dark / 10 mid / 10 bright by **CCPD filename brightness metadata**).
* +20 seeded extension (seed 2026) to reach the group plan of **50 difficult-lighting images**.
* **Caveat:** in CCPD-DB the brightness metadata is bimodal (5–60 or 180–242, no values in between), so the 'mid' group (22–39) is the upper end of the dark mode, not a true middle level.
* Filename parsed exactly per the official README: area – tilt – bbox – 4 vertices (RB, LB, LT, RT) – plate indices – brightness – blurriness.""")
code("""man = pd.read_csv(os.path.join(R, 'manifest_roi.csv'), keep_default_na=False)
print(open(os.path.join(R, 'selection_rule.txt')).read())
display(man.groupby(['group', 'roi_status']).size().rename('n').reset_index())
display(man[man.roi_status != 'ok'][['id', 'group', 'plate_w_px', 'plate_h_px', 'roi_status', 'roi_reason']])""")
code("display(Image(os.path.join(R, 'figures/contact_sheet_roi.jpg'), width=1100))")

md("""## 3. Methods – live demonstration on one crop
* **CLAHE**: BGR→LAB, CLAHE on L only (a, b untouched → no hue change), clip limit 2.0, 8×2 tiles (≈55×70 px, larger than one character).
* **Top-Hat / Black-Hat** (as written in our report): L' = L + TopHat(L) − BlackHat(L) after CLAHE; elliptical SE of size 2× the stroke width estimated per crop with a distance transform.
* **Homomorphic**: f = i·r ⇒ ln f = ln i + ln r. log1p(L) → reflect-pad → FFT → Gaussian high-emphasis filter H = (γH−γL)[1−exp(−c·D²/D0²)] + γL with γL = 0.5 (illumination attenuated), γH = 1.5 (reflectance/edges boosted), D0 = 10 → inverse FFT → expm1 → robust 0.5–99.5 % normalisation. Applied to L only.""")
code("""crop = cv2.imread(os.path.join(R, 'crops/original/S06.png'))
variants = {'Original': crop, 'CLAHE': m2.apply_clahe(crop),
            'CLAHE + Top/Black-Hat': m2.apply_clahe_tophat(crop)[0], 'Homomorphic': m2.apply_homomorphic(crop)[0]}
fig, ax = plt.subplots(2, 4, figsize=(18, 5))
for j, (k, v) in enumerate(variants.items()):
    ax[0, j].imshow(cv2.cvtColor(v, cv2.COLOR_BGR2RGB)); ax[0, j].set_title(k); ax[0, j].axis('off')
    ax[1, j].hist(m2.luminance(v).ravel(), 64, (0, 255), color='k'); ax[1, j].set_title('L histogram')
plt.tight_layout(); plt.show()
# reproducibility check against the saved run
for k, f in [('CLAHE', 'clahe'), ('Homomorphic', 'homomorphic')]:
    saved = cv2.imread(os.path.join(R, f'crops/{f}/S06.png'))
    print(k, 'identical to saved result:', np.array_equal(saved, variants[k]))""")

md("""## 4. Parameter choice
Parameters were fixed **before** OCR was run, by visual inspection of these sweeps on 3 crops (S06 dark, S17 mid, S23 bright). Larger clip limits / γH visibly amplify JPEG 8×8 blocks in dark crops; the chosen values are the moderate settings.""")
code("""display(Image(os.path.join(R, 'figures/sweep_clahe.png'), width=1100))
display(Image(os.path.join(R, 'figures/sweep_homomorphic.png'), width=1100))""")

md("""## 5. Evaluation – OCR Character Error Rate
* Engine: **Tesseract 5.5.4**, `chi_sim` (tessdata_fast), `--oem 1 --psm 8`, character whitelist (31 province characters + 24 letters + 10 digits), grayscale input. Identical for all methods.
* OCR settings were selected on a **separate calibration set** of 20 CCPD-DB images (seed 7, disjoint from the 50 test images) using only unprocessed crops (grid in `ocr_calibration_grid.csv`). Four settings tied at CER 0.507; tie broken by zero empty outputs and native resolution.
* Ground truth: plate indices in the CCPD filename decoded with the official `provinces` / `alphabets` / `ads` tables.
* Normalisation: upper-case; keep only CJK ideographs, A–Z, 0–9 (drops spaces and the '·' separator).
* **CER = Σ Levenshtein(OCR, GT) / Σ |GT|** over all images of the subset. `CER_alnum_6` ignores the Chinese province character.
* 95 % CI of ΔCER (method − original): paired bootstrap over images, 5000 resamples.""")
code("""print(open(os.path.join(R, 'ocr_settings.txt')).read())
summ = pd.read_csv(os.path.join(R, 'metrics_summary_ocr.csv'), keep_default_na=False)
display(summ)""")
code("display(Image(os.path.join(R, 'figures/cer_by_group.png'), width=900))")
code("""ocr = pd.read_csv(os.path.join(R, 'ocr_results_per_image.csv'), keep_default_na=False)
piv = ocr.pivot(index='id', columns='method', values='edit_dist')[['original', 'clahe', 'clahe_tophat', 'homomorphic']]
piv['group'] = ocr.drop_duplicates('id').set_index('id').group
display(piv)""")

md("## 6. Auxiliary image measures (descriptive only – not evidence of OCR performance)")
code("""aux = pd.read_csv(os.path.join(R, 'metrics_aux_per_image.csv'))
cols = ['L_mean', 'local_contrast', 'illum_cv', 'sat_hi_frac', 'sat_lo_frac', 'chroma_shift']
display(aux.groupby(['group', 'method'])[cols].mean().round(3))
print('Top/Black-Hat: mean stroke estimate %.1f px, mean clipped fraction %.1f %%' %
      (aux.diag_stroke_px.mean(), 100 * aux.diag_tophat_clip_frac.mean()))
print('Homomorphic NaN/Inf pixels in total:', int(aux.diag_nan_or_inf.sum()))""")
code("display(Image(os.path.join(R, 'figures/luminance_histograms.png'), width=1100))")

md("""## 7. Successful, unchanged and failed examples
Selection rule: largest per-image change of OCR edit distance vs. the original (per method / group), plus cases where all variants are identical. Observations in `review/visual_review.csv` are based only on what is visible in the panels.""")
code("""rv = pd.read_csv(os.path.join(PROJECT, 'review/visual_review.csv'))
display(rv)
for cat in ['success', 'unchanged', 'failure']:
    display(Image(os.path.join(R, f'figures/representative_{cat}.png'), width=1100))""")

md("""## 8. Findings and limitations
* **Overall (n = 50)** no method gives a statistically clear OCR improvement: CLAHE 0.606 and Homomorphic 0.614 vs. original 0.620 (both CIs include 0). CLAHE + Top/Black-Hat is worse (0.663, CI [−0.003, +0.091]).
* **Dark plates (n = 20):** Homomorphic has the lowest CER (0.500 vs 0.557) and strongly reduces illumination non-uniformity (illum_cv 0.374 → 0.146).
* **Bright/glare plates (n = 20):** nothing helps; saturated (clipped) characters cannot be recovered and Homomorphic/Top-Hat add haze or halos.
* **Main failure causes (visual):** saturation from glare, JPEG 8×8 block amplification in very dark crops, halos from Top/Black-Hat, low resolution (S02/S04 upsampled 3×).
* **OCR limitation:** Tesseract is a general document OCR; it almost never reads the Chinese province character and has 0/50 exact matches. CER differences are therefore small and partly reflect OCR behaviour, not only image quality.
* Crops come from CCPD annotations (temporary rectification), not Module 1. Small sample (50), single dataset (CCPD-DB), parameters chosen visually on 3 crops of the test set.
* No deep-learning model was used in Module 2.""")

md("""## 9. Re-running
```bash
.venv/bin/python src/run_pipeline.py --out results_rerun --stages roi,enhance,sweep,ocr,metrics,figures,calibrate
```
The runner refuses to overwrite existing files, so always use a new `--out` folder.""")

nb["cells"] = C
nb["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nbf.write(nb, out)
print("wrote", out)
