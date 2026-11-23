# Integrating Module 2 into the group pipeline

```
Module 1 (Gül Berra)          Module 2 (Elif)                 Module 3 (Beyza)            Module 4 (Hakan)
perspective-corrected  ──►  illumination correction  ──►  Sauvola binarization  ──►  projection + CCA
plate crop (BGR)            same size, same orientation      + morphological opening      character boxes
```

## Contract
| | Module 2 input (from Module 1) | Module 2 output (to Module 3) |
|---|---|---|
| Type | `uint8` BGR (gray is accepted and converted) | `uint8` BGR, **same height/width** as the input |
| Geometry | rectified, front-facing plate | unchanged: no resize, crop or rotation |
| Files | any `.png/.jpg/.bmp` in one folder | `.png` (lossless) with the same base name, plus `module2_log.csv` |

The parameters were tuned on 440×140 crops (the 440 mm × 140 mm plate aspect). If Module 1 produces a different size, resizing to 440×140 before Module 2 keeps the CLAHE tile size and the homomorphic cut-off at the tested scale.

## Python API (for the group notebook)
```python
import sys; sys.path.insert(0, "path/to/elif_illumination_project/src")
import module2 as m2

corrected, diag = m2.correct_illumination(plate_bgr, method="clahe")
# methods: "original", "global_he", "clahe", "clahe_tophat", "homomorphic"
```

## Command line (folder → folder)
```bash
.venv/bin/python src/module2_cli.py --input-dir <module1_crops> --method clahe --out-dir <module2_out_clahe>
.venv/bin/python src/module2_cli.py --input-dir <module1_crops> --method homomorphic --out-dir <module2_out_homomorphic>
```
The CLI was verified to produce byte-identical output to the experiment pipeline (`results_v2/crops/clahe`, `results_v2/crops/homomorphic`).

## Which method to pass to Module 3 (based on `results_v2`, n = 50)
| Method | Recommendation | Evidence |
|---|---|---|
| **CLAHE** | **default** | Best overall CER (0.606 vs 0.620). Never clearly worse than the original in any subset. |
| Homomorphic | worth testing for dark / unevenly lit plates | Lowest CER on dark plates (0.500 vs 0.557) and in the exploratory uneven stratum (0.463 vs 0.526). Worse on bright plates. |
| Global HE | do not use | Significantly worse: 0.754, Δ 95 % CI [+0.071, +0.197]. |
| CLAHE + Top/Black-Hat | do not use before Sauvola | Worse (0.663). About 18 % of pixels clip and halos appear around strokes, which would turn into false foreground. |

Module 3 should ideally be evaluated on both CLAHE and Homomorphic output, because OCR CER measures the whole chain and not binarization quality alone.

## Still to do together with the group
- Re-run `run_pipeline.py` with Module 1's real crops: put them in place of `crops/original/` in a new output folder.
- Run Module 3 on `module2_out_clahe/` and report its binarization results per Module 2 method.
- Merge into the group `.ipynb` and fill the full contribution table.
