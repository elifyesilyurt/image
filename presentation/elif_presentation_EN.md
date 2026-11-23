# Elif Yeşilyurt – Module 2: Illumination Correction (presentation material, EN)

* **Slides (7):** https://claude.ai/artifact/FLaJMMf9tFN4gpaCeVch6o (private until shared from its Share menu; export to PPTX/PDF from there). Source copy: `presentation/slides_source/`.
* **Speaking time:** about 2 minutes (296 words at ~140 wpm), which leaves room for the other three members in the 10-minute group video.
* All numbers come from `results_v2/` (n = 50).

## Contribution-table row (for the group slide)
| Student | Assigned task | Outputs |
|---|---|---|
| Elif Yeşilyurt (240610008) | Module 2 – Illumination Correction | Global HE, CLAHE, CLAHE+Top/Black-Hat and Homomorphic filtering on the same 50 CCPD-DB plate crops; OCR-CER evaluation with bootstrap CIs; success/failure analysis; `correct_illumination()` API for Module 3; notebook + README |

## Script (one paragraph per slide; also in each slide's speaker notes)
1. **Cover / pipeline.** Hi, I’m Elif. I did Module 2, illumination correction. It takes the straightened plate from Module 1 and hands Module 3 a plate whose lighting no longer hides the characters.
2. **Problem & data.** At night plates are too dark; with glare they are washed out. I used 50 MIT-licensed crops from CCPD-DB. Every method gets the same 440 by 140 crop, and I measure Tesseract’s character error rate with identical settings.
3. **Methods.** All methods change only the lightness channel, so colours stay. Global equalisation uses one mapping for the whole plate; CLAHE equalises small tiles with a clip limit. Top-Hat and Black-Hat come from our report. Homomorphic filtering takes a logarithm, so illumination and reflectance add up; then, in the Fourier domain, it damps the slow illumination and boosts the edges.
4. **Results.** Global equalisation is the only method significantly worse than doing nothing: the error rate rises from 0.62 to 0.75. CLAHE, its adaptive version, avoids this and is best overall, though not significantly. On dark plates, homomorphic filtering is best at 0.50.
5. **Uneven light (exploratory).** Our module targets uneven light, so I split the plates by how uneven they are. On the 25 most uneven, homomorphic filtering lowers the error from 0.53 to 0.46, and the interval excludes zero. Two caveats: I defined this split after the first results, and these plates are also dark.
6. **Failure cases.** Four failure causes, all visible here. Glare clips pixels, so nothing can be recovered. In dark JPEGs, enhancement also amplifies the compression blocks. Top-Hat creates halos. And global equalisation over-stretches until the strokes merge.
7. **Conclusion.** So: CLAHE as the default for Module 3, homomorphic for dark or uneven plates, and avoid global equalisation and Top-Hat. Limits: 50 images, temporary crops, and an OCR engine not built for plates. Next, we rerun this with Module 1’s real crops. Thank you.

## Likely questions – my module
1. **What is the difference between CLAHE and homomorphic filtering?**
   CLAHE works in the spatial domain. It equalises the histogram of each tile, clips the histogram at a limit (2.0) so noise is not over-amplified, and blends the tiles bilinearly. Homomorphic filtering works in the frequency domain. It models the image as illumination × reflectance, takes the log so the two become a sum, and uses a Gaussian high-emphasis filter: low frequencies × 0.5 (illumination) and high frequencies × 1.5 (edges). In short, CLAHE redistributes contrast locally, while homomorphic filtering separates out the lighting.
2. **Why only the L channel of LAB?**
   L carries lightness, and a/b carry colour. Equalising R, G and B separately shifts hues. Our mean chroma shift was below 1 unit (of 255) for CLAHE and homomorphic filtering.
3. **Why is global HE so much worse than CLAHE?**
   It applies one mapping to the whole crop. A dark-dominated histogram is stretched across the full range, so the background turns flat grey, JPEG blocks become as bright as strokes and characters saturate (S06: CER 0.14 → 0.86). CLAHE's tiles and clip limit prevent this.
4. **How did you choose the parameters? Did you tune them on the results?**
   The image parameters were fixed by eye on a sweep over 3 crops before any OCR was run. The OCR settings were chosen on a separate 20-image calibration set, using unprocessed crops only. Nothing was tuned on CER.
5. **Is the improvement significant?**
   Over all 50 images, no: all CIs for CLAHE and homomorphic filtering include 0. Global HE is significantly worse (CI +0.071 … +0.197). Homomorphic filtering on the uneven stratum has a CI of −0.126 … −0.006, but that split is exploratory and overlaps with darkness.
6. **Why is CER high even for the best method?**
   Tesseract is a document OCR. It almost never reads the Chinese province character (0/50 exact matches), and some plates are saturated, so information was lost at capture.
7. **Why not use deep learning?**
   The guideline makes it optional, and our report listed it as an optional ablation. Module 2 is classical by design.
8. **What does Module 3 receive from you?**
   A crop of the same size and orientation, as a lossless PNG, via `correct_illumination(img, method)` or `module2_cli.py`. Recommended method: CLAHE, or homomorphic filtering for dark or uneven plates (`INTEGRATION.md`).

## Likely questions – the whole project (every member must be able to explain the main steps)
* **Module 1 – Perspective correction (Gül Berra):** filtering → Canny edges → Progressive Probabilistic Hough Transform to find the plate's boundary lines → intersections give 4 corners → homography maps them to a W×H rectangle. Why: a skewed plate breaks character isolation and OCR.
* **Module 2 – Illumination correction (me):** see above.
* **Module 3 – Binarization & noise removal (Beyza):** Sauvola adaptive thresholding, which sets a local threshold from the mean and standard deviation of each neighbourhood, so it survives leftover lighting changes. It is compared with global Otsu, and morphological opening then removes specks.
* **Module 4 – Character segmentation (Hakan):** a vertical projection profile (column sums, where gaps between characters are minima) plus connected-component analysis, filtering components by area, aspect ratio and position to drop frame edges and bolts.
* **Evaluation:** OCR CER (Tesseract) before and after enhancement. Character Segmentation Accuracy = correctly segmented / ground-truth characters, where a box counts as correct at IoU ≥ 0.75 on a manually labelled subset (Module 4).
* **Data:** CCPD (Xu et al., ECCV 2018, MIT licence). The group plan is about 350 images: 200 normal, 100 strongly angled and 50 difficult lighting (my 50).
* **Changes vs. the method plan (to mention briefly):** Module 2 used temporary crops from CCPD corner annotations until Module 1's output is ready. A global HE baseline was added to make the report's global-vs-adaptive comparison concrete. Top/Black-Hat was implemented as planned but is not recommended.
