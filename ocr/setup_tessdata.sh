#!/usr/bin/env bash
# Fetch the Tesseract language data used in this project into ocr/tessdata.
# chi_sim: tesseract-ocr/tessdata_fast (main); eng/osd: copied from the Homebrew tesseract install.
set -euo pipefail
cd "$(dirname "$0")/tessdata"
[ -f chi_sim.traineddata ] || curl -sSL -o chi_sim.traineddata https://github.com/tesseract-ocr/tessdata_fast/raw/main/chi_sim.traineddata
for l in eng osd; do [ -f $l.traineddata ] || cp "$(brew --prefix)/share/tessdata/$l.traineddata" .; done
shasum -a 256 -c ../tessdata.sha256 || echo "WARNING: checksum differs from the version used in the reported run"
