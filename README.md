# Sudoku Solver — Computer Vision

End-to-end computer vision pipeline to detect, extract, and solve Sudoku puzzles from real-world images · 15/16 success rate

---

## Overview

This project implements a fully automated pipeline that processes real-world images of Sudoku puzzles and solves them. The system handles challenging conditions including poor lighting, shadows, rotated grids, broken grid lines, and different handwriting styles — achieving a 15/16 success rate across a diverse test set.

Built as part of a Computer Vision course at The American University in Cairo.

---

## Results

| Metric | Value |
|--------|-------|
| Test Images | 16 |
| Successful Solves | 15 |
| Success Rate | 93.75% |

**Handles:**
- ✅ Various lighting conditions and shadows
- ✅ Rotated and skewed grids
- ✅ Broken or dashed grid lines
- ✅ Different handwriting styles
- ✅ Thin digits (like "1")

---

## Tech Stack

| Layer | Technology |
|-------|------------|
| Language | Python |
| Vision | OpenCV (cv2) |
| Numerics | NumPy |
| Visualization | Matplotlib |

---

## Pipeline

The system runs five sequential stages:

### 1. Image Preprocessing
- Grayscale conversion
- CLAHE enhancement for lighting normalization
- Gaussian blur for noise reduction
- Adaptive thresholding for shadow handling
- Morphological operations to connect broken lines

### 2. Grid Detection
Two-stage detection strategy:
- **Standard:** Gaussian blur + adaptive threshold, finds largest square contour
- **Dark fallback:** CLAHE + dilation for poorly lit images
- **Rubber band technique:** Convex hull approximation for broken borders
- Perspective transformation to warp detected grid to a perfect square

### 3. Cell Extraction
- Divides warped grid into 81 individual cells (9×9)
- Dynamic margins (8% of cell size) to remove grid lines
- Preserves digit content while cutting away borders

### 4. Digit Recognition
- Empty cell detection via connected component analysis
- Multi-template matching: 54 templates (6 variations per digit 1–9)
- Contour-based matching combining template correlation with shape analysis
- Confidence scoring: weighted combination of 6 metrics
- Special handling to distinguish similar digits (e.g. 1 vs 2)

**Confidence thresholds:**
| Level | Score |
|-------|-------|
| High | ≥ 0.60 |
| Good | 0.50 – 0.59 |
| Low | 0.38 – 0.49 |
| Rejected | < 0.38 |

### 5. Puzzle Solving
- Validates grid for duplicate entries
- Backtracking algorithm to recursively solve valid puzzles
- Side-by-side visualization of original, recognized, and solved grids

---

## Installation
```bash
pip install opencv-python numpy matplotlib
```

---

## Usage

### Single Image
```python
recognized, solved, confidence, status = process_sudoku_image_updated(
    "01.jpg",
    multi_templates
)

if status == "Success":
    print("Recognized Grid:", recognized)
    print("Solved Grid:", solved)
```

### Batch Processing
```python
image_files = [f"{i:02d}.jpg" for i in range(1, 17)]

for filename in image_files:
    recognized, solved, confidence, status = process_sudoku_image_updated(
        filename, multi_templates
    )
    print(f"{filename}: {status}")
```

---

## Authors

- **Omar Abdelhady** — [@3-bhd](https://github.com/3-bhd)

The American University in Cairo — Computer Vision Course
