# Sudoku Solver - Computer Vision Project

A comprehensive computer vision solution for detecting, extracting, recognizing, and solving Sudoku puzzles from images using OpenCV and Python.

## Overview

This project implements an end-to-end pipeline that processes images of Sudoku puzzles and automatically solves them. It handles various image conditions including poor lighting, shadows, rotated grids, and different handwriting styles.

 **Features**  
   We apply grayscale conversion, CLAHE enhancement, Gaussian blur, adaptive thresholding, and morphological closing.  
   This makes the grid lines and numbers clearer and reduces noise so contour detection becomes more reliable.
   
- **Robust Grid Detection**: Detects Sudoku grids even in challenging conditions (shadows, rotation, broken lines)
- **Smart Preprocessing**: Uses CLAHE and adaptive thresholding to handle uneven lighting
- **Perspective Correction**: Automatically warps skewed grids to perfect squares
- **Advanced Digit Recognition**: Multi-template matching with contour analysis for accurate digit identification
- **Puzzle Validation**: Checks for duplicates and invalid configurations
- **Backtracking Solver**: Solves valid Sudoku puzzles automatically
- **Visual Debugging**: Comprehensive visualization at each pipeline stage

## Requirements
```python
opencv-python (cv2)
numpy
matplotlib
```

Install dependencies:
```bash
pip install opencv-python numpy matplotlib
```

## Project Structure

The pipeline consists of five main stages:

### 1. Image Preprocessing
- Converts to grayscale
- Applies CLAHE (Contrast Limited Adaptive Histogram Equalization) for lighting normalization
- Uses Gaussian blur to reduce noise
- Applies adaptive thresholding to handle shadows
- Morphological operations to connect broken lines

### 2. Grid Detection and Extraction
- **Two-stage detection strategy**:
  - Standard approach for well-lit images
  - Dark fallback with dilation for poorly lit images
- **Smart contour finding**:
  - Attempts to find perfect square loops
  - Uses "rubber band" technique (convex hull) for broken borders
- **Perspective transformation**: Warps detected grid to perfect square dimensions

### 3. Cell Extraction
- Divides warped grid into 81 individual cells (9x9)
- Uses dynamic margins (8% of cell size) to remove grid lines
- Preserves digit content while cutting away borders

### 4. Digit Recognition
- **Empty cell detection**: Uses connected component analysis
- **Digit extraction**: Finds and crops the main digit contour with intelligent filtering
- **Multi-template matching**: 54 templates (6 variations per digit 1-9)
  - Thin, medium, and bold variants
  - SIMPLEX and DUPLEX font styles
- **Contour-based matching**: Combines template matching with shape analysis
  - Area ratio, perimeter, aspect ratio
  - Extent, solidity, number of contours (holes)
  - Hu Moments for rotation-invariant shape descriptors
- **Confidence scoring**: Weighted combination of multiple metrics
- **Special handling**: Distinguishes between similar digits (1 vs 2)

### 5. Puzzle Solving
- **Validation**: Checks for duplicates in rows, columns, and 3x3 boxes
- **Backtracking algorithm**: Recursively solves valid puzzles
- **Result visualization**: Shows original, recognized, and solved grids side-by-side

## Usage

### Basic Usage
```python
# Process a single image
recognized, solved, confidence, status = process_sudoku_image_updated(
    "01.jpg", 
    multi_templates
)

if status == "Success":
    print("Recognized Grid:")
    print(recognized)
    print("\nSolved Grid:")
    print(solved)
```

### Batch Processing
```python
# Process multiple images
image_files = [f"{i:02d}.jpg" for i in range(1, 17)]

for filename in image_files:
    recognized, solved, confidence, status = process_sudoku_image_updated(
        filename, 
        multi_templates
    )
    print(f"{filename}: {status}")
```

### Visualization
```python
# Visualize results
img = cv2.imread("01.jpg")
visualize_results(img, recognized, solved, confidence)
```

## Key Functions

### Grid Detection
- `robust_preprocess(image_path)`: Preprocesses image for analysis
- `get_sudoku_grid(img)`: Detects grid using two-stage strategy
- `find_grid_contour(processed_img, img_area)`: Finds grid boundary
- `four_point_transform(image, pts)`: Warps grid to square

### Cell Processing
- `extract_cells_from_grid(warped_img)`: Extracts 9x9 cell array
- `preprocess_cell(cell)`: Prepares cell for digit recognition
- `is_cell_empty(cell)`: Detects empty cells

### Digit Recognition
- `extract_digit_from_cell(cell)`: Extracts and normalizes digit
- `create_multiple_templates_per_digit()`: Creates 54 templates
- `match_digit_with_contour_analysis(digit_img, templates)`: Recognizes digit with confidence score

### Puzzle Solving
- `is_valid(grid, row, col, num)`: Validates number placement
- `solve_sudoku(grid)`: Solves puzzle using backtracking
- `validate_sudoku(grid)`: Checks grid validity

## Algorithm Details

### Grid Detection Strategy

1. **Standard Method**: Works for 15/16 test images
   - Simple Gaussian blur + adaptive threshold
   - Finds largest square contour

2. **Dark Fallback**: Handles poorly lit images
   - CLAHE for brightness enhancement
   - Dilation to connect faint lines
   - Higher block size for adaptive threshold

3. **Rubber Band Technique**: Handles broken borders
   - Collects significant contour points
   - Constructs convex hull
   - Approximates hull to square

### Digit Recognition Approach

**Multi-Metric Scoring** (weights):
- 35% - Max correlation coefficient
- 15% - Average of top 3 correlation scores
- 10% - Max cross-correlation
- 10% - Max inverted squared difference
- 20% - Max contour similarity
- 10% - Average contour similarity

**Contour Features**:
- Area ratio, perimeter, aspect ratio
- Extent (contour/bounding box ratio)
- Solidity (contour/convex hull ratio)
- Number of contours (hole detection)
- Hu Moments (first 3 invariants)

**Special Cases**:
- Rejects digits with combined score < 0.38
- Prefers digit 1 over 2 when score difference is 0-0.1

## Performance

The system successfully handles:
- ✅ Various lighting conditions
- ✅ Rotated and skewed grids
- ✅ Broken or dashed grid lines
- ✅ Different handwriting styles
- ✅ Shadows and reflections
- ✅ Thin digits (like "1")

**Confidence Thresholds**:
- High confidence: ≥ 0.60 (dark green)
- Good confidence: 0.50 - 0.59 (green)
- Low confidence: 0.38 - 0.49 (orange)
- Rejected: < 0.38 (red)
