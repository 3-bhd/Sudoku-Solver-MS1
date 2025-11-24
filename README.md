# Sudoku Grid Extraction – Milestone 1

This repository contains our work for Milestone 1 of the Computer Vision project.  
The goal of this phase is to take a real Sudoku image, clean it, detect the outer frame, identify the four corners, and finally straighten the grid into a usable square that will later be passed to OCR.

## Overview

Our pipeline goes through three main steps:

1. **Preprocessing**  
   We apply grayscale conversion, CLAHE enhancement, Gaussian blur, adaptive thresholding, and morphological closing.  
   This makes the grid lines and numbers clearer and reduces noise so contour detection becomes more reliable.

2. **Grid Detection and Corner Extraction**  
   We search the binary image for contours and try to approximate the Sudoku frame as a four-point polygon.  
   If the standard thresholding fails, we use a fallback approach based on stronger enhancement, dilation, and convex-hull grouping of significant contour points.  
   This allows us to handle difficult lighting conditions or weak grid lines.

3. **Perspective Warping**  
   After obtaining the four corners, we order them and apply a perspective transform.  
   The output is a square, top-down view of the Sudoku grid, ready for OCR in the next milestone.

## Code Structure
main_milestone1.py # Visualization pipeline

preprocessing.py # Preprocessing steps

grid_detection.py # Contour extraction + fallback method

warping.py # Perspective transform

Project Test Cases-1/ # Input test images

outputs/ # Output of the test images

## What the Script Shows

Running the main file displays three stages for all test images:

- the **original image**  
- the **preprocessed binary image**  
- the **warped, straightened grid** (with the method used: Standard or Fallback)

This makes it easy to verify performance across different lighting and image qualities.

## Notes

The current implementation meets all requirements for Milestone 1:  
preprocessing, outer frame isolation, corner detection, and grid straightening.

Milestone 2 will build on this by adding digit extraction and a basic OCR approach using pattern matching.
