import cv2
from preprocessing import robust_preprocess
from grid_detection import get_sudoku_grid
from warping import four_point_transform
from cell_processing import extract_cells_from_grid, preprocess_cell, is_cell_empty
from ocr_templates import create_multiple_templates_per_digit
from ocr_matching import extract_digit_from_cell, match_digit_with_contour_analysis
from sudoku_solver import validate_sudoku, solve_sudoku
from visualization import visualize_results
import numpy as np
import os

templates = create_multiple_templates_per_digit()

def process(image_path):
    img = cv2.imread(image_path)

    contour, strategy = get_sudoku_grid(img)
    if contour is None:
        return None, None, None, "Grid detection failed"

    warped = four_point_transform(img, contour.reshape(4, 2))
    cells = extract_cells_from_grid(warped)

    grid = np.zeros((9, 9), dtype=int)
    conf_grid = np.zeros((9, 9), dtype=float)

    for r in range(9):
        for c in range(9):
            cell = preprocess_cell(cells[r][c])

            if is_cell_empty(cell):
                continue

            digit_img = extract_digit_from_cell(cell)
            if digit_img is not None:
                digit, conf, _ = match_digit_with_contour_analysis(
                    digit_img, templates, os.path.basename(image_path), (r, c)
                )
                grid[r][c] = digit
                conf_grid[r][c] = conf

    ok, msg = validate_sudoku(grid)
    if not ok:
        return grid, None, conf_grid, msg

    solved = grid.copy()
    if solve_sudoku(solved):
        return grid, solved, conf_grid, "Success"
    else:
        return grid, None, conf_grid, "Unsolvable"

if __name__ == "__main__":
    for i in range(1, 17):
        name = f"images/{i:02d}.jpg"
        r, s, c, status = process(name)
        print(name, status)
