import cv2
import numpy as np

def extract_cells_from_grid(warped_img, grid_size=9):
    h, w = warped_img.shape[:2]
    cell_h = h // grid_size
    cell_w = w // grid_size

    margin_y = max(2, int(0.08 * cell_h))
    margin_x = max(2, int(0.08 * cell_w))

    cells = []

    for r in range(grid_size):
        row_cells = []
        for c in range(grid_size):
            y1 = r * cell_h + margin_y
            y2 = (r + 1) * cell_h - margin_y
            x1 = c * cell_w + margin_x
            x2 = (c + 1) * cell_w - margin_x

            row_cells.append(warped_img[y1:y2, x1:x2])
        cells.append(row_cells)

    return np.array(cells)


def preprocess_cell(cell):
    if len(cell.shape) == 3:
        cell = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)

    denoised = cv2.bilateralFilter(cell, 9, 75, 75)

    adaptive = cv2.adaptiveThreshold(
        denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 21, 4
    )

    _, otsu = cv2.threshold(denoised, 0, 255,
                            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    combined = cv2.bitwise_and(adaptive, otsu)

    kernel = np.ones((2, 2), np.uint8)
    opened = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel)

    labels, stats, _ = cv2.connectedComponentsWithStats(
        opened, connectivity=8
    )[1:4]

    mask = np.zeros_like(opened)

    cell_area = opened.size

    for i in range(1, stats.shape[0]):
        area = stats[i, cv2.CC_STAT_AREA]
        if area > cell_area * 0.005:
            mask[labels == i] = 255

    return mask

def white_ratio(img):
        return float(np.sum(img == 255)) / img.size

    r_otsu = white_ratio(th_otsu)
    r_adapt = white_ratio(th_adapt)

    # Target: some white, but not full snow
    candidates = []
    if 0.02 < r_otsu < 0.35:
        candidates.append((th_otsu, abs(r_otsu - 0.15)))
    if 0.02 < r_adapt < 0.35:
        candidates.append((th_adapt, abs(r_adapt - 0.15)))

    if candidates:
        thresh = min(candidates, key=lambda x: x[1])[0]
    else:
        # Fallback: choose closer to target ratio
        thresh = th_otsu if abs(r_otsu - 0.15) < abs(r_adapt - 0.15) else th_adapt

    # 5) Morphology to clean and thicken digits
    kernel = np.ones((2, 2), np.uint8)
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=1)

    # 6) Remove tiny components (pepper noise)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(cleaned, connectivity=8)
    min_area = 0.002 * cleaned.size  # 0.2% of cell area

    mask = np.zeros_like(cleaned)
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            mask[labels == i] = 255

    return mask

def is_cell_empty(cell, threshold_area_ratio=0.01):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(cell)

    cell_area = cell.size

    for i in range(1, num_labels):
        if (stats[i, cv2.CC_STAT_AREA] / cell_area) >= threshold_area_ratio:
            return False

    return True

