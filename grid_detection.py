# grid_detection.py
import cv2
import numpy as np

def find_grid_contour(processed_img, img_area):
    contours, _ = cv2.findContours(
        processed_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    # Attempt 1: 4-corner contour
    for c in contours:
        area = cv2.contourArea(c)
        if area > (img_area * 0.90) or area < (img_area * 0.02):
            continue

        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        if len(approx) == 4:
            return approx

    # Attempt 2: rubber-band convex hull
    significant_points = []
    for c in contours:
        area = cv2.contourArea(c)
        if (img_area * 0.001) < area < (img_area * 0.90):
            for pt in c:
                significant_points.append(pt)

    if significant_points:
        significant_points = np.array(significant_points).reshape(-1, 1, 2)
        hull = cv2.convexHull(significant_points)

        peri = cv2.arcLength(hull, True)
        approx = cv2.approxPolyDP(hull, 0.02 * peri, True)

        if len(approx) == 4:
            return approx

        rect = cv2.minAreaRect(hull)
        box = cv2.boxPoints(rect)
        return np.int32(box).reshape(4, 1, 2)

    return None


def get_sudoku_grid(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_area = img.shape[0] * img.shape[1]

    # Strategy A: Standard threshold
    blurred = cv2.GaussianBlur(gray, (9, 9), 0)
    thresh_standard = cv2.adaptiveThreshold(
        blurred, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        11, 2
    )

    contour = find_grid_contour(thresh_standard, img_area)
    if contour is not None:
        return contour, "Standard"

    # Strategy B: Dark/Fallback
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    blurred_dark = cv2.GaussianBlur(enhanced, (5, 5), 0)
    thresh_dark = cv2.adaptiveThreshold(
        blurred_dark, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        23, 2
    )

    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(thresh_dark, kernel, iterations=2)

    contour = find_grid_contour(dilated, img_area)
    if contour is not None:
        return contour, "Dark/Fallback"

    return None, "Failed"
