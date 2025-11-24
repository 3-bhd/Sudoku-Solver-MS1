# main_milestone1.py
import cv2
import os
import math
import matplotlib.pyplot as plt

from preprocessing import robust_preprocess
from grid_detection import get_sudoku_grid
from wraping import four_point_transform

IMAGE_DIR = "Project Test Cases-1"


# -----------------------------
# Helper: load all image names
# -----------------------------
def get_all_images(folder):
    """
    Returns a sorted list of all .jpg/.jpeg/.png images in the folder.
    Sorting is numeric-friendly (01.jpg before 10.jpg).
    """
    valid_ext = (".jpg", ".jpeg", ".png")
    files = [f for f in os.listdir(folder) if f.lower().endswith(valid_ext)]

    # numeric-friendly sort
    def sort_key(name):
        base = os.path.splitext(name)[0]
        return int(base) if base.isdigit() else base

    return sorted(files, key=sort_key)


# -----------------------------
# Helper: dynamic clean grid
# -----------------------------
def make_grid(n):
    """
    Picks a near-square grid (rows x cols) for clean plotting.
    """
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)
    return rows, cols


# -----------------------------
# 1) SHOW ORIGINAL IMAGES
# -----------------------------
def show_originals(image_files):
    n = len(image_files)
    rows, cols = make_grid(n)

    plt.figure(figsize=(4 * cols, 4 * rows))
    plt.suptitle("Original Images", fontsize=20)

    for idx, filename in enumerate(image_files):
        img_path = os.path.join(IMAGE_DIR, filename)
        img = cv2.imread(img_path)

        plt.subplot(rows, cols, idx + 1)
        if img is not None:
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title(filename)
        plt.axis("off")

    plt.tight_layout()
    plt.show()


# -----------------------------
# 2) SHOW PREPROCESSED IMAGES
# -----------------------------
def show_preprocessed(image_files):
    n = len(image_files)
    rows, cols = make_grid(n)

    plt.figure(figsize=(4 * cols, 4 * rows))
    plt.suptitle("Preprocessed (Binary) Images", fontsize=20)

    for idx, filename in enumerate(image_files):
        img_path = os.path.join(IMAGE_DIR, filename)
        _, processed = robust_preprocess(img_path)

        plt.subplot(rows, cols, idx + 1)
        if processed is not None:
            plt.imshow(processed, cmap="gray")
        plt.title(filename)
        plt.axis("off")

    plt.tight_layout()
    plt.show()


# -----------------------------
# 3) SHOW WARPED GRIDS
# -----------------------------
def show_warped(image_files):
    n = len(image_files)
    rows, cols = make_grid(n)

    plt.figure(figsize=(4 * cols, 4 * rows))
    plt.suptitle("Warped & Straightened Sudoku Grids", fontsize=20)

    for idx, filename in enumerate(image_files):
        img_path = os.path.join(IMAGE_DIR, filename)
        img = cv2.imread(img_path)

        plt.subplot(rows, cols, idx + 1)

        if img is None:
            plt.title(f"{filename}\nERROR")
            plt.axis("off")
            continue

        contour, strategy = get_sudoku_grid(img)

        if contour is not None:
            warped = four_point_transform(img, contour.reshape(4, 2))
            plt.imshow(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB))
            plt.title(f"{filename}\n{strategy}")
        else:
            plt.title(f"{filename}\nFAILED")

        plt.axis("off")

    plt.tight_layout()
    plt.show()


# -----------------------------
# MAIN EXECUTION
# -----------------------------
if __name__ == "__main__":
    image_files = get_all_images(IMAGE_DIR)

    if not image_files:
        print(f"No images found in folder: {IMAGE_DIR}")
        exit()

    print(f"Found {len(image_files)} images in '{IMAGE_DIR}'")

    show_originals(image_files)
    show_preprocessed(image_files)
    show_warped(image_files)
