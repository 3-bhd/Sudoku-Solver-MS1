# main_milestone1.py
import cv2
import os
import matplotlib.pyplot as plt

from preprocessing import robust_preprocess
from grid_detection import get_sudoku_grid
from wraping import four_point_transform

# Local folder containing the test images
IMAGE_DIR = "Project Test Cases-1"

def run_preprocess_demo(image_files):
    plt.figure(figsize=(20, 40))
    plot_idx = 1

    for filename in image_files:
        img_path = os.path.join(IMAGE_DIR, filename)

        original, processed = robust_preprocess(img_path)
        if original is None:
            print(f"Warning: Could not open {img_path}")
            continue

        plt.subplot(8, 4, plot_idx)
        plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
        plt.title(f"Orig: {filename}")
        plt.axis("off")
        plot_idx += 1

        plt.subplot(8, 4, plot_idx)
        plt.imshow(processed, cmap="gray")
        plt.title(f"Processed: {filename}")
        plt.axis("off")
        plot_idx += 1

    plt.tight_layout()
    plt.show()


def run_warp_demo(image_files):
    plt.figure(figsize=(20, 40))
    plot_idx = 1

    print("--- Checking Output Dimensions ---")
    for filename in image_files:
        img_path = os.path.join(IMAGE_DIR, filename)

        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: Could not open {img_path}")
            continue

        contour, strategy = get_sudoku_grid(img)

        if contour is not None:
            warped = four_point_transform(img, contour.reshape(4, 2))

            h, w = warped.shape[:2]
            print(f"{filename}: {w}x{h} -> {'SQUARE' if w==h else 'RECTANGLE'}")

            plt.subplot(8, 4, plot_idx)
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            plt.title(f"{filename} ({strategy})")
            plt.axis("off")
            plot_idx += 1

            plt.subplot(8, 4, plot_idx)
            plt.imshow(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB))
            plt.title(f"Warped: {w}x{h}")
            plt.axis("off")
            plot_idx += 1

        else:
            print(f"FAILED: {filename}")
            plt.subplot(8, 4, plot_idx)
            plt.title("FAILED")
            plt.axis("off")
            plot_idx += 2

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    image_files = [f"{i:02d}.jpg" for i in range(1, 17)]

    run_preprocess_demo(image_files)
    run_warp_demo(image_files)
