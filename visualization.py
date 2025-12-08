import cv2
import matplotlib.pyplot as plt

def visualize_results(original_img, recognized_grid, solved_grid, conf_grid):
    fig = plt.figure(figsize=(18, 6))

    # Original
    ax1 = plt.subplot(1, 3, 1)
    ax1.imshow(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB))
    ax1.set_title("Original Image")
    ax1.axis('off')

    # Recognized
    ax2 = plt.subplot(1, 3, 2)
    ax2.set_title("Recognized Grid")
    ax2.axis('off')

    for r in range(9):
        for c in range(9):
            d = recognized_grid[r][c]
            if d != 0:
                ax2.text(c+0.5, r+0.5, str(d),
                         ha='center', va='center', fontsize=16,
                         color='blue')

    ax2.set_xlim(0, 9)
    ax2.set_ylim(9, 0)
    ax2.grid(True)

    # Solved
    if solved_grid is not None:
        ax3 = plt.subplot(1, 3, 3)
        ax3.set_title("Solved Grid")
        ax3.axis('off')

        for r in range(9):
            for c in range(9):
                d = solved_grid[r][c]

                color = 'blue' if recognized_grid[r][c] != 0 else 'green'
                ax3.text(c+0.5, r+0.5, str(d),
                         ha='center', va='center',
                         fontsize=16, color=color)

        ax3.set_xlim(0, 9)
        ax3.set_ylim(9, 0)
        ax3.grid(True)

    plt.tight_layout()
    plt.show()
