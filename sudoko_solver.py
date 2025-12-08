import numpy as np

def is_valid(grid, row, col, num):
    if num in grid[row]: return False
    if num in grid[:, col]: return False

    br, bc = 3 * (row//3), 3 * (col//3)
    box = grid[br:br+3, bc:bc+3]
    if num in box: return False

    return True


def solve_sudoku(grid):
    for r in range(9):
        for c in range(9):
            if grid[r][c] == 0:
                for num in range(1, 10):
                    if is_valid(grid, r, c, num):
                        grid[r][c] = num
                        if solve_sudoku(grid): return True
                        grid[r][c] = 0
                return False
    return True


def validate_sudoku(grid):
    for i in range(9):
        row = [x for x in grid[i] if x != 0]
        if len(row) != len(set(row)):
            return False, f"Duplicate in row {i}"

        col = [x for x in grid[:, i] if x != 0]
        if len(col) != len(set(col)):
            return False, f"Duplicate in col {i}"

    for br in range(0, 9, 3):
        for bc in range(0, 9, 3):
            box = grid[br:br+3, bc:bc+3].flatten()
            box = [x for x in box if x != 0]
            if len(box) != len(set(box)):
                return False, f"Duplicate in 3×3 box ({br},{bc})"

    return True, "Valid"
