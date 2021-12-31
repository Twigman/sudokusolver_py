import numpy as np


DIM = 3
DIM_GAME = DIM*DIM
CONTROL_VEC = np.arange(1, DIM * DIM + 1)
CHECKSUM = np.sum(CONTROL_VEC)


def is_valid(v: np.array) -> bool():
    v = np.unique(v)

    if len(v) != 9:
        return False
    elif np.sum(v) != CHECKSUM:
        return False
    else:
        return True


def check_square(A: np.array, n: int) -> bool:
    '''
    0 1 2
    3 4 5
    6 7 8
    '''
    v = np.zeros(9)
    i = 0

    for col in range(DIM * (n % DIM), DIM * (n % DIM) + DIM):
        for row in range(n - (n % DIM), n - (n % DIM) + DIM):
            v[i] = A[row, col]
            i += 1

    return is_valid(v)


def check_row(A: np.array, row: int) -> bool:
    return is_valid(A[row])


def check_col(A: np.array, col: int) -> bool:
    return is_valid(A[ : , col])


def print_sudoku(A: np.array):
    row_counter = 0
    col_counter = 0

    for row in range(0, DIM_GAME):
        for col in range(0, DIM_GAME):
            print(str(A[row, col]) + '  ', end='')
            col_counter += 1
            if col_counter % DIM == 0 and col_counter % DIM_GAME != 0:
                print('|  ', end='')
        print('\n')
        row_counter += 1
        if row_counter % DIM == 0 and row_counter < DIM_GAME:
            print('-------------------------------')


def solve(A: np.array) -> np.array:

    A[3,3] = 5

    print(A)


A = np.zeros((DIM_GAME, DIM_GAME))

example1 = np.array([
    [0, 0, 9, 0, 0, 0, 0, 1, 5],
    [5, 0, 0, 4, 0, 9, 7, 0, 0],
    [4, 7, 3, 5, 6, 1, 9, 0, 0],
    [0, 0, 0, 7, 4, 0, 0, 9, 6],
    [0, 0, 0, 0, 0, 0, 0, 8, 0],
    [0, 0, 4, 8, 3, 0, 1, 5, 0],
    [1, 3, 5, 9, 0, 0, 0, 0, 2],
    [0, 0, 6, 2, 5, 7, 0, 3, 0],
    [7, 2, 0, 0, 1, 0, 0, 0, 9]
])


#A = solve(example1)
print_sudoku(example1)