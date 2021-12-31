import numpy as np
from random import choice


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


def vectorize_square(A: np.array, n: int) -> np.array:
    '''
    0 1 2
    3 4 5
    6 7 8
    '''
    v = np.zeros(9)
    i = 0

    for row in range(n - (n % DIM), n - (n % DIM) + DIM):
        for col in range(DIM * (n % DIM), DIM * (n % DIM) + DIM):
            v[i] = A[row, col]
            i += 1

    return v


def check_square(A: np.array, n: int) -> bool:
    '''
    0 1 2
    3 4 5
    6 7 8
    '''
    return is_valid(vectorize_square(A, n))


def check_row(A: np.array, row: int) -> bool:
    return is_valid(A[row])


def check_col(A: np.array, col: int) -> bool:
    return is_valid(A[ : , col])


def max_values_in_row(A: np.array) -> (int, int, np.array):
    max_len = 0
    row = 0
    vals = np.zeros(1)
    for i in range(0, DIM_GAME):
        v = A[i]
        # remove 0 values
        v = v[v != 0]
        
        if len(v) > max_len and len(v) < DIM_GAME:
            max_len = len(v)
            row = i
            vals = v
    return (row, max_len, vals)


def max_values_in_col(A: np.array) -> (int, int, np.array):
    max_len = 0
    col = 0
    vals = np.zeros(1)
    for j in range(0, DIM_GAME):
        v = A[ : , j]
        # remove 0 values
        v = v[v != 0]
        
        if len(v) > max_len and len(v) < DIM_GAME:
            max_len = len(v)
            col = j
            vals = v
    return (col, max_len, vals)


def max_values_in_square(A: np.array) -> (int, int, np.array):
    max_len = 0
    square = 0
    vals = np.zeros(1)
    for i in range(0, DIM_GAME):
        v = vectorize_square(A, i)
        # remove 0 values
        v = v[v != 0]
        
        if len(v) > max_len and len(v) < DIM_GAME:
            max_len = len(v)
            square = i
            vals = v
    return (square, max_len, vals)


def make_guess(vals: np.array) -> int:
    return choice([i for i in range(1, DIM_GAME + 1) if i not in vals])


def enter_guess_in_row(A: np.array, guess: int, row: int) -> (np.array, int):
    col = np.where(A[row] == 0)
    A[row, col] = guess
    
    return A, row, col



def is_valid_value(A: np.array, row: int, col: int) -> bool:
    # TODO
    # row
    # col
    # square
    return False

def is_solved(A: np.array) -> bool:
    return False


def collect_vec_info(v: np.array) -> (np.array, int):
    '''
    Returns useful vector information.

    Args:
        v (np.array): input vector
    Returns:
        np.array: vector without zero values (clean vector)
        int: clean vector length
    '''
    # remove 0 values
    v = v[v != 0]

    return v, len(v)



def choose_field_in_row(A: np.array, row: int) -> (int, int):
    # fields which contain 0
    colList = np.where(A[row] == 0)
    
    for col in colList:
        # collect col information
        v, v_len = collect_vec_info(A[ : , col])
        # collect row information
        square = get_square(row, col)
        v_square = vectorize_square(A, square)


    return None


def simple_field_pick(A: np.array) -> (int, int, np.array):
    square, max_square_val, square_vals = max_values_in_square(A)
    row, max_row_val, row_vals = max_values_in_row(A)
    col, max_col_val, col_vals = max_values_in_col(A)

    choose_field_in_row(A, row)

    #if max_square_val == 0 and max_row_val == 0 and max_col_val == 0:
        # solved!
        # TODO
    #if max_square_val > max_row_val and max_square_val > max_col_val:
        # square

        #guess = make_guess(square_vals)
        # TODO
        
    #elif max_row_val > max_col_val:
        # row
        #guess = make_guess(row_vals)
        #A, i, j = enter_guess_in_row(A, guess, row)
        # TODO
    #else:
        # col
        #guess = make_guess(col_vals)
        # TODO
    return None



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


def get_square(row, col) -> int:
    factor = (row - (row % DIM)) / DIM
    summand = (col - (col % DIM)) / DIM

    return factor * DIM + summand


def solve(A: np.array) -> np.array:
    #row, col = simple_field_pick(A)
    simple_field_pick(A)

    return A

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


A = solve(example1)
#print_sudoku(example1)