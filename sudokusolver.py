import numpy as np
from random import choice


DIM = 3
DIM_GAME = DIM*DIM
CONTROL_VEC = np.arange(1, DIM * DIM + 1)
CHECKSUM = np.sum(CONTROL_VEC)


def is_valid(v: np.array) -> bool():
    v = np.unique(v)

    if len(v) != DIM_GAME:
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


def row_with_max_number_of_values(A: np.array) -> tuple[int, np.array]:
    '''
    Returns the row index for which the most values are available.

    Args:
        A (np.array): sudoku as a matrix
    Returns:
        int: row index
        np.array: values which are included
    '''
    row = 0
    vals = np.zeros(1)
    for i in range(0, DIM_GAME):
        v = A[i]
        # remove 0 values
        v = v[v != 0]
        
        if len(v) > len(vals) and len(v) < DIM_GAME:
            row = i
            vals = v
    return row, vals


def col_with_max_number_of_values(A: np.array) -> tuple[int, np.array]:
    '''
    Returns the col index which for which the most values are available.

    Args:
        A (np.array): sudoku as a matrix
    Returns:
        int: col index
        np.array: values which are included
    '''
    col = 0
    vals = np.zeros(1)
    for j in range(0, DIM_GAME):
        v = A[ : , j]
        # remove 0 values
        v = v[v != 0]
        
        if len(v) > len(vals) and len(v) < DIM_GAME:
            col = j
            vals = v
    return col, vals


def square_with_max_number_of_values(A: np.array) -> tuple[int, np.array]:
    '''
    Returns the number of the square which contains the most values.

    Args:
        A (np.array): sudoku as a matrix
    Returns:
        int: square number
        np.array: values which are included
    '''
    square = 0
    vals = np.zeros(0)
    for i in range(0, DIM_GAME):
        v = vectorize_square(A, i)
        # remove 0 values
        v = v[v != 0]
        
        if len(v) > len(vals) and len(v) < DIM_GAME:
            square = i
            vals = v
    return square, vals


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


def collect_vec_info(v: np.array) -> np.array:
    '''
    Returns useful vector information.

    Args:
        v (np.array): input vector
    Returns:
        np.array: vector without zero values (clean vector)
    '''
    # remove 0 values
    return v[v != 0]


def choose_field_in_row(A: np.array, row: int) -> tuple[int, np.array]:
    '''
    Returns the col index for which the most values are available.

    Args:
        A (np.array): sudoku as a matrix
        row (int): row to look for the best column in it
    Returns:
        int: column index for which most of the information is provided
        np.array: numbers to exclude from guess
    '''
    # fields which contain 0
    colList = np.where(A[row] == 0)
    best_col = -1
    best_col_info = np.zeros(0)
    
    for col in np.nditer(colList):
        # collect col information
        v_col = collect_vec_info(A[ : , col])
        # collect square information
        v_clean_square = collect_vec_info(vectorize_square(A, get_square(row, col)))
        col_info = np.unique(np.concatenate((v_col, v_clean_square)))

        if len(col_info) > len(best_col_info):
            best_col = int(col)
            best_col_info = col_info

    return best_col, best_col_info


def choose_field_in_col(A: np.array, col: int) -> tuple[int, np.array]:
    '''
    Returns the col index for which the most values are available.

    Args:
        A (np.array): sudoku as a matrix
        col (int): col to look for the best row in it
    Returns:
        int: row index for which most of the information is provided
        np.array: numbers to exclude from guess
    '''
    # fields which contain 0
    rowList = np.where(A[ : , col] == 0)
    best_row = -1
    best_row_info = np.zeros(0)
    
    for row in np.nditer(rowList):
        # collect row information
        v_row = collect_vec_info(A[row])
        # collect square information
        v_clean_square = collect_vec_info(vectorize_square(A, get_square(row, col)))
        row_info = np.unique(np.concatenate((v_row, v_clean_square)))

        if len(row_info) > len(best_row_info):
            best_row = int(row)
            best_row_info = row_info

    return best_row, best_row_info


def choose_field_in_square(A: np.array, square: int) -> tuple[int, int, np.array]:
    '''
    Returns the indices for a field in a square for which the most values are available.

    Args:
        A (np.array): sudoku as a matrix
        square (int): square to look for the best field in it
    Returns:
        int: field row index for which most of the information is provided
        int: field col index for which most of the information is provided
        np.array: numbers to exclude from guess
    '''
    best_row = -1
    best_col = -1
    v_info = np.zeros(0)

    v_square = vectorize_square(A, square)
    # find 0 value indices
    indexList = np.where(v_square == 0)
    
    for i in np.nditer(indexList):
        # reconstruct row and col index
        row, col = reconstruct_field_out_of_vectorized_square_index(square, i)
        # collect row information
        v_row = collect_vec_info(A[row])
        # collect col information
        v_col = collect_vec_info(A[ : , col])
        v = np.unique(np.concatenate((v_row, v_col)))

        if len(v) > len(v_info):
            best_row = int(row)
            best_col = int(col)
            v_info = v
    return best_row, best_col, v_info


def reconstruct_index_area_for_square(square: int) -> tuple[int, int, int, int]:
    '''
    Reconstructs the original sudoku matrix index area for the square.

    Args:
        square (int): square number
    Returns:
        int: row start index
        int: row end index
        int: col start index
        int: col end index
    '''
    row_range_start = int(square / DIM) * DIM
    row_range_end = row_range_start + DIM - 1
    col_range_start = (square % DIM) * DIM
    col_range_end = col_range_start + DIM - 1

    print(row_range_start)
    print(row_range_end)
    print(col_range_start)
    print(col_range_end)


def reconstruct_field_out_of_vectorized_square_index(square: int, index: int) -> tuple[int, int]:
    '''
    Reconstructs the original sudoku matrix index out of the given information.

    Args:
        square (int): square number
        index (int): index of the element in the vectorized square
    Returns:
        int: row index
        int: col index
    '''
    if index < 0 or index > DIM_GAME:
        # not a valid index
        return -1, -1
    else:
        # reconstruct row
        # row in square + row in sudoku
        row = int(index / DIM) + int(square / DIM) * DIM
        # col in square + col in sudoku
        col = (index % DIM) + (square % DIM) * DIM
        return row, col


def pick_promising_fields(A: np.array) -> list:
    '''
    Picks three fields depending on the max. number of values provided for a row/col/square.

    Args:
        A (np.array): Sudoku as a matrix
    Returns:
        tuple: row and col index for the best row-pick
        tuple: row and col index for the best col-pick
        tuple: row and col index for the best square-pick
    '''
    # completed rows, cols and squares are excluded
    max_square, square_vals = square_with_max_number_of_values(A)
    max_row, row_vals = row_with_max_number_of_values(A)
    max_col, col_vals = col_with_max_number_of_values(A)

    # choose the best field in the detected row/col/square
    max_row_best_col, max_row_best_col_info = choose_field_in_row(A, max_row)
    max_col_best_row, max_col_best_row_info = choose_field_in_col(A, max_col)
    max_square_best_row, max_square_best_col, max_square_info = choose_field_in_square(A, max_square)

    # accumulate inforation
    col_vals = np.unique(np.concatenate((col_vals, max_col_best_row_info)))
    row_vals = np.unique(np.concatenate((row_vals, max_row_best_col_info)))
    square_vals = np.unique(np.concatenate((square_vals, max_square_info)))

    return [(max_row, max_row_best_col, row_vals), (max_col_best_row, max_col, col_vals), (max_square_best_row, max_square_best_col, square_vals)]


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

    return int(factor * DIM + summand)


def do_safe_guess_only(A: np.array, guessList: list) -> np.array:
    '''
    Enter only safe guesses into the sudoku.

    Args:
        A (np.array): sudoku as a matrix
        guessList (list): one list entry consists of row index, column index and excluded values for the guess.
    Returns:
        np.array: sudoku with the entered guesses
    '''
    # TODO
    return A


def solve(A: np.array) -> np.array:
    '''
    Solves the passed sudoku.

    Args:
        A (np.array): sudoku to solve
    Returns:
        np.array: solved sudoku
    '''
    promising_fields = pick_promising_fields(A)
    A = do_safe_guess_only(A, promising_fields)

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