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
    row = -1
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
    col = -1
    vals = np.zeros(0)
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
    square = -1
    vals = np.zeros(0)
    for i in range(0, DIM_GAME):
        v = vectorize_square(A, i)
        # remove 0 values
        v = v[v != 0]
        
        if len(v) > len(vals) and len(v) < DIM_GAME:
            square = i
            vals = v
    return square, vals


def make_a_guess(vals: np.array) -> int:
    return choice([i for i in range(1, DIM_GAME + 1) if i not in vals])


def find_missing_number(vals: np.array) -> int:
    return int(CHECKSUM - np.sum(vals))




'''
def enter_guess_in_row(A: np.array, guess: int, row: int) -> (np.array, int):
    col = np.where(A[row] == 0)
    A[row, col] = guess
    
    return A, row, col
'''



def is_valid_value(A: np.array, row: int, col: int) -> bool:
    # TODO
    # row
    # col
    # square
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
    
    if len(colList[0]) > 0:
        for col in np.nditer(colList):
            # collect col information
            v_col = collect_vec_info(A[ : , col])
            # collect square information
            v_clean_square = collect_vec_info(vectorize_square(A, get_square_for_coords(row, col)))
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
    
    if len(rowList[0]) > 0:
        for row in np.nditer(rowList):
            # collect row information
            v_row = collect_vec_info(A[row])
            # collect square information
            v_clean_square = collect_vec_info(vectorize_square(A, get_square_for_coords(row, col)))
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
    
    if len(indexList[0]) > 0:
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

    return row_range_start, row_range_end, col_range_start, col_range_end


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


def calc_relative_row_in_square(row: int) -> int:
    '''
    Calculates the row relative to a square.

    Args:
        row (int): 
    Returns:
        int: row
    '''
    return row % DIM


def count_blank_fields_in_row_in_square(square: int, v: np.array, row: int) -> int:
    '''
    Counts blank fields in a row in a square.

    Args:
        square (int): square number
        v (np.array): verctorized square
        row (int): row in which to search
    Returns:
        int: amount of blank fields
    '''
    counter = 0
    blank_field_indices = np.where(v == 0)

    for index in np.nditer(blank_field_indices):
        # is the calculated row equals the asked row
        if int(index / DIM) == row:
            counter += 1
    return counter


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


def get_square_for_coords(row, col) -> int:
    factor = (row - (row % DIM)) / DIM
    summand = (col - (col % DIM)) / DIM

    return int(factor * DIM + summand)


def do_safe_guesses_only(A: np.array, guessList: list) -> tuple[np.array, int]:
    '''
    Enter only safe guesses into the sudoku.

    Args:
        A (np.array): sudoku as a matrix
        guessList (list): one list entry consists of row index, column index and excluded values for the guess.
    Returns:
        np.array: sudoku with the entered guesses
        int: number of inserts
    '''
    inserts = 0
    for guess in guessList:
        if len(guess[2]) == 8:
            # only one number is missing
            A[guess[0], guess[1]] = find_missing_number(guess[2])
            inserts += 1
    return A, inserts


def insert_values(A: np.array, vals: list) -> tuple[np.array, int]:
    '''
    Inserts values in the sudoku.

    Args:
        A (np.array): sudoku as a matrix
        vals (list): row, col, value
    Returns:
        np.array: sudoku with the entered guesses
        int: number of inserts
    '''
    inserts = 0
    for val in vals:
        A[val[0], val[1]] = val[2]
        inserts += 1
    return A, inserts


def is_solved_fast_check(promissing_fields: list) -> bool:
    complete_counter = 0
    for field in promissing_fields:
        if field[0] == -1 and field[1] == -1:
            complete_counter += 1
        else:
            return False
    if complete_counter == 3:
        return True


def intersec_values(value_list: list, square_arr: np.array) -> np.array:
    '''
    Finds the intersection of all arrays.

    Args:
        value_list (list): list of np.arrays (row or column collection)
        square_arr (np.array): values in the square
    Returns:
        np.array: Intersection of all np.arrays
    '''
    # store the first array for the loop
    intersec_arr = value_list[0]

    for i in range(1, len(value_list)):
        intersec_arr = np.intersect1d(intersec_arr, value_list[i])

    # remove values which are already in the square
    return np.setdiff1d(intersec_arr, square_arr)


def look_for_solutions_by_crossing_lines_in_a_square(A: np.array) -> list:
    '''
    Looks for clear solutions in squares by checking the crossing rows and columns.

    Args:
        A (np.array): sudoku as a matrix
    Returns:
        row: row
        col: col
        solution: unabiguous value
    '''
    square = 4
    vals = []
    #for square in range(DIM_GAME):
    print('checking square ' + str(square))
    v_square = vectorize_square(A, square)
    v_square_info = collect_vec_info(v_square)
    zero_indices = np.where(v_square == 0)

    # check fields in square for solutions
    if len(zero_indices[0]) > 0:
        for zero_index in np.nditer(zero_indices):
            # reset field info
            row_info_list = []
            col_info_list = []
            # current index
            row, col = reconstruct_field_out_of_vectorized_square_index(square, zero_index)
            print('field: ' + str(row) + ', ' + str(col))
            # get coords
            row_start, row_end, col_start, col_end = reconstruct_index_area_for_square(square)

            # check rows
            for r in range(row_start, row_end + 1):
                # expect of the current index
                if r == row:
                    continue
                else:
                    row_info_list.append(collect_vec_info(A[r]))

            intersec_row = intersec_values(row_info_list, v_square_info)

            # insert value if this is the only blank field in this row in the square
            if len(intersec_row) > 0 and count_blank_fields_in_row_in_square(square, v_square, calc_relative_row_in_square(row)) == 1:
                print('intersec_row')
                print(intersec_row)
                vals.append((row, col, intersec_row[0]))
                # continue with the next row
                continue

            # check cols if a number was found
            if len(intersec_row) > 0:
                for c in range(col_start, col_end + 1):
                    if c == col:
                        # expect of the current index
                        continue
                    else:
                        # store column values 
                        col_info_list.append(collect_vec_info(A[ : , c]))

                intersec_col = intersec_values(col_info_list, v_square_info)
                print(intersec_col)

    print(vals)
    return vals


def is_solved(A: np.array) -> bool:
    '''
    Checks if the sudoku is completed.

    Args:
        A (np.array): sudoku as a matrix
    Returns:
        bool: True if the sudoku is complete; False if fields are blank
    '''
    # fields which contain 0
    rowList, colList = np.where(A == 0)

    if len(rowList) == 0 and len(colList) == 0:
        # TODO?
        # check the sudoku
        return True
    else:
        return False


def solve(A: np.array) -> np.array:
    '''
    Solves the passed sudoku.

    Args:
        A (np.array): sudoku to solve
    Returns:
        np.array: solved sudoku
    '''
    iteration = 0
    inserts_per_iter = -1

    while inserts_per_iter != 0:
        promising_fields = pick_promising_fields(A)
        print(promising_fields)
        A, inserts_per_iter = do_safe_guesses_only(A, promising_fields)
        intersec_solutions = look_for_solutions_by_crossing_lines_in_a_square(A)
        A, inserts = insert_values(A, intersec_solutions)
        inserts_per_iter += inserts
        #print(indirect_solutions)
        iteration += 1

    print('solving stopped after ' + str(iteration) + ' iterations')
    return A

A = np.zeros((DIM_GAME, DIM_GAME))

example1 = np.array([
    [0, 0, 9, 3, 0, 0, 0, 1, 5],
    [5, 0, 0, 4, 0, 9, 7, 0, 0],
    [4, 7, 3, 5, 6, 1, 9, 0, 0],
    [0, 0, 0, 7, 4, 0, 0, 9, 6],
    [0, 0, 0, 0, 0, 0, 0, 8, 0],
    [0, 0, 4, 8, 3, 0, 1, 5, 0],
    [1, 3, 5, 9, 0, 0, 0, 0, 2],
    [0, 0, 6, 2, 5, 7, 0, 3, 0],
    [7, 2, 0, 0, 1, 0, 0, 0, 9]
])

example2 = np.array([
    [0, 3, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 3, 0],
    [0, 0, 0, 1, 0, 2, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 3, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 4, 0, 0, 0]
])

example3 = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 0, 0],
    [0, 0, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0]
])

#print_sudoku(example1)
#A = solve(example2)
#A = solve(example1)
A = solve(example3)
#print_sudoku(A)
#print_sudoku(example1)
#look_for_solutions_by_crossing_lines_in_a_square(example1)
#print('--------------------------------------------')
#res = look_for_solutions_by_crossing_lines_in_a_square(test1)
#print(res)