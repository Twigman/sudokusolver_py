import numpy as np

DIM_SQUARE = 3
DIM_GAME = 3

def check_square(A: np.array, n: int) -> bool:
    
    # 0 1 2
    # 3 4 5
    # 6 7 8

    # 0:    (0, 0) - (0, 2)
    #       (1, 0) - (1, 2)
    #       (2, 0) - (2, 2)

    # 1:    (0, 3) - (0, 5)
    #       (1, 3) - (1, 5)
    #       (2, 3) - (2, 5)

    # 2:    (0, 6) - (0, 8)
    #       (1, 6) - (1, 8)
    #       (2, 6) - (2, 8)

    # 5:    (3, 6) - (3, 8)
    #       (4, 6) - (4, 8)
    #       (5, 6) - (5, 8)

    # [row, col]
    # vertikal
    for col in range(DIM_SQUARE * (n % DIM_SQUARE), DIM_SQUARE * (n % DIM_SQUARE) + DIM_SQUARE):
        # horizontal
        print('\n')
        for row in range(n - (n % DIM_SQUARE), n - (n % DIM_SQUARE) + DIM_SQUARE):
            print(str(row) + ', ' + str(col))
        
    return False


def solve(A: np.array):
    A[3,3] = 5

    print(A)


A = np.zeros((DIM_SQUARE*DIM_GAME, DIM_SQUARE*DIM_GAME))

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

check_square(A, 6)
#solve(example1)