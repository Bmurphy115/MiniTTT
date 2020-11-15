import numpy as np


def create_board(board_len):
    board = np.zeros((board_len, board_len))
    return board


def select_space(player, board, move):
    if player == 'X':
        token = 1
    else:
        token = -1
    row, col = move
    board[row][col] = token
    return


def available_moves(board):
    moves = []
    for i in range(len(board)):
        for j in range(len(board)):
            if board[i][j] == 0:
                moves.append((i, j))
    return moves


def game_is_over(board):

    # Tie
    if not np.any(board.flatten() == 0):
        return 'T'

    board_t = board.T

    for i in range(len(board)):
        x_row = [1 for _ in range(len(board))]
        o_row = [-1 for _ in range(len(board))]

        if np.all(board[i] == x_row):
            return 'X'
        elif np.all(board[i] == o_row):
            return 'O'
        elif np.all(board_t[i] == x_row):
            return 'X'
        elif np.all(board_t[i] == o_row):
            return 'O'

    condition1 = True
    condition2 = True
    first1 = board[0][0]
    first2 = board[len(board)-1][0]

    for i in range(len(board)):
        if board[i][i] != first1:
            condition1 = False
        if board[len(board)-1-i][i] != first2:
            condition2 = False

        if condition1 is False and condition2 is False:
            return 'N'

    if condition1 is True and first1 != 0:
        if first1 == 1:
            return 'X'
        else:
            return 'O'

    if condition2 is True and first2 != 0:
        if first2 == 1:
            return 'X'
        else:
            return 'O'

    return 'N'


def discount_weights(move_len, discount_rate, win):
    weights = []
    for i in range(move_len):
        weights.append(discount_rate**(move_len-i-1))

    weights = np.array(weights)

    if win is False:
        weights *= -1

    return weights


def availability_mask(board):

    mask = np.ones((len(board), len(board)))

    for i, row in enumerate(board):
        for j, cell in enumerate(row):
            if cell != 0:
                mask[i][j] = 0

    return mask





