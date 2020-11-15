import game_mechanics as gm
import numpy as np
import os
import time


class Environment:
    def __init__(self, board_len):
        self.board_len = board_len
        self.board = gm.create_board(board_len)

        self.board_history = {'X': [],
                              'O': []}

        self.move_history = {'X': [],
                             'O': []}

        # Basically player sees board_history[player][i] before making move_history[player][i] move

    def step(self, player, move):

        self.board_history[player].append(self.board.copy())
        self.move_history[player].append(move)

        gm.select_space(player, self.board, move)
        obs = self.board
        winner = gm.game_is_over(self.board)

        if winner == 'N':
            reward = 0
        elif winner == player:
            reward = 1
        else:
            reward = -1

        return obs, reward, winner

    def reset(self):
        self.board = gm.create_board(self.board_len)
        self.board_history = {'X': [],
                              'O': []}
        self.move_history = {'X': [],
                             'O': []}

    def render(self, final_board):

        symbol_dict = {1: 'X', -1: 'O', 0: ' '}

        game_len = len(self.board_history['X']) + len(self.board_history['O'])
        board = gm.create_board(self.board_len)

        turn = 'X'
        x_it = 0
        o_it = 0

        while x_it + o_it < game_len:

            for i, row in enumerate(board):
                print('\n')
                row_str = ""
                for j, cell in enumerate(row):
                    row_str += f" {symbol_dict[cell]} |"

                row_str = row_str[:-1]
                print(row_str)
                if i < len(board)-1:
                    divider = "_"*len(row_str)
                    print(divider)

            print('-------------------------------------------')
            if turn == 'X':
                gm.select_space(turn, board, self.move_history[turn][x_it])
                x_it += 1
            else:
                gm.select_space(turn, board, self.move_history[turn][o_it])
                o_it += 1

            if turn == 'X':
                turn = 'O'
            else:
                turn = 'X'

            time.sleep(1)

        # Render final board:
        for i, row in enumerate(final_board):
            print('\n')
            row_str = ""
            for j, cell in enumerate(row):
                row_str += f" {symbol_dict[cell]} |"

            row_str = row_str[:-1]
            print(row_str)
            if i < len(final_board) - 1:
                divider = "_" * len(row_str)
                print(divider)

        print("=====================================================================================")











