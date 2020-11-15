import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import game_mechanics as gm
import random


class Brain(nn.Module):
    def __init__(self, board_len):
        super().__init__()

        self.board_len = board_len

        input_dim = self.board_len * self.board_len

        self.model = nn.Sequential(
            nn.Linear(input_dim, 4 * input_dim),
            nn.ReLU(),
            nn.Linear(4 * input_dim, 10 * input_dim),
            nn.ReLU(),
            nn.Linear(10 * input_dim, 20 * input_dim),
            nn.ReLU(),
            nn.Linear(20 * input_dim, 4 * input_dim),
            nn.ReLU(),
            nn.Linear(4 * input_dim, input_dim),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        x = torch.tensor(x)
        x = x.view(-1, self.board_len*self.board_len).float()
        return self.model(x)


class MiniTTT:
    def __init__(self, board_len):
        self.brain = Brain(board_len)
        self.board_len = board_len
        self.symbol = None

        self.optimizer = optim.Adam(self.brain.parameters(), lr=0.0001)
        self.loss = nn.CrossEntropyLoss()

    def decide(self, board, epsilon):

        random_select = (random.random() < epsilon)
        
        if random_select == False:
            probabilities = self.brain(board).detach().numpy()[0]
            mask = gm.availability_mask(board)
            mask = mask.flatten()
            probabilities *= mask
            move = np.unravel_index(np.argmax(probabilities), (self.board_len, self.board_len))

        else:
            available_moves = gm.available_moves(board)
            move = random.choice(available_moves)

        return move

    def load_brain(self, subject=None, path=None):
        if path is not None:
            self.brain.load_state_dict(torch.load(path))
        elif subject is not None:
            self.brain.load_state_dict(subject.brain.state_dict())
        else:
            print("BrainError: No subject or path passed for brain probing, loading default brain")

        return

    def save_brain(self, name='brain'):
        torch.save(self.brain.state_dict(), name)
        return

    def train(self, state_sequence, move_sequence, win, discount_rate):

        discounted_weights = gm.discount_weights(len(state_sequence), discount_rate, win)

        for i, board in enumerate(state_sequence):

            self.optimizer.zero_grad()

            prediction = self.brain(board)

            # Creating y
            # chosen = torch.argmax(prediction)
            move = move_sequence[i]

            chosen = move[0] * self.board_len + move[1]

            y = [chosen]
            y = torch.tensor(y)

            prediction_loss = self.loss(prediction, y)
            prediction_loss.backward()

            for param in self.brain.parameters():
                param.grad *= discounted_weights[i]

            self.optimizer.step()

