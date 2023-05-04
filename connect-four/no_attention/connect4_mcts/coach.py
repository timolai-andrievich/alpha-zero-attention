from typing import Tuple

import numpy as np

from . import policy
from . import game
from . import mcts


States = np.ndarray
Policies = np.ndarray
WDLs = np.ndarray


class Buffer:
    def __init__(self, max_size: int):
        assert max_size > 0
        self.max_size = max_size
        self.size = 0
        self.oldest = 0
        self.states = np.zeros(
            (self.max_size, game.Game.STATE_LAYERS, game.Game.STATE_HEIGHT, game.Game.STATE_WIDTH), np.float32)
        self.y_pol = np.zeros((self.max_size, game.Game.NUM_ACTIONS), np.float32)
        self.y_wdl = np.zeros((self.max_size, 3), np.float32)

    def insert_tuple(self, state: np.ndarray, y_pol: np.ndarray, y_wdl: np.ndarray):
        if self.size < self.max_size:
            self.states[self.size] = state
            self.y_pol[self.size] = y_pol
            self.y_wdl[self.size] = y_wdl
            self.size += 1
        else:
            self.states[self.oldest] = state
            self.y_pol[self.oldest] = y_pol
            self.y_wdl[self.oldest] = y_wdl
            self.oldest = (self.oldest + 1) % self.max_size

    def get_batch(self, sample_size: int) -> Tuple[States, Policies, WDLs]:
        assert self.size > 0
        indicies = np.random.randint(0, self.size, sample_size)
        return self.states[indicies], self.y_pol[indicies], self.y_wdl[indicies]


class Coach:
    def __init__(self, model: policy.Model, max_buffer_size: int):
        self.model = model
        self.buffer = Buffer(max_buffer_size)

    def generate_game(self, c_puct: float, n_playouts: int, temp: float):
        g = game.Game()
        tree = mcts.MCTS(c_puct)
        while not g.is_terminal():
            probs, wdl = tree.run(g, self.model.policy_function, n_playouts)
            mask = np.zeros_like(probs, np.float32)
            for move in g.get_legal_moves():
                mask[move] = 1
            probs = probs * mask
            probs /= probs.sum()
            if temp < 1e-6:
                move = np.argmax(probs)
                probs = np.zeros_like(probs, np.float32)
                probs[move] = 1
            else:
                probs = np.power(probs, 1 / temp)
                probs /= probs.sum()
            state = g.get_state()
            self.buffer.insert_tuple(state, probs, wdl)
            move = np.random.choice(len(probs), p=probs)
            g.make_move(move)
            tree.make_move(move)

    def generate_games(self, n_games: int, c_puct: float, n_playouts: int, temp: float):
        for _ in range(n_games):
            self.generate_game(c_puct, n_playouts, temp)

    def train(self, batch_size: int, minibatch_size: int):
        states, y_pol, y_wdl = self.buffer.get_batch(batch_size)
        self.model.train(states, y_pol, y_wdl, minibatch_size)

    def train_epochs(self, batch_size: int, minibatch_size: int, epochs: int):
        for _ in range(epochs):
            self.train(batch_size, minibatch_size)

