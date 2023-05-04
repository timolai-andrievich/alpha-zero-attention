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
        self.states = []
        self.y_pol = []
        self.y_wdl = []

    def insert_tuple(self, state: np.ndarray, y_pol: np.ndarray, y_wdl: np.ndarray):
        if self.size < self.max_size:
            self.states.append(state)
            self.y_pol.append(y_pol)
            self.y_wdl.append(y_wdl)
            self.size += 1
        else:
            self.states[self.oldest] = state
            self.y_pol[self.oldest] = y_pol
            self.y_wdl[self.oldest] = y_wdl
            self.oldest = (self.oldest + 1) % self.max_size

    def get_batch(self, sample_size: int) -> Tuple[States, Policies, WDLs]:
        assert self.size > 0
        indicies = np.random.randint(0, self.size, sample_size)
        states = []
        pol = []
        wdl = []
        for i in indicies:
            states.append(self.states[i])
            pol.append(self.y_pol[i])
            wdl.append(self.y_wdl[i])
        return states, pol, wdl


class Coach:
    def __init__(self, model: policy.Model, max_buffer_size: int):
        self.model = model
        self.buffer = Buffer(max_buffer_size)

    def generate_game(self, c_puct: float, n_playouts: int, temp: float):
        g = game.Game()
        tree = mcts.MCTS(c_puct)
        states, pols, wdls = [], [], []
        while not g.is_terminal():
            probs, wdl = tree.run(g, self.model.policy_function, n_playouts, states)
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
            states.append(state)
            pols.append(probs)
            wdls.append(wdl)
            self.buffer.insert_tuple(np.array(states, np.float32), np.array(pols, np.float32), np.array(wdls, np.float32))
            move = np.random.choice(len(probs), p=probs)
            g.make_move(move)
            tree.make_move(move)
        print(f'Generated {len(states)}-move game.')

    def generate_games(self, n_games: int, c_puct: float, n_playouts: int, temp: float):
        for _ in range(n_games):
            self.generate_game(c_puct, n_playouts, temp)

    def train(self, batch_size: int, minibatch_size: int):
        states, y_pol, y_wdl = self.buffer.get_batch(batch_size)
        return self.model.train(states, y_pol, y_wdl, minibatch_size)

    def train_epochs(self, batch_size: int, minibatch_size: int, epochs: int):
        losses = []
        for _ in range(epochs):
            loss = self.train(batch_size, minibatch_size)
            losses.append(loss)
        return np.mean(losses)

