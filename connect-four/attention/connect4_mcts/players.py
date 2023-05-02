import random

import numpy as np

from .policy import Model
from .game import Game
from .mcts import MCTS


class Player:
    def __init__(self, game: Game):
        self.game: Game = game.copy()

    def get_move(self) -> int:
        pass

    def make_move(self, move: int):
        pass


class RandomPlayer(Player):
    def get_move(self) -> int:
        legal_moves = self.game.get_legal_moves()
        return random.choice(legal_moves)

    def make_move(self, move: int):
        self.game.make_move(move)


class ModelPlayer(Player):
    def __init__(self, game: Game, model: Model):
        self.game: Game = game.copy()
        self.model: Model = model

    def get_move(self) -> int:
        probs, _wdl = self.model.policy_function(self.game.get_state())
        legal_moves = self.game.get_legal_moves()
        filtered_probs = np.zeros_like(probs, np.float32)
        filtered_probs[legal_moves] = probs[legal_moves]
        filtered_probs /= filtered_probs.sum()
        return np.random.choice(len(probs), p=filtered_probs)

    def make_move(self, move: int):
        self.game.make_move(move)


class MctsPlayer(Player):
    def __init__(self, game: Game, model: Model, c_puct: float, n_playouts: int, temp: float, states: np.ndarray):
        self.game: Game = game.copy()
        self.model: Model = model
        self.c_puct = c_puct
        self.n_playouts = n_playouts
        self.tree = MCTS(self.c_puct)
        self.temp = temp
        self.states = states

    def get_move(self) -> int:
        probs, _wdl = self.tree.run(
            self.game, self.model.policy_function, self.n_playouts, self.states)
        if self.temp < 1e-2:
            new_probs = np.zeros_like(probs, np.float32)
            new_probs[np.argmax(probs)] = 1
            probs = new_probs
        else:
            probs = np.power(probs, 1 / self.temp)
            probs = probs / probs.sum()
        return np.random.choice(len(probs), p=probs)

    def make_move(self, move: int):
        self.game.make_move(move)
        self.states = np.append(self.states, self.game.get_state()[None, ...], axis=0)
        self.tree.make_move(move)
