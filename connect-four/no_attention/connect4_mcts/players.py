"""Contains several utility classes to encapsulate choosing the next action.
"""
import random

import numpy as np

from .policy import Model
from .game import Game
from .mcts import MCTS


class Player:
    """Interface for the player classes.
    """
    def __init__(self, game: Game):
        """Dummy method for the interface.

        Args:
            game (Game): The game position to be played from.
        """
        self.game: Game = game.copy()

    def get_move(self) -> int:
        """Dummy interface method. Returns the move ID chosen by the agent.

        Returns:
            int: Move ID.
        """

    def make_move(self, move: int):
        """Dummy interface method.

        Args:
            move (int): The ID of the move to be made.
        """


class RandomPlayer(Player):
    """Chooses a random legal move each position."""

    def __init__(self, game: Game): # pylint: disable=useless-parent-delegation
        """Chooses a random legal move each position.

        Args:
            game (Game): Game to be played.
        """
        super().__init__(game)

    def get_move(self) -> int:
        """Uniformly sample the next move.

        Returns:
            int: The ID of the next move.
        """
        legal_moves = self.game.get_legal_moves()
        return random.choice(legal_moves)

    def make_move(self, move: int):
        """Make move in the internal game state.

        Args:
            move (int): The move ID.
        """
        self.game.make_move(move)


class ModelPlayer(Player):
    """Plays based on raw model outputs, without the MCTS.
    """
    def __init__(self, game: Game, model: Model):
        """Plays based on raw model outputs, without the MCTS.

        Args:
            game (Game): The game to be played.
            model (Model): The model to be used.
        """
        super().__init__(game)
        self.game: Game = game.copy()
        self.model: Model = model

    def get_move(self) -> int:
        """Uses the model to sample the next move from the legal
        moves using the probabilities outputted by model.

        Returns:
            int: The move ID.
        """
        probs, _wdl = self.model.policy_function(self.game.get_state())
        legal_moves = self.game.get_legal_moves()
        filtered_probs = np.zeros_like(probs, np.float32)
        filtered_probs[legal_moves] = probs[legal_moves]
        filtered_probs /= filtered_probs.sum()
        return np.random.choice(len(probs), p=filtered_probs)

    def make_move(self, move: int):
        """Make move in the internal game state.

        Args:
            move (int): The move ID.
        """
        self.game.make_move(move)


class MctsPlayer(Player):
    """Uses Monte-Carlo Tree search to pick the next move.
    """
    def __init__(
        self, game: Game, model: Model, c_puct: float, n_playouts: int, temp: float
    ):
        """Uses Monte-Carlo Tree search to pick the next move.

        Args:
            game (Game): The game to be played.
            model (Model): The model to be used for policy.
            c_puct (float): Confidence hyperparameter for UCB.
            n_playouts (int): The number of playouts for the simulations.
            temp (float): The temperature hyperparameter.
        """
        super().__init__(game)
        self.game: Game = game.copy()
        self.model: Model = model
        self.c_puct = c_puct
        self.n_playouts = n_playouts
        self.tree = MCTS(self.c_puct)
        self.temp = temp

    def get_move(self) -> int:
        """Uses the MCTS to sample the next move from the legal
        moves using the probabilities outputted by MCTS.

        Returns:
            int: The move ID.
        """
        probs, _wdl = self.tree.run(
            self.game, self.model.policy_function, self.n_playouts
        )
        if self.temp < 1e-2:
            new_probs = np.zeros_like(probs, np.float32)
            new_probs[np.argmax(probs)] = 1
            probs = new_probs
        else:
            probs = np.power(probs, 1 / self.temp)
            probs = probs / probs.sum()
        return np.random.choice(len(probs), p=probs)

    def make_move(self, move: int):
        """Make move in the internal game state.

        Args:
            move (int): The move ID.
        """
        self.game.make_move(move)
        self.tree.make_move(move)
