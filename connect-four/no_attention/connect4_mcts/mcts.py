from typing import List, Tuple, Dict
import math

import numpy as np

from .game import Game


Policy = np.ndarray
WDL = np.ndarray


class Node:
    """Represents a node in the Monte-Carlo Search Tree.
    """
    def __init__(self, parent, prior: float, c_puct: float):
        """Creates a new node.

        Args:
            parent (Node): The parent of the new node. If the node
            is root, the parent is None.
            prior (float): The prior estimate of the node value. Used in UCB.
            c_puct (float): The confidence hyperparameter in the UCB.
        """
        self.parent: Node = parent
        self.prior: float = prior
        self.c_puct: float = c_puct
        self.children: Dict[int, Node] = {}
        self.visits: int = 0
        self.results: np.ndarray = np.zeros(3, np.float32)

    def _avg(self) -> float:
        """Returns the expected value of the node.

        Returns:
            float: The expected value of the node.
        """
        results_to_score: np.ndarray = np.array([1, 0.5, 0])
        if self.visits == 0:
            return 0
        return self.results.dot(results_to_score) / self.visits

    def select(self) -> Tuple[int, any]:
        """Returns the node with the highest UCB score.

        Returns:
            Tuple[int, any]: (Move ID, Node)
        """
        return max(self.children.items(), key=lambda x: x[1].ucb_score())

    def expand(self, moves_with_probs: Dict[int, float]):
        """Expands the node with the given probabilities as priors.
        If the move is illegal it should not be in the dictionary.

        Args:
            moves_with_probs (Dict[int, float]): Dictionary with move ids as
            keys and prior probabilities as values.

        Raises:
            RuntimeError: Raises RuntimeError if the node is already has children.
        """
        if not self.is_leaf():
            raise RuntimeError("Trying to expand non-leaf node")
        for move, prior in moves_with_probs.items():
            self.children[move] = Node(self, prior, self.c_puct)

    def is_root(self) -> bool:
        """Returns true if the node is the root node of the tree.

        Returns:
            bool: Returns true if the node is the root node of the tree, false otherwise.
        """
        return self.parent is None

    def is_leaf(self) -> bool:
        """Returns true if the node is the leaf node.

        Returns:
            bool: Returns true if the node is the leaf node, false otherwise.
        """
        return len(self.children) == 0

    def ucb_score(self) -> float:
        """Returns the UCB score of the node.

        Raises:
            RuntimeError: If the node is the root node.

        Returns:
            float: The UCB score of the node.
        """
        if self.is_root():
            raise RuntimeError("Trying to get UCB score of the root node")
        return -self._avg() + self.c_puct * self.prior * math.sqrt(
            self.parent.visits
        ) / (self.visits + 1)

    def update(self, score: np.ndarray):
        """Updates the node value using the new score.

        Args:
            score (np.ndarray): The new score acquired during playouts.
        """
        self.visits += 1
        self.results += score

    def update_recursive(self, new_score):
        """Recursively update the node and all of the ancestors nodes.

        Args:
            new_score (np.ndarray): The score acquired during playouts.
        """
        self.update(new_score)
        if not self.is_root():
            self.parent.update_recursive(new_score[::-1])


class MCTS:
    """Encapsulates tree-related part of the MCTS code.
    """
    def __init__(self, c_puct: float):
        """Creates new Monte-Carlo Tree Search instance.

        Args:
            c_puct (float): The confidence hyperparameter of the UCB formula.
        """
        self.c_puct: float = c_puct
        self.root_node: Node = Node(None, 0, self.c_puct)

    def run(self, game: Game, policy_function, iterations: int) -> Tuple[Policy, WDL]:
        """Runs playouts and returns improved move and win-draw-lose probabilities.

        Args:
            game (Game): The game to be played.
            policy_function: The policy function to estimate move
            probabilities and wdl probabilities.
            iterations (int): The number of playouts to be played.

        Returns:
            Tuple[Policy, WDL]: Improved move and win-draw-lose probabilities.
        """
        for _ in range(iterations):
            self.simulate(game.copy(), policy_function)
        policy = np.zeros(Game.NUM_ACTIONS, np.float32)
        for move, node in self.root_node.children.items():
            policy[move] = node.visits / (self.root_node.visits - 1)
        wdl = self.root_node.results / self.root_node.results.sum()
        return policy, wdl

    def simulate(self, game: Game, policy_function):
        """Simulates one playout.

        Args:
            game (Game): The game to be played.
            policy_function: The policy function.
        """
        node: Node = self.root_node
        while not node.is_leaf():
            move, next_node = node.select()
            game.make_move(move)
            node = next_node
        if not game.is_terminal():
            policy, wdl = policy_function(game.get_state())
            moves_with_probs = {move: policy[move] for move in game.get_legal_moves()}
            node.expand(moves_with_probs)
        else:
            wdl = np.array([0, 0, 1], np.float32)
        node.update_recursive(wdl)

    def make_move(self, move: int):
        """Makes a move. Helps reduce computation costs,
        as previously computed values are preserved.

        Args:
            move (int): The move id.
        """
        if move not in self.root_node.children:
            self.root_node = Node(None, 0, self.c_puct)
        else:
            self.root_node = self.root_node.children[move]
            self.root_node.parent = None
