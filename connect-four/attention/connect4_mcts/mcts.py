from typing import List, Tuple, Dict
import math

import numpy as np

from .game import Game


Policy = np.ndarray
WDL = np.ndarray


class Node:
    def __init__(self, parent, prior: float, c_puct: float):
        self.parent: Node = parent
        self.prior: float = prior
        self.c_puct: float = c_puct
        self.children: Dict[int, Node] = {}
        self.visits: int = 0
        self.results: np.ndarray = np.zeros(3, np.float32)

    def _avg(self) -> float:
        results_to_score: np.ndarray = np.array([1, .5, 0])
        if self.visits == 0:
            return 0
        return self.results.dot(results_to_score) / self.visits

    def select(self) -> Tuple[int, any]:
        return max(self.children.items(), key=lambda x: x[1].ucb_score())

    def expand(self, moves_with_probs: Dict[int, float]):
        if not self.is_leaf():
            raise RuntimeError('Trying to expand non-leaf node')
        for move, prior in moves_with_probs.items():
            self.children[move] = Node(self, prior, self.c_puct)

    def is_root(self) -> bool:
        return self.parent == None

    def is_leaf(self) -> bool:
        return len(self.children) == 0

    def ucb_score(self) -> float:
        if self.is_root():
            raise RuntimeError('Trying to get UCB score of the root node')
        return -self._avg() + self.c_puct * self.prior * math.sqrt(self.parent.visits) / (self.visits + 1)

    def update(self, score: np.ndarray):
        self.visits += 1
        self.results += score

    def update_recursive(self, new_score):
        self.update(new_score)
        if not self.is_root():
            self.parent.update_recursive(new_score[::-1])


class MCTS:
    def __init__(self, c_puct: float):
        self.c_puct: float = c_puct
        self.root_node: Node = Node(None, 0, self.c_puct)

    def run(self, game: Game, policy_function, iterations: int, states) -> Tuple[Policy, WDL]:
        for _ in range(iterations):
            self.simulate(game.copy(), policy_function, states)
        policy = np.zeros(Game.NUM_ACTIONS, np.float32)
        for move, node in self.root_node.children.items():
            policy[move] = node.visits / (self.root_node.visits - 1)
        wdl = self.root_node.results / self.root_node.results.sum()
        return policy, wdl

    def simulate(self, game: Game, policy_function, states):
        node: Node = self.root_node
        while not node.is_leaf():
            move, next_node = node.select()
            game.make_move(move)
            node = next_node
        if not game.is_terminal():
            state = np.array(list(states) + [game.get_state()])
            policy, wdl = policy_function(state)
            moves_with_probs = {move: policy[move] for move in game.get_legal_moves()}
            node.expand(moves_with_probs)
        else:
            wdl = np.array([0, 0, 1], np.float32)
        node.update_recursive(wdl)

    def make_move(self, move: int):
        if move not in self.root_node.children:
            self.root_node = Node(None, 0, self.c_puct)
        else:
            self.root_node = self.root_node.children[move]
            self.root_node.parent = None
