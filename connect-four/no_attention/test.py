import numpy as np
import os
import pytest
import torch

from connect4_mcts.game import Game, GameResult
from connect4_mcts.mcts import Node, MCTS
from connect4_mcts.policy import ConvLayer, SqueezeExcitation, ResidualBlock, Network, Model
from connect4_mcts.coach import Coach
from connect4_mcts.players import RandomPlayer, MctsPlayer, ModelPlayer


np.random.seed(42)


def test_game():
    game = Game()
    assert not game.is_terminal()
    assert not game._is_full()
    assert game.get_legal_moves() == list(range(Game.BOARD_WIDTH))
    assert game.get_winner() == GameResult.GameOngoing
    for _ in range(6):
        game.make_move(0)
    with pytest.raises(RuntimeError):
        game.make_move(0)
    assert game.get_legal_moves() == list(range(1, Game.BOARD_WIDTH))
    for _ in range(3):
        game.make_move(1)
        game.make_move(2)
    game.make_move(1)
    assert game.get_winner() == GameResult.RedWins


def test_node():
    root = Node(None, 0, 1)
    assert root.is_leaf()
    assert root.is_root()
    game = Game()
    moves_with_probs = {
        move: 1 / len(game.get_legal_moves()) for move in game.get_legal_moves()
    }
    root.expand(moves_with_probs)
    _action, node = root.select()
    node.update_recursive(np.array([0, 1, 1], np.float32))
    assert (root.results == np.array([1, 1, 0])).all()
    assert (node.results == np.array([0, 1, 1])).all()


def test_tree():
    def uniform_policy(state):
        return np.ones(Game.NUM_ACTIONS) / Game.NUM_ACTIONS, np.ones(3) / 3
    game = Game()
    initial_state = game.get_state()
    tree = MCTS(2 ** .5)
    policy, wdl = tree.run(game, uniform_policy, 100)
    assert (game.get_state() == initial_state).all()


def test_conv_layer():
    conv = ConvLayer(16, 4)
    x = torch.randn((10, 16, 8, 8))
    x = conv(x)
    assert x.size() == (10, 4, 8, 8)


def test_squeeze_exitation():
    se = SqueezeExcitation(16)
    x = torch.randn((10, 16, 8, 8))
    x = se(x)
    assert x.size() == (10, 16, 8, 8)


def test_res_block():
    block = ResidualBlock(16)
    x = torch.randn((10, 16, 8, 8))
    x = block(x)
    assert x.size() == (10, 16, 8, 8)


def test_net():
    net = Network(128, 10)
    x = torch.randn(
        (10, Game.STATE_LAYERS, Game.STATE_HEIGHT, Game.STATE_WIDTH))
    pol, wdl = net(x)
    assert pol.size() == (10, Game.NUM_ACTIONS)
    assert wdl.size() == (10, 3)


def test_model():
    model = Model(8, 1, 1e-4, 'cpu')
    state = np.random.randn(
        Game.STATE_LAYERS, Game.STATE_HEIGHT, Game.STATE_WIDTH)
    pol, wdl = model.policy_function(state)
    assert pol.shape == (Game.NUM_ACTIONS,)
    assert wdl.shape == (3,)
    states = np.random.randn(
        10, Game.STATE_LAYERS, Game.STATE_HEIGHT, Game.STATE_WIDTH)
    y_wdl = np.random.randn(10, 3)
    y_wdl = y_wdl / np.sum(y_wdl, -1, keepdims=True)
    y_pol = np.random.randn(10, Game.NUM_ACTIONS)
    y_pol = y_pol / np.sum(y_pol, -1, keepdims=True)
    model.train(states, y_pol, y_wdl, 3)
    before_save = model.policy_function(state)
    model.save('test.pt')
    new_model = Model(8, 1, 1e-3, 'cpu')
    new_model.load('test.pt')
    if os.path.exists('test.pt'):
        os.remove('test.pt')
    after_save = new_model.policy_function(state)
    assert (before_save[0] == after_save[0]).all()
    assert (before_save[1] == after_save[1]).all()


def test_coach():
    model = Model(8, 1, 1e-4, 'cpu')
    coach = Coach(model, 10)
    coach.generate_game(2 ** .5, 10, 1)
    coach.generate_game(2 ** .5, 10, 0)
    coach.generate_games(2, 2 ** .5, 10, 0)
    coach.train(16, 4)
    coach.train_epochs(16, 4, 1)


def test_random_player():
    game = Game()
    player = RandomPlayer(game)
    while not game.is_terminal():
        move = player.get_move()
        game.make_move(move)
        player.make_move(move)


def test_model_player():
    game = Game()
    model = Model(8, 1, 1e-4, 'cpu')
    player = ModelPlayer(game, model)
    while not game.is_terminal():
        move = player.get_move()
        game.make_move(move)
        player.make_move(move)


def test_mcts_player_with_temp_zero():
    game = Game()
    model = Model(8, 1, 1e-4, 'cpu')
    player = MctsPlayer(game, model, 2 ** .5, 10, 0)
    while not game.is_terminal():
        move = player.get_move()
        game.make_move(move)
        player.make_move(move)


def test_mcts_player():
    game = Game()
    model = Model(8, 1, 1e-4, 'cpu')
    player = MctsPlayer(game, model, 2 ** .5, 10, 1)
    while not game.is_terminal():
        move = player.get_move()
        game.make_move(move)
        player.make_move(move)
