#!python3
from flask import Flask, request, abort, send_from_directory
from flask_cors import CORS
import numpy as np
import waitress

from connect4_mcts.game import Game
from connect4_mcts.policy import Model
from connect4_mcts.players import MctsPlayer

SVELTE_STATIC_PATH = './client/build'


class ModelWrapper:
    def __init__(self, *args, **kwargs):
        self.model: Model = Model(*args, **kwargs)
        self.model.load('model.pt')

    def get_action(self):
        options = {
            'c_puct': 2 ** .5,
            'n_playouts': 80,
            'temp': 0.,
        }
        if not request.is_json:
            abort(400)
        options.update(request.get_json())
        if not isinstance(options['temp'], float) and not isinstance(options['temp'], int) or options["temp"] < 0:
            return f'Invalid value for temp: {options["temp"]}', 400
        if not isinstance(options['n_playouts'], int):
            return f'Invalid value for n_playouts: {options["n_playouts"]}', 400
        if not isinstance(options['c_puct'], float):
            return f'Invalid value for c_puct: {options["c_puct"]}', 400
        if 'board' not in options:
            return 'Board not provided', 400
        states_and_board = options['board']
        try:
            pass
            # TODO: proper validation
            # if len(board) != Game.BOARD_HEIGHT:
            #     return f'Board is not of height {Game.BOARD_HEIGHT}', 400
            # for row in board:
            #     if len(row) != Game.BOARD_WIDTH:
            #         return f'Not all rows are of width {Game.BOARD_WIDTH}, row {row} is of width {len(row)}', 400
        except TypeError:
            return 'Wrong type passed as board', 400
        try:
            previous_boards = states_and_board[:-1]
            states = []
            for board in previous_boards:
                states.append(Game(board).get_state())
            states = np.array(states)
            board = states_and_board[-1]
            game = Game(board)
            player = MctsPlayer(
                game, self.model, options['c_puct'], options['n_playouts'], options['temp'], states)
            move = player.get_move()
        except Exception as e:
            return f'Something went wrong: {e}', 400
        return {'move': move}


def create_app():
    m = ModelWrapper(128, 10, 1e-4, 'cpu')
    app = Flask(__name__)
    CORS(app)

    @app.route('/')
    def svelte_base():
        return send_from_directory(SVELTE_STATIC_PATH, 'index.html')

    @app.route('/<path:path>')
    def svelte_path(path):
        return send_from_directory(SVELTE_STATIC_PATH, path)
    app.add_url_rule('/getAction', 'get_action',
                     m.get_action, methods=['POST'])
    return app


def main():
    app = create_app()
    waitress.serve(app, port=40080)


if __name__ == '__main__':
    main()
