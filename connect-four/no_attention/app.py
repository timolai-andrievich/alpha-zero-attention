#!python3
"""The flask application file."""
from flask import Flask, request, abort, send_from_directory
from flask_cors import CORS
import numpy as np
import waitress

from connect4_mcts.game import Game
from connect4_mcts.policy import Model
from connect4_mcts.players import MctsPlayer

SVELTE_STATIC_PATH = "./client/build"


class ModelWrapper:
    """Wraps the model, and provides methods for certain paths."""
    def __init__(self, *args, **kwargs):
        """Wraps the model, and provides methods for certain paths.
        Arguments are passed into a model.
        """
        self.model: Model = Model(*args, **kwargs)
        self.model.load("model.pt")

    def get_action(self):
        """Flask application method to get the next move from the game state.

        Returns:
            Dict: The json with the response.
        """
        options = {
            "c_puct": 2**0.5,
            "n_playouts": 180,
            "temp": 0.0,
        }
        if not request.is_json:
            abort(400)
        options.update(request.get_json())
        if (
            not isinstance(options["temp"], float)
            and not isinstance(options["temp"], int)
            or options["temp"] < 0
        ):
            return f'Invalid value for temp: {options["temp"]}', 400
        if not isinstance(options["n_playouts"], int):
            return f'Invalid value for n_playouts: {options["n_playouts"]}', 400
        if not isinstance(options["c_puct"], float):
            return f'Invalid value for c_puct: {options["c_puct"]}', 400
        if "board" not in options:
            return "Board not provided", 400
        board = options["board"]
        try:
            if len(board) != Game.BOARD_HEIGHT:
                return f"Board is not of height {Game.BOARD_HEIGHT}", 400
            for row in board:
                if len(row) != Game.BOARD_WIDTH:
                    return (
                        f"Not all rows are of width {Game.BOARD_WIDTH}, row {row} is of width {len(row)}",
                        400,
                    )
        except TypeError:
            return "Wrong type passed as board", 400
        try:
            game = Game(board)
            player = MctsPlayer(
                game,
                self.model,
                options["c_puct"],
                options["n_playouts"],
                options["temp"],
            )
            move = player.get_move()
        except Exception as exception: # pylint: disable=(broad-exception-caught)
            return f"Something went wrong: {exception}", 400
        return {"move": move}


def create_app() -> Flask:
    """Creates the Flask app.

    Returns:
        Flask: App.
    """
    model = ModelWrapper(128, 10, 1e-4, "cpu")
    app = Flask(__name__)
    CORS(app)
    @app.route("/")
    def svelte_base():
        return send_from_directory(SVELTE_STATIC_PATH, "index.html")

    @app.route("/<path:path>")
    def svelte_path(path):
        return send_from_directory(SVELTE_STATIC_PATH, path)

    app.add_url_rule("/getAction", "get_action", model.get_action, methods=["POST"])
    return app


def main():
    """The entry function of the script.
    """
    app = create_app()
    waitress.serve(app, port=40080)


if __name__ == "__main__":
    main()
