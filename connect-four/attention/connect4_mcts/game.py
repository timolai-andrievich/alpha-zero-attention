"""Contains the game logic and everything related, such as the state generation.
"""
import copy
import enum
from typing import List, Optional

import numpy as np


Move = int


class GameResult(enum.IntEnum):
    """Enumerates the results of the game: 
        - RED_WINS
        - RED_LOSES
        - DRAW
        - GAME_ONGOING
    """
    RED_WINS = 1
    RED_LOSES = -1
    DRAW = 0
    GAME_ONGOING = 2


class Cell(enum.IntEnum):
    """Enumerates types of cells on the board:
        - RED (The first player)
        - YELLOW (The second player)
        - EMPTY
    """
    RED = 1
    YELLOW = -1
    EMPTY = 0


class Game:
    """Encapsulates the game logic.
    """
    SYMBOLS_REQUIRED_IN_ROW: int = 4

    BOARD_HEIGHT: int = 6
    BOARD_WIDTH: int = 7

    STATE_HEIGHT: int = 6
    STATE_WIDTH: int = 7
    STATE_LAYERS: int = 4

    NUM_ACTIONS: int = BOARD_WIDTH

    _DEFAULT_BOARD: List[List[Cell]] = [
        [Cell.EMPTY for _column in range(7)] for _row in range(6)
    ]

    def __init__(self, board: Optional[List[List[Cell]]] = None):
        """Creates a game with a given board. If no board is provided, empty board is used.

        Args:
            board (Optional[List[List[Cell]]], optional): The board to use. Defaults to None.
        """
        if board is None:
            board = self._DEFAULT_BOARD
        self._board: List[List[Cell]] = copy.deepcopy(board)

    def get_state(self) -> np.ndarray:
        """Creates a state to be passed into the policy function.

        Returns:
            np.ndarray: The state that represents the current position.
        """
        assert Game.STATE_LAYERS == 4
        result = np.zeros(
            (Game.STATE_LAYERS, Game.STATE_HEIGHT, Game.STATE_WIDTH), np.float32
        )
        for row in range(Game.BOARD_HEIGHT):
            for column in range(Game.BOARD_WIDTH):
                cell: Cell = self._board[row][column]
                result[0, row, column] = 1
                if cell == Cell.RED:
                    result[1, row, column] = 1
                    result[2, row, column] = 1
                elif cell == Cell.YELLOW:
                    result[1, row, column] = -1
                    result[3, row, column] = 1
        return result

    def _get_next_move_color(self) -> Cell:
        nonempty_cells_count: int = 0
        for row in range(Game.BOARD_HEIGHT):
            for column in range(Game.BOARD_WIDTH):
                cell: Cell = self._board[row][column]
                if cell in (Cell.RED, Cell.YELLOW):
                    nonempty_cells_count += 1
        if nonempty_cells_count % 2 == 0:
            return Cell.RED
        return Cell.YELLOW

    def make_move(self, move: Move):
        """Make a move with ID `move`. In the case of "Connect Four",
        id is just a column in which the chip will be dropped.

        Args:
            move (Move): The id of the move to be made.

        Raises:
            RuntimeError: Move is illegal (wrong id or the move cannot 
            be made due to the rules of the game.)
        """
        assert Game.BOARD_HEIGHT > 0
        column: int = move
        if not self._in_bounds(0, column):
            raise RuntimeError(f"Column {column} is out of bounds for this board")
        if self._board[0][column] != Cell.EMPTY:
            raise RuntimeError(
                f"Couldn't make a move, as column {column} is already full"
            )
        row = 0
        while (
            row + 1 < Game.BOARD_HEIGHT and self._board[row + 1][column] == Cell.EMPTY
        ):
            row += 1
        next_color: Cell = self._get_next_move_color()
        self._board[row][column] = next_color

    def _is_full(self) -> bool:
        for row in range(Game.BOARD_HEIGHT):
            for column in range(Game.BOARD_WIDTH):
                cell: Cell = self._board[row][column]
                if cell == Cell.EMPTY:
                    return False
        return True

    def _in_bounds(self, row: int, column: int) -> bool:
        return 0 <= row < Game.BOARD_HEIGHT and 0 <= column < Game.BOARD_WIDTH

    def _check_from_cell_south(self, row: int, column: int) -> bool:
        if not self._in_bounds(row, column):
            return False
        starting_cell: Cell = self._board[row][column]
        if starting_cell == Cell.EMPTY:
            return False
        for delta in range(1, Game.SYMBOLS_REQUIRED_IN_ROW):
            if not self._in_bounds(row + delta, column):
                return False
            cell: Cell = self._board[row + delta][column]
            if cell != starting_cell:
                return False
        return True

    def _check_from_cell_east(self, row: int, column: int) -> bool:
        if not self._in_bounds(row, column):
            return False
        starting_cell: Cell = self._board[row][column]
        if starting_cell == Cell.EMPTY:
            return False
        for delta in range(1, Game.SYMBOLS_REQUIRED_IN_ROW):
            if not self._in_bounds(row, column + delta):
                return False
            cell: Cell = self._board[row][column + delta]
            if cell != starting_cell:
                return False
        return True

    def _check_from_cell_southeast(self, row: int, column: int) -> bool:
        if not self._in_bounds(row, column):
            return False
        starting_cell: Cell = self._board[row][column]
        if starting_cell == Cell.EMPTY:
            return False
        for delta in range(1, Game.SYMBOLS_REQUIRED_IN_ROW):
            if not self._in_bounds(row + delta, column + delta):
                return False
            cell: Cell = self._board[row + delta][column + delta]
            if cell != starting_cell:
                return False
        return True

    def _check_from_cell_southwest(self, row: int, column: int) -> bool:
        if not self._in_bounds(row, column):
            return False
        starting_cell: Cell = self._board[row][column]
        if starting_cell == Cell.EMPTY:
            return False
        for delta in range(1, Game.SYMBOLS_REQUIRED_IN_ROW):
            if not self._in_bounds(row + delta, column - delta):
                return False
            cell: Cell = self._board[row + delta][column - delta]
            if cell != starting_cell:
                return False
        return True

    def _check_from_cell_in_all_directions(self, row: int, column: int):
        if not self._in_bounds(row, column):
            return False
        if self._board[row][column] == Cell.EMPTY:
            return False
        return (
            self._check_from_cell_east(row, column)
            or self._check_from_cell_south(row, column)
            or self._check_from_cell_southeast(row, column)
            or self._check_from_cell_southwest(row, column)
        )

    def is_terminal(self) -> bool:
        """Returns true if the game is in the terminal position.

        Returns:
            bool: Whether the game is into a terminal position or not.
        """
        for row in range(Game.BOARD_HEIGHT):
            for column in range(Game.BOARD_WIDTH):
                if self._check_from_cell_in_all_directions(row, column):
                    return True
        return self._is_full()

    def get_winner(self) -> GameResult:
        """Returns the game result in current position.

        Returns:
            GameResult: The result of the game.
        """
        def game_result_from_cell(cell: Cell) -> GameResult:
            match cell:
                case Cell.RED:
                    return GameResult.RED_WINS
                case Cell.YELLOW:
                    return GameResult.RED_LOSES

        for row in range(Game.BOARD_HEIGHT):
            for column in range(Game.BOARD_WIDTH):
                cell: Cell = self._board[row][column]
                if self._check_from_cell_in_all_directions(row, column):
                    return game_result_from_cell(cell)
        if self._is_full():
            return GameResult.DRAW
        return GameResult.GAME_ONGOING

    def get_legal_moves(self) -> List[Move]:
        """Returs the list of ids of legal moves.

        Returns:
            List[Move]: Ids of legal moves.
        """
        assert Game.BOARD_HEIGHT > 0
        result: List[int] = []
        for column in range(Game.BOARD_WIDTH):
            if self._board[0][column] == Cell.EMPTY:
                result.append(column)
        return result

    def copy(self) -> any:
        """Returns the deep copy of the current game.

        Returns:
            Game: Deep copy of this game.
        """
        return Game(self._board)
