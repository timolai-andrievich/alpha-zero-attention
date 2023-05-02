import copy
import enum
from typing import List, Optional

import numpy as np


Move = int


class GameResult(enum.IntEnum):
    RedWins = 1
    RedLoses = -1
    Draw = 0
    GameOngoing = 2


class Cell(enum.IntEnum):
    Red = 1
    Yellow = -1
    Empty = 0


class Game:
    SYMBOLS_REQUIRED_IN_ROW: int = 4

    BOARD_HEIGHT: int = 6
    BOARD_WIDTH: int = 7

    STATE_HEIGHT: int = 6
    STATE_WIDTH: int = 7
    STATE_LAYERS: int = 4

    NUM_ACTIONS: int = BOARD_WIDTH

    _DEFAULT_BOARD: List[List[Cell]] = [
        [Cell.Empty for _column in range(7)] for _row in range(6)
    ]

    def __init__(self, board: Optional[List[List[Cell]]] = None):
        if board is None:
            board = self._DEFAULT_BOARD
        self._board: List[List[Cell]] = copy.deepcopy(board)

    def get_state(self) -> np.ndarray:
        assert Game.STATE_LAYERS == 4
        result = np.zeros((Game.STATE_LAYERS, Game.STATE_HEIGHT, Game.STATE_WIDTH), np.float32)
        for row in range(Game.BOARD_HEIGHT):
            for column in range(Game.BOARD_WIDTH):
                cell: Cell = self._board[row][column]
                result[0, row, column] = 1
                if cell == Cell.Red:
                    result[1, row, column] = 1
                    result[2, row, column] = 1
                elif cell == Cell.Yellow:
                    result[1, row, column] = -1
                    result[3, row, column] = 1
        return result

    def _get_next_move_color(self) -> Cell:
        nonempty_cells_count: int = 0
        for row in range(Game.BOARD_HEIGHT):
            for column in range(Game.BOARD_WIDTH):
                cell: Cell = self._board[row][column]
                if cell == Cell.Red or cell == Cell.Yellow:
                    nonempty_cells_count += 1
        if nonempty_cells_count % 2 == 0:
            return Cell.Red
        else:
            return Cell.Yellow

    def make_move(self, move: Move):
        assert Game.BOARD_HEIGHT > 0
        column: int = move
        if not self._in_bounds(0, column):
            raise RuntimeError(f"Column {column} is out of bounds for this board")
        if self._board[0][column] != Cell.Empty:
            raise RuntimeError(
                f"Couldn't make a move, as column {column} is already full"
            )
        row = 0
        while row + 1 < Game.BOARD_HEIGHT and self._board[row + 1][column] == Cell.Empty:
            row += 1
        next_color: Cell = self._get_next_move_color()
        self._board[row][column] = next_color

    def _is_full(self) -> bool:
        for row in range(Game.BOARD_HEIGHT):
            for column in range(Game.BOARD_WIDTH):
                cell: Cell = self._board[row][column]
                if cell == Cell.Empty:
                    return False
        return True

    def _in_bounds(self, row: int, column: int) -> bool:
        return 0 <= row < Game.BOARD_HEIGHT and 0 <= column < Game.BOARD_WIDTH

    def _check_from_cell_south(self, row: int, column: int) -> bool:
        if not self._in_bounds(row, column):
            return False
        starting_cell: Cell = self._board[row][column]
        if starting_cell == Cell.Empty:
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
        if starting_cell == Cell.Empty:
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
        if starting_cell == Cell.Empty:
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
        if starting_cell == Cell.Empty:
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
        if self._board[row][column] == Cell.Empty:
            return False
        return (
            self._check_from_cell_east(row, column)
            or self._check_from_cell_south(row, column)
            or self._check_from_cell_southeast(row, column)
            or self._check_from_cell_southwest(row, column)
        )

    def is_terminal(self) -> bool:
        for row in range(Game.BOARD_HEIGHT):
            for column in range(Game.BOARD_WIDTH):
                if self._check_from_cell_in_all_directions(row, column):
                    return True
        if self._is_full():
            return True
        else:
            return False

    def get_winner(self) -> GameResult:
        def game_result_from_cell(cell: Cell) -> GameResult:
            match cell:
                case Cell.Red:
                    return GameResult.RedWins
                case Cell.Yellow:
                    return GameResult.RedLoses
        for row in range(Game.BOARD_HEIGHT):
            for column in range(Game.BOARD_WIDTH):
                cell: Cell = self._board[row][column]
                if self._check_from_cell_in_all_directions(row, column):
                    return game_result_from_cell(cell)
        if self._is_full():
            return GameResult.Draw
        else:
            return GameResult.GameOngoing

    def get_legal_moves(self) -> List[Move]:
        assert Game.BOARD_HEIGHT > 0
        result: List[int] = []
        for column in range(Game.BOARD_WIDTH):
            if self._board[0][column] == Cell.Empty:
                result.append(column)
        return result
    
    def copy(self) -> any:
        return Game(self._board)
