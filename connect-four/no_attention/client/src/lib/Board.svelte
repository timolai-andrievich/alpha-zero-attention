<script lang="ts">
    import { createEventDispatcher } from 'svelte';

    const BOARD_HEIGHT = 6;
    const BOARD_WIDTH = 7;
    const IN_ROW = 4;
    let dispatch = createEventDispatcher();
    enum Cell {
        Empty,
        Red,
        Blue
    }
    const directions = ['south', 'east', 'southEast', 'southWest'];
    let board: Array<Array<Cell>> = Array(BOARD_HEIGHT)
        .fill(0)
        .map((_) => Array(BOARD_WIDTH).fill(Cell.Empty));
    let classBoard: Array<Array<String>> = Array(BOARD_HEIGHT)
        .fill(0)
        .map((_) => Array(BOARD_WIDTH).fill('empty'));
    export function reset() {
        board = Array(BOARD_HEIGHT)
            .fill(0)
            .map((_) => Array(BOARD_WIDTH).fill(Cell.Empty));
        classBoard = Array(BOARD_HEIGHT)
            .fill(0)
            .map((_) => Array(BOARD_WIDTH).fill('empty'));
    }
    function classFromCell(cell: Cell): String {
        switch (cell) {
            case Cell.Empty:
                return 'empty';
            case Cell.Red:
                return 'redPlayer';
            case Cell.Blue:
                return 'bluePlayer';
        }
    }
    function getLowestEmptyOrHighlightedRowNumber(column: number): number | null {
        if (board[0][column] != Cell.Empty) {
            return null;
        }
        let result: number = 0;
        for (let i = 1; i < BOARD_HEIGHT; i++) {
            if (board[i][column] == Cell.Empty) {
                result = i;
            } else {
                break;
            }
        }
        return result;
    }
    function highlightColumn(column: number) {
        let row: number | null = getLowestEmptyOrHighlightedRowNumber(column);
        if (row == null) {
            return;
        }
        classBoard[row][column] = 'highlighted';
    }
    function undoHighlight(column: number) {
        let row: number | null = getLowestEmptyOrHighlightedRowNumber(column);
        if (row == null) {
            return;
        }
        if (classBoard[row][column] == 'highlighted') {
            classBoard[row][column] = 'empty';
        }
    }
    function getCurrentPlayerColor(): Cell {
        let sum = 0;
        for (let row of board) {
            for (let cell of row) {
                if (cell != Cell.Empty) {
                    sum += 1;
                }
            }
        }
        if (sum % 2 == 0) {
            return Cell.Red;
        } else {
            return Cell.Blue;
        }
    }
    function insideBounds(column: number, row: number): boolean {
        return 0 <= column && column < BOARD_WIDTH && 0 <= row && row < BOARD_HEIGHT;
    }
    function checkInDirection(column: number, row: number, direction: string): boolean {
        if (!insideBounds(column, row)) {
            return false;
        }
        if (board[row][column] == Cell.Empty) {
            return false;
        }
        const startingCell = board[row][column];
        switch (direction) {
            case 'south':
                if (!insideBounds(column, row + IN_ROW - 1)) {
                    return false;
                }
                for (let delta: number = 0; delta < IN_ROW; delta++) {
                    if (board[row + delta][column] != startingCell) {
                        return false;
                    }
                }
                return true;

            case 'east':
                if (!insideBounds(column + IN_ROW - 1, row)) {
                    return false;
                }
                for (let delta: number = 0; delta < IN_ROW; delta++) {
                    if (board[row][column + delta] != startingCell) {
                        return false;
                    }
                }
                return true;

            case 'southEast':
                if (!insideBounds(column + IN_ROW - 1, row + IN_ROW - 1)) {
                    return false;
                }
                for (let delta: number = 0; delta < IN_ROW; delta++) {
                    if (board[row + delta][column + delta] != startingCell) {
                        return false;
                    }
                }
                return true;

            case 'southWest':
                if (!insideBounds(column - IN_ROW + 1, row + IN_ROW - 1)) {
                    return false;
                }
                for (let delta: number = 0; delta < IN_ROW; delta++) {
                    if (board[row + delta][column - delta] != startingCell) {
                        return false;
                    }
                }
                return true;

            default:
                return false;
        }
    }

    function checkIfFinished(): boolean {
        for (let column: number = 0; column < BOARD_WIDTH; column++) {
            for (let row: number = 0; row < BOARD_HEIGHT; row++) {
                for (const direction of directions.values()) {
                    if (checkInDirection(column, row, direction)) {
                        return true;
                    }
                }
            }
        }
        return isFull();
    }

    function getWinnerMessage(): string {
        for (let column: number = 0; column < BOARD_WIDTH; column++) {
            for (let row: number = 0; row < BOARD_HEIGHT; row++) {
                for (const direction of directions.values()) {
                    if (checkInDirection(column, row, direction)) {
                        switch (board[row][column]) {
                            case Cell.Blue:
                                return 'Blue won';
                            case Cell.Red:
                                return 'Red won';
                        }
                    }
                }
            }
        }
        return 'Draw';
    }

    function isFull(): boolean {
        for (let column: number = 0; column < BOARD_WIDTH; column++) {
            for (let row: number = 0; row < BOARD_HEIGHT; row++) {
                if (board[row][column] == Cell.Empty) {
                    return false;
                }
            }
        }
        return true;
    }
    function isLegalMove(column: number): boolean {
        return (
            column >= 0 &&
            column < BOARD_WIDTH &&
            board[0][column] == Cell.Empty &&
            !checkIfFinished()
        );
    }
    function getIntFromCell(cell: Cell) {
        switch (cell) {
            case Cell.Empty:
                return 0;
            case Cell.Red:
                return 1;
            case Cell.Blue:
                return -1;
        }
    }
    function getState(): Array<Array<number>> {
        let result = Array(BOARD_HEIGHT)
            .fill(0)
            .map((_) => Array(BOARD_WIDTH).fill(0));
        for (let row = 0; row < BOARD_HEIGHT; row++) {
            for (let column = 0; column < BOARD_WIDTH; column++) {
                result[row][column] = getIntFromCell(board[row][column]);
            }
        }
        return result;
    }
    function makeMove(column: number) {
        if (!isLegalMove(column)) {
            return;
        }
        let nextCell: Cell = getCurrentPlayerColor();
        let row = getLowestEmptyOrHighlightedRowNumber(column);
        if (row == null) {
            return;
        }
        board[row][column] = nextCell;
        classBoard[row][column] = classFromCell(nextCell);
        highlightColumn(column);
        if (checkIfFinished()) {
            dispatch('finished', { gameOverMessage: getWinnerMessage() });
        } else {
            if (event) {
                dispatch('moved', { gameState: getState() });
            }
        }
    }
    export function makeAIMove(column: number) {
        if (!isLegalMove(column)) {
            return;
        }
        let nextCell: Cell = getCurrentPlayerColor();
        let row = getLowestEmptyOrHighlightedRowNumber(column);
        if (row == null) {
            return;
        }
        board[row][column] = nextCell;
        classBoard[row][column] = classFromCell(nextCell);
        if (checkIfFinished()) {
            dispatch('finished', { gameOverMessage: getWinnerMessage() });
        }
    }
</script>

<div id="canvas">
    {#each [...Array(BOARD_HEIGHT).keys()] as row}
        {#each [...Array(BOARD_WIDTH).keys()] as column}
            <cell
                style="row: {row}; column: {column};"
                on:mouseenter={() => highlightColumn(column)}
                on:mouseleave={() => undoHighlight(column)}
                on:click={() => makeMove(column, true)}
                on:focus={() => {}}
                on:keypress={() => {}}
                class={classBoard[row][column]}
            />
        {/each}
    {/each}
</div>

<style>
    #canvas {
        width: 83vmin;
        height: 71vmin;
        display: grid;
        gap: 1vmin;
        grid-template-columns: repeat(7, 1fr);
    }

    cell {
        width: 11vmin;
        height: 11vmin;
        border-radius: 50%;
    }

    cell.empty {
        background-color: black;
    }

    cell.highlighted {
        background-color: orange;
    }

    cell.redPlayer {
        background-color: red;
    }

    cell.bluePlayer {
        background-color: blue;
    }
</style>
