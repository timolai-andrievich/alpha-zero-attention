<script lang="ts">
    import Board from '../lib/Board.svelte';
    enum State {
        Thinking,
        GameOver,
        NoOverlay
    }
    const apiUrl = '';
    let state = State.NoOverlay;
    let board: any;
    let gameOverMessage: String;
    async function timeout(delay: number): Promise<void> {
        return new Promise((x) => setTimeout(x, delay));
    }
    async function think(state: Array<Array<Array<number>>>): Promise<number> {
        console.log("Sending as state: ", state);
        let response = await fetch(`${apiUrl}/getAction`, {
            method: 'POST',
            body: JSON.stringify({ board: state, temp: 0.0 }),
            headers: { 'content-type': 'application/json' }
        });
        // TODO Error handling
        let json = await response.json();
        return json.move;
    }
</script>

<main>
    <div id="boardContainer">
        <overlay class={state == State.NoOverlay ? 'inactive' : 'active'}>
            {#if state == State.Thinking}
                Thinking...
            {:else if state == State.GameOver}
                Game Over!
            {/if}
        </overlay>
        <Board
            bind:this={board}
            on:finished={(event) => {
                state = State.GameOver;
                gameOverMessage = event.detail.gameOverMessage;
            }}
            on:moved={(event) => {
                state = State.Thinking;
                think(event.detail.gameState).then((action) => {
                    state = State.NoOverlay;
                    board.makeAIMove(action, false);
                });
            }}
        />
    </div>
    <button
        on:click={() => {
            board.reset();
            state = State.NoOverlay;
        }}>New Game</button
    >
</main>

<style>
    main {
        display: flex;
        align-items: center;
        margin: auto;
        width: fit-content;
        flex-direction: column;
    }

    #boardContainer {
        display: flex;
        align-items: center;
        margin: auto;
        width: fit-content;
        position: relative;
        user-select: none;
    }

    overlay {
        position: absolute;
        width: 100%;
        height: 100%;
        display: flex;
        align-items: center;
        align-content: center;
        justify-content: space-evenly;
        font-size: 10vmin;
        font-weight: bold;
        flex-direction: column;
    }

    overlay.inactive {
        display: none;
    }

    overlay.active {
        display: flex;
        flex: 1;
        background-color: rgba(255, 255, 255, 0.75);
    }

    button {
        font: inherit;
        background-color: transparent;
        border: none;
        font-size: 10vmin;
        font-weight: bold;
    }

    button:hover {
        color: darkblue;
        transition: color 100ms ease;
    }
</style>
