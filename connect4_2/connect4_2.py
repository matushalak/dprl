import numpy as np

def create_grid(sizeX=6, sizeY=7):
    return np.zeros((sizeX, sizeY), dtype=int)

def reset(grid):
    return np.zeros(grid.shape, dtype=int)

def play(grid_, column, player=None):
    """
    Play at the given column. If no player is provided, determine which player must play.
    Returns the new grid and a reward: +1 (win), -1 (loss), 0 (draw or ongoing).
    """
    grid = grid_.copy()
    if player is None:
        player = get_player_to_play(grid)

    if can_play(grid, column):
        row = grid.shape[0] - 1 - np.sum(np.abs(grid[:, column]), dtype=int)
        grid[row, column] = player
    else:
        raise Exception(f"Error: Column {column} is full")

    if has_won(grid, player, row, column):
        return grid, 1 if player == 1 else -1
    elif not any(can_play(grid, col) for col in range(grid.shape[1])):  # Draw
        return grid, 0
    else:
        return grid, 0

def random_opponent_move(grid):
    """Choose a random valid move for the opponent."""
    moves = valid_move(grid)
    return np.random.choice(moves) if moves else None

def can_play(grid, column):
    """Check if the given column is free."""
    return np.sum(np.abs(grid[:, column])) < len(grid[:, column])

def valid_move(grid):
    """Return a list of valid columns for the next move."""
    return [i for i in range(grid.shape[1]) if can_play(grid, i)]

def has_won(grid, player, row, column):
    """
    Check if the player has won after placing a piece.
    """
    player += 1
    grid += 1
    row_str = ''.join(grid[row, :].astype(str).tolist())
    col_str = ''.join(grid[:, column].astype(str).tolist())
    up_diag_str = ''.join(np.diagonal(grid, offset=(column - row)).astype(str).tolist())
    down_diag_str = ''.join(np.diagonal(np.rot90(grid), offset=-grid.shape[1] + (column + row) + 1).astype(str).tolist())
    grid -= 1

    victory_pattern = str(player) * 4
    if victory_pattern in row_str or victory_pattern in col_str or victory_pattern in up_diag_str or victory_pattern in down_diag_str:
        return True
    return False

def get_player_to_play(grid):
    """Determine which player's turn it is."""
    player_1 = 0.5 * np.abs(np.sum(grid - 1))
    player_2 = 0.5 * np.sum(grid + 1)
    return 1 if player_1 > player_2 else -1

def to_state(grid):
    """Convert the grid to a string representation for MCTS."""
    grid += 1
    res = ''.join(grid.astype(str).flatten().tolist())
    grid -= 1
    return res
