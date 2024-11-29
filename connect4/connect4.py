import numpy as np

def create_grid(sizeX=6, sizeY=7):
    """Create an empty Connect 4 grid."""
    return np.zeros((sizeX, sizeY), dtype=int)

def reset(grid):
    """Reset the grid to an empty state."""
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

    # Check for a winner
    if has_won(grid, player, row, column):
        return grid, player  # Return the player who won
    elif not any(can_play(grid, col) for col in range(grid.shape[1])):  # Draw
        return grid, 0
    else:
        return grid, 0  # Game continues

def random_opponent_move(grid):
    """Choose a random valid move for the opponent."""
    moves = valid_move(grid)
    if not moves:
        return None
    return np.random.choice(moves)

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
    def check_direction(delta_row, delta_col):
        count = 1
        for direction in [1, -1]:
            r, c = row, column
            while True:
                r += direction * delta_row
                c += direction * delta_col
                if 0 <= r < grid.shape[0] and 0 <= c < grid.shape[1]:
                    if grid[r, c] == player:
                        count += 1
                    else:
                        break
                else:
                    break
        return count >= 4

    # Check all directions
    return (
        check_direction(0, 1) or  # Horizontal
        check_direction(1, 0) or  # Vertical
        check_direction(1, 1) or  # Diagonal up-right
        check_direction(1, -1)    # Diagonal up-left
    )

def get_player_to_play(grid):
    """Determine which player's turn it is."""
    num_player1 = np.count_nonzero(grid == 1)
    num_player_minus1 = np.count_nonzero(grid == -1)
    return 1 if num_player1 <= num_player_minus1 else -1

def to_state(grid):
    """Convert the grid to a string representation for MCTS."""
    return ''.join(grid.flatten().astype(str).tolist())



