import numpy as np  

def create_grid(sizeX=6, sizeY=7):
    """
    Create an empty Connect 4 grid.

    Parameters:
    - sizeX (int): Number of rows in the grid (default is 6).
    - sizeY (int): Number of columns in the grid (default is 7).

    Returns:
    - grid (ndarray): A NumPy 2D array initialized with zeros, representing an empty grid.
    
    The grid uses the following conventions:
    - 0 represents an empty cell.
    - 1 represents a cell occupied by Player 1 (AI).
    - -1 represents a cell occupied by Player -1 (random).
    """
    # Initialize a 2D array of zeros with dimensions (sizeX, sizeY)
    return np.zeros((sizeX, sizeY), dtype=int)

def reset(grid):
    """
    Reset the grid to an empty state without changing its dimensions.

    Parameters:
    - grid (ndarray): The current game grid.

    Returns:
    - grid (ndarray): A new grid with the same shape as the input grid, filled with zeros.
    """
    # Create a new array of zeros with the same shape as the input grid
    return np.zeros(grid.shape, dtype=int)

def play(grid_, column, player=None):
    """
    Place a player's piece into the specified column and update the grid.

    Parameters:
    - grid_ (ndarray): The current game grid.
    - column (int): The index of the column where the piece will be placed.
    - player (int, optional): The player making the move (1 or -1). If None, the function determines whose turn it is.

    Returns:
    - grid (ndarray): The updated game grid after the move.
    - winner (int): The result after the move:
        - 1 or -1 if the player has won.
        - 0 if the game is a draw or ongoing.
    
    Raises:
    - Exception: If the specified column is full and no more moves can be made in that column.
    """
    # Create a copy of the grid to avoid modifying the original grid outside this function
    grid = grid_.copy()
    
    # Determine which player's turn it is if not specified
    if player is None:
        player = get_player_to_play(grid)

    # Check if a move can be made in the specified column
    if can_play(grid, column):
        # Calculate the row index where the player's piece will land
        # The piece lands on top of the highest occupied cell in the column
        # Count the number of occupied cells in the column to find the next available row from the bottom
        row = grid.shape[0] - 1 - np.sum(np.abs(grid[:, column]), dtype=int)
        # Place the player's piece in the calculated row and specified column
        grid[row, column] = player
    else:
        # If the column is full, raise an exception to indicate no moves can be made in this column
        raise Exception(f"Error: Column {column} is full")

    # After making the move, check if the player has won the game
    if has_won(grid, player, row, column):
        # The current player has won; return the updated grid and the player's number
        return grid, player
    elif not any(can_play(grid, col) for col in range(grid.shape[1])):
        # If there are no valid moves left (the grid is full), it's a draw
        return grid, 0  # 0 indicates a draw
    else:
        # The game is still ongoing; no winner yet
        return grid, 0  # 0 indicates the game continues

def random_opponent_move(grid):
    """
    Select a random valid move for the opponent.

    Parameters:
    - grid (ndarray): The current game grid.

    Returns:
    - column (int or None): The column index where the opponent will play, or None if no valid moves are available.
    """
    # Get the list of valid moves (columns that are not full)
    moves = valid_move(grid)
    if not moves:
        # If there are no valid moves, return None
        return None
    # Randomly select one of the available columns for the move
    return np.random.choice(moves)

def can_play(grid, column):
    """
    Check if a move can be made in the specified column.

    Parameters:
    - grid (ndarray): The current game grid.
    - column (int): The index of the column to check.

    Returns:
    - can_play (bool): True if the column is not full; False if the column is full.
    """
    # A column is playable if the number of occupied cells is less than the number of rows
    # Sum the absolute values in the column to count occupied cells
    return np.sum(np.abs(grid[:, column])) < len(grid[:, column])

def valid_move(grid):
    """
    Get a list of all valid moves (columns where a piece can be placed).

    Parameters:
    - grid (ndarray): The current game grid.

    Returns:
    - valid_moves (list): A list of column indices that are not full.
    """
    # Iterate over all columns and collect the indices of those that are playable
    return [i for i in range(grid.shape[1]) if can_play(grid, i)]

def has_won(grid, player, row, column):
    """
    Determine if the player has won the game after making a move at (row, column).

    Parameters:
    - grid (ndarray): The current game grid.
    - player (int): The player number (1 or -1).
    - row (int): The row index where the piece was placed.
    - column (int): The column index where the piece was placed.

    Returns:
    - has_won (bool): True if the player has won; False otherwise.

    The function checks for four consecutive pieces in all directions:
    - Horizontal
    - Vertical
    - Diagonal (both directions)
    """
    def check_direction(delta_row, delta_col):
        """
        Check for four consecutive pieces in a specific direction.

        Parameters:
        - delta_row (int): The row increment (can be -1, 0, or 1).
        - delta_col (int): The column increment (can be -1, 0, or 1).

        Returns:
        - win_found (bool): True if four consecutive pieces are found in the direction; False otherwise.
        """
        count = 1  # Start counting from 1 to include the newly placed piece
        for direction in [1, -1]:  # Check both positive and negative directions
            r, c = row, column  # Start from the position of the new piece
            while True:
                # Move to the next cell in the specified direction
                r += direction * delta_row
                c += direction * delta_col
                # Check if the new position is within the bounds of the grid
                if 0 <= r < grid.shape[0] and 0 <= c < grid.shape[1]:
                    if grid[r, c] == player:
                        # If the cell contains the player's piece, increment the count
                        count += 1
                    else:
                        # If the cell is empty or contains the opponent's piece, stop checking in this direction
                        break
                else:
                    # If the new position is outside the grid, stop checking in this direction
                    break
        # Return True if four or more consecutive pieces are found
        return count >= 4

    # Check all four possible directions for a winning sequence
    return (
        check_direction(0, 1) or   # Horizontal check 
        check_direction(1, 0) or   # Vertical check 
        check_direction(1, 1) or   # Diagonal check 
        check_direction(1, -1)     # Diagonal check
    )

def get_player_to_play(grid):
    """
    Determine which player's turn it is based on the current state of the grid.

    Parameters:
    - grid (ndarray): The current game grid.

    Returns:
    - player (int): 1 if it's Player 1's turn; -1 if it's Player -1's turn.

    The function counts the number of pieces for each player:
    - If the counts are equal or Player 1 has fewer pieces, it's Player 1's turn.
    - If Player -1 has fewer pieces, it's Player -1's turn.
    """
    # Count the number of pieces placed by Player 1 (represented by 1)
    num_player1 = np.count_nonzero(grid == 1)
    # Count the number of pieces placed by Player -1 (represented by -1)
    num_player_minus1 = np.count_nonzero(grid == -1)
    # Decide whose turn it is based on the counts
    return 1 if num_player1 <= num_player_minus1 else -1

def to_state(grid):
    """
    Convert the grid to a string representation, useful for hashing or state comparison.

    Parameters:
    - grid (ndarray): The current game grid.

    Returns:
    - state_str (str): A string representation of the grid.

    The grid is flattened into a 1D array, converted to strings, and concatenated.
    This can be used in Monte Carlo Tree Search (MCTS) algorithms to represent states.
    """
    # Flatten the grid to a 1D array
    flattened_grid = grid.flatten()
    # Convert each cell value to a string
    string_grid = flattened_grid.astype(str)
    # Join all strings into a single string without separators
    return ''.join(string_grid.tolist())
