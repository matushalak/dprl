import numpy as np  
import random      
from connect4.node import Node
from connect4.connect4 import create_grid, play, valid_move, get_player_to_play

# - create_grid: to create an empty Connect 4 grid
# - play: to make a move in the game
# - valid_move: to get valid moves from the current state
# - get_player_to_play: to determine whose turn it is

def random_play(grid):
    """
    Play a completely random game starting from the given state.

    Parameters:
    - grid (ndarray): The current game grid.

    Returns:
    - winner (int): The winner of the game:
        - 1 or -1 if a player wins.
        - 0 if the game ends in a draw.
    """
    grid = grid.copy()  # Copy the grid to avoid modifying the original state
    while True:
        moves = valid_move(grid)  # Get a list of valid columns where a move can be made
        if len(moves) == 0:
            # If there are no valid moves left, the game is a draw
            return 0  # Return 0 to indicate a draw
        selected_move = random.choice(moves)  # Randomly select one of the valid moves
        grid, winner = play(grid, selected_move)  # Apply the move and get the updated grid and winner status
        if winner != 0:
            # If there is a winner (1 or -1), return the winner
            return winner
        # If the game is ongoing (winner == 0), continue the loop and make another move

def train_mcts_during(mcts, training_time):
    """
    Train the Monte Carlo Tree Search (MCTS) for a specified amount of time.

    Parameters:
    - mcts (Node): The root node of the MCTS tree.
    - training_time (int): The amount of time to train, in milliseconds.

    Returns:
    - mcts (Node): The updated MCTS tree after training.
    """
    import time  # Import time module to measure time intervals
    start = int(round(time.time() * 1000))  # Record the start time in milliseconds
    current = start  # Initialize current time to start time
    while (current - start) < training_time:
        # Continue training until the specified training time has elapsed
        mcts = train_mcts_once(mcts)  # Perform one iteration of MCTS and update the tree
        current = int(round(time.time() * 1000))  # Update the current time
    return mcts  # Return the updated MCTS tree

def train_mcts_once(mcts=None):
    """
    Perform one iteration of the MCTS algorithm, which includes:
    - Selection
    - Expansion
    - Simulation
    - Backpropagation

    Parameters:
    - mcts (Node, optional): The root node of the MCTS tree. If None, a new tree is created.

    Returns:
    - mcts (Node): The updated MCTS tree after one iteration.
    """
    if mcts is None:
        # If no MCTS tree is provided, create a new root node with the initial game state
        initial_state = create_grid()  # Create an empty Connect 4 grid
        mcts = Node(initial_state, winner=0, move=None, parent=None)  # Initialize root node with no move and no parent

    node = mcts  # Start from the root node for selection

    # Selection Phase:
    # Traverse the tree by selecting child nodes until reaching a leaf node (node with no children)
    while node.children:
        # Use the best_child method to select the child node with the highest win rate
        node = node.best_child()  # Move to the best child node

    # At this point, node is a leaf node (no children)

    # Expansion Phase:
    # If the node represents a non-terminal state (game is not over), expand it
    winner = node.winner  # Get the winner status at this node
    if winner == 0:  # If the game is ongoing (no winner yet)
        moves = valid_move(node.state)  # Get all valid moves from the current state
        for move in moves:
            # For each valid move, create a new child node representing the state after the move
            new_state, new_winner = play(node.state, move)  # Apply the move to get the new state and winner status
            child_node = Node(new_state, winner=new_winner, move=move, parent=node)  # Create a new child node
            node.children.append(child_node)  # Add the child node to the current node's children list

        # Simulation Phase:
        # Randomly select one of the child nodes and perform a simulation (playout)
        node = random.choice(node.children)  # Select a random child node to simulate from
        victorious = simulate_random_playout(node.state)  # Simulate a random playout from this state and get the winner
    else:
        # If the node is terminal (game has ended), use the winner directly
        victorious = winner  # Set the victorious player to the winner at this node

    # Backpropagation Phase:
    # Update the statistics (win counts, game counts) for the nodes along the path from the simulation node back to the root
    while node:
        node.games += 1  # Increment the number of games simulated from this node
        # Determine which player is to play at this node
        current_player = get_player_to_play(node.state)
        # If the victorious player is the opponent of the player to play at this node
        if victorious == current_player * -1:
            node.win += 1  # Increment the win count for this node
        # Move up to the parent node
        node = node.parent

    return mcts  # Return the updated MCTS tree

def simulate_random_playout(state):
    """
    Simulate a random playout (game) from the given state to the end of the game.

    Parameters:
    - state (ndarray): The game grid representing the current state.

    Returns:
    - winner (int): The winner of the simulated game:
        - 1 or -1 if a player wins.
        - 0 if the game ends in a draw.
    """
    grid = state.copy()  # Copy the state to avoid modifying the original grid
    current_player = get_player_to_play(grid)  # Determine whose turn it is to play
    while True:
        moves = valid_move(grid)  # Get a list of valid moves from the current state
        if not moves:
            # If there are no valid moves left, the game is a draw
            return 0  # Return 0 to indicate a draw
        move = random.choice(moves)  # Randomly select a valid move
        grid, winner = play(grid, move, player=current_player)  # Apply the move and get the updated grid and winner
        if winner != 0:
            # If there is a winner (1 or -1), return the winner
            return winner  # End the simulation and return the winner
        # Switch to the other player for the next move
        current_player = -current_player
