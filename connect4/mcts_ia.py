import numpy as np
import random
from connect4.node import Node
from connect4.connect4 import create_grid, play, valid_move, get_player_to_play

def random_play(grid):
    """Play a completely random game starting from the given state."""
    grid = grid.copy()
    while True:
        moves = valid_move(grid)
        if len(moves) == 0:
            return 0  # Draw
        selected_move = random.choice(moves)
        grid, winner = play(grid, selected_move)
        if winner != 0:
            return winner  # Return the winner

def train_mcts_during(mcts, training_time):
    """Train MCTS for a specified amount of time (in milliseconds)."""
    import time
    start = int(round(time.time() * 1000))
    current = start
    while (current - start) < training_time:
        mcts = train_mcts_once(mcts)
        current = int(round(time.time() * 1000))
    return mcts

# def train_mcts_once(mcts=None):
#     """Perform one iteration of MCTS (selection, expansion, simulation, backpropagation)."""
#     if mcts is None:
#         initial_state = create_grid()
#         mcts = Node(initial_state, 0, move=None, parent=None)

#     node = mcts

#     # Selection Phase
#     while node.children:
#         node = node.best_child()

#     # Expansion Phase
#     winner = node.winner
#     if winner == 0:  # Non-terminal node
#         moves = valid_move(node.state)
#         for move in moves:
#             new_state, new_winner = play(node.state, move)
#             child_node = Node(new_state, new_winner, move=move, parent=node)
#             node.children.append(child_node)

#     # Simulation Phase
#     node = random.choice(node.children)
#     victorious = simulate_random_playout(node.state)

#     # Backpropagation Phase
#     while node:
#         node.games += 1
#         if victorious == get_player_to_play(node.state) * -1:
#             node.win += 1
#         node = node.parent

#     return mcts


def train_mcts_once(mcts=None):
    """Perform one iteration of MCTS (selection, expansion, simulation, backpropagation)."""
    if mcts is None:
        initial_state = create_grid()
        mcts = Node(initial_state, winner=0, move=None, parent=None)

    node = mcts

    # Selection Phase
    while node.children:
        node = node.best_child()

    # Expansion Phase
    winner = node.winner
    if winner == 0:  # Non-terminal node
        moves = valid_move(node.state)
        for move in moves:
            new_state, new_winner = play(node.state, move)
            child_node = Node(new_state, winner=new_winner, move=move, parent=node)
            node.children.append(child_node)

        # Simulation Phase
        node = random.choice(node.children)
        victorious = simulate_random_playout(node.state)
    else:
        # Terminal node: use the winner directly
        victorious = winner

    # Backpropagation Phase
    while node:
        node.games += 1
        if victorious == get_player_to_play(node.state) * -1:
            node.win += 1
        node = node.parent

    return mcts


def simulate_random_playout(state):
    """Simulate a random playout from the given state."""
    grid = state.copy()
    current_player = get_player_to_play(grid)
    while True:
        moves = valid_move(grid)
        if not moves:
            return 0  # Draw
        move = random.choice(moves)
        grid, winner = play(grid, move, player=current_player)
        if winner != 0:
            return winner
        current_player = -current_player