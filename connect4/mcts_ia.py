# # import numpy as np
# # import random
# # from connect4.mcts import Node
# # from connect4.connect4 import create_grid, play, valid_move, get_player_to_play

# # def heuristic_play(grid):
# #     """
# #     Simulate a game using a heuristic: prioritize winning moves, then blocking moves,
# #     then random moves if no immediate win/loss is possible.
# #     """
# #     while True:
# #         moves = valid_move(grid)
# #         if len(moves) == 0:
# #             return 0  # Draw
# #         player_to_play = get_player_to_play(grid)
# #         # Prioritize winning moves
# #         for move in moves:
# #             _, winner = play(grid, move, player=player_to_play)
# #             if winner == player_to_play:
# #                 return winner
# #         # Block opponent's winning moves
# #         for move in moves:
# #             _, winner = play(grid, move, player=-player_to_play)
# #             if winner == -player_to_play:
# #                 return player_to_play
# #         # Otherwise, play randomly
# #         selected_move = random.choice(moves)
# #         grid, winner = play(grid, selected_move)
# #         if np.abs(winner) > 0:
# #             return winner

# # def train_mcts_during(mcts, training_time):
# #     """
# #     Train MCTS for a specified amount of time (in milliseconds).
# #     """
# #     import time
# #     start = int(round(time.time() * 1000))
# #     current = start
# #     while (current - start) < training_time:
# #         mcts = train_mcts_once(mcts)
# #         current = int(round(time.time() * 1000))
# #     return mcts

# # def train_mcts_once(mcts=None):
# #     """
# #     Perform one iteration of MCTS (selection, expansion, simulation, backpropagation).
# #     """
# #     if mcts is None:
# #         mcts = Node(create_grid(), 0, None, None)

# #     node = mcts

# #     # Selection Phase
# #     while node.children is not None:
# #         ucts = [child.get_uct() for child in node.children]
# #         if None in ucts:
# #             node = random.choice(node.children)
# #         else:
# #             node = node.children[np.argmax(ucts)]

# #     # Expansion Phase
# #     moves = valid_move(node.state)
# #     if len(moves) > 0:
# #         if node.winner == 0:  # Expand only if the node is not terminal
# #             states = [(play(node.state, move), move) for move in moves]
# #             node.set_children([Node(state_winning[0], state_winning[1], move=move, parent=node) for state_winning, move in states])

# #             # Simulation Phase
# #             winner_nodes = [n for n in node.children if n.winner]
# #             if len(winner_nodes) > 0:
# #                 node = winner_nodes[0]
# #                 victorious = node.winner
# #             else:
# #                 node = random.choice(node.children)
# #                 victorious = heuristic_play(node.state)
# #         else:
# #             victorious = node.winner

# #         # Backpropagation Phase
# #         parent = node
# #         while parent is not None:
# #             parent.games += 1
# #             if victorious != 0 and get_player_to_play(parent.state) != victorious:
# #                 parent.win += 1
# #             print(f"Backpropagation - Node {parent.move}: Wins={parent.win}, Games={parent.games}")
# #             parent = parent.parent
# #     else:
# #         print('No valid moves, expanded all.')

# #     return mcts


# import numpy as np
# import random
# from connect4.mcts import Node
# from connect4.connect4 import create_grid, play, valid_move, get_player_to_play

# def random_play(grid):
#     """
#     Play a completely random game starting from the given state.
#     Returns the winner (1 for AI, -1 for Opponent, 0 for draw).
#     """
#     while True:
#         moves = valid_move(grid)
#         if len(moves) == 0:
#             return 0  # Draw
#         selected_move = random.choice(moves)
#         grid, winner = play(grid, selected_move)
#         if np.abs(winner) > 0:
#             return winner  # Return the winner

# def train_mcts_during(mcts, training_time):
#     """
#     Train MCTS for a specified amount of time (in milliseconds).
#     """
#     import time
#     start = int(round(time.time() * 1000))
#     current = start
#     while (current - start) < training_time:
#         mcts = train_mcts_once(mcts)
#         current = int(round(time.time() * 1000))
#     return mcts

# def train_mcts_once(mcts=None):
#     """
#     Perform one iteration of MCTS (selection, expansion, simulation, backpropagation).
#     """
#     if mcts is None:
#         mcts = Node(create_grid(), 0, None, None)

#     node = mcts

#     # Selection Phase
#     while node.children is not None:
#         ucts = [child.get_uct() for child in node.children]
#         if None in ucts:
#             node = random.choice(node.children)
#         else:
#             node = node.children[np.argmax(ucts)]

#     # Expansion Phase
#     moves = valid_move(node.state)
#     if len(moves) > 0:
#         if node.winner == 0:  # Expand only if the node is not terminal
#             states = [(play(node.state, move), move) for move in moves]
#             node.set_children([Node(state_winning[0], state_winning[1], move=move, parent=node) for state_winning, move in states])

#             # Simulation Phase
#             winner_nodes = [n for n in node.children if n.winner]
#             if len(winner_nodes) > 0:
#                 node = winner_nodes[0]
#                 victorious = node.winner
#             else:
#                 node = random.choice(node.children)
#                 victorious = random_play(node.state)  # Use purely random simulation
#         else:
#             victorious = node.winner

#         # Backpropagation Phase
#         parent = node
#         while parent is not None:
#             parent.games += 1
#             if victorious != 0 and get_player_to_play(parent.state) != victorious:
#                 parent.win += 1
#             parent = parent.parent
#     else:
#         print('No valid moves, expanded all.')

#     return mcts


import numpy as np
import random
from connect4.mcts import Node
from connect4.connect4 import create_grid, play, valid_move, get_player_to_play

def random_play(grid):
    """
    Play a completely random game starting from the given state.
    Returns the winner (1 for AI, -1 for Opponent, 0 for draw).
    """
    while True:
        moves = valid_move(grid)
        if len(moves) == 0:
            return 0  # Draw
        selected_move = random.choice(moves)
        grid, winner = play(grid, selected_move)
        if np.abs(winner) > 0:
            return winner  # Return the winner

def train_mcts_during(mcts, training_time):
    """
    Train MCTS for a specified amount of time (in milliseconds).
    """
    import time
    start = int(round(time.time() * 1000))
    current = start
    while (current - start) < training_time:
        mcts = train_mcts_once(mcts)
        current = int(round(time.time() * 1000))
    return mcts

def train_mcts_once(mcts=None):
    """
    Perform one iteration of MCTS (selection, expansion, simulation, backpropagation).
    """
    if mcts is None:
        mcts = Node(create_grid(), 0, None, None)

    node = mcts

    # Selection Phase
    while node.children is not None:
        ucts = [child.get_uct() for child in node.children]
        print(f"UCT Values: {ucts}")  # Debugging UCT values
        if None in ucts:
            node = random.choice(node.children)
        else:
            node = node.children[np.argmax(ucts)]

    # Expansion Phase
    moves = valid_move(node.state)
    if len(moves) > 0:
        if node.winner == 0:  # Expand only if the node is not terminal
            states = [(play(node.state, move), move) for move in moves]
            node.set_children([Node(state_winning[0], state_winning[1], move=move, parent=node) for state_winning, move in states])

            # Simulation Phase
            winner_nodes = [n for n in node.children if n.winner]
            if len(winner_nodes) > 0:
                node = winner_nodes[0]
                victorious = node.winner
            else:
                node = random.choice(node.children)
                victorious = random_play(node.state)  # Use purely random simulation
        else:
            victorious = node.winner

        # Backpropagation Phase
        parent = node
        while parent is not None:
            parent.games += 1
            if victorious != 0 and get_player_to_play(parent.state) != victorious:
                parent.win += 1
            print(f"Backpropagation - Node {parent.move}: Wins={parent.win}, Games={parent.games}")  # Debugging
            parent = parent.parent
    else:
        print('No valid moves, expanded all.')

    return mcts
