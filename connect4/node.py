
# import numpy as np
# import math

# class Node:
#     def __init__(self, state, winner, move=None, parent=None):
#         self.state = state  # The game state at this node
#         self.winner = winner  # The winner at this state, if any
#         self.move = move  # The move that led to this state
#         self.parent = parent  # Parent node
#         self.children = []  # List of child nodes
#         self.win = 0  # Number of wins from this node
#         self.games = 0  # Number of simulations from this node

#     def get_uct(self, c=math.sqrt(2)):
#         """Calculate the UCT value for this node."""
#         if self.games == 0:
#             return float('inf')  # Encourage exploration of unvisited nodes
#         exploitation = self.win / self.games
#         exploration = c * math.sqrt(math.log(self.parent.games) / self.games)
#         return exploitation + exploration

#     def set_children(self, children):
#         """Set the child nodes of this node."""
#         self.children = children

#     def select_move(self):
#         """Select the best move based on UCT."""
#         if not self.children:
#             return None, None
#         ucts = [child.get_uct() for child in self.children]
#         best_index = np.argmax(ucts)
#         return self.children[best_index], self.children[best_index].move

#     def get_children_with_move(self, move):
#         """Get the child node corresponding to a given move."""
#         for child in self.children:
#             if child.move == move:
#                 return child
#         raise Exception(f"Move {move} not found among children nodes.")

#     def is_fully_expanded(self):
#         """Check if the node is fully expanded."""
#         return len(self.children) == len(valid_move(self.state))

#     def best_child(self):
#         """Return the child with the highest win rate."""
#         if not self.children:
#             return None
#         win_rates = [child.win / child.games if child.games > 0 else 0 for child in self.children]
#         best_index = np.argmax(win_rates)
#         return self.children[best_index]



import numpy as np
import math
from connect4.connect4 import create_grid, play, valid_move, get_player_to_play


class Node:
    def __init__(self, state, winner, move=None, parent=None):
        self.state = state  # The game state at this node
        self.winner = winner  # The winner at this state, if any
        self.move = move  # The move that led to this state
        self.parent = parent  # Parent node
        self.children = []  # List of child nodes
        self.win = 0  # Number of wins from this node
        self.games = 0  # Number of simulations from this node

    def get_uct(self, c=math.sqrt(2)):
        """Calculate the UCT value for this node."""
        if self.games == 0:
            return float('inf')  # Encourage exploration of unvisited nodes
        exploitation = self.win / self.games
        exploration = c * math.sqrt(math.log(self.parent.games) / self.games)
        return exploitation + exploration

    def set_children(self, children):
        """Set the child nodes of this node."""
        self.children = children

    def select_move(self):
        """Select the best move based on UCT."""
        if not self.children:
            return None, None
        ucts = [child.get_uct() for child in self.children]
        best_index = np.argmax(ucts)
        return self.children[best_index], self.children[best_index].move

    def get_children_with_move(self, move):
        """Get the child node corresponding to a given move."""
        for child in self.children:
            if child.move == move:
                return child
        raise Exception(f"Move {move} not found among children nodes.")

    def is_fully_expanded(self):
        """Check if the node is fully expanded."""
        return len(self.children) == len(valid_move(self.state))

    def best_child(self):
        """Return the child with the highest UCT value."""
        if not self.children:
            return None
        ucts = [child.get_uct() for child in self.children]
        best_index = np.argmax(ucts)
        return self.children[best_index]

