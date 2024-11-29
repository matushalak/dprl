import numpy as np 
import math        

from connect4.connect4 import create_grid, play, valid_move, get_player_to_play

class Node:
    """
    A class representing a node in the Monte Carlo Tree Search (MCTS) tree.

    Attributes:
    - state (ndarray): The game state (grid) at this node.
    - winner (int or None): The winner at this state:
        - None if the game is ongoing.
        - 0 if the game is a draw.
        - 1 or -1 if there is a winner.
    - move (int or None): The move (column index) that led to this state.
    - parent (Node or None): The parent node in the tree.
    - children (list): A list of child Node objects representing possible future states.
    - win (int): The number of wins recorded from this node during simulations.
    - games (int): The total number of simulations/games played from this node.
    """

    def __init__(self, state, winner, move=None, parent=None):
        """
        Initialize a new Node.

        Parameters:
        - state (ndarray): The game state at this node.
        - winner (int or None): The winner at this state.
        - move (int, optional): The move that led to this state (column index).
        - parent (Node, optional): The parent node in the tree.
        """
        self.state = state            # Store the current game state (grid)
        self.winner = winner          # Store the winner status at this state
        self.move = move              # The move that led to this state from the parent
        self.parent = parent          # Reference to the parent node
        self.children = []            # Initialize an empty list for child nodes
        self.win = 0                  # Initialize win count to 0
        self.games = 0                # Initialize games (simulations) count to 0

    def get_uct(self, c=math.sqrt(2)):
        """
        Calculate the Upper Confidence Bound for Trees (UCT) value for this node.

        Parameters:
        - c (float): The exploration parameter (default is sqrt(2)).

        Returns:
        - uct_value (float): The calculated UCT value.

        The UCT formula balances exploration and exploitation:
        UCT = (win / games) + c * sqrt(ln(parent.games) / games)

        - The first term (win / games) is the exploitation term (average reward).
        - The second term encourages exploration of less-visited nodes.
        """
        if self.games == 0:
            # If the node has not been visited yet, return infinity to ensure it gets explored
            return float('inf')  # Encourage exploration of unvisited nodes

        # Calculate the exploitation term (average win rate)
        exploitation = self.win / self.games

        # Calculate the exploration term
        # Ensure that parent.games > 0 to avoid division by zero
        if self.parent is not None and self.parent.games > 0:
            exploration = c * math.sqrt(math.log(self.parent.games) / self.games)
        else:
            # If parent.games is 0, set exploration to 0 to avoid division by zero
            exploration = 0

        # Sum the exploitation and exploration terms to get the UCT value
        return exploitation + exploration

    def set_children(self, children):
        """
        Set the child nodes of this node.

        Parameters:
        - children (list): A list of Node objects to be set as children.
        """
        self.children = children  # Assign the list of children to self.children

    def select_move(self):
        """
        Select the best move based on the highest UCT value among the children.

        Returns:
        - best_child (Node or None): The child node with the highest UCT value.
        - move (int or None): The move (column index) associated with the best child.

        If there are no children, returns (None, None).
        """
        if not self.children:
            # If the node has no children, return None
            return None, None

        # Calculate the UCT value for each child
        ucts = [child.get_uct() for child in self.children]

        # Find the index of the child with the highest UCT value
        best_index = np.argmax(ucts)

        # Return the best child node and its associated move
        return self.children[best_index], self.children[best_index].move

    def get_children_with_move(self, move):
        """
        Get the child node corresponding to a given move.

        Parameters:
        - move (int): The move (column index) to find among the children.

        Returns:
        - child (Node): The child node corresponding to the given move.

        Raises:
        - Exception: If no child with the specified move is found.
        """
        for child in self.children:
            if child.move == move:
                # Return the child node if the move matches
                return child

        # If the move is not found among the children, raise an exception
        raise Exception(f"Move {move} not found among children nodes.")

    def is_fully_expanded(self):
        """
        Check if the node is fully expanded.

        Returns:
        - is_expanded (bool): True if all possible moves from this state have been explored.

        A node is fully expanded if the number of children equals the number of valid moves.
        """
        # Get the number of valid moves from the current state
        valid_moves = valid_move(self.state)

        # Compare the number of children with the number of valid moves
        return len(self.children) == len(valid_moves)

    def best_child(self):
        """
        Return the child with the highest UCT value.

        Returns:
        - best_child (Node or None): The child node with the highest UCT value.

        If there are no children, returns None.
        """
        if not self.children:
            # If there are no children, return None
            return None

        # Calculate the UCT value for each child
        ucts = [child.get_uct() for child in self.children]

        # Find the index of the child with the highest UCT value
        best_index = np.argmax(ucts)

        # Return the best child node
        return self.children[best_index]
