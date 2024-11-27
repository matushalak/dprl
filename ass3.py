import connect4
from connect4.mcts_ia import train_mcts_once, train_mcts_during
from connect4.connect4 import create_grid, play, valid_moves
from collections import defaultdict
import numpy as np


def utils_print(grid):
    """Print the Connect 4 grid in a readable format."""
    print_grid = grid.astype(str)
    print_grid[print_grid == '-1'] = 'X'  # Opponent
    print_grid[print_grid == '1'] = 'O'   # AI
    print_grid[print_grid == '0'] = ' '   # Empty space
    res = str(print_grid).replace("'", "")
    res = res.replace('[[', '[')
    res = res.replace(']]', ']')
    print(' ' + res)
    print('  ' + ' '.join('0123456'))  # Column indices


def visualize_probabilities(grid, probabilities):
    """Display action-winning probabilities for each column."""
    print("Action probabilities (winning chances):")
    for col, prob in enumerate(probabilities):
        print(f"Column {col}: {prob:.2f}")


if __name__ == '__main__':
    # Initialize MCTS tree
    mcts = None

    # Train the MCTS algorithm
    for i in range(100):  # Number of training iterations
        mcts = train_mcts_once(mcts)

    print('Training finished')

    # Initialize game
    while True:
        grid = create_grid()
        round = 0
        training_time = 2000  # Training time for MCTS during each move
        node = mcts
        utils_print(grid)

        # Track convergence metrics
        stats = defaultdict(lambda: {'visits': 0, 'rewards': 0})

        while True:
            if (round % 2) == 0:  # AI's turn
                probabilities = [
                    child.win / child.games if child.games > 0 else 0
                    for child in node.children
                ]
                visualize_probabilities(grid, probabilities)

                move = int(input("Enter your move (column index 0-6): "))
                if move not in valid_moves(grid):
                    print("Invalid move. Try again.")
                    continue

                # Update MCTS node
                node = train_mcts_during(node, training_time).get_children_with_move(move)
            else:  # Random opponent's turn
                move = np.random.choice(valid_moves(grid))
                print(f"Opponent chooses column {move}")

            # Play the move and update the game state
            grid, winner = play(grid, move)
            utils_print(grid)

            # Update convergence metrics
            stats[move]['visits'] += 1
            stats[move]['rewards'] += 1 if winner == 1 else -1 if winner == -1 else 0

            # Check for a terminal condition
            if winner != 0:
                print('Winner:', 'X (Opponent)' if winner == -1 else 'O (AI)' if winner == 1 else 'Draw')
                break

            round += 1

        # Print convergence insights after the game
        print("Convergence Metrics:")
        for move, data in stats.items():
            print(f"Column {move}: Visits={data['visits']}, Rewards={data['rewards']}")

