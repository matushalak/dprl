import pygame
from connect4_2.mcts_ia2 import train_mcts_once, train_mcts_during
from connect4_2.connect4_2 import *
import numpy as np

# Pygame visualization
def visualize_board_pygame(grid):
    """
    Visualize the Connect 4 board using pygame.
    """
    pygame.init()
    # Constants
    CELL_SIZE = 100
    ROWS, COLS = grid.shape
    WIDTH, HEIGHT = COLS * CELL_SIZE, (ROWS + 1) * CELL_SIZE
    RADIUS = CELL_SIZE // 2 - 5

    # Colors
    BLUE = (0, 0, 255)
    BLACK = (0, 0, 0)
    RED = (255, 0, 0)
    YELLOW = (255, 255, 0)

    # Initialize screen
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Connect 4")
    screen.fill(BLUE)

    # Draw the board
    for row in range(ROWS):
        for col in range(COLS):
            pygame.draw.circle(
                screen,
                BLACK,
                (col * CELL_SIZE + CELL_SIZE // 2, (row + 1) * CELL_SIZE + CELL_SIZE // 2),
                RADIUS,
            )

    # Add pieces
    for row in range(ROWS):
        for col in range(COLS):
            piece = grid[row, col]
            if piece == 1:
                color = YELLOW  # AI
            elif piece == -1:
                color = RED  # Opponent
            else:
                continue
            pygame.draw.circle(
                screen,
                color,
                (col * CELL_SIZE + CELL_SIZE // 2, (row + 1) * CELL_SIZE + CELL_SIZE // 2),
                RADIUS,
            )

    pygame.display.update()

def utils_print(grid):
    """
    Print the Connect 4 grid in the terminal.
    """
    print_grid = grid.astype(str)
    print_grid[print_grid == '-1'] = 'X'
    print_grid[print_grid == '1'] = 'O'
    print_grid[print_grid == '0'] = ' '
    res = str(print_grid).replace("'", "").replace("[[", "[").replace("]]", "]")
    print(' ' + res)
    print('  ' + ' '.join('0123456'))

if __name__ == '__main__':
    # Initialize the MCTS tree
    mcts = None

    print("Training MCTS...")
    # Train MCTS with multiple iterations
    for _ in range(1000):
        mcts = train_mcts_once(mcts)
    print("Training complete.")

    # Start a new game
    grid = create_grid()
    round = 0
    training_time = 2000  # Time for MCTS to train during each move (in milliseconds)
    node = mcts

    # Pygame loop setup
    pygame.init()
    running = True
    visualize_board_pygame(grid)

    while running:
        if round % 2 == 0:  # AI Player's turn
            print("AI's turn...")
            best_child, move = node.select_move()
            print(f"AI chooses column {move}")
        else:  # Opponent's turn (random opponent)
            move = random_opponent_move(grid)
            print(f"Opponent chooses column {move}")

        # Apply the move and update the grid
        grid, winner = play(grid, move)
        utils_print(grid)

        # Visualize the board in pygame
        visualize_board_pygame(grid)

        # Update the MCTS node
        node = train_mcts_during(node, training_time).get_children_with_move(move)

        # Check for a winner or draw
        if winner != 0:
            print('Winner:', 'AI Player' if winner == 1 else 'Random Opponent' if winner == -1 else 'Draw')
            running = False  # Exit the loop
            break

        round += 1

    pygame.quit()
