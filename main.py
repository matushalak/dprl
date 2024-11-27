# # ## matplotlib visualisation
# # from connect4.mcts_ia import train_mcts_once, train_mcts_during
# # from connect4.connect4 import *
# # import numpy as np
# # import matplotlib.pyplot as plt

# # # Visualization function
# # def visualize_board_with_probabilities(grid, probabilities):
# #     fig, ax = plt.subplots(figsize=(7, 6))
# #     rows, cols = grid.shape

# #     # Draw the board grid
# #     for row in range(rows):
# #         for col in range(cols):
# #             piece = grid[row, col]
# #             color = 'white'  # Default empty cell
# #             if piece == 1:
# #                 color = 'blue'  # AI player
# #             elif piece == -1:
# #                 color = 'red'  # Opponent
# #             circle = plt.Circle((col, rows - row - 1), 0.4, color=color, edgecolor='black')
# #             ax.add_artist(circle)

# #     # Draw probabilities above the columns
# #     for col, prob in enumerate(probabilities):
# #         plt.text(col, rows, f"{prob:.2f}", ha='center', va='center', fontsize=12, color='black')

# #     # Set board aesthetics
# #     ax.set_xlim(-0.5, cols - 0.5)
# #     ax.set_ylim(-0.5, rows + 0.5)
# #     ax.set_xticks(range(cols))
# #     ax.set_yticks([])
# #     ax.set_xticklabels(range(cols))
# #     ax.set_aspect('equal')
# #     plt.gca().invert_yaxis()
# #     plt.title("Connect 4 - Current Game State")
# #     plt.show()

# # # Utility to print the grid
# # def utils_print(grid):
# #     print_grid = grid.astype(str)
# #     print_grid[print_grid == '-1'] = 'X'
# #     print_grid[print_grid == '1'] = 'O'
# #     print_grid[print_grid == '0'] = ' '
# #     res = str(print_grid).replace("'", "").replace("[[", "[").replace("]]", "]")
# #     print(' ' + res)
# #     print('  ' + ' '.join('0123456'))

# # if __name__ == '__main__':
# #     # Initialize MCTS tree
# #     mcts = None

# #     # Train MCTS
# #     print("Training MCTS...")
# #     for _ in range(1000):  # Increase training iterations
# #         mcts = train_mcts_once(mcts)
# #     print("Training finished.")

# #     # Start a new game
# #     grid = create_grid()
# #     round = 0
# #     training_time = 2000  # Time for MCTS to train during each move
# #     node = mcts
# #     utils_print(grid)

# #     while True:
# #         if (round % 2) == 0:  # Player's turn (AI)
# #             # Calculate probabilities
# #             probabilities = [
# #                 child.win / child.games if child.games > 0 else 0
# #                 for child in (node.children or [])
# #             ]
# #             probabilities += [0] * (grid.shape[1] - len(probabilities))  # Fill for unvisited columns

# #             visualize_board_with_probabilities(grid, probabilities)
# #             best_child, move = node.select_move()
# #             print(f"Player (AI) chooses column {move}")
# #         else:  # Opponent's turn
# #             move = random_opponent_move(grid)
# #             print(f"Opponent (Random) chooses column {move}")

# #         # Apply the move and print the grid
# #         grid, winner = play(grid, move)
# #         utils_print(grid)

# #         # Update MCTS node
# #         try:
# #             node = train_mcts_during(node, training_time).get_children_with_move(move)
# #         except Exception as e:
# #             print(f"Error updating MCTS tree: {e}")
# #             break

# #         # Check for a winner
# #         if winner != 0:
# #             print('Winner:', 'AI (Player)' if winner == 1 else 'Random Opponent' if winner == -1 else 'Draw')
# #             break

# #         round += 1

# # # ### pygame visualisation
# # # import pygame
# # # from connect4.mcts_ia import train_mcts_once, train_mcts_during
# # # from connect4.connect4 import *
# # # import numpy as np

# # # # Visualization function using pygame
# # # def visualize_board_with_probabilities_pygame(grid, probabilities):
# # #     """
# # #     Visualize the Connect 4 board and probabilities using pygame.
# # #     """
# # #     pygame.init()
# # #     # Constants
# # #     CELL_SIZE = 100
# # #     ROWS, COLS = grid.shape
# # #     WIDTH, HEIGHT = COLS * CELL_SIZE, (ROWS + 1) * CELL_SIZE
# # #     RADIUS = CELL_SIZE // 2 - 5

# # #     # Colors
# # #     BLUE = (0, 0, 255)
# # #     BLACK = (0, 0, 0)
# # #     RED = (255, 0, 0)
# # #     YELLOW = (255, 255, 0)
# # #     WHITE = (255, 255, 255)

# # #     # Initialize pygame screen
# # #     screen = pygame.display.set_mode((WIDTH, HEIGHT))
# # #     pygame.display.set_caption("Connect 4 - AI vs Random Opponent")

# # #     # Fill the screen background
# # #     screen.fill(BLUE)

# # #     # Draw the board grid
# # #     for row in range(ROWS):
# # #         for col in range(COLS):
# # #             pygame.draw.circle(
# # #                 screen,
# # #                 BLACK,
# # #                 (col * CELL_SIZE + CELL_SIZE // 2, (row + 1) * CELL_SIZE + CELL_SIZE // 2),
# # #                 RADIUS,
# # #             )

# # #     # Draw pieces
# # #     for row in range(ROWS):
# # #         for col in range(COLS):
# # #             piece = grid[row, col]
# # #             if piece == 1:  # AI Player
# # #                 color = YELLOW
# # #             elif piece == -1:  # Opponent
# # #                 color = RED
# # #             else:  # Empty cell
# # #                 continue
# # #             pygame.draw.circle(
# # #                 screen,
# # #                 color,
# # #                 (col * CELL_SIZE + CELL_SIZE // 2, (row + 1) * CELL_SIZE + CELL_SIZE // 2),
# # #                 RADIUS,
# # #             )

# # #     # Draw probabilities above the columns
# # #     for col, prob in enumerate(probabilities):
# # #         text_color = WHITE if prob < 0.5 else BLACK  # Choose text color for visibility
# # #         font = pygame.font.SysFont("monospace", 24)
# # #         label = font.render(f"{prob:.2f}", True, text_color)
# # #         screen.blit(label, (col * CELL_SIZE + CELL_SIZE // 4, CELL_SIZE // 4))

# # #     pygame.display.update()

# # # # Utility to print the grid in terminal
# # # def utils_print(grid):
# # #     print_grid = grid.astype(str)
# # #     print_grid[print_grid == '-1'] = 'X'
# # #     print_grid[print_grid == '1'] = 'O'
# # #     print_grid[print_grid == '0'] = ' '
# # #     res = str(print_grid).replace("'", "").replace("[[", "[").replace("]]", "]")
# # #     print(' ' + res)
# # #     print('  ' + ' '.join('0123456'))

# # # if __name__ == '__main__':
# # #     # Initialize the MCTS tree
# # #     mcts = None

# # #     print("Training MCTS...")
# # #     for _ in range(1000):  # Adjust number of iterations as needed
# # #         mcts = train_mcts_once(mcts)
# # #     print("Training complete.")

# # #     # Start a new game
# # #     grid = create_grid()
# # #     round = 0
# # #     training_time = 2000  # Time for MCTS to train during each move (milliseconds)
# # #     node = mcts

# # #     # Initialize pygame
# # #     pygame.init()
# # #     running = True

# # #     while running:
# # #         if round % 2 == 0:  # AI Player's turn
# # #             print("AI's turn...")
# # #             # Calculate probabilities
# # #             probabilities = [
# # #                 child.win / child.games if child.games > 0 else 0
# # #                 for child in (node.children or [])
# # #             ]
# # #             probabilities += [0] * (grid.shape[1] - len(probabilities))  # Fill for unvisited columns

# # #             # Visualize the board and probabilities
# # #             visualize_board_with_probabilities_pygame(grid, probabilities)
# # #             best_child, move = node.select_move()
# # #             print(f"AI chooses column {move}")
# # #         else:  # Opponent's turn (random opponent)
# # #             move = random_opponent_move(grid)
# # #             print(f"Opponent chooses column {move}")

# # #         # Apply the move and update the grid
# # #         grid, winner = play(grid, move)
# # #         utils_print(grid)

# # #         # Update MCTS node
# # #         try:
# # #             node = train_mcts_during(node, training_time).get_children_with_move(move)
# # #         except Exception as e:
# # #             print(f"Error updating MCTS tree: {e}")
# # #             break

# # #         # Check for a winner
# # #         if winner != 0:
# # #             print('Winner:', 'AI (Player)' if winner == 1 else 'Random Opponent' if winner == -1 else 'Draw')
# # #             running = False
# # #             break

# # #         round += 1

# # #     pygame.quit()


# from connect4.mcts_ia import train_mcts_once, train_mcts_during
# from connect4.connect4 import *
# import numpy as np
# import pygame

# # Pygame visualization
# def draw_board_with_pygame(grid, probabilities):
#     pygame.init()
#     screen_width, screen_height = 700, 600
#     screen = pygame.display.set_mode((screen_width, screen_height))
#     pygame.display.set_caption("Connect 4 - AI vs Random Opponent")
#     clock = pygame.time.Clock()
#     rows, cols = grid.shape
#     cell_size = screen_width // cols

#     # Draw the board
#     screen.fill((0, 0, 255))  # Blue background
#     for row in range(rows):
#         for col in range(cols):
#             color = (0, 0, 0)  # Black for empty cells
#             if grid[row, col] == 1:
#                 color = (255, 0, 0)  # Red for AI
#             elif grid[row, col] == -1:
#                 color = (255, 255, 0)  # Yellow for Random Opponent
#             pygame.draw.circle(screen, color, (col * cell_size + cell_size // 2, row * cell_size + cell_size // 2 + 50), cell_size // 3)

#     # Display probabilities above columns
#     font = pygame.font.SysFont("Arial", 20)
#     for col, prob in enumerate(probabilities):
#         text = font.render(f"{prob:.2f}", True, (255, 255, 255))
#         screen.blit(text, (col * cell_size + cell_size // 4, 10))

#     pygame.display.flip()
#     clock.tick(30)

# def utils_print(grid):
#     print_grid = grid.astype(str)
#     print_grid[print_grid == '-1'] = 'X'
#     print_grid[print_grid == '1'] = 'O'
#     print_grid[print_grid == '0'] = ' '
#     res = str(print_grid).replace("'", "").replace("[[", "[").replace("]]", "]")
#     print(' ' + res)
#     print('  ' + ' '.join('0123456'))

# if __name__ == '__main__':
#     # Initialize MCTS tree
#     mcts = None

#     # Train MCTS
#     print("Training MCTS...")
#     for _ in range(1000):  # Increase training iterations
#         mcts = train_mcts_once(mcts)
#     print("Training finished.")

#     # Start a new game
#     grid = create_grid()
#     round = 0
#     training_time = 2000  # Time for MCTS to train during each move
#     node = mcts

#     utils_print(grid)

#     while True:
#         if (round % 2) == 0:  # Player's turn (AI)
#             # Calculate probabilities
#             probabilities = [
#                 child.win / child.games if child.games > 0 else 0
#                 for child in (node.children or [])
#             ]
#             probabilities += [0] * (grid.shape[1] - len(probabilities))  # Fill for unvisited columns

#             draw_board_with_pygame(grid, probabilities)
#             best_child, move = node.select_move()
#             print(f"Player (AI) chooses column {move}")
#         else:  # Opponent's turn
#             move = random_opponent_move(grid)
#             print(f"Opponent (Random) chooses column {move}")

#         # Apply the move and print the grid
#         grid, winner = play(grid, move)
#         utils_print(grid)

#         # Update MCTS node
#         try:
#             node = train_mcts_during(node, training_time).get_children_with_move(move)
#         except Exception as e:
#             print(f"Error updating MCTS tree: {e}")
#             break

#         # Check for a winner
#         if winner != 0:
#             print('Winner:', 'AI (Player)' if winner == 1 else 'Random Opponent' if winner == -1 else 'Draw')
#             break

#         round += 1


from connect4.mcts_ia import train_mcts_once, train_mcts_during
from connect4.connect4 import *
import numpy as np
import pygame

# Pygame visualization
def draw_board_with_pygame(grid, probabilities):
    pygame.init()
    screen_width, screen_height = 700, 600
    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption("Connect 4 - AI vs Random Opponent")
    clock = pygame.time.Clock()
    rows, cols = grid.shape
    cell_size = screen_width // cols

    # Draw the board
    screen.fill((0, 0, 255))  # Blue background
    for row in range(rows):
        for col in range(cols):
            color = (0, 0, 0)  # Black for empty cells
            if grid[row, col] == 1:
                color = (255, 0, 0)  # Red for AI
            elif grid[row, col] == -1:
                color = (255, 255, 0)  # Yellow for Random Opponent
            pygame.draw.circle(screen, color, (col * cell_size + cell_size // 2, row * cell_size + cell_size // 2 + 50), cell_size // 3)

    # Display probabilities above columns
    font = pygame.font.SysFont("Arial", 20)
    for col, prob in enumerate(probabilities):
        text = font.render(f"{prob:.2f}", True, (255, 255, 255))
        screen.blit(text, (col * cell_size + cell_size // 4, 10))

    pygame.display.flip()
    clock.tick(30)

if __name__ == '__main__':
    # Initialize MCTS tree
    mcts = None

    # Train MCTS
    print("Training MCTS...")
    for _ in range(5000):  # Increase training iterations
        mcts = train_mcts_once(mcts)
    print("Training finished.")

    # Simulate multiple games
    ai_wins = 0
    random_wins = 0
    draws = 0

    for game in range(10):  # Simulate 10 games for testing
        print(f"Starting game {game + 1}...")
        grid = create_grid()
        round = 0
        training_time = 2000  # Time for MCTS to train during each move
        node = mcts

        while True:
            if (round % 2) == 0:  # Player's turn (AI)
                # Calculate probabilities
                probabilities = [
                    child.win / child.games if child.games > 0 else 0
                    for child in (node.children or [])
                ]
                probabilities += [0] * (grid.shape[1] - len(probabilities))  # Fill for unvisited columns

                draw_board_with_pygame(grid, probabilities)
                best_child, move = node.select_move()
                print(f"Player (AI) chooses column {move}")
            else:  # Opponent's turn
                move = random_opponent_move(grid)
                print(f"Opponent (Random) chooses column {move}")

            # Apply the move
            grid, winner = play(grid, move)

            # Update MCTS node
            try:
                node = train_mcts_during(node, training_time).get_children_with_move(move)
            except Exception as e:
                print(f"Error updating MCTS tree: {e}")
                break

            # Check for a winner
            if winner != 0:
                if winner == 1:
                    ai_wins += 1
                elif winner == -1:
                    random_wins += 1
                else:
                    draws += 1
                print('Winner:', 'AI (Player)' if winner == 1 else 'Random Opponent' if winner == -1 else 'Draw')
                break

            round += 1

    print(f"Final Results: AI Wins: {ai_wins}, Random Wins: {random_wins}, Draws: {draws}")
    pygame.quit()
