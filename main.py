from connect4.mcts_ia import train_mcts_once, train_mcts_during 
from connect4.connect4 import * 
from connect4.node import Node  
import numpy as np  
import pygame  
import random  

# Pygame visualization function
def draw_board_with_pygame(grid, probabilities):
    """
    Draws the current state of the game grid using Pygame,
    along with the probabilities of selecting each column.

    Parameters:
    - grid (ndarray): The current game grid.
    - probabilities (list): A list of probabilities associated with each column.
    """
    pygame.init()  # Initialize all imported Pygame modules

    # Set up the display window dimensions
    screen_width, screen_height = 700, 600  # Width and height of the game window in pixels
    screen = pygame.display.set_mode((screen_width, screen_height))  # Create a display surface
    pygame.display.set_caption("Connect 4 - AI vs Random Opponent")  # Set the window title
    clock = pygame.time.Clock()  # Create a Clock object to manage the frame rate

    # Get the number of rows and columns from the grid shape
    rows, cols = grid.shape  # rows = number of rows, cols = number of columns
    cell_size = screen_width // cols  # Determine the size of each cell based on the window width

    # Draw the board background
    screen.fill((0, 0, 255))  # Fill the screen with blue color (RGB: 0, 0, 255)

    # Draw the pieces on the board
    for row in range(rows):
        for col in range(cols):
            # Set the default color for empty cells
            color = (0, 0, 0)  # Black color

            # Determine the color based on the player occupying the cell
            if grid[row, col] == 1:
                color = (0, 255, 0)  # Green color for AI player (Player 1)
            elif grid[row, col] == -1:
                color = (255, 255, 0)  # Yellow color for Random Opponent (Player -1)

            # Calculate the position to draw the circle (piece)
            # Note: (rows - row - 1) in y-coordinate to draw from bottom to top
            position = (col * cell_size + cell_size // 2, (rows - row - 1) * cell_size + cell_size // 2)
            # Draw the circle representing the piece
            pygame.draw.circle(screen, color, position, cell_size // 3)

    # Display the probabilities above each column
    font = pygame.font.SysFont("Arial", 20)  # Create a Font object
    for col, prob in enumerate(probabilities):
        # Render the probability text for each column
        text = font.render(f"{prob:.2f}", True, (255, 255, 255))  # White color text
        # Blit the text onto the screen at the appropriate position
        screen.blit(text, (col * cell_size + cell_size // 4, 10))

    # Update the full display Surface to the screen
    pygame.display.flip()
    clock.tick(30)  # Limit the frame rate to 30 frames per second

# Main execution block
if __name__ == '__main__':
    # Initialize MCTS tree
    mcts = None  # Start with an empty tree

    # Offline training phase
    print("Training MCTS...")
    for _ in range(5000):  # Number of iterations for offline training; can be adjusted
        mcts = train_mcts_once(mcts)  # Perform one iteration of MCTS training
    print("Training finished.")

    # Initialize counters for game results
    ai_wins = 0       # Counter for AI wins
    random_wins = 0   # Counter for Random Opponent wins
    draws = 0         # Counter for draws

    # Simulate multiple games
    for game in range(100):  # Simulate 100 games; adjust the number as needed
        print(f"Starting game {game + 1}...")
        grid = create_grid()  # Create a new empty game grid
        round = 0  # Initialize round counter to ensure the AI starts first
        training_time = 100  # Training time for MCTS during each move in milliseconds
        node = mcts  # Start from the root of the MCTS tree

        while True:
            if (round % 2) == 0:  # AI's turn (even rounds)
                # Train MCTS during AI's turn for the specified training time
                node = train_mcts_during(node, training_time)

                # Calculate the probabilities of winning for each child (move)
                probabilities = [
                    child.win / child.games if child.games > 0 else 0  # Avoid division by zero
                    for child in node.children
                ]
                # If there are fewer probabilities than columns, fill the rest with zeros
                probabilities += [0] * (grid.shape[1] - len(probabilities))

                # Visualize the current board state and move probabilities
                draw_board_with_pygame(grid, probabilities)

                # Select the best move based on the MCTS results
                best_child = node.best_child()
                if best_child is None or best_child.move not in valid_move(grid):
                    # If no valid move is found, print an error and break the loop
                    print(f"No valid move found by AI! Valid Moves: {valid_move(grid)}")
                    break
                move = best_child.move  # The column where the AI decides to place its piece
                print(f"Player (AI) chooses column {move}")
            else:  # Opponent's turn (odd rounds)
                # Randomly select a valid move for the opponent
                move = random_opponent_move(grid)
                if move is None:
                    # If no valid moves are left, print a message and break the loop
                    print("No valid moves left for opponent.")
                    break
                print(f"Opponent (Random) chooses column {move}")

            # Apply the move to the game grid
            current_player = get_player_to_play(grid)  # Determine whose turn it is
            grid, winner = play(grid, move, player=current_player)  # Update the grid and check for a winner

            # Update the MCTS node to the child corresponding to the move made
            try:
                node = node.get_children_with_move(move)
            except Exception as e:
                # If an error occurs (e.g., move not found among children), print an error and break
                print(f"Error updating MCTS tree: {e}")
                break

            # Check for a winner or if the game is a draw
            if winner != 0:
                # Update the appropriate counter based on who won
                if winner == 1:
                    ai_wins += 1
                elif winner == -1:
                    random_wins += 1
                else:
                    draws += 1
                # Print the result of the game
                print('Winner:', 'AI (Player)' if winner == 1 else 'Random Opponent' if winner == -1 else 'Draw')
                break  # Exit the game loop as the game has concluded

            round += 1  # Increment the round counter to switch turns

    # After all games have been simulated, print the final results
    print(f"Final Results: AI Wins: {ai_wins}, Random Wins: {random_wins}, Draws: {draws}")
    pygame.quit()  # Quit Pygame






# # main.py

# from connect4.mcts_ia import train_mcts_once, train_mcts_during
# from connect4.connect4 import *
# from connect4.node import Node
# import numpy as np
# import pygame
# import random

# # Pygame visualization function remains the same
# # def draw_board_with_pygame(grid, probabilities):
# #     # ... [Same as before] ...
# #     pass  # For brevity, code is omitted; use your existing function


# def draw_board_with_pygame(grid, probabilities):
#     """
#     Draws the current state of the game grid using Pygame,
#     along with the probabilities of selecting each column.

#     Parameters:
#     - grid (ndarray): The current game grid.
#     - probabilities (list): A list of probabilities associated with each column.
#     """
#     pygame.init()  # Initialize all imported Pygame modules

#     # Set up the display window dimensions
#     screen_width, screen_height = 700, 600  # Width and height of the game window in pixels
#     screen = pygame.display.set_mode((screen_width, screen_height))  # Create a display surface
#     pygame.display.set_caption("Connect 4 - AI vs Random Opponent")  # Set the window title
#     clock = pygame.time.Clock()  # Create a Clock object to manage the frame rate

#     # Get the number of rows and columns from the grid shape
#     rows, cols = grid.shape  # rows = number of rows, cols = number of columns
#     cell_size = screen_width // cols  # Determine the size of each cell based on the window width

#     # Draw the board background
#     screen.fill((0, 0, 255))  # Fill the screen with blue color (RGB: 0, 0, 255)

#     # Draw the pieces on the board
#     for row in range(rows):
#         for col in range(cols):
#             # Set the default color for empty cells
#             color = (0, 0, 0)  # Black color

#             # Determine the color based on the player occupying the cell
#             if grid[row, col] == 1:
#                 color = (0, 255, 0)  # Green color for AI player (Player 1)
#             elif grid[row, col] == -1:
#                 color = (255, 255, 0)  # Yellow color for Random Opponent (Player -1)

#             # Calculate the position to draw the circle (piece)
#             # Note: (rows - row - 1) in y-coordinate to draw from bottom to top
#             position = (col * cell_size + cell_size // 2, (rows - row - 1) * cell_size + cell_size // 2)
#             # Draw the circle representing the piece
#             pygame.draw.circle(screen, color, position, cell_size // 3)

#     # Display the probabilities above each column
#     font = pygame.font.SysFont("Arial", 20)  # Create a Font object
#     for col, prob in enumerate(probabilities):
#         # Render the probability text for each column
#         text = font.render(f"{prob:.2f}", True, (255, 255, 255))  # White color text
#         # Blit the text onto the screen at the appropriate position
#         screen.blit(text, (col * cell_size + cell_size // 4, 10))

#     # Update the full display Surface to the screen
#     pygame.display.flip()
#     clock.tick(30)  # Limit the frame rate to 30 frames per second

# if __name__ == '__main__':
#     # Initialize the hardcoded board
#     hardcodedB = np.array([
#         [1, 1, 1, 2, 0, 2, 0],
#         [1, 2, 2, 2, 0, 1, 0],
#         [2, 1, 1, 1, 0, 2, 0],
#         [1, 2, 2, 2, 0, 1, 0],
#         [2, 2, 2, 1, 0, 1, 0],
#         [1, 1, 2, 1, 0, 2, 0]
#     ])

#     # Convert to match code's player representations
#     initial_state = hardcodedB.copy()
#     initial_state[hardcodedB == 2] = -1  # Convert player 2 to -1

#     # Initialize MCTS tree starting from initial_state
#     mcts = Node(initial_state, winner=0, move=None, parent=None)

#     # Simulate multiple games
#     ai_wins = 0
#     random_wins = 0
#     draws = 0

#     for game in range(100):  # Simulate one game for testing
#         print(f"Starting game {game + 1}...")
#         grid = initial_state.copy()
#         training_time = 100  # Time for MCTS to train during each move (in milliseconds)
#         node = mcts  # Start from the node corresponding to initial_state

#         while True:
#             current_player = get_player_to_play(grid)
#             if current_player == 1:  # AI's turn
#                 # Train MCTS during AI's turn
#                 node = train_mcts_during(node, training_time)

#                 # Calculate probabilities
#                 probabilities = [
#                     child.win / child.games if child.games > 0 else 0
#                     for child in node.children
#                 ]
#                 probabilities += [0] * (grid.shape[1] - len(probabilities))  # Fill for unvisited columns

#                 draw_board_with_pygame(grid, probabilities)

#                 # Select the best move
#                 best_child = node.best_child()
#                 if best_child is None or best_child.move not in valid_move(grid):
#                     print(f"No valid move found by AI! Valid Moves: {valid_move(grid)}")
#                     break
#                 move = best_child.move
#                 print(f"Player (AI) chooses column {move}")
#             else:  # Opponent's turn
#                 move = random_opponent_move(grid)
#                 if move is None:
#                     print("No valid moves left for opponent.")
#                     break
#                 print(f"Opponent (Random) chooses column {move}")

#             # Apply the move
#             grid, winner = play(grid, move, player=current_player)

#             # Update MCTS node
#             try:
#                 node = node.get_children_with_move(move)
#             except Exception as e:
#                 print(f"Error updating MCTS tree: {e}")
#                 break

#             # Check for a winner
#             if winner != 0:
#                 if winner == 1:
#                     ai_wins += 1
#                 elif winner == -1:
#                     random_wins += 1
#                 else:
#                     draws += 1
#                 print('Winner:', 'AI (Player)' if winner == 1 else 'Random Opponent' if winner == -1 else 'Draw')
#                 break

#             # Continue to the next turn; current_player will be updated based on the new grid state

#     print(f"Final Results: AI Wins: {ai_wins}, Random Wins: {random_wins}, Draws: {draws}")
#     pygame.quit()
