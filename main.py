# main.py

from connect4.mcts_ia import train_mcts_once, train_mcts_during
from connect4.connect4 import *
from connect4.node import Node
import numpy as np
import pygame
import random

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
                color = (0, 255, 0)  # Green for AI
            elif grid[row, col] == -1:
                color = (255, 255, 0)  # Yellow for Random Opponent
            pygame.draw.circle(screen, color, (col * cell_size + cell_size // 2, (rows - row - 1) * cell_size + cell_size // 2), cell_size // 3)

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

    # Offline training
    print("Training MCTS...")
    for _ in range(1000):  # Adjust the number of iterations as needed
        mcts = train_mcts_once(mcts)
    print("Training finished.")

    # Simulate multiple games
    ai_wins = 0
    random_wins = 0
    draws = 0

    for game in range(100):  # Simulate one game for testing
        print(f"Starting game {game + 1}...")
        grid = create_grid()
        round = 0  # Ensure the AI starts
        training_time = 100  # Time for MCTS to train during each move (in milliseconds)
        node = mcts

        while True:
            if (round % 2) == 0:  # AI's turn
                # Train MCTS during AI's turn
                node = train_mcts_during(node, training_time)

                # Calculate probabilities
                probabilities = [
                    child.win / child.games if child.games > 0 else 0
                    for child in node.children
                ]
                probabilities += [0] * (grid.shape[1] - len(probabilities))  # Fill for unvisited columns

                draw_board_with_pygame(grid, probabilities)

                # Select the best move
                best_child = node.best_child()
                if best_child is None or best_child.move not in valid_move(grid):
                    print(f"No valid move found by AI! Valid Moves: {valid_move(grid)}")
                    break
                move = best_child.move
                print(f"Player (AI) chooses column {move}")
            else:  # Opponent's turn
                move = random_opponent_move(grid)
                if move is None:
                    print("No valid moves left for opponent.")
                    break
                print(f"Opponent (Random) chooses column {move}")

            # Apply the move
            current_player = get_player_to_play(grid)
            grid, winner = play(grid, move, player=current_player)

            # Update MCTS node
            try:
                node = node.get_children_with_move(move)
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
