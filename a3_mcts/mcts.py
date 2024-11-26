# @matushalak
# Monte-Carlo Tree Search Connect Four Game
'''
Connect Four is a board-size
6 x 7, representation B \in \R^{6 X 7}
'''
from numpy import ndarray, unique, fliplr, array, where, zeros
from numpy.random import randint
from string import ascii_letters as ascl
import argparse
from collections import defaultdict

def string_board(B:ndarray, symbols:dict[int:str, int:str] = ['⚽️','🍎']
                 ) -> str:
    # printed board with symbols
    header = [' | '+l for l in ascl[:B.shape[1]]]
    PrB = ' '+ ''.join(header)+ ' |  ' +'\n'
    string_board = ''
    for i, row in enumerate(B):
        string_board += ''.join([str(n) for n in row]) #hashable board representation
        symb = [f'{symbols[n]} |' for n in row]
        PrB += str(i)+ ' |' + ''.join(symb) + ' ' + str(i) + '\n'
    PrB += len(' '+ ''.join(header)+ ' |  ') * '-' + '\n'
    PrB += ' '+ ''.join(header)+ ' |  '
    print(PrB)
    return string_board #hashable board representation


def evaluate_board(B:ndarray):
    # only loop through pivots and look at 4x4 slices of B
    pivots = ([1,2,3],[1,2,3,4])
    winner = None
    for ir in pivots[0]:
        for ic in pivots[1]:
            # upper end of slice is EXCLUSIVE!, want 4 x 4 slice!
            Bslice = B[ir-1:ir+3,
                       ic-1:ic+3]
            
            unique_rows = [unique(row).tolist() for row in Bslice]
            unique_cols = [unique(col).tolist() for col in Bslice.T] # transpose to get cols
            rows = min(unique_rows ,key = len)
            cols = min(unique_cols, key = len)
            diag = unique(Bslice.diagonal())
            antidiag = unique(fliplr(Bslice).diagonal())

            sub_res = min([rows, cols, diag, antidiag], key = len)
            if len(sub_res) == 1 and sub_res[0] != 0:
                # breakpoint()
                winner = sub_res[0]
                return winner
    if 0 not in B:
        return 0 # draw if no free pieces left
    else:
        return winner # None if no winner (no connect 4), but still pieces left


def available_moves(B:ndarray):
    return [i for i,c in enumerate(B.T) if 0 in c] # transpose for columns


def moves(player:int, B:ndarray, possible_moves:list[int]) -> list[ndarray]:
    assert player in (1,2), 'Needs to be either player 1 or player 2!!!'
    resulting_states = []
    for m in possible_moves:
        # find last row in which 0 is still present
        B_new = B.copy()[where(B[:,m] == 0)[0][-1], m]
        B_new[where(B[:,m] == 0)[0][-1], m] = player
        resulting_states.append(B_new)
    return resulting_states

# TODO: finish
class TreeNode():
    def __init__(self, 
                 B:ndarray, 
                 player = 2):
        self.player = player
        self.amoves = available_moves(B)

        self.root = B # root
        self.children = moves(self.player, self.STATE, self.amoves) # children

# TODO: finish
def MCTS(startB:ndarray):
    # So that  no error thrown if new board state indexed
    state_node_map = defaultdict(None)
    sB = string_board(startB)
    B = startB
    while True:
        # Player turn
        # store possible board configurations in state_node_map
        state_node_map[sB] = TreeNode(B)
        # Oponent turn
        # results in B & sB
        if state_node_map[sB]:
            state = state_node_map[sB]
        else:
            state_node_map[sB] = TreeNode(B)

        pass


# State = current Board config
def build_tree(STATE:ndarray):
    pass


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-sym', nargs='+', type = str, default = ['⚽️','🍎'])
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    symbols = [' '+s if s.isalnum() else s for s in args.sym]
    d = {0:'  ', 1:symbols[0], 2:symbols[1]}
    # Board
    # hardcoded for assignment
    hardcodedB = array([[1, 1, 1, 2, 0, 2, 0],
                        [1, 2, 2, 2, 0, 1, 0],
                        [2, 1, 1, 1, 0, 2, 0],
                        [1, 2, 2, 2, 0, 1, 0],
                        [2, 2, 2, 1, 0, 1, 0],
                        [1, 1, 2, 1, 0, 2, 0]])
    zeroB = zeros((6,7), dtype=int) # for the real game

    # B = randint(1, 3, (6,7), dtype=int) # for testing & evaluation
    # B = zeroB
    B = hardcodedB

    sB = string_board(B, d)
    # winner = evaluate_board(B)
    winner = evaluate_board(B)
    print(available_moves(B))
    if winner:
        print(f'WinnerID:{winner} -> {d[winner]}')
    else:
        print('No winner yet!')
