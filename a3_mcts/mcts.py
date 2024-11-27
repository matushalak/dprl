# @matushalak
# Monte-Carlo Tree Search Connect Four Game
'''
Connect Four is a board-size
6 x 7, representation B \in \R^{6 X 7}
'''
from numpy import ndarray, unique, fliplr, array, where, zeros
from numpy.random import randint, choice
from string import ascii_letters as ascl
import argparse
from collections import defaultdict
import os

def string_board(B:ndarray, symbols:dict[int:str, int:str] = {0:'  ', 1:'🍎', 2:'⚽️'}) -> str:
    # printed board with symbols
    header = [' | '+l for l in ascl[:B.shape[1]]]
    PrB = ' '+ ''.join(header)+ ' |  ' +'\n'
    string_board = ''
    for i, row in enumerate(B):
        string_board += ''.join([str(n) for n in row]) #hashable board representation
        symb = [f'{symbols[n]} |' for n in row]
        PrB += str(i)+ ' |' + ''.join(symb) + ' ' + str(i) + '\n'
    PrB += len(' '+ ''.join(header)+ ' |  ') * '-' + '\n'
    PrB += ' '+ ''.join(header)+ ' |  '+'\n'
    # clear previous output in terminal
    os.system('clear')
    print(PrB, end='\r')
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
    if not (B == 0).any():
        return 0 # draw if no free pieces left
    else:
        return winner # None if no winner (no connect 4), but still pieces left


class TreeNode():
    def __init__(self, 
                 B:ndarray,
                 parent = None): # that's how you identify root node!!!
        self.player = self.turn(B)
        self.amoves = self.available_moves(B)
        self.parent = parent

        # only increase w back-up
        self.depth = 0 if not parent else parent.depth + 1
        self.Q = 0
        self.visits = 0

        self.board = B # root
        self.strB = ''.join([str(n) for n in self.B.ravel()])
        # children are Treenodes themselves, self is parent of its children
        # Qs are Q-values associated with each possible child
        self.children, self.Qs = self.moves(self.player, self.board, self.amoves, parent = self) 

    def available_moves(self, B:ndarray):
        return [i for i,c in enumerate(B.T) if 0 in c] # transpose for columns

    def turn(self, B:ndarray):
        counts = [(B == 1).sum(), (B == 2).sum()]
        # breakpoint()
        if len(set(counts)) == 2:
            return counts.index(min(counts)) + 1
        else:
            return 1 # if same then 1st players turn 

    def moves(self, player:int, B:ndarray, possible_moves:list[int], parent:'TreeNode'
              ) -> list['TreeNode']:
        assert player in (1,2), 'Needs to be either player 1 or player 2!!!'
        children = []
        Qs = dict() # Q-values for each possible action a_j(children) from state x_i (parent)
        for m in possible_moves:
            # find last row in which 0 is still present
            B_new = B.copy()
            B_new[where(B_new[:,m] == 0)[0][-1], m] = player
            strB_new = ''.join([str(n) for n in B_new.ravel()])
            Qs[strB_new] = 0 # initiate with 0 Q-values
            # child node
            child_node = TreeNode(B_new, parent)
            children.append(child_node)
        return children, Qs

    # Returns -1 (loss), 0(draw), +1 (win)
    def rollout(self)->int:
        rollB = self.board.copy()
        while True:
            winner = evaluate_board(rollB)
            if winner is not None:
                break

            player = self.turn(rollB)
            move = choice(self.available_moves(rollB))
            rollB[where(rollB[:,move] == 0)[0][-1], move] = player

        string_board(rollB)
        match winner:
            case 0: # draw
                return 0
            case 1: # agent won
                return 1 
            case 2: # agent lost
                return -1 
            case _:
                return ValueError('Should always be draw / win / loss')


# TODO: finish
# TODO, multiple rollouts / node - especially EARLY in the game
def MCTS(startB:ndarray):
    '''
    Main function performing Monte Carlo Tree Search Algorithm 
    with Upper Confidence Trees (UCT) & min-max algorithm to consider playing optimal oponent

    1) Selection:
        - on nodes previously seen, choose action according to UCB rule 
        (apply on each previously seen node, balances exploration & exploitation)
        - for Node.depth == 0, need to random rollout on all possible moves 
        (ideally multiple times, decreasing as a function of Node depth)
    2) Expansion
        - when leaf Node reached add child(ren) of this node
    3) Simulation
        - random rollout from one (random) child of leaf Node until end
    4) Back-up
        - update Q-value & visits for Nodes in Selection & expansion
        Q(state, action) = ∑_x' [#x' reached from (x,a)] . (
        immediate reward (0) + max/min{based on p1/p2}_a' Q(x', a')) // # a taken in state x

        Q(state, action) = All maximal future rewards, possibly achieved by action weighed by their frequency // frequency of action in given state
            => Gives an estimate of Value / Action in a state
        - if oponent actions kept negative: use max(p1)-min(p2) approach! 
            -> max reward for worst-case scenario
    '''
    # So that  no error thrown if new board state indexed
    # stores hashable reference to every possible state (which has its depth, children, etc.)
    state_node_map = defaultdict(lambda: None)
    sB = string_board(startB)
    B = startB
    while True:
        # turn determined within treenode_class
        # store possible board configurations in state_node_map
        if state_node_map[sB]:
            state = state_node_map[sB]
        else:
            state_node_map[sB] = TreeNode(B)
            state = state_node_map[sB]

        if state.depth == 0:
            for ch in state.children:
                r = ...
                node.parent = ch
                # Backup
                while node.parent is not None:
                    ...

        breakpoint()

        state_node_map[sB] = TreeNode(B)
        # Oponent turn
        # results in B & sB
        pass


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-sym', nargs='+', type = str, default = ['🍎','⚽️'])
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

    # B = randint(0, 3, (6,7), dtype=int) # for testing & evaluation
    # B = zeroB
    B = hardcodedB

    sB = string_board(B, d)
    # winner = evaluate_board(B)
    winner = evaluate_board(B)
    MCTS(B)
    if winner:
        print(f'WinnerID:{winner} -> {d[winner]}')
    else:
        print('No winner yet!')
