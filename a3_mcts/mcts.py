# @matushalak
# Monte-Carlo Tree Search Connect Four Game
'''
Connect Four is a board-size
6 x 7, representation B \in \R^{6 X 7}
'''
from numpy import ndarray, unique, fliplr, array, where, zeros, mean, argmax, log
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

# this tree Class does the 1/3 of heavy lifting
class TreeNode():
    def __init__(self, 
                 B:ndarray,
                 parent = None): # that's how you identify root node!!!
        self.player = self.turn(B)
        # Actions := Free fields / available moves
        self.actions = self.available_moves(B)
        self.parent = parent
        self.parent_move = None # for root

        self.board = B # root
        self.strB = ''.join([str(n) for n in B.ravel()]) # hashable representation

        # only increase w back-up
        self.depth = 0 if not parent else parent.depth + 1
        # only for ultimate leaf nodes of MCT -1 / 0 / 1
        self.reward = 0
        self.state_visits = 0 # #times this state visited

        # Q-values for each (current_state, possible_action) pair
        self.Qs = dict.fromkeys(self.actions, zeros(len(self.actions)))
        # Visits for each (state, action) pair
        self.visits = dict.fromkeys(self.actions, zeros(len(self.actions)))
        # children are Treenodes themselves, self is parent of its children
        self.children = dict.fromkeys(self.actions, None) # added on a need-to-add basis

    # Only Add child when needed, no need to generate self.children every time a node is accessed
    def add_child(self, action:int) -> 'TreeNode':
        assert self.player in (1,2), 'Needs to be either player 1 or player 2!!!'
        B_new:ndarray = self.board.copy()
        B_new[where(B_new[:,action] == 0)[0][-1], 
              action] = self.player
        # child node
        ChildNode = TreeNode(B_new, self) # self is a Parent
        ChildNode.parent_move = action # can call this up later
        self.children[action] = ChildNode
        return ChildNode # for use outside

    def available_moves(self, B:ndarray):
        return [i for i,c in enumerate(B.T) if 0 in c] # transpose for columns

    def turn(self, B:ndarray):
        counts = [(B == 1).sum(), (B == 2).sum()]
        # breakpoint()
        if len(set(counts)) == 2:
            return counts.index(min(counts)) + 1
        else:
            return 1 # if same then 1st players turn 

    # Returns -1 (loss), 0(draw), +1 (win)
    def rollout(self)->int:
        rollB = self.board.copy()
        # end of the game, terminal node
        if self.available_moves(rollB) == []:
            self.reward = evaluate_board(rollB)
            return self.reward

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
# this MCTS function does 2/3 of heavy lifting
def MCTS(startB:ndarray, c:float = 2**0.5):
    '''
    Main function performing Monte Carlo Tree Search Algorithm 
    with Upper Confidence Trees (UCT) & min-max algorithm to consider playing optimal oponent

    1) Selection:
        - on nodes previously seen, choose action according to UCB rule 
        (apply on each previously seen node, balances exploration & exploitation)
        - for Node.depth == 0, need to random rollout on all possible moves 
        (ideally multiple times, decreasing as a function of Node depth)
    2) Expansion:
        - when leaf Node reached add child(ren) of this node
    3) Simulation:
        - random rollout from one (random) child of leaf Node until end
    4) Back-up:
    XXX Q-VAlUES NOT ASSOCIAteD w CHILD / STATE !!!! ONLY w STATE-ACTION PAIR!!!! XXX
        - update Q-value & visits for Nodes in Selection & expansion
            -> for deterministic games Go, Chess, TicTacToe, Connect4, we dont need to keep track of 
            different possible states reached by taking action: always 1 Action --> 1 State
        Q(state, action) = immediate reward + max/min{based on p1/p2}_a' Q(x', a'))

        Q(state, action) = Sum of rewards achieved by action weighed by their frequency // frequency of action in given state
            => Gives an estimate of Value / Action in a state
        - if oponent actions kept negative: use max(p1)-min(p2) approach! 
            -> max reward for worst-case scenario
    '''
    # So that  no error thrown if new board state indexed
    # stores hashable reference to every possible state (which has its depth, children, etc.)
    state_node_map = defaultdict(lambda: None)
    sB = string_board(startB)
    # store possible board configurations in state_node_map
    StartState:TreeNode = TreeNode(startB)
    state_node_map[sB] = StartState

    # Starting case - expand all children, and use average of 5 rollout rewards for initial Q-values
    for action in StartState.actions:
        # if unexplored action from root node
        if StartState.visits[action] == 0:
            Ch:TreeNode = StartState.add_child(action)
            state_node_map[Ch.strB] = Ch
            # random rollout reward from (x,a)->initial estimate of Q(x,a) == here x.Q[a] 
            StartState.Qs[action] += mean([Ch.rollout() for _ in 5]) # average of 5 random rollouts, for more accurate initial Q-vals
            # n(x, a) == here x.n[a]
            StartState.visits[action] += 1

            # N(x) == x.N
            Ch.state_visits += 1
            StartState.state_visits += 1

    # in MCTS - AFTER Backpropagation: Always start again at root Node!
    State : TreeNode = StartState
    while True:
        # Terminal node, no point in looking at its actions
        if State.reward != 0:
            break
        
        # UCB(x, a) = argmax_a{ Q(x,a) + c. √[ln(x.state_visits) / (x.visits[action])]
        UCB = argmax([State.Qs[a] + c * (log(State.state_visits) / State.visits[action])**0.5
                      for a in State.actions])
        
        # TODO complete this logic -> what to do when leaf node(add child) vs not leaf node (continue into next state in while loop)
        # Continue onto child with highest UCB
        State = State.children[State.actions[UCB]]


        breakpoint()

        # for Ch in State.children:
        #     r = Ch.rollout
        #     Node = Ch
        #     # Backup up to root node, Q-value of root note not really meaningful
        #     while Node.parent is not None:
        #         Node.visits += 1
                    
        # else:
        #     sB = Node.strB
        #     B = Node.board
        #     if state_node_map[sB] is None:
        #         state_node_map[sB] = TreeNode(B)
        #         State = state_node_map[sB]

        # pass T


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
