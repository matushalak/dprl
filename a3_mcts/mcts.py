# @matushalak
# Monte-Carlo Tree Search Connect Four Game
'''
Connect Four is a board-size
6 x 7, representation B in R^{6 X 7}
'''
from numpy import ndarray, unique, fliplr, array, where, zeros, mean, argmax, argmin, log
from numpy.random import randint, choice
from string import ascii_letters as ascl
import argparse
from collections import defaultdict
import os
import time

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
                winner = sub_res[0]
                return winner
    if not (B == 0).any():
        return 0 # draw if no free pieces left
    else:
        return winner # None if no winner (no connect 4), but still pieces left

# this tree Class does the 1/3 of heavy lifting
class McNode():
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
        self.reward = evaluate_board(B)
        self.state_visits = 0 # #times this state visited, small init for UCT calc

        # Q-values for each (current_state, possible_action) pair
        self.Qs = {a:0 for a in self.actions}
        # Visits for each (state, action) pair from this state, initialize with small value for UCT calc
        self.visits = {a:1e-2 for a in self.actions}
        # children are TreeNodes themselves, self is parent of its children
        self.children = {a:None for a in self.actions} # added on a need-to-add basis

    # Only Add child when needed, no need to generate self.children every time a node is accessed
    def add_child(self, action:int) -> tuple['McNode', str]:
        assert self.player in (1,2), 'Needs to be either player 1 or player 2!!!'
        B_new:ndarray = self.board.copy()
        B_new[where(B_new[:,action] == 0)[0][-1], 
              action] = self.player
        # child node
        ChildNode = McNode(B_new, self) # self is a Parent
        # action that created you
        ChildNode.parent_move = action # can call this up later
        self.children[action] = ChildNode
        return ChildNode, ChildNode.strB # for use outside

    def available_moves(self, B:ndarray):
        return [i for i,c in enumerate(B.T) if 0 in c] # transpose for columns

    def turn(self, B:ndarray):
        counts = [(B == 1).sum(), (B == 2).sum()]
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

        match winner:
            case 0: # draw
                return 0
            case 1: # agent won
                return 1 
            case 2: # agent lost
                return -1 
            case _:
                return ValueError('Should always be draw / win / loss')


# this MCTS function does 2/3 of heavy lifting
def MCTS(startB:ndarray, SNmap:defaultdict[str:McNode|None], c:float = 2**0.5, iterations:float = 3e3
         ) -> tuple[int, defaultdict]:
    '''
    Main function performing Monte Carlo Tree Search Algorithm 
    with Upper Confidence Trees (UCT) & min-max algorithm to consider playing optimal oponent

    XXX Run this function for every move!!! XXX
    returns:
        tuple(bests move based on #iterations (int); 
              *updated* SNmap with all the nodes ever encountered)
    ----------------------
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
    # get this from the board
    # sB = ''.join([str(n) for n in startB.ravel()])
    sB = string_board(startB)

    # first try to access the state from previous simulations
    if SNmap[sB] is not None:
        StartState:McNode = SNmap[sB]
    # new unexplored board position
    else:
        StartState:McNode = McNode(startB)
        # stores hashable reference to every possible state (which has its depth, children, etc.)
        # allows us to reuse previously built tree
        SNmap[sB] = StartState

    # in MCTS - AFTER Backpropagation: Always start again at root Node!
    for sim in range(int(iterations)): # 1000 iterations for each move
        State : McNode = StartState
        path:list[McNode] = [State]
        while True:
            # visualize Monte Carlo Tree 'thinking' (running simulations & figuring out best move)
            # if sim == int(iterations)-1:
            #     _ = string_board(State.board)
            #     time.sleep(0.25)
            
            # Terminal Node -> can use backpropagation of direct rewards straight away
            if State.reward is not None:
                break

            # Minimax approach (Maximizes for player 1 nodes, Minimizes for player 2 nodes)
            minimax:function = argmax if State.player == 1 else argmin
            
            # UCB(x, a) = argmax_a{ Q(x,a) + c. √[ln(x.state_visits) / (x.visits[action])]
            # will weigh unexplored options more initially because of visit initialization w small number
            # UCB here is the action chosen
            UCB:int = State.actions[minimax(ucbv :=
                [State.Qs[a] + c * (max(1e-1, log(State.state_visits)) / State.visits[a])**0.5
                 for a in State.actions])]
            
            # print(State.strB, State.depth, UCB, ucbv)
            
            # Add if unvisited Child node
            if State.children[UCB] == None:
                State, sB = State.add_child(UCB)
            else:
                # Continue onto child with highest UCB, if visited
                State = State.children[UCB]
                
            # path we are following in this iteration, will use to backpropagate along this path
            path.append(State)

            # restart at Leaf Node
            try:
                if SNmap[sB] is None:
                    SNmap[sB] = State
                    # print(f'{sim} simulation done')
                    break

            except AttributeError:
                breakpoint()
   
        # Immediate Reward if terminal node, else Rollout
        # State.reward is WHO IS THE WINNER, NOT ACTUAL REWARD!!!!
        reward_map = {0:0,1:1,2:-1}
        r = reward_map[State.reward] if State.reward is not None else State.rollout()
        for Step in path[::-1]:
            Step.state_visits += 1

            if Step.parent is not None:
                # add 1 visit to action that created you
                Step.parent.visits[Step.parent_move] += 1
                # incremental average - converges to true average
                Step.parent.Qs[Step.parent_move] += (r - Step.parent.Qs[Step.parent_move] # adjustment toward new reward
                                                        ) / Step.parent.visits[Step.parent_move] # diminishing updates over time
    else:
        if State.reward is not None:
            return State.reward, State.board, SNmap
        else:
            # need to get most visited, minimax already reflected in the UCB rule!!!
            best_action = max(StartState.visits.items(), key = lambda item: item[1])[0]
            
            # Return best action, tree map
            # return best_action, SNmap
            
            # Return board corresponding to best action, tree map
            
            return  StartState.children[best_action].reward, StartState.children[best_action].board, SNmap
            
def game(start:ndarray, opponent:str = 'random'):
    # Tree dictionary from which we re-use previously seen nodes
    tree_dict : defaultdict[str: McNode | None] = defaultdict(lambda: None)
    winner = evaluate_board(start)
    B :ndarray = start
    
    while winner is None:
        match opponent:
            case 'random':
                # sB = string_board(B)
                # time.sleep(0.25)
                sB = ''.join([str(n) for n in B.ravel()])
                # player w more less pieces' turn, if equal pieces, P1 starts
                try:
                    turn : int = tree_dict[sB].player
                except AttributeError:
                    turn = 1
                if turn == 1:
                    winner, B, tree_dict = MCTS(B, tree_dict) 
                else: # random P2 turn
                    rand_move = choice(tree_dict[sB].actions) # random action choice
                    # still run MCTS, but dont'update board & winner based on optimal simulation, 
                    # just update the tree
                    _, _, tree_dict = MCTS(B, tree_dict) 
                    if tree_dict[sB].children[rand_move] is None:
                        tree_dict[sB].add_child(rand_move)
                    winner, B = tree_dict[sB].children[rand_move].reward, tree_dict[sB].children[rand_move].board
                
            case 'optimal':
                # resulting child node becomes root node at next move
                winner, B, tree_dict = MCTS(B, tree_dict) 

            case 'human':
                pass
        
        if winner is not None:
            _ = string_board(B)
            print(f'WinnerID:{winner} -> {d[winner]}')
            return winner, B, tree_dict
    

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-sym', nargs='+', type = str, default = ['🍎','⚽️'])
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    symbols = [' '+s if s.isalnum() else s for s in args.sym]
    d = {0:'  ', 1:symbols[0], 2:symbols[1]}
    
    # Board(s)
    # hardcoded for assignment
    hardcodedB = array([[1, 1, 1, 2, 0, 2, 0],
                        [1, 2, 2, 2, 0, 1, 0],
                        [2, 1, 1, 1, 0, 2, 0],
                        [1, 2, 2, 2, 0, 1, 0],
                        [2, 2, 2, 1, 0, 1, 0],
                        [1, 1, 2, 1, 0, 2, 0]])
    # for the real game
    zeroB = zeros((6,7), dtype=int) 
    # for quicktesting & evaluation
    # B = randint(0, 3, (6,7), dtype=int) 
    # B = zeroB
    B = hardcodedB

    # play game
    w, winB, Tree = game(B)
    # print(winB)
    # w, winB, Tree = game(B, opponent='optimal')
    # w, winB, Tree = game(B, opponent='human')
