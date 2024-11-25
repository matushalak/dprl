# @matushalak
# Monte-Carlo Tree Search Connect Four Game
'''
Connect Four is a board-size
6 x 7, representation B \in \R^{6 X 7}
'''
from numpy import ndarray, unique, fliplr
from numpy.random import randint
from string import ascii_letters as ascl
import argparse

def print_board(B:ndarray, symbols:dict[int:str, int:str]):
    # printed board with symbols
    header = [' | '+l for l in ascl[:B.shape[1]]]
    PrB = ' '+ ''.join(header)+ ' |  ' +'\n'
    for i, row in enumerate(B):
        symb = [f'{symbols[n]} |' for n in row]
        PrB += str(i)+ ' |' + ''.join(symb) + ' ' + str(i) + '\n'
    PrB += len(' '+ ''.join(header)+ ' |  ') * '-' + '\n'
    PrB += ' '+ ''.join(header)+ ' |  '
    print(PrB)

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
    return winner # None if no winner (no connect 4)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-sym', nargs='+', type = str, default = ['⚽️','🍎'])
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    symbols = [' '+s if s.isalnum() else s for s in args.sym]
    d = {0:'  ', 1:symbols[0], 2:symbols[1]}
    # Board
    # B = np.zeros((6,7), dtype=int)
    B = randint(0, 3, (6,7), dtype=int) # for testing & evaluation
    print_board(B, d)
    winner = evaluate_board(B)
    if winner:
        print(f'WinnerID:{winner} -> {d[winner]}')
    else:
        print('No winner yet!')
