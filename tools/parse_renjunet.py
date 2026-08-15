#!/usr/bin/env python3
"""Parse RenjuNet XML game database into training samples.

Output: <out>.npz with keys states[N,4,15,15], policies[N,225], winners[N].

Selection (matches the original pipeline):
- rule=1 (Classic / freestyle, no forbidden moves)
- swap in ('', '-')  (no swap games: black/white identity stays fixed)
- at least 10 moves

Coordinate mapping: RenjuNet writes moves as "<col-letter><row-digit>"
e.g. h8 -> col=7, row=7 -> move=7*15+7. State encoding mirrors
Board.encode_state: channels [current stones, opponent stones, last move,
all-ones if current player is black].

Usage: python3 parse_renjunet.py <renjunet.xml> <out.npz> [--limit N]
"""
import argparse
import re
import numpy as np

SIZE = 15
XML_GAME_RE = re.compile(r'<game\b([^>]*)>(.*?)</game>', re.S)
ATTR_RE = re.compile(r'(\w+)="([^"]*)"')
MOVE_RE = re.compile(r'([a-o])(\d{1,2})')


def parse_game(attrs_text, body, board, states_list, probs_list, winners_list):
    attrs = dict(ATTR_RE.findall(attrs_text))
    if attrs.get('rule', '') != '1':
        return
    swap = attrs.get('swap', '')
    if swap not in ('', '-'):
        return
    m = re.search(r'<move>([^<]*)</move>', body)
    if not m:
        return
    tokens = m.group(1).split()
    if len(tokens) < 10:
        return
    moves = []
    for tok in tokens:
        mm = MOVE_RE.fullmatch(tok)
        if not mm:
            return
        col = ord(mm.group(1)) - ord('a')
        row = int(mm.group(2)) - 1
        if not (0 <= col < SIZE and 0 <= row < SIZE):
            return
        moves.append(row * SIZE + col)
    # remove duplicates (illegal repeats)
    seen, clean = set(), []
    for mv in moves:
        if mv not in seen:
            seen.add(mv)
            clean.append(mv)
    moves = clean
    if len(moves) < 10:
        return

    bresult = attrs.get('bresult', '')
    winner = -1
    if bresult == '1':
        winner = 1
    elif bresult == '0':
        winner = 2

    board.reset()
    board.play(moves[0])  # black first
    for k in range(len(moves) - 1):
        # player to move for this state (before playing moves[k+1])
        player_to_move = board.current_player
        state = board.current_state()
        states_list.append(state.reshape(-1))
        p = np.zeros(SIZE * SIZE, dtype=np.float32)
        p[moves[k + 1]] = 1.0
        probs_list.append(p)
        z = 1.0 if winner == player_to_move else (-1.0 if winner != -1 else 0.0)
        winners_list.append(z)
        board.play(moves[k + 1])
        end, w = board.game_end()
        if end:
            break


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('xml')
    ap.add_argument('out')
    ap.add_argument('--limit', type=int, default=None)
    args = ap.parse_args()

    class Board:
        def __init__(self):
            self.width = self.height = SIZE
            self.n_in_row = 5
            self.players = [1, 2]
            self.reset()

        def reset(self):
            self.states = {}
            self.current_player = 1
            self.availables = list(range(SIZE * SIZE))
            self.last_move = -1

        def play(self, mv):
            self.states[mv] = self.current_player
            self.availables.remove(mv)
            self.current_player = 2 if self.current_player == 1 else 1
            self.last_move = mv

        def game_end(self):
            win, winner = self.has_a_winner()
            if win:
                return True, winner
            if not self.availables:
                return True, -1
            return False, -1

        def has_a_winner(self):
            if self.last_move < 0:
                return False, -1
            r, c = divmod(self.last_move, SIZE)
            p = self.states.get(self.last_move, -1)
            if p == -1:
                return False, -1
            for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
                cnt = 1
                for s in range(1, 5):
                    rr, cc = r + dr * s, c + dc * s
                    if not (0 <= rr < SIZE and 0 <= cc < SIZE):
                        break
                    if self.states.get(rr * SIZE + cc) != p:
                        break
                    cnt += 1
                for s in range(1, 5):
                    rr, cc = r - dr * s, c - dc * s
                    if not (0 <= rr < SIZE and 0 <= cc < SIZE):
                        break
                    if self.states.get(rr * SIZE + cc) != p:
                        break
                    cnt += 1
                if cnt >= 5:
                    return True, p
            return False, -1

        def current_state(self):
            sq = np.zeros((4, SIZE, SIZE), dtype=np.float32)
            if self.states:
                for mv, pl in self.states.items():
                    r, c = divmod(mv, SIZE)
                    if pl == self.current_player:
                        sq[0, r, c] = 1.0
                    else:
                        sq[1, r, c] = 1.0
                if self.last_move >= 0:
                    r, c = divmod(self.last_move, SIZE)
                    sq[2, r, c] = 1.0
            if len(self.states) % 2 == 0:
                sq[3, :, :] = 1.0
            return sq

    board = Board()
    states, probs, winners = [], [], []
    with open(args.xml, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    for i, m in enumerate(XML_GAME_RE.finditer(content)):
        parse_game(m.group(1), m.group(2), board, states, probs, winners)
        if args.limit and len(winners) >= args.limit:
            break
        if (i + 1) % 20000 == 0:
            print('games {}, samples {}'.format(i + 1, len(winners)))
    print('total samples: {}'.format(len(winners)))
    np.savez(args.out,
             states=np.array(states, dtype=np.float32),
             policies=np.array(probs, dtype=np.float32),
             winners=np.array(winners, dtype=np.float32))
    print('saved', args.out)


if __name__ == '__main__':
    main()
