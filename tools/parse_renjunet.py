#!/usr/bin/env python3
"""Convert audited RenjuNet records to grouped freestyle training examples.

Default source rules are Gomoku Pro/Swap2/Swap (7,11,28), NOT Classic Renju
(rule 1). These sources use different opening/overline rules, so metadata
explicitly disables value supervision; their moves can be policy pretraining
examples. The first five opening placements are never policy targets.

All games are validated before writing samples. Illegal repeats, unknown
results, moves after freestyle termination and contradictory terminal results
reject a whole record. Complete duplicate/D4-equivalent games are removed.

python tools/parse_renjunet.py data/renjunet.xml --bin-prefix data/gomoku_v2
python tools/parse_renjunet.py data/renjunet.xml dataset.npz
"""
import argparse
from collections import Counter
from dataclasses import dataclass
import gzip
import hashlib
import html
import json
from pathlib import Path
import re
import struct
import tempfile

import numpy as np

from gomoku import BOARD, CELLS, Board

XML_GAME_RE = re.compile(r'<game\b([^>]*)>(.*?)</game>', re.S)
ATTR_RE = re.compile(r'(\w+)="([^"]*)"')
MOVE_RE = re.compile(r'([a-o])([1-9]|1[0-5])')
DEFAULT_RULES = {'7', '11', '28'}


def canonical_moves(moves):
    """Stable bytes for all eight square symmetries, preserving move order."""
    points = np.asarray(moves, dtype=np.uint16)
    rows, cols = points // BOARD, points % BOARD
    variants = []
    for _ in range(4):
        variants.append((rows * BOARD + cols).astype('u1').tobytes())
        variants.append((rows * BOARD + BOARD-1-cols).astype('u1').tobytes())
        rows, cols = cols, BOARD-1-rows
    return min(variants)


@dataclass(frozen=True)
class Game:
    moves: tuple
    winner: int
    group: int
    rule: str


def validate_game(attrs, body, rules=DEFAULT_RULES, min_moves=10):
    """Return (Game, None) or (None, rejection reason). No partial samples."""
    if attrs.get('rule') not in rules:
        return None, 'other_rule'
    if attrs.get('swap', '') not in ('', '-'):
        return None, 'swap_opening'
    result = attrs.get('bresult')
    if result not in ('1', '0', '0.5'):
        return None, 'unknown_result'
    match = re.search(r'<move>([^<]*)</move>', body)
    if match is None:
        return None, 'missing_moves'
    tokens = match[1].split()
    if len(tokens) < min_moves:
        return None, 'short_game'
    moves = []
    for token in tokens:
        move = MOVE_RE.fullmatch(token)
        if move is None:
            return None, 'invalid_coordinate'
        moves.append((int(move[2])-1)*BOARD + ord(move[1])-ord('a'))
    if len(set(moves)) != len(moves):
        return None, 'repeated_move'
    winner = {'1': 1, '0': 2, '0.5': -1}[result]
    board = Board()
    for index, move in enumerate(moves):
        board.play(move)
        ended, actual = board.game_end()
        if ended:
            if actual != winner:
                return None, 'terminal_result_conflict'
            if index != len(moves)-1:
                return None, 'moves_after_terminal'
    key = canonical_moves(moves)
    group = int.from_bytes(hashlib.blake2b(key, digest_size=8).digest(), 'little')
    return Game(tuple(moves), winner, group, attrs['rule']), None


def read_games(path, rules=DEFAULT_RULES, min_moves=10, limit=None, skip_opening=5):
    if min_moves < 1 or skip_opening < 0 or (limit is not None and limit < 1):
        raise ValueError('min_moves/limit must be positive; skip_opening must be nonnegative')
    opener = gzip.open if str(path).endswith('.gz') else open
    with opener(path, 'rt', encoding='utf-8') as file:
        content = file.read()
    games, conflicts, counts = {}, set(), Counter()
    for match in XML_GAME_RE.finditer(content):
        counts['records'] += 1
        attrs = {key: html.unescape(value) for key, value in ATTR_RE.findall(match[1])}
        game, rejection = validate_game(attrs, match[2], rules, min_moves)
        if rejection:
            counts[rejection] += 1
            continue
        if len(game.moves) <= skip_opening:
            counts['no_training_moves'] += 1
            continue
        key = canonical_moves(game.moves)
        if key in conflicts:
            counts['conflicting_duplicate'] += 1
        elif key in games:
            if games[key].winner == game.winner:
                counts['duplicate_game'] += 1
            else:
                del games[key]
                conflicts.add(key)
                counts['conflicting_duplicate'] += 2
        else:
            games[key] = game
    selected, samples = [], 0
    for game in games.values():
        selected.append(game)
        samples += len(game.moves)-skip_opening
        # Keep games intact. A sample limit may exceed its target by one game.
        if limit is not None and samples >= limit:
            break
    counts['accepted_games'], counts['samples'] = len(selected), samples
    counts['excluded_by_limit'] = len(games)-len(selected)
    return selected, dict(counts)


def game_samples(game, skip_opening=5):
    count = max(0, len(game.moves)-skip_opening)
    states = np.empty((count, 4, BOARD, BOARD), dtype='<f4')
    policies = np.zeros((count, CELLS), dtype='<f4')
    winners = np.empty(count, dtype='<f4')
    board = Board()
    for index, move in enumerate(game.moves):
        if index >= skip_opening:
            row = index-skip_opening
            states[row] = board.current_state()
            policies[row, move] = 1
            winners[row] = 0 if game.winner == -1 else (1 if game.winner == board.current_player else -1)
        board.play(move)
    groups = np.full(count, game.group, dtype='<u8')
    return states, policies, winners, groups


def write_dataset(games, counts, source, output=None, bin_prefix=None, skip_opening=5, rules=DEFAULT_RULES):
    n = counts['samples']
    if skip_opening < 0:
        raise ValueError('skip_opening must be nonnegative')
    if n != sum(max(0, len(game.moves)-skip_opening) for game in games):
        raise ValueError('sample count disagrees with complete games and skip_opening')
    if n == 0:
        raise ValueError('no usable training samples')
    metadata = {
        'format_version': 2, 'samples': n, 'games': len(games),
        'source': str(source), 'source_rules': sorted(rules),
        'target_rules': 'freestyle', 'value_supervision': 'disabled',
        'value_weight': 0.0, 'skip_opening': skip_opening,
        'grouping': 'blake2b64 of D4 canonical complete move sequence',
        'deduplicated': True, 'statistics': counts,
    }
    if bin_prefix is not None:
        prefix = Path(bin_prefix)
        prefix.parent.mkdir(parents=True, exist_ok=True)
        specs = [('states', 4*CELLS), ('probs', CELLS), ('winners', 1), ('game_ids', 1)]
        with tempfile.TemporaryDirectory(prefix='.dataset-', dir=prefix.parent) as temporary:
            paths = [Path(temporary)/f'{name}.bin' for name, _ in specs]
            files = [path.open('wb') for path in paths]
            try:
                for file, (_, width) in zip(files, specs):
                    file.write(struct.pack('<Q', n*width))
                for game in games:
                    for file, values in zip(files, game_samples(game, skip_opening)):
                        file.write(values.tobytes())
            finally:
                for file in files:
                    file.close()
            for path, (name, _) in zip(paths, specs):
                path.replace(f'{prefix}_{name}.bin')
        Path(f'{prefix}_meta.json').write_text(json.dumps(metadata, indent=2)+'\n')
    if output is not None:
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='.dataset-', dir=output.parent) as temporary:
            specs = [('states', (n, 4, BOARD, BOARD), '<f4'),
                     ('policies', (n, CELLS), '<f4'), ('winners', (n,), '<f4'),
                     ('game_ids', (n,), '<u8')]
            arrays = {key: np.lib.format.open_memmap(Path(temporary)/f'{key}.npy', mode='w+', dtype=dtype, shape=shape)
                      for key, shape, dtype in specs}
            offset = 0
            for game in games:
                rows = game_samples(game, skip_opening)
                length = len(rows[0])
                for key, values in zip(arrays, rows):
                    arrays[key][offset:offset+length] = values
                offset += length
            for array in arrays.values():
                array.flush()
            np.savez(output, **arrays, metadata=np.asarray(json.dumps(metadata)))
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('xml', type=Path)
    parser.add_argument('out', type=Path, nargs='?')
    parser.add_argument('--bin-prefix', type=Path, help='stream raw C++ data files directly, with game IDs and metadata')
    parser.add_argument('--limit', type=int, help='sample limit; final whole game may exceed it')
    parser.add_argument('--rules', default=','.join(sorted(DEFAULT_RULES)), help='comma separated RenjuNet rule IDs; Classic Renju is 1')
    parser.add_argument('--min-moves', type=int, default=10)
    parser.add_argument('--skip-opening', type=int, default=5)
    args = parser.parse_args()
    if args.out is None and args.bin_prefix is None:
        parser.error('provide output.npz or --bin-prefix')
    rules = set(args.rules.split(','))
    games, counts = read_games(args.xml, rules, args.min_moves, args.limit, args.skip_opening)
    metadata = write_dataset(games, counts, args.xml, args.out, args.bin_prefix, args.skip_opening, rules)
    print(json.dumps(metadata, indent=2))


if __name__ == '__main__':
    main()
