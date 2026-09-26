#!/usr/bin/env python3
"""Validate and stream NPZ arrays into C++ float32 data and uint64 game IDs.

Files contain [little endian u64 element count][little endian array data].
NPZ members are extracted one at a time to temporary files and memory mapped,
so conversion and validation do not copy the entire dataset into RAM.
Missing game IDs are preserved as missing, never invented per sample.
"""
import argparse
from contextlib import contextmanager
import json
import math
from pathlib import Path
import shutil
import struct
import tempfile
import zipfile

import numpy as np

CELLS = 225


def terminal_positions(states):
    """Vectorized freestyle five-or-more test for both relative stone planes."""
    stones = states.reshape(-1, 4, 15, 15)[:, :2].astype(bool)
    terminal = np.zeros(len(stones), dtype=bool)
    for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
        rows, cols = 15-4*dr, 15-4*abs(dc)
        start_col = 4 if dc < 0 else 0
        runs = np.ones((len(stones), 2, rows, cols), dtype=bool)
        for offset in range(5):
            row, col = offset*dr, start_col+offset*dc
            runs &= stones[:, :, row:row+rows, col:col+cols]
        terminal |= runs.any(axis=(1, 2, 3))
    return terminal


@contextmanager
def mapped_npz(path):
    with tempfile.TemporaryDirectory(prefix='gomoku-npz-') as temporary, zipfile.ZipFile(path) as archive:
        arrays = {}
        for member in archive.infolist():
            name = member.filename
            if not name.endswith('.npy') or Path(name).name != name:
                raise ValueError(f'unsupported NPZ member: {name}')
            key = name[:-4]
            if key not in ('states', 'policies', 'probs', 'winners', 'game_ids', 'metadata'):
                continue
            if key in arrays:
                raise ValueError(f'duplicate NPZ member: {name}')
            target = Path(temporary)/name
            with archive.open(member) as source, target.open('wb') as output:
                shutil.copyfileobj(source, output, length=1024*1024)
            arrays[key] = np.load(target, mmap_mode='r', allow_pickle=False)
        yield arrays


def validate_arrays(states, probs, winners, game_ids=None, chunk_size=4096, check_terminal=True):
    if chunk_size < 1:
        raise ValueError('chunk_size must be positive')
    if winners.ndim != 1:
        raise ValueError('winners must be a nonempty vector')
    n = len(winners)
    if n == 0 or winners.shape != (n,):
        raise ValueError('winners must be a nonempty vector')
    if states.shape not in ((n, 4*CELLS), (n, 4, 15, 15)) or probs.shape != (n, CELLS):
        raise ValueError(f'unexpected shapes: {states.shape}, {probs.shape}, {winners.shape}')
    if any(array.dtype.kind not in 'biuf' for array in (states, probs, winners)):
        raise ValueError('state, policy, and winner arrays must contain real numeric values')
    if game_ids is not None and (game_ids.shape != (n,) or game_ids.dtype.kind not in 'ui'):
        raise ValueError('game_ids must be an integer vector with one ID per sample')
    group_outcomes = {}
    for start in range(0, n, chunk_size):
        # Reshape only one chunk: reshaping a full Fortran-order input would
        # silently allocate a second dense dataset.
        s = states[start:start+chunk_size].reshape(-1, 4, CELLS)
        p = probs[start:start+chunk_size]
        w = winners[start:start+chunk_size]
        if not np.isfinite(s).all() or not np.isfinite(p).all() or not np.isfinite(w).all():
            raise ValueError(f'non-finite data near sample {start}')
        if np.any((s != 0) & (s != 1)) or np.any(s[:, 0]+s[:, 1] > 1):
            raise ValueError(f'invalid stone planes near sample {start}')
        if np.any(p < 0) or np.any(np.abs(p.sum(1)-1) > 1e-5):
            raise ValueError(f'invalid policy probabilities near sample {start}')
        occupied = s[:, 0]+s[:, 1]
        if np.any((p*occupied).sum(1) > 1e-6):
            raise ValueError(f'policy targets occupied cells near sample {start}')
        if np.any((w < -1) | (w > 1)):
            raise ValueError(f'value targets outside [-1,1] near sample {start}')
        count = occupied.sum(1)
        black = count % 2 == 0
        if np.any(s[:, 3] != black[:, None]):
            raise ValueError(f'wrong current-player plane near sample {start}')
        if np.any(s[:, 0].sum(1) != count//2):
            raise ValueError(f'wrong stone counts for player to move near sample {start}')
        last = s[:, 2]
        if np.any(last.sum(1) != (count > 0)) or np.any(last > s[:, 1]):
            raise ValueError(f'last move must belong to the opponent near sample {start}')
        if game_ids is not None and np.any(game_ids[start:start+chunk_size] < 0):
            raise ValueError('game IDs must be nonnegative')
        if game_ids is not None:
            ids, inverse = np.unique(game_ids[start:start+chunk_size], return_inverse=True)
            black_outcome = w*np.where(black, 1, -1)
            low, high = np.full(len(ids), np.inf), np.full(len(ids), -np.inf)
            np.minimum.at(low, inverse, black_outcome)
            np.maximum.at(high, inverse, black_outcome)
            if np.any(high-low > 1e-5):
                raise ValueError('outcomes disagree within a game; winners must use current-player perspective')
            for group, outcome in zip(ids, low):
                previous = group_outcomes.setdefault(int(group), float(outcome))
                if abs(previous-outcome) > 1e-5:
                    raise ValueError('outcomes disagree within a game across chunks')
        if check_terminal and np.any(terminal_positions(s)):
            raise ValueError(f'policy sample is already terminal under freestyle rules near sample {start}')
    return n


def validate_metadata(metadata):
    if not isinstance(metadata, dict):
        raise ValueError('metadata must contain a JSON object')
    if metadata.get('target_rules', 'freestyle') != 'freestyle':
        raise ValueError('metadata target_rules must match freestyle Gomoku')
    weight = metadata.get('value_weight', 0.0)
    if isinstance(weight, bool) or not isinstance(weight, (int, float)) or not math.isfinite(weight) or not 0 <= weight <= 1:
        raise ValueError('metadata value_weight must be a number in [0,1]')
    if weight > 0 and metadata.get('target_rules') != 'freestyle':
        raise ValueError('positive value_weight requires explicit freestyle target_rules')
    supervision = metadata.get('value_supervision', 'enabled' if weight else 'disabled')
    if not isinstance(supervision, str) or (supervision == 'disabled' and weight > 0):
        raise ValueError('metadata value_supervision conflicts with value_weight')
    return dict(metadata, value_weight=weight, value_supervision=supervision)


def convert(source, prefix):
    prefix = Path(prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    with mapped_npz(source) as data:
        required = {'states', 'winners'}
        if not required.issubset(data) or not ('policies' in data or 'probs' in data):
            raise ValueError('NPZ needs states, policies (or probs), and winners')
        states, winners = data['states'], data['winners']
        probs = data['policies'] if 'policies' in data else data['probs']
        groups = data.get('game_ids')
        n = validate_arrays(states, probs, winners, groups)
        metadata = validate_metadata(json.loads(str(data['metadata'].item())) if 'metadata' in data else {})
        metadata.update(format_version=2, samples=n, source=str(source))
        # Unspecified source rules cannot establish valid value supervision.
        metadata.setdefault('value_supervision', 'disabled')
        metadata.setdefault('value_weight', 0.0)
        metadata['has_game_ids'] = groups is not None
        arrays = [('states', states, '<f4'), ('probs', probs, '<f4'), ('winners', winners, '<f4')]
        if groups is not None:
            arrays.append(('game_ids', groups, '<u8'))
        elif Path(f'{prefix}_game_ids.bin').exists():
            raise ValueError('output has existing game IDs; choose a fresh prefix for ungrouped data')
        with tempfile.TemporaryDirectory(prefix='.convert-', dir=prefix.parent) as temporary:
            for name, array, dtype in arrays:
                path = Path(temporary)/f'{name}.bin'
                with path.open('wb') as output:
                    output.write(struct.pack('<Q', array.size))
                    for start in range(0, n, 4096):
                        output.write(np.asarray(array[start:start+4096], dtype=dtype).tobytes(order='C'))
            for name, _, _ in arrays:
                (Path(temporary)/f'{name}.bin').replace(f'{prefix}_{name}.bin')
        Path(f'{prefix}_meta.json').write_text(json.dumps(metadata, indent=2)+'\n')
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('npz', type=Path)
    parser.add_argument('prefix', type=Path)
    args = parser.parse_args()
    try:
        print(json.dumps(convert(args.npz, args.prefix), indent=2))
    except (ValueError, KeyError, zipfile.BadZipFile) as error:
        parser.exit(2, f'conversion failed: {error}\n')


if __name__ == '__main__':
    main()
