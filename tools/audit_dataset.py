#!/usr/bin/env python3
"""Audit C++ datasets without loading their dense arrays into RAM.

Reports physical tensor validity, whole-game groups, and the expected fraction
of games leaked by a random sample split. Optional --state-overlap estimates
D4-equivalent position overlap after a reproducible whole-game split; even
clean game splits can share opening positions. This measures dataset leakage,
not playing strength or whether a network has overfit.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import struct

import numpy as np

from npz_to_bin import terminal_positions, validate_arrays


def mapped_array(prefix, name, dtype, width=1):
    path = Path(f'{prefix}_{name}.bin')
    with path.open('rb') as file:
        header = file.read(8)
    if len(header) != 8:
        raise ValueError(f'{path}: truncated header')
    count, = struct.unpack('<Q', header)
    if path.stat().st_size != 8 + count*np.dtype(dtype).itemsize or count % width:
        raise ValueError(f'{path}: invalid payload size')
    if count == 0:
        raise ValueError(f'{path}: empty dataset')
    shape = (count//width, width) if width > 1 else (count,)
    return np.memmap(path, dtype=dtype, mode='r', offset=8, shape=shape)


def infer_game_ids(states):
    n = len(states)
    if n == 0:
        raise ValueError('cannot infer games from an empty dataset')
    counts = np.empty(n, dtype=np.int16)
    for start in range(0, n, 4096):
        counts[start:start+4096] = states[start:start+4096, :450].sum(1)
    boundaries = np.r_[True, np.diff(counts) != 1]
    if counts[0] > 1 or np.any(counts[boundaries] > 1):
        raise ValueError('cannot infer legacy games: expected games starting at zero or one stone')
    # Check adjacent positions really extend the previous board, not shuffled
    # positions with an accidentally matching stone count.
    for start in range(1, n, 4096):
        end = min(n, start+4096)
        keep = ~boundaries[start:end]
        before = states[start-1:end-1].reshape(-1, 4, 225)[keep]
        after = states[start:end].reshape(-1, 4, 225)[keep]
        if np.any(after[:, 0] != before[:, 1]) or np.any(after[:, 1]-before[:, 0] != after[:, 2]):
            raise ValueError('cannot infer legacy games: positions do not form consecutive legal moves')
    return np.cumsum(boundaries, dtype=np.uint64)-1


def canonical_state_hash(state):
    array = state.reshape(4, 15, 15).astype(np.uint8)
    variants = []
    for rotation in range(4):
        rotated = np.rot90(array, rotation, axes=(1, 2))
        variants.append(np.packbits(rotated).tobytes())
        variants.append(np.packbits(rotated[:, :, ::-1]).tobytes())
    return hashlib.blake2b(min(variants), digest_size=16).digest()


def splitmix64(value):
    """Same unsigned arithmetic as src/dataset.cpp's mix64."""
    mask = (1 << 64)-1
    value = (value+0x9e3779b97f4a7c15) & mask
    value = ((value ^ (value >> 30))*0xbf58476d1ce4e5b9) & mask
    value = ((value ^ (value >> 27))*0x94d049bb133111eb) & mask
    return value ^ (value >> 31)


def holdout_game_ids(groups, fraction=0.1, seed=1):
    """Whole-game holdout, byte-for-byte reproducible with the C++ trainer."""
    if not math.isfinite(fraction) or not 0 <= fraction < 1:
        raise ValueError('validation fraction must be in [0,1)')
    if not isinstance(seed, int) or not 0 <= seed < (1 << 64):
        raise ValueError('seed must be an unsigned 64-bit integer')
    ids = sorted(set(map(int, groups)))
    if fraction == 0:
        return set()
    if len(ids) < 2:
        raise ValueError('validation requires at least two games')
    # C++ llround rounds positive half-integers upward, unlike Python round.
    count = min(len(ids)-1, max(1, math.floor(len(ids)*fraction+0.5)))
    ids.sort(key=lambda group: (splitmix64(group ^ seed), group))
    return set(ids[:count])


def audit(prefix, fraction=0.1, state_overlap=False, seed=1):
    if not 0 < fraction < 1:
        raise ValueError('validation fraction must be between 0 and 1')
    states = mapped_array(prefix, 'states', '<f4', 900)
    probs = mapped_array(prefix, 'probs', '<f4', 225)
    winners = mapped_array(prefix, 'winners', '<f4')
    has_groups = Path(f'{prefix}_game_ids.bin').exists()
    groups = mapped_array(prefix, 'game_ids', '<u8') if has_groups else infer_game_ids(states)
    n = validate_arrays(states, probs, winners, groups, check_terminal=False)
    terminal_count = 0
    terminal_groups = set()
    for start in range(0, n, 4096):
        terminal = terminal_positions(states[start:start+4096])
        terminal_count += int(terminal.sum())
        terminal_groups.update(groups[start:start+4096][terminal].tolist())
    ids, sizes = np.unique(groups, return_counts=True)
    metadata_path = Path(f'{prefix}_meta.json')
    report = {
        'prefix': str(prefix), 'samples': n, 'games': len(ids),
        'group_source': 'explicit' if has_groups else 'inferred_consecutive_legal_games',
        'tensor_validation': 'passed',
        'freestyle_training_validation': 'failed' if terminal_count else 'passed',
        'terminal_policy_samples': terminal_count,
        'games_with_terminal_policy_samples': len(terminal_groups),
        'mean_samples_per_game': float(sizes.mean()),
        'sample_split_expected_leaked_game_fraction': float(np.mean(1-(1-fraction)**sizes-fraction**sizes)),
        'winner_counts': {str(float(value)): int(count) for value, count in zip(*np.unique(winners, return_counts=True))},
        'metadata': json.loads(metadata_path.read_text()) if metadata_path.exists() else None,
    }
    if state_overlap:
        validation_groups = holdout_game_ids(ids, fraction, seed)
        train_hashes, validation = set(), []
        for index, state in enumerate(states):
            key = canonical_state_hash(state)
            if int(groups[index]) in validation_groups:
                validation.append((key, int(state[:450].sum())))
            else:
                train_hashes.add(key)
        overlap = sum(key in train_hashes for key, _ in validation)
        late = [(key, count) for key, count in validation if count >= 10]
        report['position_overlap'] = {
            'seed': seed, 'split_algorithm': 'splitmix64(game_id ^ seed), round-half-up group count',
            'validation_samples': len(validation),
            'D4_overlap_samples': overlap,
            'D4_overlap_fraction': overlap/max(1, len(validation)),
            'validation_samples_after_purge': len(validation)-overlap,
            'train_samples': n-len(validation),
            'validation_samples_at_least_10_stones': len(late),
            'D4_overlap_at_least_10_stones': sum(key in train_hashes for key, _ in late),
        }
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('prefix', type=Path)
    parser.add_argument('--validation-fraction', type=float, default=0.1)
    parser.add_argument('--state-overlap', action='store_true')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    report = json.dumps(audit(args.prefix, args.validation_fraction, args.state_overlap, args.seed), indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report+'\n')
    print(report)


if __name__ == '__main__':
    main()
