#!/usr/bin/env python3
"""Convert a renju-net style .npz ({states, probs, winners}) into the raw
float32 .bin files the C++ trainer loads via --mix <prefix>.

Reconstructed after data loss; format verified against the original file
sizes: each output is [u64 element count][raw float32 values].

  <prefix>_states.bin  : N x (4*15*15)
  <prefix>_probs.bin   : N x 225
  <prefix>_winners.bin : N
"""
import argparse
import struct
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('npz')
    ap.add_argument('prefix')
    args = ap.parse_args()

    data = np.load(args.npz)
    states = np.asarray(data['states'])
    probs = np.asarray(data['policies'] if 'policies' in data else data['probs'])
    winners = np.asarray(data['winners'])
    n = len(winners)
    states = states.astype(np.float32).reshape(n, -1)
    probs = probs.astype(np.float32).reshape(n, -1)
    winners = winners.astype(np.float32).reshape(n)
    if states.shape[1] != 4 * 15 * 15 or probs.shape[1] != 15 * 15:
        raise SystemExit(f'unexpected shapes: states={states.shape} probs={probs.shape}')
    if not np.isfinite(states).all() or not np.isfinite(probs).all():
        raise SystemExit('non-finite values in input')
    for name, arr in [('_states', states), ('_probs', probs), ('_winners', winners)]:
        path = args.prefix + name + '.bin'
        with open(path, 'wb') as f:
            f.write(struct.pack('<Q', arr.size))
            f.write(arr.tobytes())
        print(f'wrote {path}  {arr.shape}')
    print(f'samples: {n}')


if __name__ == '__main__':
    main()
