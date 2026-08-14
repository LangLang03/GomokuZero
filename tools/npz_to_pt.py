#!/usr/bin/env python3
"""Convert a renju-net style .npz ({states, probs, winners}) into a torch
checkpoint dict {states, probs, winners} saved as mix_data.pt.

Reconstructed after data loss; matches the original keys seen in the
Python training pipeline.
"""
import argparse
import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('npz')
    ap.add_argument('out')
    args = ap.parse_args()

    data = np.load(args.npz)
    t_states = torch.from_numpy(np.asarray(data['states']).astype(np.float32))
    t_probs = torch.from_numpy(np.asarray(data['probs']).astype(np.float32))
    t_win = torch.from_numpy(np.asarray(data['winners']).astype(np.float32))
    torch.save({'states': t_states, 'probs': t_probs, 'winners': t_win}, args.out)
    print(f'saved {args.out}  {t_win.numel()} samples')


if __name__ == '__main__':
    main()
