# -*- coding: utf-8 -*-
"""Export Python-trained state_dict to raw float32 binary files readable by C++.

Each layer saved as <out_dir>/<key>.bin: [4 bytes = element count][raw float32].
Usage: python3 export_model.py best_policy.model [out_dir]
"""
import sys
import os
import struct
import torch


def main():
    model = sys.argv[1] if len(sys.argv) > 1 else 'best_policy.model'
    out_dir = sys.argv[2] if len(sys.argv) > 2 else 'cpp_model'
    os.makedirs(out_dir, exist_ok=True)

    try:
        state = torch.load(model, map_location='cpu', weights_only=False)
    except Exception as e:
        print('not a torch state_dict:', e)
        return 1
    if not isinstance(state, dict):
        if hasattr(state, 'state_dict'):
            state = state.state_dict()
            print('extracted state_dict from TorchScript module')
        else:
            print('unexpected model format:', type(state))
            return 1

    for key, tensor in state.items():
        if not isinstance(tensor, torch.Tensor):
            print('skipping non-tensor:', key)
            continue
        flat = tensor.detach().cpu().numpy().ravel().astype('float32')
        out = os.path.join(out_dir, key.replace('.', '_') + '.bin')
        with open(out, 'wb') as f:
            f.write(struct.pack('<I', flat.size))
            f.write(flat.tobytes())
        print('wrote', out, flat.size)
    return 0


if __name__ == '__main__':
    sys.exit(main())
