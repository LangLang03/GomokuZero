#!/usr/bin/env python3
"""Export the 15x15 Gomoku policy/value network to a single ONNX file.

The PyTorch side loads the raw .bin tensor set written by the C++ trainer.
Inference exports store batch normalisation folded into the convolution.
Training checkpoints retain the original convolutions and BN statistics;
this exporter reads those authoritative tensors when training.meta exists.
Network dimensions are inferred from saved tensor headers."""

import argparse
import hashlib
import math
import re
from dataclasses import dataclass
import struct
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

@dataclass(frozen=True)
class Architecture:
    board: int = 15
    channels: int = 192
    res_blocks: int = 12
    value_hidden: int = 64

    @property
    def cells(self):
        return self.board * self.board


def read_tensor(directory, name, shape=None):
    path = Path(directory) / name
    with path.open('rb') as file:
        header = file.read(4)
        if len(header) != 4:
            raise ValueError(f'{path}: truncated tensor header')
        count, = struct.unpack('<I', header)
        if path.stat().st_size != 4 + count * 4:
            raise ValueError(f'{path}: tensor size does not match header')
        data = np.fromfile(file, dtype='<f4', count=count)
    if shape is not None and count != math.prod(shape):
        raise ValueError(f'{path}: expected {math.prod(shape)} values, got {count}')
    if not np.isfinite(data).all():
        raise ValueError(f'{path}: non-finite tensor values')
    return torch.from_numpy(data.reshape(shape) if shape is not None else data)


def infer_architecture(directory):
    directory = Path(directory)
    if (directory / 'training.meta').exists():
        metadata = dict(line.split() for line in (directory/'training.meta').read_text().splitlines() if line.strip())
        if metadata.get('version') not in ('1', '2'):
            raise ValueError('unsupported training checkpoint version')
        conv_count = int(metadata.get('conv_count', '0'))
        if conv_count < 3 or (conv_count-3) % 2 or metadata.get('fc_count') != '3':
            raise ValueError('unsupported checkpoint architecture')
        channels = read_tensor(directory, 'training_conv0_bias.bin').numel()
        cells = read_tensor(directory, 'training_fc0_bias.bin').numel()
        hidden = read_tensor(directory, 'training_fc1_bias.bin').numel()
        blocks = (conv_count-3)//2
    else:
        channels = read_tensor(directory, 'stem_bias.bin').numel()
        cells = read_tensor(directory, 'act_fc1_bias.bin').numel()
        hidden = read_tensor(directory, 'val_fc1_bias.bin').numel()
        indices = sorted(int(match[1]) for path in directory.glob('res*_a_weight.bin')
                         if (match := re.fullmatch(r'res(\d+)_a_weight\.bin', path.name)))
        if indices != list(range(len(indices))):
            raise ValueError('residual block indices must be contiguous')
        blocks = len(indices)
    board = math.isqrt(cells)
    if board*board != cells or channels < 1 or hidden < 1:
        raise ValueError('invalid network dimensions')
    return Architecture(board, channels, blocks, hidden)


class ResBlock(nn.Module):
    def __init__(self, config=Architecture()) -> None:
        super().__init__()
        self.conv_a = nn.Conv2d(config.channels, config.channels, 3, padding=1)
        self.bn_a = nn.BatchNorm2d(config.channels)
        self.conv_b = nn.Conv2d(config.channels, config.channels, 3, padding=1)
        self.bn_b = nn.BatchNorm2d(config.channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn_a(self.conv_a(x)))
        out = self.bn_b(self.conv_b(out))
        return F.relu(x + out)


class GomokuNet(nn.Module):
    def __init__(self, config=Architecture()) -> None:
        super().__init__()
        self.stem = nn.Conv2d(4, config.channels, 3, padding=1)
        self.stem_bn = nn.BatchNorm2d(config.channels)
        self.trunk = nn.ModuleList(ResBlock(config) for _ in range(config.res_blocks))
        self.policy_conv1 = nn.Conv2d(config.channels, 2, 1)
        self.policy_bn = nn.BatchNorm2d(2)
        self.act_fc1 = nn.Linear(2 * config.cells, config.cells)
        self.val_conv1 = nn.Conv2d(config.channels, 1, 1)
        self.val_bn = nn.BatchNorm2d(1)
        self.val_fc1 = nn.Linear(config.cells, config.value_hidden)
        self.val_fc2 = nn.Linear(config.value_hidden, 1)

    def forward(self, state: torch.Tensor):
        h = F.relu(self.stem_bn(self.stem(state)))
        for block in self.trunk:
            h = block(h)
        policy = F.relu(self.policy_bn(self.policy_conv1(h))).flatten(1)
        log_policy = F.log_softmax(self.act_fc1(policy), dim=1)
        value = F.relu(self.val_bn(self.val_conv1(h))).flatten(1)
        value = F.relu(self.val_fc1(value))
        return log_policy, torch.tanh(self.val_fc2(value))


def set_identity_batch_norm(module: nn.Module) -> None:
    """Batch norm already folded into the weights: make it a no-op."""
    for bn in module.modules():
        if isinstance(bn, nn.BatchNorm2d):
            with torch.no_grad():
                bn.weight.fill_(1.0)
                bn.bias.zero_()
                bn.running_mean.zero_()
                bn.running_var.fill_(1.0 - bn.eps)


def layer_shapes(config=Architecture()):
    channels, cells, hidden = config.channels, config.cells, config.value_hidden
    shapes = {'stem': ((channels, 4, 3, 3), (channels,))}
    for index in range(config.res_blocks):
        for suffix in ('a', 'b'):
            shapes[f'res{index}_{suffix}'] = ((channels, channels, 3, 3), (channels,))
    shapes.update({
        'policy_conv1': ((2, channels, 1, 1), (2,)),
        'val_conv1': ((1, channels, 1, 1), (1,)),
        'act_fc1': ((cells, 2*cells), (cells,)),
        'val_fc1': ((hidden, cells), (hidden,)),
        'val_fc2': ((1, hidden), (1,)),
    })
    return shapes


def load_state_from_bin(directory: Path, config=None) -> dict:
    """Read folded inference exports or authoritative raw training tensors."""
    directory = Path(directory)
    config = config or infer_architecture(directory)
    training = (directory/'training.meta').exists()
    state, conv_index, fc_index = {}, 0, 0
    batch_norm_of = {'stem': 'stem_bn', 'policy_conv1': 'policy_bn', 'val_conv1': 'val_bn'}
    for layer, (w_shape, b_shape) in layer_shapes(config).items():
        if layer.startswith('res'):
            index, suffix = layer[3:].split('_')
            target, bn_name = f'trunk.{index}.conv_{suffix}', f'trunk.{index}.bn_{suffix}'
        else:
            target, bn_name = layer, batch_norm_of.get(layer)
        if training:
            if bn_name is not None:
                prefix = f'training_conv{conv_index}'
                conv_index += 1
            else:
                prefix = f'training_fc{fc_index}'
                fc_index += 1
        else:
            prefix = layer
        for part, shape in [('weight', w_shape), ('bias', b_shape)]:
            state[f'{target}.{part}'] = read_tensor(directory, f'{prefix}_{part}.bin', shape)
        if bn_name is not None:
            channels = w_shape[0]
            if training:
                for part, filename in [('weight', 'bn_gamma'), ('bias', 'bn_beta'),
                                       ('running_mean', 'bn_mean'), ('running_var', 'bn_var')]:
                    tensor = read_tensor(directory, f'{prefix}_{filename}.bin', (channels,))
                    if part == 'running_var' and (tensor < 0).any():
                        raise ValueError(f'{prefix}: negative BN variance')
                    state[f'{bn_name}.{part}'] = tensor
            else:
                state[f'{bn_name}.weight'] = torch.ones(channels)
                state[f'{bn_name}.bias'] = torch.zeros(channels)
                state[f'{bn_name}.running_mean'] = torch.zeros(channels)
                state[f'{bn_name}.running_var'] = torch.full((channels,), 1.0-1e-5)
            state[f'{bn_name}.num_batches_tracked'] = torch.tensor(0, dtype=torch.long)
    return state


def load_state(path: Path):
    try:
        return torch.jit.load(str(path), map_location="cpu").state_dict()
    except RuntimeError:
        state = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(state, dict):
            raise TypeError(f"unsupported checkpoint object: {type(state)!r}")
        return state


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path, nargs="?", default=None)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--bin-dir",
        type=Path,
        default=None,
        help="load weights from the C++ trainer's .bin directory",
    )
    args = parser.parse_args()

    import onnx
    import onnxruntime as ort
    if args.bin_dir is not None and args.source is not None:
        parser.error('provide only one of source and --bin-dir')
    if args.bin_dir is not None:
        config = infer_architecture(args.bin_dir)
        state = load_state_from_bin(args.bin_dir, config)
    elif args.source is not None:
        state = load_state(args.source)
        cells = state['act_fc1.bias'].numel()
        blocks = {key.split('.')[1] for key in state if key.startswith('trunk.')}
        config = Architecture(math.isqrt(cells), state['stem.weight'].shape[0],
                              len(blocks), state['val_fc1.bias'].numel())
        if config.cells != cells:
            raise ValueError('policy output must be a square board')
    else:
        parser.error("provide either a checkpoint or --bin-dir")
    model = GomokuNet(config).eval()
    model.load_state_dict(state, strict=True)
    board, cells = config.board, config.cells

    parameters = sum(p.numel() for p in model.parameters())
    print(f"parameters: {parameters:,}")

    example = torch.zeros(1, 4, board, board, dtype=torch.float32)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        (example,),
        str(args.output),
        input_names=["state"],
        output_names=["log_policy", "value"],
        dynamic_axes={
            "state": {0: "batch"},
            "log_policy": {0: "batch"},
            "value": {0: "batch"},
        },
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
        external_data=False,
    )

    exported = onnx.load(args.output)
    onnx.checker.check_model(exported)
    if args.bin_dir is not None:
        digest = hashlib.sha256()
        source_files = sorted(args.bin_dir.glob('*.bin'))
        if (args.bin_dir/'training.meta').exists():
            source_files.append(args.bin_dir/'training.meta')
        for path in source_files:
            digest.update(path.name.encode()+b'\0')
            with path.open('rb') as file:
                for chunk in iter(lambda: file.read(1024*1024), b''):
                    digest.update(chunk)
        source_sha = digest.hexdigest()
        source = str(args.bin_dir)
    else:
        source_sha = hashlib.sha256(args.source.read_bytes()).hexdigest()
        source = str(args.source)
    metadata = {
        "source": source,
        "source_sha256": source_sha,
        "board_size": f"{board}x{board}",
        "channels": str(config.channels),
        "res_blocks": str(config.res_blocks),
        "policy_output": "log_softmax",
        "rules": "freestyle",
        "state_encoding": "current_player,opponent,last_move,black_to_move",
        "legal_policy": "mask_occupied_and_renormalize",
    }
    del exported.metadata_props[:]
    for key, value in metadata.items():
        item = exported.metadata_props.add()
        item.key = key
        item.value = value
    onnx.save(exported, args.output)

    rng = np.random.default_rng(20260814)
    test_input = np.zeros((3, 4, board, board), dtype=np.float32)
    for batch in range(3):
        occupied = rng.choice(cells, size=20 + batch * 7, replace=False)
        split = len(occupied) // 2
        test_input[batch, 0].flat[occupied[:split]] = 1.0
        test_input[batch, 1].flat[occupied[split:]] = 1.0
        test_input[batch, 2].flat[occupied[-1]] = 1.0
        if len(occupied) % 2 == 0:
            test_input[batch, 3].fill(1.0)
    with torch.no_grad():
        expected = model(torch.from_numpy(test_input))
    session = ort.InferenceSession(str(args.output), providers=["CPUExecutionProvider"])
    actual = session.run(None, {"state": test_input})
    policy_error = float(np.max(np.abs(actual[0] - expected[0].numpy())))
    value_error = float(np.max(np.abs(actual[1] - expected[1].numpy())))
    if policy_error > 2e-4 or value_error > 2e-4:
        raise RuntimeError(
            f"ONNX verification failed: policy={policy_error}, value={value_error}"
        )
    print(f"saved: {args.output}")
    print(f"sha256: {hashlib.sha256(args.output.read_bytes()).hexdigest()}")
    print(f"max_abs_error: log_policy={policy_error:.8g}, value={value_error:.8g}")


if __name__ == "__main__":
    main()
