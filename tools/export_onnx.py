#!/usr/bin/env python3
"""Export the 15x15 Gomoku policy/value network to a single ONNX file.

The PyTorch side loads the raw .bin tensor set written by the C++ trainer.
That format stores batch normalisation folded into the convolution, so the
batch-norm modules here are initialised to identity (var = 1 - eps, which
makes the fold factor exactly 1)."""

import argparse
import hashlib
import struct
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from torch import nn
from torch.nn import functional as F

# Mirrors src/net_config.h.
BOARD = 15
CELLS = BOARD * BOARD
CHANNELS = 192
RES_BLOCKS = 12
VALUE_HIDDEN = 64


class ResBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv_a = nn.Conv2d(CHANNELS, CHANNELS, 3, padding=1)
        self.bn_a = nn.BatchNorm2d(CHANNELS)
        self.conv_b = nn.Conv2d(CHANNELS, CHANNELS, 3, padding=1)
        self.bn_b = nn.BatchNorm2d(CHANNELS)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn_a(self.conv_a(x)))
        out = self.bn_b(self.conv_b(out))
        return F.relu(x + out)


class GomokuNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.stem = nn.Conv2d(4, CHANNELS, 3, padding=1)
        self.stem_bn = nn.BatchNorm2d(CHANNELS)
        self.trunk = nn.ModuleList(ResBlock() for _ in range(RES_BLOCKS))
        self.policy_conv1 = nn.Conv2d(CHANNELS, 2, 1)
        self.policy_bn = nn.BatchNorm2d(2)
        self.act_fc1 = nn.Linear(2 * CELLS, CELLS)
        self.val_conv1 = nn.Conv2d(CHANNELS, 1, 1)
        self.val_bn = nn.BatchNorm2d(1)
        self.val_fc1 = nn.Linear(CELLS, VALUE_HIDDEN)
        self.val_fc2 = nn.Linear(VALUE_HIDDEN, 1)

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


def layer_shapes() -> dict:
    shapes = {
        "stem": ((CHANNELS, 4, 3, 3), (CHANNELS,)),
        "policy_conv1": ((2, CHANNELS, 1, 1), (2,)),
        "val_conv1": ((1, CHANNELS, 1, 1), (1,)),
        "act_fc1": ((CELLS, 2 * CELLS), (CELLS,)),
        "val_fc1": ((VALUE_HIDDEN, CELLS), (VALUE_HIDDEN,)),
        "val_fc2": ((1, VALUE_HIDDEN), (1,)),
    }
    for i in range(RES_BLOCKS):
        shapes[f"res{i}_a"] = ((CHANNELS, CHANNELS, 3, 3), (CHANNELS,))
        shapes[f"res{i}_b"] = ((CHANNELS, CHANNELS, 3, 3), (CHANNELS,))
    return shapes


def load_state_from_bin(directory: Path) -> dict:
    """Load the raw .bin tensor set written by the C++ trainer.

    Each file is [uint32 count][float32 values...].
    """

    def read_bin(name: str, shape):
        path = directory / name
        with open(path, "rb") as f:
            (count,) = struct.unpack("<I", f.read(4))
            data = np.frombuffer(f.read(), dtype="<f4")
            if len(data) != count:
                raise ValueError(f"{name}: header {count} != {len(data)} values")
        return torch.from_numpy(data.reshape(shape).copy())

    state = {}
    # Only the convolutions carry a batch-norm module.
    batch_norm_of = {"stem": "stem_bn", "policy_conv1": "policy_bn",
                     "val_conv1": "val_bn"}

    for layer, (w_shape, b_shape) in layer_shapes().items():
        if layer.startswith("res"):
            index = layer[3:layer.index("_")]
            target = f"trunk.{index}.conv_{layer[-1]}"
            bn_name = f"trunk.{index}.bn_{layer[-1]}"
        else:
            target = layer
            bn_name = batch_norm_of.get(layer)
        state[f"{target}.weight"] = read_bin(f"{layer}_weight.bin", w_shape)
        state[f"{target}.bias"] = read_bin(f"{layer}_bias.bin", b_shape)

        if bn_name is not None:
            # The convolution already has batch norm folded in, so the module
            # is a no-op: var = 1 - eps makes the fold factor exactly 1.
            out_channels = w_shape[0]
            state[f"{bn_name}.weight"] = torch.ones(out_channels)
            state[f"{bn_name}.bias"] = torch.zeros(out_channels)
            state[f"{bn_name}.running_mean"] = torch.zeros(out_channels)
            state[f"{bn_name}.running_var"] = torch.full((out_channels,),
                                                         1.0 - 1e-5)
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

    model = GomokuNet().eval()
    if args.bin_dir is not None:
        model.load_state_dict(load_state_from_bin(args.bin_dir), strict=True)
        set_identity_batch_norm(model)
    elif args.source is not None:
        model.load_state_dict(load_state(args.source), strict=True)
    else:
        parser.error("provide either a checkpoint or --bin-dir")

    parameters = sum(p.numel() for p in model.parameters())
    print(f"parameters: {parameters:,}")

    example = torch.zeros(1, 4, BOARD, BOARD, dtype=torch.float32)
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
        source_sha = hashlib.sha256(
            b"".join(p.read_bytes() for p in sorted(args.bin_dir.glob("*.bin")))
        ).hexdigest()
        source = str(args.bin_dir)
    else:
        source_sha = hashlib.sha256(args.source.read_bytes()).hexdigest()
        source = str(args.source)
    metadata = {
        "source": source,
        "source_sha256": source_sha,
        "board_size": f"{BOARD}x{BOARD}",
        "channels": str(CHANNELS),
        "res_blocks": str(RES_BLOCKS),
        "policy_output": "log_softmax",
    }
    del exported.metadata_props[:]
    for key, value in metadata.items():
        item = exported.metadata_props.add()
        item.key = key
        item.value = value
    onnx.save(exported, args.output)

    rng = np.random.default_rng(20260814)
    test_input = np.zeros((3, 4, BOARD, BOARD), dtype=np.float32)
    for batch in range(3):
        occupied = rng.choice(CELLS, size=20 + batch * 7, replace=False)
        split = (len(occupied) + 1) // 2
        test_input[batch, 0].flat[occupied[:split]] = 1.0
        test_input[batch, 1].flat[occupied[split:]] = 1.0
        test_input[batch, 2].flat[occupied[-1]] = 1.0
        if batch % 2 == 0:
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
