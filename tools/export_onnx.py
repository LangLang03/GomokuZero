#!/usr/bin/env python3
"""Export the 15x15 Gomoku policy/value network to a single ONNX file."""

import argparse
import hashlib
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from torch import nn
from torch.nn import functional as F


class GomokuNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(4, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.act_conv1 = nn.Conv2d(128, 4, kernel_size=1)
        self.act_fc1 = nn.Linear(4 * 15 * 15, 15 * 15)
        self.val_conv1 = nn.Conv2d(128, 2, kernel_size=1)
        self.val_fc1 = nn.Linear(2 * 15 * 15, 64)
        self.val_fc2 = nn.Linear(64, 1)

    def forward(self, state: torch.Tensor):
        trunk = F.relu(self.conv1(state))
        trunk = F.relu(self.conv2(trunk))
        trunk = F.relu(self.conv3(trunk))
        policy = F.relu(self.act_conv1(trunk)).flatten(1)
        log_policy = F.log_softmax(self.act_fc1(policy), dim=1)
        value = F.leaky_relu(self.val_conv1(trunk), 0.01).flatten(1)
        value = F.leaky_relu(self.val_fc1(value), 0.01)
        return log_policy, torch.tanh(self.val_fc2(value))


def load_state(path: Path):
    try:
        return torch.jit.load(str(path), map_location="cpu").state_dict()
    except RuntimeError:
        state = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(state, dict):
            raise TypeError(f"unsupported checkpoint object: {type(state)!r}")
        return state


def load_state_from_bin(directory: Path):
    """Load weights from raw .bin files (C++ trainer's native format).

    Each file: [uint32 count][float32 values...], named <layer>_weight.bin /
    <layer>_bias.bin, matching tools/export_model.py output.
    """
    import struct

    layer_shapes = {
        "conv1": ((32, 4, 3, 3), (32,)),
        "conv2": ((64, 32, 3, 3), (64,)),
        "conv3": ((128, 64, 3, 3), (128,)),
        "act_conv1": ((4, 128, 1, 1), (4,)),
        "act_fc1": ((225, 900), (225,)),
        "val_conv1": ((2, 128, 1, 1), (2,)),
        "val_fc1": ((64, 450), (64,)),
        "val_fc2": ((1, 64), (1,)),
    }

    def read_bin(name: str, shape):
        path = directory / name
        with open(path, "rb") as f:
            (count,) = struct.unpack("<I", f.read(4))
            data = np.frombuffer(f.read(), dtype="<f4")
            assert len(data) == count, f"{name}: count {count} != data {len(data)}"
        return torch.from_numpy(data.reshape(shape))

    state = {}
    for layer, (w_shape, b_shape) in layer_shapes.items():
        state[f"{layer}.weight"] = read_bin(f"{layer}_weight.bin", w_shape)
        state[f"{layer}.bias"] = read_bin(f"{layer}_bias.bin", b_shape)
    return state


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--bin-dir",
        type=Path,
        default=None,
        help="load weights from a directory of raw .bin files instead of source",
    )
    args = parser.parse_args()

    model = GomokuNet().eval()
    if args.bin_dir is not None:
        model.load_state_dict(load_state_from_bin(args.bin_dir), strict=True)
    else:
        model.load_state_dict(load_state(args.source), strict=True)
    example = torch.zeros(1, 4, 15, 15, dtype=torch.float32)
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
    source_sha = (
        hashlib.sha256(b"".join(p.read_bytes() for p in sorted(args.bin_dir.glob("*.bin")))).hexdigest()
        if args.bin_dir is not None
        else hashlib.sha256(args.source.read_bytes()).hexdigest()
    )
    metadata = {
        "source": str(args.bin_dir if args.bin_dir is not None else args.source),
        "source_sha256": source_sha,
        "board_size": "15x15",
        "policy_output": "log_softmax",
    }
    del exported.metadata_props[:]
    for key, value in metadata.items():
        item = exported.metadata_props.add()
        item.key = key
        item.value = value
    onnx.save(exported, args.output)

    rng = np.random.default_rng(20260814)
    test_input = np.zeros((3, 4, 15, 15), dtype=np.float32)
    for batch in range(3):
        occupied = rng.choice(15 * 15, size=20 + batch * 7, replace=False)
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
