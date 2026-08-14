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
        value = F.relu(self.val_conv1(trunk)).flatten(1)
        value = F.relu(self.val_fc1(value))
        return log_policy, torch.tanh(self.val_fc2(value))


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
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    model = GomokuNet().eval()
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
    metadata = {
        "source": str(args.source),
        "source_sha256": hashlib.sha256(args.source.read_bytes()).hexdigest(),
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
