#!/usr/bin/env python3
"""Produce fp16 and int8 variants of an exported policy/value net.

Quantization is only worth it if the engine still picks the same moves, so this
reports the numeric drift next to the size and the latency:

  * policy top-1 agreement with the fp32 model
  * mean KL between the fp32 and the quantized move distribution
  * value MSE difference

Calibration uses positions from the front of the dataset; the quality numbers
are measured on positions from the end of it, so the two never overlap.

Usage:
  python3 tools/quantize_onnx.py --src runs/256x16.onnx --out-dir runs/quantized \
      --calibration data/gomoku_v2_states.bin --samples 256
"""
import argparse
import os
import time

import numpy as np
import onnx
import onnxruntime as ort
from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_static
from onnxruntime.transformers import float16

PLANES = 4
BOARD = 15
CELLS = BOARD * BOARD
FLOATS_PER_SAMPLE = PLANES * CELLS


def read_states(path, offset_samples, count_samples):
    """Read a slice of a dataset written as [u64 floats][float32 ...].

    A negative offset counts back from the end, which is how the evaluation
    slice is kept disjoint from the calibration slice.
    """
    with open(path, "rb") as handle:
        total = int(np.fromfile(handle, dtype=np.uint64, count=1)[0])
        available = total // FLOATS_PER_SAMPLE
        start = available + offset_samples if offset_samples < 0 else min(offset_samples, available)
        start = max(0, min(start, available))
        take = min(count_samples, available - start)
        handle.seek(8 + start * FLOATS_PER_SAMPLE * 4)
        data = np.fromfile(handle, dtype=np.float32, count=take * FLOATS_PER_SAMPLE)
    if data.size != take * FLOATS_PER_SAMPLE:
        raise SystemExit(f"{path}: short read")
    return data.reshape(-1, PLANES, BOARD, BOARD)


class Reader(CalibrationDataReader):
    def __init__(self, states):
        self.states = states
        self.index = 0

    def get_next(self):
        if self.index >= len(self.states):
            return None
        item = {"state": self.states[self.index][None].astype(np.float32)}
        self.index += 1
        return item


def up_to_date(src, dst):
    return os.path.exists(dst) and os.path.getmtime(dst) >= os.path.getmtime(src)


def make_fp16(src, dst):
    model = onnx.load(src)
    converted = float16.convert_float_to_float16(model, keep_io_types=True)
    # The converter appends the input Cast node instead of inserting it at the
    # front, so the graph is no longer topologically sorted and the ONNX checker
    # rejects it. ORT's own helper re-sorts it.
    from onnxruntime.transformers.onnx_model import OnnxModel

    wrapper = OnnxModel(converted)
    wrapper.topological_sort()
    # onnxruntime 1.28 removed OnnxModel.save_model; the public writer is
    # save_model_to_file (save() keeps the model in memory only).
    wrapper.save_model_to_file(dst)
    return dst


def make_int8(src, dst, calibration):
    quantize_static(
        src,
        dst,
        Reader(calibration),
        quant_format=QuantFormat.QDQ,
        per_channel=True,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
    )
    return dst


def session(path, threads):
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(path, options, providers=["CPUExecutionProvider"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", default="runs/256x16.onnx")
    parser.add_argument("--out-dir", default="runs/quantized")
    parser.add_argument("--calibration", default="data/gomoku_v2_states.bin")
    parser.add_argument("--samples", type=int, default=256)
    parser.add_argument("--eval-samples", type=int, default=192)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--only", choices=("fp16", "int8"), default=None)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(args.src))[0]
    fp16_path = os.path.join(args.out_dir, f"{stem}_fp16.onnx")
    int8_path = os.path.join(args.out_dir, f"{stem}_int8.onnx")

    calibration = read_states(args.calibration, 0, args.samples)
    # Quality is measured away from the calibration window.
    evaluation = read_states(args.calibration, -args.eval_samples, args.eval_samples)
    print(f"calibration {calibration.shape} | evaluation {evaluation.shape} (disjoint)")

    print("\n== building variants ==")
    started = time.monotonic()
    if args.only != "int8":
        if up_to_date(args.src, fp16_path):
            print(f"fp16  {fp16_path}  {os.path.getsize(fp16_path)/1e6:.1f} MB  (reused)")
        else:
            make_fp16(args.src, fp16_path)
            print(f"fp16  {fp16_path}  {os.path.getsize(fp16_path)/1e6:.1f} MB  ({time.monotonic()-started:.1f}s)")
    if args.only != "fp16":
        started = time.monotonic()
        if up_to_date(args.src, int8_path):
            print(f"int8  {int8_path}  {os.path.getsize(int8_path)/1e6:.1f} MB  (reused)")
        else:
            make_int8(args.src, int8_path, calibration)
            print(f"int8  {int8_path}  {os.path.getsize(int8_path)/1e6:.1f} MB  ({time.monotonic()-started:.1f}s)")

    variants = [("fp32", args.src)]
    if args.only != "int8" and os.path.exists(fp16_path):
        variants.append(("fp16", fp16_path))
    if args.only != "fp16" and os.path.exists(int8_path):
        variants.append(("int8", int8_path))

    print("\n== inference cost (batch 1, batch 8) ==")
    outputs = {}
    for name, path in variants:
        try:
            onnx.checker.check_model(onnx.load(path))
        except Exception as error:  # noqa: BLE001 - reported, not fatal: ORT is the judge
            print(f"  warning: {name} fails the ONNX checker: {error}")
        sess = session(path, args.threads)
        blank = np.zeros((1, PLANES, BOARD, BOARD), dtype=np.float32)
        sess.run(None, {"state": blank})
        row = [f"{name:5s} {os.path.getsize(path)/1e6:6.1f} MB"]
        for batch in (1, 8):
            tensor = np.repeat(blank, batch, axis=0)
            sess.run(None, {"state": tensor})
            t0 = time.monotonic()
            for _ in range(3):
                sess.run(None, {"state": tensor})
            ms = (time.monotonic() - t0) / 3 * 1000
            row.append(f"b{batch}: {ms:7.0f} ms/call ({ms/batch:6.0f} ms/pos)")
        print("  " + "  ".join(row))

        logits, values = [], []
        for sample in evaluation:
            policy, value = sess.run(None, {"state": sample[None].astype(np.float32)})
            logits.append(policy[0])
            values.append(float(value.reshape(-1)[0]))
        outputs[name] = (np.array(logits), np.array(values))

    if "fp32" not in outputs:
        return
    base_logits, base_values = outputs["fp32"]
    base_top = base_logits.argmax(axis=1)
    base_probs = np.exp(base_logits - base_logits.max(axis=1, keepdims=True))
    base_probs /= base_probs.sum(axis=1, keepdims=True)

    print("\n== drift vs fp32 ==")
    print(f"  {'variant':6s} {'top1 agree':>11s} {'mean KL':>9s} {'max KL':>8s} {'value MAE':>10s}")
    for name, (logits, values) in outputs.items():
        top = logits.argmax(axis=1)
        agree = float((top == base_top).mean())
        probs = np.exp(logits - logits.max(axis=1, keepdims=True))
        probs /= probs.sum(axis=1, keepdims=True)
        kl = np.sum(base_probs * np.log(np.maximum(base_probs, 1e-9) / np.maximum(probs, 1e-9)), axis=1)
        mae = float(np.abs(values - base_values).mean())
        print(f"  {name:6s} {agree*100:10.2f}% {kl.mean():9.5f} {kl.max():8.4f} {mae:10.5f}")


if __name__ == "__main__":
    main()
