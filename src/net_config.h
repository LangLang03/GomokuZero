#pragma once

// Single source of truth for the network shape.
//
// The shape used to be duplicated in src/network.h, src/network_cuda.cpp,
// tests/dump_forward.cpp and tools/export_onnx.py, which made any change a
// guaranteed three-way inconsistency. Everything now derives from here.

namespace gomoku {

struct NetConfig {
    // Board geometry.
    static constexpr int kBoardSize = 15;
    static constexpr int kCells = kBoardSize * kBoardSize;  // 225
    static constexpr int kInputChannels = 4;

    // Shared trunk: a stem convolution followed by residual blocks. The
    // receptive field of a stack of 3x3 convolutions with padding 1 grows by
    // two per layer, so `kResBlocks` blocks (2 convolutions each) plus the
    // stem see 2 * (2 * kResBlocks + 1) + 1 cells across. Detecting five in a
    // row needs at least 9x9, and the previous 3-layer trunk only reached
    // 7x7, so long-range threats were structurally invisible.
    static constexpr int kChannels = 192;
    static constexpr int kResBlocks = 12;

    static constexpr int receptive_field() {
        return 2 * (2 * kResBlocks + 1) + 1;
    }

    // Policy head: a 1x1 convolution produces a per-cell score pair, which a
    // small fully connected layer mixes into the 225 move logits. The old
    // 900 -> 225 layer had no weight sharing across board positions and cost
    // 62% of all parameters.
    static constexpr int kPolicyFcIn = 2 * kCells;  // 450
    static constexpr int kPolicyOut = kCells;       // 225

    // Value head: a 1x1 convolution to a single plane, then two dense layers.
    static constexpr int kValueHidden = 64;
    static constexpr int kValueFcIn = kCells;  // 225
};

}  // namespace gomoku
