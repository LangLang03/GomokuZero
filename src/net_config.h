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
    // Sanitizer builds may shrink the trunk. TSan instruments every memory
    // access, so the full 12x192 trunk makes a self-play run take tens of
    // minutes; the concurrency logic under test does not depend on the shape.
#ifndef GOMOK_NET_CHANNELS
#define GOMOK_NET_CHANNELS 192
#endif
#ifndef GOMOK_NET_RES_BLOCKS
#define GOMOK_NET_RES_BLOCKS 12
#endif
    static constexpr int kChannels = GOMOK_NET_CHANNELS;
    static constexpr int kResBlocks = GOMOK_NET_RES_BLOCKS;

    // True when the trunk was shrunk for a sanitizer run, so tests can relax
    // capacity expectations without weakening the production assertions.
    static constexpr bool kShrunkForSanitizer() {
#if defined(GOMOK_NET_CHANNELS) || defined(GOMOK_NET_RES_BLOCKS)
        return true;
#else
        return false;
#endif
    }

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
