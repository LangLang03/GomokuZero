#include "network.h"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace fs = std::filesystem;
namespace {
void check(bool okay, const char* message) {
    if (!okay) throw std::runtime_error(message);
}
struct TemporaryDirectory {
    fs::path path = fs::temp_directory_path() /
        ("gomoku_save_failure_" + std::to_string(
            std::chrono::steady_clock::now().time_since_epoch().count()));
    TemporaryDirectory() { fs::create_directories(path); }
    ~TemporaryDirectory() { std::error_code ignored; fs::remove_all(path, ignored); }
};
fs::path failing_directory(const fs::path& root, const char* name, const char* filename) {
    const auto directory = root / name;
    fs::create_directories(directory);
    fs::create_symlink("/dev/full", directory / filename);
    return directory;
}
}

int main() {
    try {
        if (!fs::exists("/dev/full")) {
            std::cout << "save_test: /dev/full unavailable; fault injection skipped\n";
            return 77;
        }
        TemporaryDirectory temporary;
        gomoku::PureNet net;
        net.set_cuda_enabled(false);
        // The final eight-byte tensor remains buffered. Before flush/close
        // checks, export returned true despite this write failing with ENOSPC.
        const auto cpu = failing_directory(temporary.path, "cpu", "val_fc2_bias.bin");
        check(!net.export_inference(cpu.string()),
              "native export must propagate a buffered final-tensor write failure");
        fs::remove(cpu / "val_fc2_bias.bin");
        check(net.export_inference(cpu.string()), "native export can retry after a write failure");
        gomoku::PureNet restored;
        restored.set_cuda_enabled(false);
        check(restored.load(cpu.string()), "successful export remains loadable");

        if (net.cuda_available()) {
            net.set_cuda_enabled(true);
            const auto gpu = failing_directory(temporary.path, "gpu", "val_fc2_bias.bin");
            check(!net.export_inference(gpu.string()),
                  "CUDA export must propagate a buffered final-tensor write failure");
            const auto raw = failing_directory(temporary.path, "raw", "training_fc2_bias.bin");
            check(!net.save_training_state(raw.string(), 0),
                  "training tensor failure must prevent a successful checkpoint result");
            const auto metadata = failing_directory(temporary.path, "metadata", "training.meta");
            check(!net.save_training_state(metadata.string(), 0),
                  "training metadata flush failure must prevent checkpoint success");
            const auto optimizer = failing_directory(temporary.path, "optimizer", "training_optimizer.pt");
            check(!net.save_training_state(optimizer.string(), 0),
                  "empty optimizer archive write failure must return false without aborting");
            fs::remove(optimizer / "training_optimizer.pt");
            check(net.save_training_state(optimizer.string(), 0),
                  "optimizer archive can retry after a write failure");
            check(restored.load_training_state(optimizer.string(), nullptr),
                  "memory-serialized empty optimizer archive remains loadable");

            // Nonempty Adam moments exercise a full-size archive, rather than
            // only the small metadata archive of a fresh optimizer.
            constexpr int batch = 2, cells = gomoku::NetConfig::kCells;
            std::vector<float> states(batch * 4 * cells, 0.0f), targets(batch * cells, 0.0f);
            for (int row = 0; row < batch; ++row) {
                for (int cell = 0; cell < cells; ++cell)
                    states[row * 4 * cells + 3 * cells + cell] = 1.0f;
                targets[row * cells + cells / 2] = 1.0f;
            }
            net.train_step(states, batch, targets, {1.0f, -1.0f}, 1e-4);
            const auto trained = failing_directory(temporary.path, "optimizer_trained", "training_optimizer.pt");
            check(!net.save_training_state(trained.string(), 1),
                  "nonempty optimizer archive write failure must return false without aborting");
            fs::remove(trained / "training_optimizer.pt");
            check(net.save_training_state(trained.string(), 1), "trained optimizer archive can retry after failure");
            int resumed_batch = -1;
            check(restored.load_training_state(trained.string(), &resumed_batch) && resumed_batch == 1,
                  "memory-serialized nonempty optimizer archive preserves resume state");
        }
        std::cout << "save_test passed" << (net.cuda_available() ? " (CPU and CUDA)\n" : " (CPU; CUDA skipped)\n");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "save_test: " << error.what() << '\n';
        return 1;
    }
}
