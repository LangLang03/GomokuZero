#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include <iostream>
#include <chrono>
#include "game.h"
#include "network.h"
#include "selfplay.h"
#include "train.h"

using namespace gomoku;

static void print_usage() {
    std::cout <<
        "Usage: gomoku_train <command> [options]\n"
        "  train  --init <bin_dir> --games N --playout N --batch-games N\n"
        "         [--mix <prefix>] [--mix-ratio 0.5] [--check-freq 50]\n"
        "         [--threads N] [--int8] [--cpu] [--tag name]\n"
        "  selfplay --model <bin_dir> --playout N --games N [--batch N]\n"
        "         [--threads N] [--int8] [--cpu] (CUDA is automatic)\n"
        "         (benchmark self-play throughput)\n"
        "  human  --model <bin_dir> --playout N  (play vs AI, moves: 'r,c')\n";
}

int cmd_train(int argc, char** argv) {
    std::string init, mix, tag = "cpp";
    int games = 1000, playout = 400, batch_games = 64;
    int buffer = 10000, batch_size = 512, epochs = 5, check_freq = 50;
    int n_threads = -1;
    bool int8_inference = false;
    bool force_cpu = false;
    float mix_ratio = 0.5f;

    for (int i = 2; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--init") { if (i+1<argc) init = argv[++i]; }
        else if (a == "--mix") { if (i+1<argc) mix = argv[++i]; }
        else if (a == "--tag") { if (i+1<argc) tag = argv[++i]; }
        else if (a == "--games") { if (i+1<argc) games = std::stoi(argv[++i]); }
        else if (a == "--playout") { if (i+1<argc) playout = std::stoi(argv[++i]); }
        else if (a == "--batch-games") { if (i+1<argc) batch_games = std::stoi(argv[++i]); }
        else if (a == "--buffer") { if (i+1<argc) buffer = std::stoi(argv[++i]); }
        else if (a == "--batch-size") { if (i+1<argc) batch_size = std::stoi(argv[++i]); }
        else if (a == "--epochs") { if (i+1<argc) epochs = std::stoi(argv[++i]); }
        else if (a == "--check-freq") { if (i+1<argc) check_freq = std::stoi(argv[++i]); }
        else if (a == "--mix-ratio") { if (i+1<argc) mix_ratio = std::stof(argv[++i]); }
        else if (a == "--threads") { if (i+1<argc) n_threads = std::stoi(argv[++i]); }
        else if (a == "--int8") { int8_inference = true; }
        else if (a == "--cpu") { force_cpu = true; }
    }

    PureNet net(2e-3);
    net.set_cuda_enabled(!force_cpu);
    if (!init.empty() && !net.load(init)) {
        std::cerr << "model load failed; starting from scratch\n";
    }
    net.set_quantized_inference(int8_inference);
    if (int8_inference && net.using_cuda())
        std::cerr << "[net] --int8 is CPU-only; CUDA uses FP32/TF32\n";
    Trainer trainer(net, playout, batch_games, n_threads, 5.0f, 1.0f,
                    buffer, batch_size, epochs, check_freq, games,
                    mix, mix_ratio, tag);
    trainer.run();
    return 0;
}

int cmd_selfplay(int argc, char** argv) {
    std::string model;
    int playout = 400, games = 4, batch = 4;
    int n_threads = -1;
    bool int8_inference = false;
    bool force_cpu = false;
    for (int i = 2; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--model") { if (i+1<argc) model = argv[++i]; }
        else if (a == "--playout") { if (i+1<argc) playout = std::stoi(argv[++i]); }
        else if (a == "--games") { if (i+1<argc) games = std::stoi(argv[++i]); }
        else if (a == "--batch") { if (i+1<argc) batch = std::stoi(argv[++i]); }
        else if (a == "--threads") { if (i+1<argc) n_threads = std::stoi(argv[++i]); }
        else if (a == "--int8") { int8_inference = true; }
        else if (a == "--cpu") { force_cpu = true; }
    }
    PureNet net(2e-3);
    net.set_cuda_enabled(!force_cpu);
    if (!net.load(model)) return 1;
    net.set_quantized_inference(int8_inference);
    if (int8_inference && net.using_cuda())
        std::cerr << "[net] --int8 is CPU-only; CUDA uses FP32/TF32\n";

    BatchedSelfPlay sp(net, 5.0f, playout, batch, n_threads);
    auto t0 = std::chrono::steady_clock::now();
    int total_games = 0;
    int total_moves = 0;
    while (total_games < games) {
        auto res = sp.run_batch(1.0f);
        for (auto& g : res) if (!g.empty()) {
            total_games++;
            total_moves += static_cast<int>(g.size());
        }
        auto dt = std::chrono::duration<float>(
            std::chrono::steady_clock::now() - t0).count();
        std::cout << "played " << total_games << "/" << games
                  << " games in " << dt << "s ("
                  << dt / std::max(1,total_games) << "s/game, "
                  << static_cast<float>(total_moves) /
                         std::max(1, total_games) << " moves/game)\n";
    }
    return 0;
}

int cmd_human(int argc, char** argv) {
    std::string model;
    bool force_cpu = false;
    for (int i = 2; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--model") { if (i+1<argc) model = argv[++i]; }
        else if (a == "--cpu") { force_cpu = true; }
    }
    PureNet net(2e-3);
    net.set_cuda_enabled(!force_cpu);
    if (!net.load(model)) return 1;

    Board board;
    std::cout << "five-in-a-row vs AI. enter 'r,c' (0-14). AI is white.\n";
    while (true) {
        int r, c;
        std::cout << "your move: ";
        std::string line;
        if (!std::getline(std::cin, line)) break;
        if (sscanf(line.c_str(), "%d,%d", &r, &c) != 2) continue;
        int move = r * BOARD_SIZE + c;
        if (move < 0 || move >= BOARD_CELLS || !board.is_empty(move)) continue;
        board.play(move);
        if (board.winner()) { std::cout << "you win!\n"; break; }

        std::vector<float> state(4 * BOARD_CELLS);
        board.encode_state(state.data());
        std::vector<float> lp;
        float val;
        net.forward_one(state, lp, val);
        float best = -1e9f; int bestm = -1;
        for (int m = 0; m < BOARD_CELLS; ++m)
            if (board.is_empty(m) && lp[m] > best) { best = lp[m]; bestm = m; }
        board.play(bestm);
        std::cout << "AI plays " << bestm / BOARD_SIZE << ","
                  << bestm % BOARD_SIZE << " (value " << val << ")\n";
        if (board.winner()) { std::cout << "AI wins\n"; break; }
        if (board.move_count() == BOARD_CELLS) { std::cout << "draw\n"; break; }
    }
    return 0;
}

int main(int argc, char** argv) {
    if (argc < 2) { print_usage(); return 0; }
    std::string cmd = argv[1];
    if (cmd == "train") return cmd_train(argc, argv);
    if (cmd == "selfplay") return cmd_selfplay(argc, argv);
    if (cmd == "human") return cmd_human(argc, argv);
    print_usage();
    return 0;
}
