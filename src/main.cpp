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
        "         [--threads N] [--cpu] [--tag name]\n"
        "         [--batch-size N] [--epochs N] [--buffer N]\n"
        "         [--lr 2e-3] [--lr-final-fraction 0.1] [--lr-warmup 2]\n"
        "         [--weight-decay 1e-4] [--grad-clip 2.0]\n"
        "         [--label-smooth 0.1] [--value-loss-weight 1.0]\n"
        "         [--augment 0|1] [--eval-games N]\n"
        "         [--c-puct 5.0] [--temp-high 1.0] [--temp-low 0.5]\n"
        "         [--temp-switch 30] [--dirichlet-eps 0.25]\n"
        "         [--dirichlet-alpha 0.3] [--max-candidates 60]\n"
        "         [--virtual-loss 3.0] [--unseen-smooth 0.05]\n"
        "  selfplay --model <bin_dir> --playout N --games N [--batch N]\n"
        "         [--threads N] [--cpu] (CUDA is automatic)\n"
        "         (benchmark self-play throughput)\n"
        "  human  --model <bin_dir> --playout N  (play vs AI, moves: 'r,c')\n";
}

int cmd_train(int argc, char** argv) {
    std::string init, mix, tag = "cpp";
    int games = 1000, batch_games = 64;
    int n_threads = -1;
    bool force_cpu = false;

    SearchConfig search;
    TrainConfig train;

    for (int i = 2; i < argc; ++i) {
        std::string a = argv[i];
        auto next_int = [&]() { return i + 1 < argc ? std::stoi(argv[++i]) : 0; };
        auto next_float = [&]() { return i + 1 < argc ? std::stof(argv[++i]) : 0.0f; };
        auto next_double = [&]() { return i + 1 < argc ? std::stod(argv[++i]) : 0.0; };
        if (a == "--init") { if (i+1<argc) init = argv[++i]; }
        else if (a == "--mix") { if (i+1<argc) train.mix_data = argv[++i]; }
        else if (a == "--tag") { if (i+1<argc) train.tag = argv[++i]; tag = train.tag; }
        else if (a == "--games") { games = next_int(); train.game_batch_num = games; }
        else if (a == "--playout") { search.n_playout = next_int(); }
        else if (a == "--batch-games") { batch_games = next_int(); search.batch_games = batch_games; }
        else if (a == "--buffer") { train.buffer_size = next_int(); }
        else if (a == "--batch-size") { train.batch_size = next_int(); }
        else if (a == "--epochs") { train.epochs = next_int(); }
        else if (a == "--check-freq") { train.check_freq = next_int(); }
        else if (a == "--mix-ratio") { train.mix_ratio = next_float(); }
        else if (a == "--threads") { n_threads = next_int(); }
        else if (a == "--int8") {
            std::cerr << "[net] --int8 was removed with the new architecture; "
                         "ignoring\n";
        }
        else if (a == "--cpu") { force_cpu = true; }
        // Optimiser / schedule.
        else if (a == "--lr") { train.lr_base = next_double(); }
        else if (a == "--lr-final-fraction") { train.lr_final_fraction = next_double(); }
        else if (a == "--lr-warmup") { train.lr_warmup_batches = next_int(); }
        else if (a == "--weight-decay") { train.weight_decay = next_double(); }
        else if (a == "--grad-clip") { train.grad_clip = next_double(); }
        else if (a == "--label-smooth") { train.label_smooth = next_float(); }
        else if (a == "--value-loss-weight") { train.value_loss_weight = next_double(); }
        else if (a == "--augment") { train.augment = next_int() != 0; }
        else if (a == "--eval-games") { train.eval_games = next_int(); }
        // Search / self-play.
        else if (a == "--c-puct") { search.c_puct = next_float(); }
        else if (a == "--temp-high") { search.temp_high = next_float(); }
        else if (a == "--temp-low") { search.temp_low = next_float(); }
        else if (a == "--temp-switch") { search.temp_switch_move = next_int(); }
        else if (a == "--dirichlet-eps") { search.dirichlet_eps = next_float(); }
        else if (a == "--dirichlet-alpha") { search.dirichlet_alpha = next_float(); }
        else if (a == "--max-candidates") { search.max_candidates = next_int(); }
        else if (a == "--virtual-loss") { search.virtual_loss = next_float(); }
        else if (a == "--unseen-smooth") { search.unseen_move_smooth = next_float(); }
        else {
            std::cerr << "unknown option: " << a << "\n";
            return 2;
        }
    }
    search.n_threads = n_threads;

    PureNet net(2e-3);
    net.set_cuda_enabled(!force_cpu);
    if (!init.empty() && !net.load(init)) {
        std::cerr << "model load failed; starting from scratch\n";
    }
    std::cerr << "[net] backend: "
              << (net.using_cuda() ? "CUDA (LibTorch)" : "CPU (native)")
              << "\n";

    Trainer trainer(net, search, train);
    trainer.run();
    return 0;
}

int cmd_selfplay(int argc, char** argv) {
    std::string model;
    int playout = 400, games = 4, batch = 4;
    int n_threads = -1;
    bool force_cpu = false;
    for (int i = 2; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--model") { if (i+1<argc) model = argv[++i]; }
        else if (a == "--playout") { if (i+1<argc) playout = std::stoi(argv[++i]); }
        else if (a == "--games") { if (i+1<argc) games = std::stoi(argv[++i]); }
        else if (a == "--batch") { if (i+1<argc) batch = std::stoi(argv[++i]); }
        else if (a == "--threads") { if (i+1<argc) n_threads = std::stoi(argv[++i]); }
        else if (a == "--int8") {
            std::cerr << "[net] --int8 was removed with the new architecture; "
                         "ignoring\n";
        }
        else if (a == "--cpu") { force_cpu = true; }
    }
    PureNet net(2e-3);
    net.set_cuda_enabled(!force_cpu);
    if (!net.load(model)) return 1;

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
