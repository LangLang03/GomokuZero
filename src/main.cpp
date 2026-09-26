#include "game.h"
#include "network.h"
#include "selfplay.h"
#include "train.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <map>
#include <set>
#include <stdexcept>
#include <string>

using namespace gomoku;
namespace {
void usage() {
    std::cout <<
        "Usage: gomoku_train <train|selfplay|arena|human|export> [options]\n"
        "train: --init DIR --games N --batch-games 64 --playout 400\n"
        "  --mix PREFIX --mix-ratio 0.5 --mix-value-weight [0,1]\n"
        "  --pretrain EPOCHS --pretrain-warmup 20 --early-stop-patience 0\n"
        "  --val-fraction 0.1 --val-freq 5 --val-samples 2048 --infer-game-groups 0|1\n"
        "  --batch-size 512 --epochs 5 --buffer 10000 --check-freq 50 --tag NAME\n"
        "  --lr 0.002 --lr-final-fraction 0.1 --lr-warmup 2 --weight-decay 0.0001\n"
        "  --precision fp32|bf16 (default fp32)\n"
        "  --grad-clip 2 --label-smooth 0.1 --value-loss-weight 1 --augment 0|1 --kl-freq 5\n"
        "  --eval-games 0 --eval-freq 50 --eval-playout 400 --baseline DIR\n"
        "  --games is the absolute final batch index; --games 0 trains supervised only.\n"
        "  --epochs means fresh optimizer minibatches per self-play batch.\n"
        "selfplay: [--model DIR] --games 4 --batch 4 (random model if omitted)\n"
        "arena: --model DIR --baseline DIR --games 40 --playout 400 --opening-moves 4\n"
        "human: --model DIR --playout 400 (moves r,c, zero based)\n"
        "export: [--model DIR] --output DIR (random initialized weights if omitted)\n"
        "Search: --c-puct 5 --max-candidates 60 --virtual-loss 8 --inference-batch 32\n"
        "  --temp-high 1 --temp-low 0.5 --temp-switch 30\n"
        "  --dirichlet-eps 0.25 --dirichlet-alpha 0.3 --unseen-smooth 0\n"
        "Common: --threads N --seed 1 --cpu --help\n";
}
class Options {
public:
    Options(int argc, char** argv) {
        for (int i = 2; i < argc; ++i) {
            const std::string key = argv[i];
            if (key.size() < 3 || key.substr(0, 2) != "--") throw std::invalid_argument("expected option, got: " + key);
            if (values_.count(key)) throw std::invalid_argument("duplicate option: " + key);
            if (key == "--cpu" || key == "--help") values_[key] = "1";
            else {
                if (i + 1 == argc || std::string(argv[i + 1]).rfind("--", 0) == 0)
                    throw std::invalid_argument("missing value for " + key);
                values_[key] = argv[++i];
            }
        }
    }
    std::string str(const std::string& key, std::string fallback = "") {
        auto it = values_.find(key);
        if (it == values_.end()) return fallback;
        used_.insert(key);
        if (it->second.empty()) throw std::invalid_argument("empty value for " + key);
        return it->second;
    }
    int integer(const std::string& key, int fallback) {
        const auto text = str(key);
        if (text.empty()) return fallback;
        size_t end = 0;
        long long value = std::stoll(text, &end);
        if (end != text.size() || value < std::numeric_limits<int>::min() || value > std::numeric_limits<int>::max())
            throw std::invalid_argument("invalid integer for " + key);
        return static_cast<int>(value);
    }
    double real(const std::string& key, double fallback) {
        const auto text = str(key);
        if (text.empty()) return fallback;
        size_t end = 0;
        const double value = std::stod(text, &end);
        if (end != text.size() || !std::isfinite(value)) throw std::invalid_argument("invalid finite number for " + key);
        return value;
    }
    bool boolean(const std::string& key, bool fallback) {
        const int value = integer(key, fallback ? 1 : 0);
        if (value != 0 && value != 1) throw std::invalid_argument(key + " must be 0 or 1");
        return value != 0;
    }
    uint64_t seed() {
        const char* env = std::getenv("GOMOK_SEED");
        const auto text = str("--seed", env ? env : "1");
        if (text.empty() || text[0] == '-') throw std::invalid_argument("seed must be nonnegative");
        size_t end = 0;
        const uint64_t seed = std::stoull(text, &end);
        if (end != text.size()) throw std::invalid_argument("invalid seed");
        if (::setenv("GOMOK_SEED", std::to_string(seed).c_str(), 1) != 0) throw std::runtime_error("cannot set random seed");
        return seed;
    }
    void finish() const {
        for (const auto& entry : values_) if (!used_.count(entry.first))
            throw std::invalid_argument("unknown or inapplicable option: " + entry.first);
    }
private:
    std::map<std::string, std::string> values_;
    std::set<std::string> used_;
};
SearchConfig search_options(Options& o) {
    SearchConfig s;
    s.n_playout = o.integer("--playout", s.n_playout);
    s.n_threads = o.integer("--threads", s.n_threads);
    s.c_puct = o.real("--c-puct", s.c_puct);
    s.max_candidates = o.integer("--max-candidates", s.max_candidates);
    s.virtual_loss = o.real("--virtual-loss", s.virtual_loss);
    s.inference_batch_target = o.integer("--inference-batch", s.inference_batch_target);
    s.temp_high = o.real("--temp-high", s.temp_high);
    s.temp_low = o.real("--temp-low", s.temp_low);
    s.temp_switch_move = o.integer("--temp-switch", s.temp_switch_move);
    s.dirichlet_eps = o.real("--dirichlet-eps", s.dirichlet_eps);
    s.dirichlet_alpha = o.real("--dirichlet-alpha", s.dirichlet_alpha);
    s.unseen_move_smooth = o.real("--unseen-smooth", s.unseen_move_smooth);
    s.eval_opening_moves = o.integer("--opening-moves", s.eval_opening_moves);
    return s;
}
void load(PureNet& net, const std::string& path) {
    if (!path.empty() && !net.load(path)) throw std::runtime_error("model load failed: " + path);
}
void show_backend(const PureNet& net) {
    std::cerr << "[net] " << (net.using_cuda() ? "CUDA" : "native CPU inference")
              << "; parameters=" << net.parameter_count() << '\n';
}
int train(Options& o) {
    auto s = search_options(o);
    TrainConfig c;
    c.seed = o.seed();
    const bool cpu = o.boolean("--cpu", false);
    c.resume_dir = o.str("--init");
    c.mix_data = o.str("--mix");
    c.tag = o.str("--tag", c.tag);
    c.game_batch_num = o.integer("--games", c.game_batch_num);
    s.batch_games = o.integer("--batch-games", s.batch_games);
    c.buffer_size = o.integer("--buffer", c.buffer_size);
    c.batch_size = o.integer("--batch-size", c.batch_size);
    c.epochs = o.integer("--epochs", c.epochs);
    c.check_freq = o.integer("--check-freq", c.check_freq);
    c.mix_ratio = o.real("--mix-ratio", c.mix_ratio);
    c.mix_value_weight = o.real("--mix-value-weight", c.mix_value_weight);
    c.validation_fraction = o.real("--val-fraction", c.validation_fraction);
    c.validation_freq = o.integer("--val-freq", c.validation_freq);
    c.validation_samples = o.integer("--val-samples", c.validation_samples);
    c.infer_game_groups = o.boolean("--infer-game-groups", c.infer_game_groups);
    c.lr_base = o.real("--lr", c.lr_base);
    c.lr_final_fraction = o.real("--lr-final-fraction", c.lr_final_fraction);
    c.lr_warmup_batches = o.integer("--lr-warmup", c.lr_warmup_batches);
    c.weight_decay = o.real("--weight-decay", c.weight_decay);
    c.grad_clip = o.real("--grad-clip", c.grad_clip);
    c.label_smooth = o.real("--label-smooth", c.label_smooth);
    c.value_loss_weight = o.real("--value-loss-weight", c.value_loss_weight);
    c.augment = o.boolean("--augment", c.augment);
    const auto precision = o.str("--precision", "fp32");
    if (precision != "fp32" && precision != "bf16") throw std::invalid_argument("precision must be fp32 or bf16");
    c.mixed_precision = precision == "bf16";
    c.pretrain_epochs = o.integer("--pretrain", c.pretrain_epochs);
    c.pretrain_warmup_steps = o.integer("--pretrain-warmup", c.pretrain_warmup_steps);
    c.early_stop_patience = o.integer("--early-stop-patience", c.early_stop_patience);
    c.kl_freq = o.integer("--kl-freq", c.kl_freq);
    c.eval_games = o.integer("--eval-games", c.eval_games);
    c.eval_freq = o.integer("--eval-freq", c.eval_freq);
    c.eval_playout = o.integer("--eval-playout", c.eval_playout);
    c.baseline_dir = o.str("--baseline");
    o.finish();
    validate_train_config(c, s);
    if (cpu) throw std::invalid_argument("--cpu supports inference only; train requires CUDA");
    PureNet net;
    load(net, c.resume_dir);
    show_backend(net);
    Trainer trainer(net, s, c);
    trainer.run();
    return 0;
}
int selfplay(Options& o) {
    auto s = search_options(o);
    o.seed();
    const bool cpu = o.boolean("--cpu", false);
    const auto model = o.str("--model");
    const int games = o.integer("--games", 4);
    s.batch_games = o.integer("--batch", 4);
    o.finish();
    validate_train_config(TrainConfig{}, s);
    if (games <= 0) throw std::invalid_argument("games must be positive");
    PureNet net;
    net.set_cuda_enabled(!cpu);
    load(net, model);
    show_backend(net);
    const auto start = std::chrono::steady_clock::now();
    int played = 0, moves = 0;
    uint64_t batch_index = 0;
    while (played < games) {
        auto batch_config = s;
        batch_config.batch_games = std::min(s.batch_games, games - played);
        BatchedSelfPlay sp(net, batch_config);
        sp.set_batch_counter(batch_index++);
        const auto result = sp.run_batch();
        for (const auto& game : result) { ++played; moves += game.size(); }
        const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        std::cout << "played " << played << '/' << games << " seconds=" << seconds
                  << " seconds/game=" << seconds / played << " moves/game=" << static_cast<double>(moves) / played << '\n';
    }
    return 0;
}
int arena(Options& o) {
    auto s = search_options(o);
    o.seed();
    const bool cpu = o.boolean("--cpu", false);
    const auto model = o.str("--model"), baseline = o.str("--baseline");
    const int games = o.integer("--games", 40);
    o.finish();
    validate_train_config(TrainConfig{}, s);
    if (model.empty() || baseline.empty() || games <= 0 || games % 2)
        throw std::invalid_argument("arena needs --model, --baseline and a positive even --games count");
    PureNet candidate, opponent;
    candidate.set_cuda_enabled(!cpu); opponent.set_cuda_enabled(!cpu);
    load(candidate, model); load(opponent, baseline);
    const auto r = play_match(candidate, opponent, s, games);
    std::cout << "arena " << r.wins << "W " << r.losses << "L " << r.draws << "D score=" << r.score()
              << " first_player_win_rate=" << r.first_player_win_rate() << " games=" << r.games
              << " pairs=" << r.pair_wins << "W/" << r.pair_losses << "L/" << r.pair_draws << "D\n";
    return 0;
}
int human(Options& o) {
    auto s = search_options(o);
    o.seed();
    const bool cpu = o.boolean("--cpu", false);
    const auto model = o.str("--model");
    o.finish();
    validate_train_config(TrainConfig{}, s);
    if (model.empty()) throw std::invalid_argument("human requires --model");
    PureNet net;
    net.set_cuda_enabled(!cpu); load(net, model);
    Board board;
    std::cout << "Enter r,c (0..14); AI plays white.\n";
    while (!board.game_over()) {
        std::cout << "your move: " << std::flush;
        std::string line;
        if (!std::getline(std::cin, line)) break;
        int r, c; char extra;
        if (std::sscanf(line.c_str(), "%d,%d %c", &r, &c, &extra) != 2 ||
            r < 0 || r >= BOARD_SIZE || c < 0 || c >= BOARD_SIZE || !board.is_empty(r * BOARD_SIZE + c)) {
            std::cout << "invalid move\n"; continue;
        }
        board.play(r * BOARD_SIZE + c);
        if (board.game_over()) break;
        const int move = search_best_move(net, board, s);
        if (move < 0) throw std::runtime_error("search returned no legal move");
        board.play(move);
        std::cout << "AI plays " << move / BOARD_SIZE << ',' << move % BOARD_SIZE << '\n';
    }
    if (board.game_over()) std::cout << (board.winner() == 1 ? "you win\n" : board.winner() == 2 ? "AI wins\n" : "draw\n");
    return 0;
}
int export_model(Options& o) {
    o.seed();
    const bool cpu = o.boolean("--cpu", false);
    const auto model = o.str("--model"), output = o.str("--output");
    o.finish();
    if (output.empty()) throw std::invalid_argument("export requires --output DIR");
    PureNet net;
    net.set_cuda_enabled(!cpu); load(net, model);
    if (!net.export_inference(output)) throw std::runtime_error("cannot export model: " + output);
    return 0;
}
}  // namespace
int main(int argc, char** argv) {
    try {
        if (argc < 2 || std::string(argv[1]) == "--help") { usage(); return 0; }
        Options options(argc, argv);
        if (options.boolean("--help", false)) { usage(); return 0; }
        const std::string command = argv[1];
        if (command == "train") return train(options);
        if (command == "selfplay") return selfplay(options);
        if (command == "arena") return arena(options);
        if (command == "human") return human(options);
        if (command == "export") return export_model(options);
        throw std::invalid_argument("unknown command: " + command);
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << '\n';
        return 2;
    }
}
