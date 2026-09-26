// Read-only value generalization probe; JSON stdout, progress stderr.
// Usage: gomoku_value_evaluation MODEL [games=16] [playout=64] [seed=20260927]
#include "dataset.h"
#include "network.h"
#include "selfplay.h"
#include <algorithm>
#include <array>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

namespace {
using namespace gomoku;
namespace fs = std::filesystem;
constexpr size_t kStateFloats = 4 * BOARD_CELLS;
constexpr size_t kRecordFloats = kStateFloats + BOARD_CELLS + 1;
constexpr int kInferenceChunk = 32;
static_assert(kRecordFloats == 1126, "replay format changed");

template<class T> T number(const char* text, const char* name) {
    T result{};
    const std::string value(text);
    const auto parsed = std::from_chars(value.data(), value.data() + value.size(), result);
    if (value.empty() || parsed.ec != std::errc{} || parsed.ptr != value.data() + value.size())
        throw std::invalid_argument(std::string("invalid ") + name + ": " + value);
    return result;
}
std::string json_string(const std::string& value) {
    std::string result = "\"";
    constexpr char hex[] = "0123456789abcdef";
    for (unsigned char c : value) {
        if (c == '"' || c == '\\') { result += '\\'; result += c; }
        else if (c < 0x20) {
            result += "\\u00"; result += hex[c >> 4]; result += hex[c & 15];
        } else result += c;
    }
    return result + '"';
}

struct ReplayReference {
    bool present = false;
    size_t records = 0;
    std::unordered_set<uint64_t> states;
    std::optional<uint64_t> training_seed;
};
ReplayReference read_replay(const fs::path& directory) {
    ReplayReference reference;
    const auto path = directory / "trainer.replay";
    reference.present = fs::exists(path);
    if (reference.present) {
        const auto bytes = fs::file_size(path);
        constexpr size_t record_bytes = kRecordFloats * sizeof(float);
        if (bytes % record_bytes || bytes / record_bytes > std::numeric_limits<size_t>::max())
            throw std::runtime_error("invalid trainer.replay size");
        reference.records = static_cast<size_t>(bytes / record_bytes);
        reference.states.reserve(reference.records);
        std::ifstream input(path, std::ios::binary);
        if (!input) throw std::runtime_error("cannot open trainer.replay");
        std::array<float, kRecordFloats> record{};
        for (size_t i = 0; i < reference.records; ++i) {
            input.read(reinterpret_cast<char*>(record.data()), record_bytes);
            if (!input) throw std::runtime_error("truncated trainer.replay record");
            validate_training_record(record.data(), record.data() + kStateFloats,
                                     record.back(), i);
            reference.states.insert(canonical_state_hash(record.data()));
        }
    }
    std::ifstream metadata(directory / "trainer.meta");
    if (metadata) {
        std::string magic;
        int version = 0, batch = 0, completed = 0, horizon = 0, pretrain_horizon = 0;
        uint64_t seed = 0;
        if (!(metadata >> magic >> version >> batch >> completed >> horizon >> pretrain_horizon >> seed) ||
            magic != "GOMOKU_TRAINER" || (version != 1 && version != 2))
            throw std::runtime_error("unsupported or malformed trainer.meta");
        reference.training_seed = seed;
    }
    return reference;
}

struct Observation {
    double predicted = 0, outcome = 0;
    bool black = true;
};
struct Game {
    size_t generated_samples = 0, purged = 0;
    int winner = 0;
    std::vector<Observation> observations;
};
struct Metrics {
    double model_mse = 0, model_mae = 0, zero_mse = 0;
    double global_constant_mse = 0, colour_constant_mse = 0;
    double mean_prediction = 0, mean_outcome = 0;
    void add(const Metrics& m) {
        model_mse += m.model_mse; model_mae += m.model_mae; zero_mse += m.zero_mse;
        global_constant_mse += m.global_constant_mse; colour_constant_mse += m.colour_constant_mse;
        mean_prediction += m.mean_prediction; mean_outcome += m.mean_outcome;
    }
    void divide(double n) {
        model_mse /= n; model_mae /= n; zero_mse /= n;
        global_constant_mse /= n; colour_constant_mse /= n;
        mean_prediction /= n; mean_outcome /= n;
    }
};
Metrics measure(const Game& game, double global_constant,
                const std::array<double, 2>& colour_constant) {
    Metrics result;
    for (const auto& point : game.observations) {
        const double error = point.predicted - point.outcome;
        result.model_mse += error * error;
        result.model_mae += std::abs(error);
        result.zero_mse += point.outcome * point.outcome;
        const double global_error = global_constant - point.outcome;
        const double colour_error = colour_constant[point.black ? 0 : 1] - point.outcome;
        result.global_constant_mse += global_error * global_error;
        result.colour_constant_mse += colour_error * colour_error;
        result.mean_prediction += point.predicted;
        result.mean_outcome += point.outcome;
    }
    if (!game.observations.empty()) result.divide(game.observations.size());
    return result;
}
void print_metrics(const Metrics& m) {
    std::cout << "{\"model_mse\":" << m.model_mse << ",\"model_mae\":" << m.model_mae
              << ",\"zero_baseline_mse\":" << m.zero_mse
              << ",\"optimistic_global_constant_mse\":" << m.global_constant_mse
              << ",\"optimistic_colour_constant_mse\":" << m.colour_constant_mse
              << ",\"mean_prediction\":" << m.mean_prediction
              << ",\"mean_outcome\":" << m.mean_outcome << '}';
}
}

int main(int argc, char** argv) {
    try {
        if (argc < 2 || argc > 5 || std::string(argv[1]) == "--help") {
            std::cerr << "usage: " << argv[0] << " MODEL [games=16] [playout=64] [seed=20260927]\n";
            return argc == 2 && std::string(argv[1]) == "--help" ? 0 : 2;
        }
        const fs::path model = argv[1];
        const int requested_games = argc > 2 ? number<int>(argv[2], "games") : 16;
        const int playout = argc > 3 ? number<int>(argv[3], "playout") : 64;
        const uint64_t seed = argc > 4 ? number<uint64_t>(argv[4], "seed") : 20260927;
        if (requested_games <= 0 || playout <= 0)
            throw std::invalid_argument("games and playout must be positive");
        SearchConfig config;
        config.n_playout = playout;
        config.batch_games = std::min(requested_games, 16);
        validate_search_config(config);
        const auto replay = read_replay(model);
        if (replay.training_seed && static_cast<uint32_t>(*replay.training_seed) == static_cast<uint32_t>(seed))
            throw std::invalid_argument("evaluation seed must differ from saved training seed in its low 32 bits");
        if (::setenv("GOMOK_SEED", std::to_string(seed).c_str(), 1) != 0)
            throw std::runtime_error("cannot set evaluation seed");
        PureNet net;
        if (!net.load(model.string())) throw std::runtime_error("cannot load evaluation model");
        std::cerr << "[value-eval] replay=" << (replay.present ? "present" : "absent")
                  << " records=" << replay.records << " unique_D4_states=" << replay.states.size() << '\n';

        const auto started = std::chrono::steady_clock::now();
        std::vector<Game> games;
        games.reserve(static_cast<size_t>(requested_games));
        size_t generated_samples = 0, purged_samples = 0, kept_samples = 0;
        int black_wins = 0, white_wins = 0, draws = 0;
        uint64_t batch_index = 0;
        std::vector<float> states(kInferenceChunk * kStateFloats);
        std::vector<float> policies(kInferenceChunk * BOARD_CELLS), values(kInferenceChunk);
        while (games.size() < static_cast<size_t>(requested_games)) {
            auto batch_config = config;
            batch_config.batch_games = std::min<int>(config.batch_games,
                requested_games - static_cast<int>(games.size()));
            BatchedSelfPlay selfplay(net, batch_config);
            selfplay.set_batch_counter(batch_index++);
            const auto generated = selfplay.run_batch();
            if (generated.size() != static_cast<size_t>(batch_config.batch_games))
                throw std::runtime_error("self-play returned the wrong game count");
            for (const auto& samples : generated) {
                if (samples.empty()) throw std::runtime_error("self-play returned an empty game");
                Game game;
                game.generated_samples = samples.size();
                const float black_outcome = samples.front().z;
                if (samples.front().state[3 * BOARD_CELLS] != 1.0f ||
                    (black_outcome != -1.0f && black_outcome != 0.0f && black_outcome != 1.0f))
                    throw std::runtime_error("self-play lacks a valid black opening outcome");
                game.winner = black_outcome > 0 ? 1 : black_outcome < 0 ? 2 : 0;
                if (game.winner == 1) ++black_wins;
                else if (game.winner == 2) ++white_wins;
                else ++draws;
                std::vector<const BatchedSelfPlay::Sample*> kept;
                kept.reserve(samples.size());
                for (const auto& sample : samples) {
                    validate_training_record(sample.state.data(), sample.probs.data(), sample.z);
                    const bool black = sample.state[3 * BOARD_CELLS] != 0.0f;
                    if (sample.z != (black ? black_outcome : -black_outcome))
                        throw std::runtime_error("self-play has inconsistent outcome signs");
                    if (replay.states.count(canonical_state_hash(sample.state.data()))) ++game.purged;
                    else kept.push_back(&sample);
                }
                for (size_t offset = 0; offset < kept.size(); offset += kInferenceChunk) {
                    const int count = static_cast<int>(std::min<size_t>(kInferenceChunk, kept.size() - offset));
                    for (int i = 0; i < count; ++i)
                        std::copy(kept[offset + i]->state.begin(), kept[offset + i]->state.end(),
                                  states.begin() + static_cast<size_t>(i) * kStateFloats);
                    net.forward_batch(states.data(), count, policies.data(), values.data());
                    for (int i = 0; i < count; ++i) {
                        if (!std::isfinite(values[i]) || std::abs(values[i]) > 1.0001f)
                            throw std::runtime_error("model returned an invalid value");
                        const auto& sample = *kept[offset + i];
                        game.observations.push_back({values[i], sample.z,
                                                     sample.state[3 * BOARD_CELLS] != 0.0f});
                    }
                }
                generated_samples += samples.size();
                purged_samples += game.purged;
                kept_samples += game.observations.size();
                games.push_back(std::move(game));
            }
            std::cerr << "[value-eval] complete_games=" << games.size() << '/' << requested_games
                      << " generated_samples=" << generated_samples << " retained=" << kept_samples
                      << " purged=" << purged_samples << '\n';
        }

        // Fit diagnostic constants on evaluation labels using the same
        // equal-game weights as the primary macro MSE objective.
        size_t evaluated_games = 0;
        double weighted_outcome = 0;
        std::array<double, 2> colour_outcome{}, colour_weight{};
        for (const auto& game : games) {
            if (game.observations.empty()) continue;
            ++evaluated_games;
            const double weight = 1.0 / game.observations.size();
            for (const auto& point : game.observations) {
                weighted_outcome += weight * point.outcome;
                const int colour = point.black ? 0 : 1;
                colour_outcome[colour] += weight * point.outcome;
                colour_weight[colour] += weight;
            }
        }
        const double global_constant = evaluated_games ? weighted_outcome / evaluated_games : 0;
        std::array<double, 2> colour_constant{};
        for (int colour = 0; colour < 2; ++colour)
            if (colour_weight[colour] > 0)
                colour_constant[colour] = colour_outcome[colour] / colour_weight[colour];
        Metrics macro;
        std::vector<Metrics> per_game;
        per_game.reserve(games.size());
        for (const auto& game : games) {
            const auto m = measure(game, global_constant, colour_constant);
            per_game.push_back(m);
            if (!game.observations.empty()) macro.add(m);
        }
        if (evaluated_games) macro.divide(evaluated_games);
        const double seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - started).count();
        const char* tf32 = std::getenv("GOMOK_INFERENCE_TF32");
        std::cout << std::setprecision(12)
                  << "{\n  \"format_version\":1,\n  \"model\":" << json_string(model.string())
                  << ",\n  \"seed\":" << seed << ",\n  \"training_seed\":";
        if (replay.training_seed) std::cout << *replay.training_seed; else std::cout << "null";
        std::cout << ",\n  \"backend\":" << json_string(net.using_cuda() ? "cuda" : "native_cpu")
                  << ",\n  \"inference_tf32_setting\":" << json_string(tf32 ? tf32 : "1")
                  << ",\n  \"playout\":" << playout << ",\n  \"selfplay_batch_games\":" << config.batch_games
                  << ",\n  \"inference_chunk\":" << kInferenceChunk
                  << ",\n  \"games\":" << games.size() << ",\n  \"evaluated_games\":" << evaluated_games
                  << ",\n  \"games_without_retained_samples\":" << games.size() - evaluated_games
                  << ",\n  \"generated_samples\":" << generated_samples << ",\n  \"evaluated_samples\":" << kept_samples
                  << ",\n  \"purged_replay_D4_overlap\":" << purged_samples
                  << ",\n  \"replay_available\":" << (replay.present ? "true" : "false")
                  << ",\n  \"replay_records\":" << replay.records << ",\n  \"replay_unique_D4_states\":" << replay.states.size()
                  << ",\n  \"overlap_scope\":" << json_string(replay.present
                      ? "Purged overlap with saved trainer.replay only; evicted or unrecorded training positions remain unknown."
                      : "No trainer.replay: training-state overlap could not be removed.")
                  << ",\n  \"black_wins\":" << black_wins << ",\n  \"white_wins\":" << white_wins
                  << ",\n  \"draws\":" << draws
                  << ",\n  \"black_win_fraction_all_games\":" << static_cast<double>(black_wins) / games.size()
                  << ",\n  \"white_win_fraction_all_games\":" << static_cast<double>(white_wins) / games.size()
                  << ",\n  \"draw_fraction\":" << static_cast<double>(draws) / games.size()
                  << ",\n  \"first_player_win_rate_decided\":";
        if (black_wins + white_wins) std::cout << static_cast<double>(black_wins) / (black_wins + white_wins);
        else std::cout << "null";
        std::cout << ",\n  \"baseline_fit\":" << json_string(
            "Optimistic diagnostics fitted on evaluation labels with equal game weights; not independently fitted predictors.")
                  << ",\n  \"global_constant\":";
        if (evaluated_games) std::cout << global_constant; else std::cout << "null";
        std::cout << ",\n  \"black_to_move_constant\":";
        if (colour_weight[0] > 0) std::cout << colour_constant[0]; else std::cout << "null";
        std::cout << ",\n  \"white_to_move_constant\":";
        if (colour_weight[1] > 0) std::cout << colour_constant[1]; else std::cout << "null";
        std::cout << ",\n  \"macro_game_metrics\":";
        if (evaluated_games) print_metrics(macro); else std::cout << "null";
        std::cout << ",\n  \"seconds_including_selfplay\":" << seconds
                  << ",\n  \"interpretation\":" << json_string(
            "One model's on-policy outcomes, not objective game-theoretic values or strength evidence. Positions within a game are correlated; no independent-sample significance claim is made.")
                  << ",\n  \"per_game\":[\n";
        for (size_t i = 0; i < games.size(); ++i) {
            if (i) std::cout << ",\n";
            const auto& game = games[i];
            std::cout << "    {\"game\":" << i + 1 << ",\"generated_samples\":" << game.generated_samples
                      << ",\"purged\":" << game.purged << ",\"evaluated_samples\":" << game.observations.size()
                      << ",\"winner\":" << game.winner << ",\"metrics\":";
            if (game.observations.empty()) std::cout << "null"; else print_metrics(per_game[i]);
            std::cout << '}';
        }
        std::cout << "\n  ]\n}\n";
        return evaluated_games ? 0 : 3;
    } catch (const std::exception& error) {
        std::cerr << "value_evaluation: " << error.what() << '\n';
        return 2;
    }
}
