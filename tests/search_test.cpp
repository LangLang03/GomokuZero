#include "selfplay.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#include <limits>
#include <stdexcept>

using namespace gomoku;

namespace {
int failures = 0;
void check(bool okay, const char* message) {
    if (!okay) {
        std::printf("FAIL: %s\n", message);
        ++failures;
    }
}
struct Environment {
    std::string name, previous;
    bool existed;
    Environment(const char* key, const char* value) : name(key) {
        const char* old = std::getenv(key);
        existed = old != nullptr;
        if (old) previous = old;
        setenv(key, value, 1);
    }
    ~Environment() {
        if (existed) setenv(name.c_str(), previous.c_str(), 1);
        else unsetenv(name.c_str());
    }
};

BatchPositionEvaluator uniform_batch() {
    return [](const float*, int count, float* policies, float* values) {
        for (int i = 0; i < count; ++i) {
            values[i] = 0.0f;
            for (int move = 0; move < BOARD_CELLS; ++move)
                policies[i * BOARD_CELLS + move] = 1.0f / BOARD_CELLS;
        }
    };
}

bool same_games(const std::vector<std::vector<BatchedSelfPlay::Sample>>& a,
                const std::vector<std::vector<BatchedSelfPlay::Sample>>& b) {
    if (a.size() != b.size()) return false;
    for (size_t game = 0; game < a.size(); ++game) {
        if (a[game].size() != b[game].size()) return false;
        for (size_t ply = 0; ply < a[game].size(); ++ply)
            if (a[game][ply].state != b[game][ply].state ||
                a[game][ply].probs != b[game][ply].probs ||
                a[game][ply].z != b[game][ply].z) return false;
    }
    return true;
}

void validate_games(const std::vector<std::vector<BatchedSelfPlay::Sample>>& games) {
    for (const auto& game : games) {
        check(!game.empty() && game.size() <= BOARD_CELLS, "self-play completes bounded games");
        for (size_t ply = 0; ply < game.size(); ++ply) {
            const auto& sample = game[ply];
            double mass = 0.0;
            for (int move = 0; move < BOARD_CELLS; ++move) {
                const float p = sample.probs[move];
                check(std::isfinite(p) && p >= 0.0f, "concurrent target stays finite");
                check(p == 0.0f || (sample.state[move] == 0.0f &&
                      sample.state[BOARD_CELLS + move] == 0.0f),
                      "concurrent target assigns mass only to legal moves");
                mass += p;
            }
            check(std::abs(mass - 1.0) < 1e-5, "concurrent target normalizes");
            check(sample.z == 0.0f || sample.z == 1.0f || sample.z == -1.0f,
                  "self-play result is a terminal value");
            if (ply + 1 < game.size())
                check(sample.z == -game[ply + 1].z, "trajectory alternates value perspective");
        }
    }
}

PositionEvaluator uniform(float value = 0.0f) {
    return [value](const Board&, std::array<float, BOARD_CELLS>& policy, float& result) {
        policy.fill(1.0f / BOARD_CELLS);
        result = value;
    };
}
}

int main() {
    SearchConfig config;
    config.n_playout = 32;
    config.max_candidates = 4;
    config.dirichlet_eps = 0.0f;
    config.temp_high = config.temp_low = 0.0f;
    {
        Board board;
        const auto result = search_position(board, config, uniform());
        double sum = 0.0;
        int supported = 0;
        for (float p : result.policy) {
            sum += p;
            supported += p > 0.0f;
            check(std::isfinite(p) && p >= 0.0f, "finite search target");
        }
        check(result.completed_playouts == config.n_playout, "exact simulation budget");
        check(result.evaluations == config.n_playout, "quiet search evaluates distinct leaves");
        check(std::abs(sum - 1.0) < 1e-6, "visit target is normalized");
        check(supported == config.max_candidates,
              "greedy move temperature preserves multiple visited training actions");
        config.temp_high = config.temp_low = 2.0f;
        const auto warm = search_position(board, config, uniform());
        check(warm.policy == result.policy, "move temperature does not distort visit targets");
        check(board.move_count() == 0, "search does not modify caller board");
        const PositionEvaluator asymmetric = [](const Board& leaf,
                std::array<float, BOARD_CELLS>& policy, float& value) {
            policy.fill(0.0f);
            policy[112] = 0.95f;
            policy[113] = 0.05f;
            const float root_value = leaf.move_count() > 0 &&
                leaf.move_at(0) == 113 ? 1.0f : 0.0f;
            value = leaf.current_player() == 1 ? root_value : -root_value;
        };
        SearchConfig huge_loss = config;
        huge_loss.virtual_loss = 1024.0f;
        const auto ordinary = search_position(board, config, asymmetric);
        const auto no_collisions = search_position(board, huge_loss, asymmetric);
        check(no_collisions.policy == ordinary.policy,
              "serial search does not add phantom visits from virtual loss");
    }
    {
        // A winning threat is exact, even if the value network claims defeat.
        Board win;
        for (int move : {0, 30, 1, 32, 2, 34, 3, 36}) win.play(move);
        int evaluations = 0;
        const auto result = search_position(win, config,
            [&](const Board&, std::array<float, BOARD_CELLS>&, float&) { ++evaluations; });
        check(result.move == 4 && result.policy[4] == 1.0f, "immediate win is exact");
        check(evaluations == 0 && result.evaluations == 0,
              "proven wins do not spend neural inference");
        check(result.completed_playouts == config.n_playout,
              "solved root preserves simulation accounting");
        win.play(4);
        const auto terminal = search_position(win, config, uniform());
        check(terminal.move == -1 && terminal.completed_playouts == 0 &&
              terminal.evaluations == 0, "terminal board needs no search");
    }
    {
        // White threatens an edge four; black must block despite hostile priors.
        Board block;
        for (int move : {30, 0, 32, 1, 34, 2, 36, 3}) block.play(move);
        SearchConfig narrow = config;
        narrow.n_playout = 1;
        narrow.max_candidates = 1;
        const auto result = search_position(block, narrow, uniform());
        check(result.move == 4 && result.policy[4] == 1.0f,
              "forced block survives a one-node budget and candidate cap");
    }
    {
        // Exact tactical values discovered two plies below root must back up
        // with the right sign, even when every unsolved value is misleading.
        Board board;
        for (int move : {106, 0, 107, 2, 108, 4}) board.play(move);
        SearchConfig tactical = config;
        tactical.n_playout = 160;
        tactical.max_candidates = 2;
        tactical.c_puct = 1.5f;
        const auto result = search_position(board, tactical,
            [&](const Board& leaf, std::array<float, BOARD_CELLS>& policy, float& value) {
                policy.fill(0.0f);
                policy[109] = 0.5f;  // Creates an open four: either white block loses.
                policy[200] = 0.5f;
                value = leaf.current_player() == board.current_player() ? -0.25f : 0.25f;
            });
        check(result.move == 109, "search finds forced win against misleading leaf values");
        check(result.evaluations < tactical.n_playout,
              "tactical proofs replace repeated inference with exact backups");
    }
    {
        SearchConfig one = config;
        one.n_playout = 1;
        const auto result = search_position(Board{}, one,
            [](const Board&, std::array<float, BOARD_CELLS>& p, float& v) {
                p.fill(0.0f);
                p[112] = 0.75f;
                p[113] = 0.25f;
                v = 0.0f;
            });
        check(result.move == 112 && result.policy[112] == 0.75f &&
              result.policy[113] == 0.25f,
              "one-playout search falls back to legal prior without inventing visits");
    }
    {
        bool rejected = false;
        try { search_position(Board{}, config, uniform(std::numeric_limits<float>::quiet_NaN())); }
        catch (const std::runtime_error&) { rejected = true; }
        check(rejected, "invalid evaluator values fail explicitly");
    }
    {
        // An internal node's old top-1 is not the complete legal move set
        // after it becomes the root. Keep its visited child while fresh root
        // exploration can visit an action outside that former shortlist.
        Environment seed("GOMOK_SEED", "1907");
        Environment threads("GOMOK_MCTS_THREADS", "1");
        Environment leaves("GOMOK_LEAF_BATCH", "1");
        SearchConfig reuse;
        reuse.n_playout = 8;
        reuse.batch_games = 1;
        reuse.n_threads = 1;
        reuse.max_candidates = 1;
        reuse.c_puct = 100.0f;
        reuse.dirichlet_eps = 0.25f;
        BatchedSelfPlay search(reuse, uniform_batch());
        const auto games = search.run_batch(0.0f);
        validate_games(games);
        check(games.size() == 1 && games.front().size() >= 2,
              "subtree promotion reaches a second move");
        if (games.size() == 1 && games.front().size() >= 2) {
            int visited = 0;
            for (float p : games.front()[1].probs) visited += p > 0.0f;
            check(visited > 1,
                  "promoted root preserves visits and admits moves outside the old top-k");
        }
    }
    {
        Environment seed("GOMOK_SEED", "7391");
        Environment threads("GOMOK_MCTS_THREADS", "1");
        Environment leaves("GOMOK_LEAF_BATCH", "1");
        SearchConfig self;
        self.n_playout = 8;
        self.batch_games = 2;
        self.max_candidates = 4;
        self.n_threads = 1;
        self.dirichlet_eps = 0.25f;
        BatchedSelfPlay original(self, uniform_batch());
        const auto first = original.run_batch(0.0f);
        const auto second = original.run_batch(0.0f);
        validate_games(first);
        validate_games(second);
        BatchedSelfPlay resumed(self, uniform_batch());
        resumed.set_batch_counter(1);
        check(same_games(second, resumed.run_batch(0.0f)),
              "seeded resume continues at the next batch instead of duplicating old data");
        check(!same_games(first, second), "different batches produce distinct seeded games");
        int support = 0;
        for (float p : first.front().front().probs) support += p > 0.0f;
        check(support > 1, "actual greedy self-play keeps the full visit training target");
    }
    {
        Environment seed("GOMOK_SEED", "7391");
        Environment threads("GOMOK_MCTS_THREADS", "4");
        Environment leaves("GOMOK_LEAF_BATCH", "4");
        SearchConfig concurrent;
        concurrent.n_playout = 16;
        concurrent.batch_games = 3;
        concurrent.max_candidates = 4;
        concurrent.n_threads = 4;
        concurrent.unseen_move_smooth = 0.05f;
        BatchedSelfPlay search(concurrent, uniform_batch());
        const auto games = search.run_batch(0.5f);
        check(games.size() == 3, "concurrent queue accounts for every game");
        validate_games(games);
    }
    {
        SearchConfig failure;
        failure.n_playout = 8;
        failure.batch_games = 2;
        failure.max_candidates = 4;
        failure.n_threads = 2;
        bool rejected = false;
        try {
            BatchedSelfPlay search(failure,
                [](const float*, int, float*, float*) {
                    throw std::runtime_error("injected inference failure");
                });
            search.run_batch();
        } catch (const std::runtime_error&) { rejected = true; }
        check(rejected, "inference exceptions cancel and join search workers");
        rejected = false;
        try {
            BatchedSelfPlay search(failure,
                [](const float*, int count, float* policies, float* values) {
                    for (int i = 0; i < count; ++i) {
                        values[i] = std::numeric_limits<float>::quiet_NaN();
                        for (int move = 0; move < BOARD_CELLS; ++move)
                            policies[i * BOARD_CELLS + move] = 1.0f / BOARD_CELLS;
                    }
                });
            search.run_batch();
        } catch (const std::runtime_error&) { rejected = true; }
        check(rejected, "worker exceptions reach the caller without stranded retry loops");
    }
    {
        SearchConfig arena;
        arena.n_playout = 1;
        arena.max_candidates = 4;
        arena.eval_opening_moves = 4;
        const auto equal = play_match(uniform_batch(), uniform_batch(), arena, 34);
        check(equal.games == 34 && equal.wins == equal.losses,
              "identical evaluators tie under swapped-colour paired openings");
        check(equal.pair_wins == 0 && equal.pair_losses == 0 && equal.pair_draws == 17,
              "pair statistics treat opposite-colour wins as one tied opening");
        const BatchPositionEvaluator connect_center = [](const float*, int count,
                float* policies, float* values) {
            for (int i = 0; i < count; ++i) {
                values[i] = 0.0f;
                float* policy = policies + i * BOARD_CELLS;
                std::fill(policy, policy + BOARD_CELLS, 0.001f);
                for (int move = 105; move < 120; ++move)
                    policy[move] = 15.0f - std::abs(move - 112);
            }
        };
        arena.eval_opening_moves = 0;
        const auto winning = play_match(connect_center, uniform_batch(), arena, 4);
        const auto losing = play_match(uniform_batch(), connect_center, arena, 4);
        check(winning.wins == 4 && winning.pair_wins == 2 &&
              winning.pair_losses == 0 && winning.pair_draws == 0,
              "pair wins aggregate the candidate result in both colours");
        check(losing.losses == 4 && losing.pair_losses == 2 &&
              losing.pair_wins == 0 && losing.pair_draws == 0,
              "pair losses aggregate the opponent result in both colours");
        const auto incomplete = play_match(connect_center, uniform_batch(), arena, 3);
        check(incomplete.games == 3 && incomplete.pair_wins == 1,
              "the odd final game is counted only in individual results");
    }
    if (failures == 0) std::puts("search_test passed");
    return failures == 0 ? 0 : 1;
}
