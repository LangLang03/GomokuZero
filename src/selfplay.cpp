#include "selfplay.h"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <fstream>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <random>
#include <stdexcept>
#include <thread>
#include <unordered_map>
#ifdef __linux__
#include <pthread.h>
#include <sched.h>
#endif

namespace gomoku {
namespace {

constexpr int MAX_CAND = 40;
constexpr int NEAR_RADIUS = 2;
constexpr int64_t VALUE_SCALE = 1LL << 20;

enum NodeState : uint8_t {
    NODE_UNEXPANDED = 0,
    NODE_EXPANDING = 1,
    NODE_EXPANDED = 2,
};

// state publishes the immutable child range. Search counters are atomic.
struct alignas(32) GNode {
    std::atomic<int64_t> value_sum{0};
    std::atomic<int32_t> visits{0};
    std::atomic<int32_t> virtual_visits{0};
    std::atomic<uint8_t> state{NODE_UNEXPANDED};
    uint32_t child_begin = 0;
    uint8_t child_count = 0;
    int16_t action = -1;
    float prior = 0.0f;
};

static_assert(std::atomic<int64_t>::is_always_lock_free,
              "lock-free MCTS requires lock-free 64-bit atomics");

struct GameState {
    Board board;
    std::unique_ptr<GNode[]> nodes;
    size_t node_capacity = 0;
    std::atomic<uint32_t> next_node{1};
    std::atomic<int> issued{0};
    bool finished = false;
    int winner = 0;
    std::vector<std::array<float, 4 * BOARD_CELLS>> states;
    std::vector<std::array<float, BOARD_CELLS>> probs;
    std::vector<int> players;
};

struct PathRec {
    std::array<int32_t, BOARD_CELLS + 1> nodes{};
    int len = 0;
};

struct CachedEvaluation {
    std::array<float, BOARD_CELLS> policy{};
    float value = 0.0f;
};

struct LeafTask {
    int game = -1;
    int node = -1;
    int eval = -1;
    uint64_t key = 0;
    PathRec path;
    Board board;
    CachedEvaluation evaluation;
};

void initialize_node(GNode& node, int action = -1, float prior = 0.0f) {
    node.value_sum.store(0, std::memory_order_relaxed);
    node.visits.store(0, std::memory_order_relaxed);
    node.virtual_visits.store(0, std::memory_order_relaxed);
    node.child_begin = 0;
    node.child_count = 0;
    node.action = static_cast<int16_t>(action);
    node.prior = prior;
    node.state.store(NODE_UNEXPANDED, std::memory_order_relaxed);
}

void reset_tree(GameState& game) {
    game.next_node.store(1, std::memory_order_relaxed);
    game.issued.store(0, std::memory_order_relaxed);
    initialize_node(game.nodes[0]);
}

void release_virtual_loss(GameState& game, const PathRec& path) {
    for (int i = 0; i < path.len; ++i)
        game.nodes[path.nodes[i]].virtual_visits.fetch_sub(
            1, std::memory_order_relaxed);
}

void backup_path(GameState& game, const PathRec& path, float leaf_value) {
    int64_t value = static_cast<int64_t>(std::llround(
        static_cast<double>(leaf_value) * VALUE_SCALE));
    for (int i = path.len - 1; i >= 0; --i) {
        GNode& node = game.nodes[path.nodes[i]];
        node.value_sum.fetch_add(value, std::memory_order_relaxed);
        node.visits.fetch_add(1, std::memory_order_relaxed);
        node.virtual_visits.fetch_sub(1, std::memory_order_relaxed);
        value = -value;
    }
}

void nearby_moves(const Board& board, std::vector<int>& out) {
    out.clear();
    bool seen[BOARD_CELLS] = {};
    for (int ply = 0; ply < board.move_count(); ++ply) {
        const int move = board.move_at(ply);
        const int row = move / BOARD_SIZE;
        const int col = move % BOARD_SIZE;
        const int r0 = std::max(0, row - NEAR_RADIUS);
        const int r1 = std::min(BOARD_SIZE - 1, row + NEAR_RADIUS);
        const int c0 = std::max(0, col - NEAR_RADIUS);
        const int c1 = std::min(BOARD_SIZE - 1, col + NEAR_RADIUS);
        for (int r = r0; r <= r1; ++r) {
            for (int c = c0; c <= c1; ++c) {
                const int idx = r * BOARD_SIZE + c;
                if (!seen[idx] && board.is_empty(idx)) {
                    seen[idx] = true;
                    out.push_back(idx);
                }
            }
        }
    }
    if (board.move_count() == 0)
        out.push_back((BOARD_SIZE / 2) * BOARD_SIZE + BOARD_SIZE / 2);
}

uint64_t board_key(const Board& board) {
    return board.position_key();
}

#ifdef __linux__
std::vector<int> search_cpus(int inference_threads) {
    cpu_set_t allowed;
    CPU_ZERO(&allowed);
    if (sched_getaffinity(0, sizeof(allowed), &allowed) != 0) return {};
    struct Core { int cpu; int max_frequency; };
    std::map<std::pair<int, int>, Core> physical_cores;
    for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
        if (!CPU_ISSET(cpu, &allowed)) continue;
        auto read_integer = [cpu](const char* leaf, int fallback) {
            std::ifstream file("/sys/devices/system/cpu/cpu" +
                               std::to_string(cpu) + "/" + leaf);
            int value = fallback;
            file >> value;
            return value;
        };
        const int package = read_integer("topology/physical_package_id", 0);
        const int core = read_integer("topology/core_id", cpu);
        const int frequency = read_integer("cpufreq/cpuinfo_max_freq", 0);
        const auto key = std::make_pair(package, core);
        auto [it, inserted] = physical_cores.emplace(key, Core{cpu, frequency});
        if (!inserted && frequency > it->second.max_frequency)
            it->second = {cpu, frequency};
    }
    std::vector<Core> sorted;
    sorted.reserve(physical_cores.size());
    for (const auto& entry : physical_cores) sorted.push_back(entry.second);
    std::sort(sorted.begin(), sorted.end(), [](const Core& left,
                                                const Core& right) {
        if (left.max_frequency != right.max_frequency)
            return left.max_frequency > right.max_frequency;
        return left.cpu < right.cpu;
    });
    const size_t skip = std::min(sorted.size(),
        static_cast<size_t>(std::max(0, inference_threads)));
    std::vector<int> cpus;
    cpus.reserve(sorted.size() - skip);
    for (size_t i = skip; i < sorted.size(); ++i) cpus.push_back(sorted[i].cpu);
    return cpus;
}

void pin_search_worker(int worker, const std::vector<int>& cpus) {
    if (cpus.empty()) return;
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(cpus[static_cast<size_t>(worker) % cpus.size()], &set);
    pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
}
#else
std::vector<int> search_cpus(int) { return {}; }
void pin_search_worker(int, const std::vector<int>&) {}
#endif

enum class SelectionResult { Retry, NeedsInference, Terminal };

SelectionResult select_leaf(GameState& game, float c_puct, LeafTask& task,
                            float& terminal_value) {
    task.board = game.board;
    task.path.len = 1;
    task.path.nodes[0] = 0;
    int node_index = 0;
    game.nodes[0].virtual_visits.fetch_add(1, std::memory_order_relaxed);

    while (true) {
        GNode& node = game.nodes[node_index];
        const uint8_t state = node.state.load(std::memory_order_acquire);
        if (state == NODE_EXPANDING) {
            release_virtual_loss(game, task.path);
            return SelectionResult::Retry;
        }
        if (state == NODE_UNEXPANDED) {
            uint8_t expected = NODE_UNEXPANDED;
            if (!node.state.compare_exchange_strong(
                    expected, NODE_EXPANDING, std::memory_order_acq_rel,
                    std::memory_order_acquire)) {
                release_virtual_loss(game, task.path);
                return SelectionResult::Retry;
            }
            task.node = node_index;
            const int winner = task.board.winner();
            if (winner != 0 || task.board.move_count() == BOARD_CELLS) {
                node.child_begin = 0;
                node.child_count = 0;
                node.state.store(NODE_EXPANDED, std::memory_order_release);
                terminal_value = winner == 0 ? 0.0f
                    : (winner == task.board.current_player() ? 1.0f : -1.0f);
                return SelectionResult::Terminal;
            }
            task.key = board_key(task.board);
            return SelectionResult::NeedsInference;
        }

        const int child_count = node.child_count;
        if (child_count == 0) {
            task.node = node_index;
            const int winner = task.board.winner();
            terminal_value = winner == 0 ? 0.0f
                : (winner == task.board.current_player() ? 1.0f : -1.0f);
            return SelectionResult::Terminal;
        }
        const int parent_visits = node.visits.load(std::memory_order_relaxed) +
            node.virtual_visits.load(std::memory_order_relaxed);
        const float exploration = c_puct * std::sqrt(
            static_cast<float>(std::max(1, parent_visits)));
        int best_index = static_cast<int>(node.child_begin);
        float best_score = -std::numeric_limits<float>::infinity();
        for (int slot = 0; slot < child_count; ++slot) {
            const int child_index = static_cast<int>(node.child_begin) + slot;
            const GNode& child = game.nodes[child_index];
            const int visits = child.visits.load(std::memory_order_relaxed);
            const int virtual_visits = child.virtual_visits.load(
                std::memory_order_relaxed);
            const int effective_visits = visits + virtual_visits;
            const int64_t sum = child.value_sum.load(std::memory_order_relaxed) +
                static_cast<int64_t>(virtual_visits) * VALUE_SCALE;
            const float q = effective_visits == 0 ? 0.0f
                : (static_cast<float>(sum) *
                   (1.0f / static_cast<float>(VALUE_SCALE))) /
                    static_cast<float>(effective_visits);
            const float score = -q + exploration * child.prior /
                (1.0f + static_cast<float>(effective_visits));
            if (score > best_score) {
                best_score = score;
                best_index = child_index;
            }
        }
        GNode& child = game.nodes[best_index];
        child.virtual_visits.fetch_add(1, std::memory_order_relaxed);
        if (task.path.len >= static_cast<int>(task.path.nodes.size())) {
            child.virtual_visits.fetch_sub(1, std::memory_order_relaxed);
            release_virtual_loss(game, task.path);
            throw std::logic_error("MCTS path exceeds board size");
        }
        task.path.nodes[task.path.len++] = best_index;
        if (!task.board.is_empty(child.action)) {
            release_virtual_loss(game, task.path);
            throw std::logic_error("MCTS selected an occupied cell");
        }
        task.board.play(child.action);
        node_index = best_index;
    }
}

void expand_leaf(GameState& game, const LeafTask& task,
                 const CachedEvaluation& evaluation,
                 std::vector<int>& nearby,
                 std::vector<std::pair<float, int>>& candidates) {
    nearby_moves(task.board, nearby);
    candidates.clear();
    for (int move : nearby) {
        float prior = evaluation.policy[move];
        if (!std::isfinite(prior) || prior < 0.0f) prior = 0.0f;
        candidates.emplace_back(prior, move);
    }
    const int keep = std::min<int>(MAX_CAND, candidates.size());
    if (keep > 0) {
        std::partial_sort(candidates.begin(), candidates.begin() + keep,
                          candidates.end(),
                          std::greater<std::pair<float, int>>());
    }
    float prior_sum = 0.0f;
    for (int i = 0; i < keep; ++i) prior_sum += candidates[i].first;
    if (!(prior_sum > 0.0f) && keep > 0) prior_sum = static_cast<float>(keep);

    GNode& leaf = game.nodes[task.node];
    if (keep > 0) {
        const uint32_t begin = game.next_node.fetch_add(
            static_cast<uint32_t>(keep), std::memory_order_relaxed);
        if (static_cast<size_t>(begin) + keep > game.node_capacity)
            throw std::overflow_error("MCTS node arena exhausted");
        for (int i = 0; i < keep; ++i) {
            const float prior = candidates[i].first > 0.0f
                ? candidates[i].first / prior_sum : 1.0f / prior_sum;
            initialize_node(game.nodes[begin + i], candidates[i].second, prior);
        }
        leaf.child_begin = begin;
        leaf.child_count = static_cast<uint8_t>(keep);
    } else {
        leaf.child_begin = 0;
        leaf.child_count = 0;
    }
    leaf.state.store(NODE_EXPANDED, std::memory_order_release);
}

}  // namespace

BatchedSelfPlay::BatchedSelfPlay(PureNet& net, float c_puct,
                                 int n_playout, int batch_games, int n_threads)
    : net_(net), c_puct_(c_puct), n_playout_(n_playout),
      batch_games_(batch_games) {
    if (n_playout_ <= 0 || batch_games_ <= 0)
        throw std::invalid_argument("playout and batch size must be positive");
    if (n_threads <= 0) {
        const unsigned hardware_threads = std::thread::hardware_concurrency();
        n_threads = hardware_threads == 0 ? 1
            : std::min(8, static_cast<int>(hardware_threads));
    }
    // CUDA needs one inference coordinator; CPU inference uses its worker pool.
    inference_threads_ = net_.using_cuda()
        ? 1 : std::max(1, std::min(n_threads, 32));
    net_.set_inference_threads(inference_threads_);

    const char* leaf_batch_text = std::getenv("GOMOK_LEAF_BATCH");
    if (leaf_batch_text != nullptr) {
        leaf_batch_size_ = std::max(
            1, std::min(n_playout_, std::atoi(leaf_batch_text)));
    } else if (net_.using_cuda()) {
        // The 15x15 CUDA network peaks near a total batch of 128.
        constexpr int CUDA_TARGET_BATCH = 128;
        leaf_batch_size_ = std::max(1, std::min(
            n_playout_,
            (CUDA_TARGET_BATCH + batch_games_ - 1) / batch_games_));
    } else {
        leaf_batch_size_ = std::min(n_playout_, 128);
    }

    const std::vector<int> cpus = search_cpus(inference_threads_);
    const int default_mcts_threads = cpus.empty() ? 2
        : std::min(8, static_cast<int>(cpus.size()));
    const char* mcts_threads_text = std::getenv("GOMOK_MCTS_THREADS");
    mcts_threads_ = mcts_threads_text != nullptr
        ? std::atoi(mcts_threads_text) : default_mcts_threads;
    mcts_threads_ = std::max(1, std::min(mcts_threads_, 32));
}

std::vector<std::vector<BatchedSelfPlay::Sample>>
BatchedSelfPlay::run_batch(float temp) {
    const int batch = batch_games_;
    std::unique_ptr<GameState[]> games(new GameState[batch]);
    const size_t arena_capacity = 1 + static_cast<size_t>(n_playout_) * MAX_CAND;
    for (int gi = 0; gi < batch; ++gi) {
        games[gi].node_capacity = arena_capacity;
        games[gi].nodes = std::make_unique<GNode[]>(arena_capacity);
        reset_tree(games[gi]);
    }

    const bool profile = std::getenv("GOMOK_PROFILE") != nullptr;
    const char* cache_entries_text = std::getenv("GOMOK_CACHE_ENTRIES");
    const size_t max_cache_entries = cache_entries_text != nullptr
        ? static_cast<size_t>(std::max(1, std::atoi(cache_entries_text)))
        : 32768;
    std::unordered_map<uint64_t, size_t> evaluation_cache_index;
    evaluation_cache_index.reserve(max_cache_entries);
    std::vector<uint64_t> evaluation_cache_keys;
    std::vector<CachedEvaluation> evaluation_cache_values;
    evaluation_cache_keys.reserve(max_cache_entries);
    evaluation_cache_values.reserve(max_cache_entries);
    size_t next_cache_slot = 0;
    int64_t cache_hits = 0;
    int64_t total_evaluations = 0;
    int inference_calls = 0;
    int largest_batch = 0;

    const char* seed_text = std::getenv("GOMOK_SEED");
    const auto seed = seed_text != nullptr
        ? static_cast<uint32_t>(std::strtoul(seed_text, nullptr, 10))
        : static_cast<uint32_t>(
              std::chrono::steady_clock::now().time_since_epoch().count());
    std::vector<std::mt19937> rngs;
    rngs.reserve(batch);
    for (int i = 0; i < batch; ++i)
        rngs.emplace_back(seed + 0x9e3779b9U * static_cast<uint32_t>(i + 1));

    const std::vector<int> worker_cpus = search_cpus(inference_threads_);
    const char* wait_text = std::getenv("GOMOK_BATCH_WAIT_US");
    const int batch_wait_us = wait_text != nullptr
        ? std::max(0, std::atoi(wait_text)) : 100;

    // Reuse the queues and inference buffers across moves.
    const int max_target = batch * n_playout_;
    const int max_inference_capacity = std::max(
        1, std::min(max_target, batch * leaf_batch_size_));
    std::unique_ptr<LeafTask[]> tasks(new LeafTask[max_target]);
    std::unique_ptr<int[]> ready_slots(new int[max_target]);
    std::unique_ptr<int[]> completion_slots(new int[max_target]);
    std::unique_ptr<std::atomic<uint8_t>[]> published(
        new std::atomic<uint8_t>[max_target]);
    std::vector<int> inference_slots;
    inference_slots.reserve(max_inference_capacity);
    std::vector<float> eval_states(
        static_cast<size_t>(max_inference_capacity) * 4 * BOARD_CELLS);
    std::vector<float> eval_policy(
        static_cast<size_t>(max_inference_capacity) * BOARD_CELLS);
    std::vector<float> eval_value(max_inference_capacity);
    std::vector<CachedEvaluation> fresh_evaluations(max_inference_capacity);

    while (true) {
        int active_games = 0;
        for (int gi = 0; gi < batch; ++gi) {
            if (!games[gi].finished) {
                reset_tree(games[gi]);
                ++active_games;
            }
        }
        if (active_games == 0) break;

        const int target = active_games * n_playout_;
        const int inference_capacity = std::max(1, std::min(
            target, batch * leaf_batch_size_));
        for (int i = 0; i < target; ++i)
            published[i].store(0, std::memory_order_relaxed);
        std::atomic<int> ready_tail{0};
        std::atomic<int> completion_head{0};
        std::atomic<int> completion_tail{0};
        std::atomic<int> completed{0};
        std::atomic<uint64_t> claim_cursor{0};
        std::atomic<bool> worker_failed{false};
        std::exception_ptr worker_error;
        std::mutex worker_error_mutex;

        std::vector<std::thread> search_workers;
        search_workers.reserve(mcts_threads_);
        for (int worker = 0; worker < mcts_threads_; ++worker) {
            search_workers.emplace_back([&, worker] {
                pin_search_worker(worker, worker_cpus);
                try {
                    std::vector<int> worker_nearby;
                    worker_nearby.reserve(BOARD_CELLS);
                    std::vector<std::pair<float, int>> worker_candidates;
                    worker_candidates.reserve(BOARD_CELLS);
                    auto complete_one = [&]() {
                        int head = completion_head.load(
                            std::memory_order_relaxed);
                        while (head < completion_tail.load(
                                           std::memory_order_acquire)) {
                            if (completion_head.compare_exchange_weak(
                                    head, head + 1, std::memory_order_relaxed,
                                    std::memory_order_relaxed)) {
                                LeafTask& completed_task =
                                    tasks[completion_slots[head]];
                                GameState& completed_game =
                                    games[completed_task.game];
                                expand_leaf(completed_game, completed_task,
                                            completed_task.evaluation,
                                            worker_nearby,
                                            worker_candidates);
                                backup_path(completed_game,
                                            completed_task.path,
                                            completed_task.evaluation.value);
                                completed.fetch_add(
                                    1, std::memory_order_release);
                                return true;
                            }
                        }
                        return false;
                    };
                    while (!worker_failed.load(std::memory_order_relaxed)) {
                        if (complete_one()) continue;
                        int game_index = -1;
                        int simulation = -1;
                        for (int attempt = 0; attempt < batch; ++attempt) {
                            const int candidate = static_cast<int>(
                                claim_cursor.fetch_add(1,
                                    std::memory_order_relaxed) % batch);
                            GameState& game = games[candidate];
                            if (game.finished) continue;
                            int current = game.issued.load(
                                std::memory_order_relaxed);
                            while (current < n_playout_) {
                                if (game.issued.compare_exchange_weak(
                                        current, current + 1,
                                        std::memory_order_relaxed,
                                        std::memory_order_relaxed)) {
                                    game_index = candidate;
                                    simulation = current;
                                    break;
                                }
                            }
                            if (game_index >= 0) break;
                        }
                        if (game_index < 0) {
                            bool all_issued = true;
                            for (int gi = 0; gi < batch; ++gi) {
                                if (!games[gi].finished &&
                                    games[gi].issued.load(
                                        std::memory_order_relaxed) < n_playout_) {
                                    all_issued = false;
                                    break;
                                }
                            }
                            if (all_issued) {
                                if (completed.load(std::memory_order_acquire) >=
                                    target) break;
                                std::this_thread::yield();
                                continue;
                            }
                            std::this_thread::yield();
                            continue;
                        }

                        LeafTask& task = tasks[static_cast<size_t>(game_index) *
                                               n_playout_ + simulation];
                        task.game = game_index;
                        task.eval = -1;
                        while (true) {
                            float terminal_value = 0.0f;
                            const SelectionResult selected = select_leaf(
                                games[game_index], c_puct_, task,
                                terminal_value);
                            if (selected == SelectionResult::Retry) {
                                complete_one();
                                std::this_thread::yield();
                                continue;
                            }
                            if (selected == SelectionResult::Terminal) {
                                backup_path(games[game_index], task.path,
                                            terminal_value);
                                completed.fetch_add(1,
                                                    std::memory_order_release);
                            } else {
                                const int position = ready_tail.fetch_add(
                                    1, std::memory_order_relaxed);
                                if (position >= target)
                                    throw std::logic_error(
                                        "MCTS inference queue overflow");
                                ready_slots[position] = static_cast<int>(
                                    static_cast<size_t>(game_index) *
                                    n_playout_ + simulation);
                                published[position].store(
                                    1, std::memory_order_release);
                            }
                            break;
                        }
                    }
                } catch (...) {
                    {
                        std::lock_guard<std::mutex> lock(worker_error_mutex);
                        if (!worker_error) worker_error = std::current_exception();
                    }
                    worker_failed.store(true, std::memory_order_release);
                }
            });
        }

        int ready_head = 0;
        int completion_tail_local = 0;

        while (completed.load(std::memory_order_acquire) < target &&
               !worker_failed.load(std::memory_order_acquire)) {
            if (ready_head >= ready_tail.load(std::memory_order_acquire) ||
                !published[ready_head].load(std::memory_order_acquire)) {
                std::this_thread::yield();
                continue;
            }

            const auto deadline = std::chrono::steady_clock::now() +
                std::chrono::microseconds(batch_wait_us);
            int contiguous = 0;
            while (contiguous < inference_capacity) {
                const int tail = ready_tail.load(std::memory_order_acquire);
                while (ready_head + contiguous < tail &&
                       contiguous < inference_capacity &&
                       published[ready_head + contiguous].load(
                           std::memory_order_acquire)) {
                    ++contiguous;
                }
                if (contiguous >= inference_capacity ||
                    std::chrono::steady_clock::now() >= deadline) break;
                std::this_thread::yield();
            }
            if (contiguous == 0) continue;

            inference_slots.clear();
            for (int i = 0; i < contiguous; ++i)
                inference_slots.push_back(ready_slots[ready_head + i]);
            ready_head += contiguous;

            std::unordered_map<uint64_t, int> pending_evaluations;
            pending_evaluations.reserve(inference_slots.size());
            int eval_count = 0;
            for (int slot : inference_slots) {
                LeafTask& task = tasks[slot];
                const auto cached = evaluation_cache_index.find(task.key);
                if (cached != evaluation_cache_index.end()) {
                    ++cache_hits;
                    task.eval = -1;
                    continue;
                }
                const auto [pending, inserted] =
                    pending_evaluations.emplace(task.key, eval_count);
                task.eval = pending->second;
                if (inserted) {
                    task.board.encode_state(eval_states.data() +
                        static_cast<size_t>(eval_count) * 4 * BOARD_CELLS);
                    ++eval_count;
                } else {
                    ++cache_hits;
                }
            }

            if (eval_count > 0) {
                total_evaluations += eval_count;
                ++inference_calls;
                largest_batch = std::max(largest_batch, eval_count);
                net_.forward_batch_policy(eval_states.data(), eval_count,
                                          eval_policy.data(),
                                          eval_value.data());
                for (const auto& entry : pending_evaluations) {
                    const int eval = entry.second;
                    CachedEvaluation& result = fresh_evaluations[eval];
                    result.value = eval_value[eval];
                    std::memcpy(result.policy.data(), eval_policy.data() +
                        static_cast<size_t>(eval) * BOARD_CELLS,
                        BOARD_CELLS * sizeof(float));
                }
            }

            for (int slot : inference_slots) {
                LeafTask& task = tasks[slot];
                const CachedEvaluation& evaluation = task.eval >= 0
                    ? fresh_evaluations[task.eval]
                    : evaluation_cache_values[
                          evaluation_cache_index.at(task.key)];
                task.evaluation = evaluation;
                completion_slots[completion_tail_local++] = slot;
                completion_tail.store(completion_tail_local,
                                      std::memory_order_release);
            }

            // Do not evict entries until this batch has copied its results.
            for (const auto& entry : pending_evaluations) {
                const uint64_t key = entry.first;
                const CachedEvaluation& result =
                    fresh_evaluations[entry.second];
                if (evaluation_cache_values.size() < max_cache_entries) {
                    const size_t cache_slot = evaluation_cache_values.size();
                    evaluation_cache_keys.push_back(key);
                    evaluation_cache_values.push_back(result);
                    evaluation_cache_index.emplace(key, cache_slot);
                } else {
                    evaluation_cache_index.erase(
                        evaluation_cache_keys[next_cache_slot]);
                    evaluation_cache_keys[next_cache_slot] = key;
                    evaluation_cache_values[next_cache_slot] = result;
                    evaluation_cache_index.emplace(key, next_cache_slot);
                    next_cache_slot = (next_cache_slot + 1) % max_cache_entries;
                }
            }
        }

        for (std::thread& worker : search_workers) worker.join();
        if (worker_error) std::rethrow_exception(worker_error);
        if (completed.load(std::memory_order_relaxed) != target)
            throw std::logic_error("MCTS round completed the wrong playout count");

        for (int gi = 0; gi < batch; ++gi) {
            GameState& game = games[gi];
            if (game.finished) continue;
            GNode& root = game.nodes[0];
            if (root.state.load(std::memory_order_acquire) != NODE_EXPANDED ||
                root.child_count == 0) {
                game.finished = true;
                continue;
            }

            std::vector<float> weights(root.child_count);
            if (temp <= 1e-6f) {
                int best = 0;
                for (int i = 1; i < root.child_count; ++i) {
                    if (game.nodes[root.child_begin + i].visits.load(
                            std::memory_order_relaxed) >
                        game.nodes[root.child_begin + best].visits.load(
                            std::memory_order_relaxed)) best = i;
                }
                weights[best] = 1.0f;
            } else {
                const float inv_temp = 1.0f / temp;
                std::vector<float> log_weights(root.child_count);
                for (int i = 0; i < root.child_count; ++i) {
                    const float visits = static_cast<float>(
                        game.nodes[root.child_begin + i].visits.load(
                            std::memory_order_relaxed));
                    log_weights[i] = std::log(std::max(visits, 1e-10f)) *
                                     inv_temp;
                }
                const float max_log = *std::max_element(log_weights.begin(),
                                                        log_weights.end());
                for (int i = 0; i < root.child_count; ++i)
                    weights[i] = std::exp(log_weights[i] - max_log);
            }
            float weight_sum = 0.0f;
            for (float weight : weights) weight_sum += weight;
            std::discrete_distribution<int> distribution(
                weights.begin(), weights.end());
            const int pick = distribution(rngs[gi]);
            const int move = game.nodes[root.child_begin + pick].action;

            std::array<float, BOARD_CELLS> probs{};
            for (int i = 0; i < root.child_count; ++i) {
                const GNode& child = game.nodes[root.child_begin + i];
                probs[child.action] = weights[i] / weight_sum;
            }
            std::array<float, 4 * BOARD_CELLS> state{};
            game.board.encode_state(state.data());
            game.states.push_back(state);
            game.probs.push_back(probs);
            game.players.push_back(game.board.current_player());
            game.board.play(move);

            const int winner = game.board.winner();
            if (winner != 0 || game.board.move_count() == BOARD_CELLS) {
                game.finished = true;
                game.winner = winner;
            }
        }
    }

    if (profile) {
        std::fprintf(stderr,
            "[selfplay-profile] evals=%lld calls=%d avg_batch=%.1f "
            "max_batch=%d cache_hits=%lld cache_size=%zu mcts_threads=%d\n",
            static_cast<long long>(total_evaluations), inference_calls,
            inference_calls == 0 ? 0.0
                : static_cast<double>(total_evaluations) / inference_calls,
            largest_batch, static_cast<long long>(cache_hits),
            evaluation_cache_values.size(), mcts_threads_);
    }

    std::vector<std::vector<Sample>> result(batch);
    for (int gi = 0; gi < batch; ++gi) {
        const GameState& game = games[gi];
        for (size_t i = 0; i < game.states.size(); ++i) {
            Sample sample;
            sample.state = game.states[i];
            sample.probs = game.probs[i];
            sample.z = game.winner == 0 ? 0.0f
                       : (game.players[i] == game.winner ? 1.0f : -1.0f);
            result[gi].push_back(std::move(sample));
        }
    }
    return result;
}

}  // namespace gomoku
