#include "network.h"
#include "network_cuda.h"
#include <cstdint>
#include <cmath>
#include <cstring>
#include <fstream>
#include <map>
#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <limits>
#include <mutex>
#include <random>
#include <stdexcept>
#include <thread>
#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif
#ifdef __linux__
#include <pthread.h>
#include <sched.h>
#endif

namespace gomoku {

namespace {
#ifdef __linux__
std::vector<int> inference_cpus() {
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
        auto [it, inserted] = physical_cores.emplace(
            key, Core{cpu, frequency});
        if (!inserted && frequency > it->second.max_frequency)
            it->second = {cpu, frequency};
    }
    std::vector<Core> sorted;
    sorted.reserve(physical_cores.size());
    for (const auto& [key, core] : physical_cores) {
        (void)key;
        sorted.push_back(core);
    }
    std::sort(sorted.begin(), sorted.end(), [](const Core& left,
                                                const Core& right) {
        if (left.max_frequency != right.max_frequency)
            return left.max_frequency > right.max_frequency;
        return left.cpu < right.cpu;
    });
    std::vector<int> cpus;
    cpus.reserve(sorted.size());
    for (const Core& core : sorted) cpus.push_back(core.cpu);
    return cpus;
}

void pin_inference_worker(int worker) {
    static const std::vector<int> cpus = inference_cpus();
    if (worker < 0 || worker >= static_cast<int>(cpus.size())) return;
    thread_local int pinned_cpu = -1;
    if (pinned_cpu == cpus[worker]) return;
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(cpus[worker], &set);
    if (pthread_setaffinity_np(pthread_self(), sizeof(set), &set) == 0)
        pinned_cpu = cpus[worker];
}
#else
void pin_inference_worker(int) {}
#endif

#if (defined(__GNUC__) || defined(__clang__)) && defined(__x86_64__)

__attribute__((target("avx2,fma")))
void fast_conv3x3_relu(const ConvLayer& layer, const float* input,
                       std::vector<float>& output,
                       std::vector<float>& columns) {
    constexpr int side = 15;
    constexpr int cells = side * side;
    const int channels = layer.in_ch;
    const int outputs = layer.out_ch;
    const int depth = channels * 9;
    columns.resize(static_cast<size_t>(depth) * cells);

    // im2col, with whole contiguous row copies instead of a branch per pixel.
    for (int channel = 0; channel < channels; ++channel) {
        const float* plane = input + static_cast<size_t>(channel) * cells;
        for (int kr = 0; kr < 3; ++kr) {
            const int row_begin = std::max(0, 1 - kr);
            const int row_end = std::min(side, side + 1 - kr);
            for (int kc = 0; kc < 3; ++kc) {
                float* column = columns.data() +
                    static_cast<size_t>((channel * 3 + kr) * 3 + kc) * cells;
                std::fill(column, column + cells, 0.0f);
                const int col_begin = std::max(0, 1 - kc);
                const int col_end = std::min(side, side + 1 - kc);
                for (int row = row_begin; row < row_end; ++row) {
                    const float* source = plane + (row + kr - 1) * side +
                                          (col_begin + kc - 1);
                    std::memcpy(column + row * side + col_begin, source,
                                static_cast<size_t>(col_end - col_begin) *
                                    sizeof(float));
                }
            }
        }
    }

    output.resize(static_cast<size_t>(outputs) * cells);
    const __m256 zero = _mm256_setzero_ps();
    // A 6x16 tile fills the AVX2 register file while reusing every loaded
    // input vector across six output channels.
    int out = 0;
    for (; out + 6 <= outputs; out += 6) {
        int cell = 0;
        for (; cell + 16 <= cells; cell += 16) {
            __m256 sums[6][2];
            for (int lane = 0; lane < 6; ++lane) {
                sums[lane][0] = _mm256_set1_ps(layer.b[out + lane]);
                sums[lane][1] = sums[lane][0];
            }
            for (int k = 0; k < depth; ++k) {
                const __m256 x0 = _mm256_loadu_ps(
                    columns.data() + static_cast<size_t>(k) * cells + cell);
                const __m256 x1 = _mm256_loadu_ps(
                    columns.data() + static_cast<size_t>(k) * cells + cell + 8);
                for (int lane = 0; lane < 6; ++lane) {
                    const __m256 weight = _mm256_set1_ps(layer.w[
                        static_cast<size_t>(out + lane) * depth + k]);
                    sums[lane][0] = _mm256_fmadd_ps(
                        weight, x0, sums[lane][0]);
                    sums[lane][1] = _mm256_fmadd_ps(
                        weight, x1, sums[lane][1]);
                }
            }
            for (int lane = 0; lane < 6; ++lane) {
                float* destination = output.data() +
                    static_cast<size_t>(out + lane) * cells + cell;
                _mm256_storeu_ps(destination,
                                 _mm256_max_ps(sums[lane][0], zero));
                _mm256_storeu_ps(destination + 8,
                                 _mm256_max_ps(sums[lane][1], zero));
            }
        }
        for (int lane = 0; lane < 6; ++lane) {
            for (int tail = cell; tail < cells; ++tail) {
                float sum = layer.b[out + lane];
                for (int k = 0; k < depth; ++k)
                    sum += layer.w[static_cast<size_t>(out + lane) * depth + k] *
                           columns[static_cast<size_t>(k) * cells + tail];
                output[static_cast<size_t>(out + lane) * cells + tail] =
                    std::max(0.0f, sum);
            }
        }
    }
    // At most four output channels remain. Keep one full spatial vector in
    // registers so the tail does not repeatedly write partial sums.
    for (; out < outputs; ++out) {
        int cell = 0;
        for (; cell + 8 <= cells; cell += 8) {
            __m256 sum = _mm256_set1_ps(layer.b[out]);
            for (int k = 0; k < depth; ++k) {
                const __m256 input_values = _mm256_loadu_ps(
                    columns.data() + static_cast<size_t>(k) * cells + cell);
                sum = _mm256_fmadd_ps(_mm256_set1_ps(
                    layer.w[static_cast<size_t>(out) * depth + k]),
                    input_values, sum);
            }
            _mm256_storeu_ps(output.data() + static_cast<size_t>(out) * cells + cell,
                             _mm256_max_ps(sum, zero));
        }
        for (; cell < cells; ++cell) {
            float sum = layer.b[out];
            for (int k = 0; k < depth; ++k)
                sum += layer.w[static_cast<size_t>(out) * depth + k] *
                       columns[static_cast<size_t>(k) * cells + cell];
            output[static_cast<size_t>(out) * cells + cell] =
                std::max(0.0f, sum);
        }
    }
}

void quantize_conv3x3_weights(const ConvLayer& layer,
                              std::vector<int8_t>& quantized,
                              std::vector<float>& scales) {
    const int depth = layer.in_ch * 9;
    const int padded_depth = (depth + 31) & ~31;
    quantized.assign(static_cast<size_t>(layer.out_ch) * padded_depth, 0);
    scales.resize(layer.out_ch);
    for (int out = 0; out < layer.out_ch; ++out) {
        const float* weights = layer.w.data() +
                               static_cast<size_t>(out) * depth;
        float maximum = 0.0f;
        for (int k = 0; k < depth; ++k)
            maximum = std::max(maximum, std::abs(weights[k]));
        const float scale = maximum > 0.0f ? maximum / 127.0f : 1.0f;
        const float inverse_scale = 1.0f / scale;
        scales[out] = scale;
        int8_t* destination = quantized.data() +
                              static_cast<size_t>(out) * padded_depth;
        for (int k = 0; k < depth; ++k) {
            const int value = static_cast<int>(
                std::nearbyint(weights[k] * inverse_scale));
            destination[k] = static_cast<int8_t>(
                std::max(-127, std::min(127, value)));
        }
    }
}

__attribute__((target("avx2")))
int horizontal_sum_int32(__m256i value) {
    __m128i sum = _mm_add_epi32(_mm256_castsi256_si128(value),
                                _mm256_extracti128_si256(value, 1));
    sum = _mm_hadd_epi32(sum, sum);
    sum = _mm_hadd_epi32(sum, sum);
    return _mm_cvtsi128_si32(sum);
}

__attribute__((target("avx2,avxvnni")))
void quantized_conv3x3_relu(const ConvLayer& layer,
                            const std::vector<int8_t>& weights,
                            const std::vector<float>& weight_scales,
                            const float* input,
                            std::vector<float>& output,
                            std::vector<uint8_t>& quantized_input,
                            std::vector<uint8_t>& columns) {
    constexpr int side = 15;
    constexpr int cells = side * side;
    const int channels = layer.in_ch;
    const int depth = channels * 9;
    const int padded_depth = (depth + 31) & ~31;

    float maximum = 0.0f;
    for (int i = 0; i < channels * cells; ++i)
        maximum = std::max(maximum, input[i]);
    const float input_scale = maximum > 0.0f ? maximum / 255.0f : 1.0f;
    const float inverse_scale = 1.0f / input_scale;
    quantized_input.resize(static_cast<size_t>(channels) * cells);
    for (int i = 0; i < channels * cells; ++i) {
        const int value = static_cast<int>(input[i] * inverse_scale + 0.5f);
        quantized_input[i] = static_cast<uint8_t>(
            std::max(0, std::min(255, value)));
    }

    columns.assign(static_cast<size_t>(cells) * padded_depth, 0);
    for (int row = 0; row < side; ++row) {
        for (int col = 0; col < side; ++col) {
            uint8_t* destination = columns.data() +
                static_cast<size_t>(row * side + col) * padded_depth;
            for (int channel = 0; channel < channels; ++channel) {
                const uint8_t* plane = quantized_input.data() +
                                       static_cast<size_t>(channel) * cells;
                for (int kernel_row = 0; kernel_row < 3; ++kernel_row) {
                    const int source_row = row + kernel_row - 1;
                    if (source_row < 0 || source_row >= side) continue;
                    for (int kernel_col = 0; kernel_col < 3; ++kernel_col) {
                        const int source_col = col + kernel_col - 1;
                        if (source_col < 0 || source_col >= side) continue;
                        destination[(channel * 3 + kernel_row) * 3 +
                                    kernel_col] =
                            plane[source_row * side + source_col];
                    }
                }
            }
        }
    }

    output.resize(static_cast<size_t>(layer.out_ch) * cells);
    int out = 0;
    for (; out + 2 <= layer.out_ch; out += 2) {
        const int8_t* weight[2] = {
            weights.data() + static_cast<size_t>(out + 0) * padded_depth,
            weights.data() + static_cast<size_t>(out + 1) * padded_depth,
        };
        const float combined_scale[2] = {
            input_scale * weight_scales[out + 0],
            input_scale * weight_scales[out + 1],
        };
        float* destination[2] = {
            output.data() + static_cast<size_t>(out + 0) * cells,
            output.data() + static_cast<size_t>(out + 1) * cells,
        };
        int cell = 0;
        for (; cell + 5 <= cells; cell += 5) {
            const uint8_t* values = columns.data() +
                static_cast<size_t>(cell) * padded_depth;
            __m256i a00 = _mm256_setzero_si256();
            __m256i a01 = _mm256_setzero_si256();
            __m256i a02 = _mm256_setzero_si256();
            __m256i a03 = _mm256_setzero_si256();
            __m256i a04 = _mm256_setzero_si256();
            __m256i a10 = _mm256_setzero_si256();
            __m256i a11 = _mm256_setzero_si256();
            __m256i a12 = _mm256_setzero_si256();
            __m256i a13 = _mm256_setzero_si256();
            __m256i a14 = _mm256_setzero_si256();
            for (int k = 0; k < padded_depth; k += 32) {
                const __m256i w0 = _mm256_loadu_si256(
                    reinterpret_cast<const __m256i*>(weight[0] + k));
                const __m256i w1 = _mm256_loadu_si256(
                    reinterpret_cast<const __m256i*>(weight[1] + k));
                const __m256i x0 = _mm256_loadu_si256(
                    reinterpret_cast<const __m256i*>(values + k));
                const __m256i x1 = _mm256_loadu_si256(
                    reinterpret_cast<const __m256i*>(values + padded_depth + k));
                const __m256i x2 = _mm256_loadu_si256(
                    reinterpret_cast<const __m256i*>(values + 2 * padded_depth + k));
                const __m256i x3 = _mm256_loadu_si256(
                    reinterpret_cast<const __m256i*>(values + 3 * padded_depth + k));
                const __m256i x4 = _mm256_loadu_si256(
                    reinterpret_cast<const __m256i*>(values + 4 * padded_depth + k));
                a00 = _mm256_dpbusd_epi32(a00, x0, w0);
                a01 = _mm256_dpbusd_epi32(a01, x1, w0);
                a02 = _mm256_dpbusd_epi32(a02, x2, w0);
                a03 = _mm256_dpbusd_epi32(a03, x3, w0);
                a04 = _mm256_dpbusd_epi32(a04, x4, w0);
                a10 = _mm256_dpbusd_epi32(a10, x0, w1);
                a11 = _mm256_dpbusd_epi32(a11, x1, w1);
                a12 = _mm256_dpbusd_epi32(a12, x2, w1);
                a13 = _mm256_dpbusd_epi32(a13, x3, w1);
                a14 = _mm256_dpbusd_epi32(a14, x4, w1);
            }
            const __m256i sums[2][5] = {
                {a00, a01, a02, a03, a04},
                {a10, a11, a12, a13, a14}};
            for (int output_lane = 0; output_lane < 2; ++output_lane)
                for (int cell_lane = 0; cell_lane < 5; ++cell_lane)
                    destination[output_lane][cell + cell_lane] = std::max(
                        0.0f, static_cast<float>(horizontal_sum_int32(
                            sums[output_lane][cell_lane])) *
                            combined_scale[output_lane] +
                            layer.b[out + output_lane]);
        }
        for (; cell < cells; ++cell) {
            const uint8_t* values = columns.data() +
                static_cast<size_t>(cell) * padded_depth;
            __m256i accumulators[2] = {
                _mm256_setzero_si256(), _mm256_setzero_si256()};
            for (int k = 0; k < padded_depth; k += 32) {
                const __m256i input_values = _mm256_loadu_si256(
                    reinterpret_cast<const __m256i*>(values + k));
                for (int output_lane = 0; output_lane < 2; ++output_lane) {
                    accumulators[output_lane] = _mm256_dpbusd_epi32(
                        accumulators[output_lane], input_values,
                        _mm256_loadu_si256(reinterpret_cast<const __m256i*>(
                            weight[output_lane] + k)));
                }
            }
            for (int output_lane = 0; output_lane < 2; ++output_lane) {
                destination[output_lane][cell] = std::max(
                    0.0f,
                    static_cast<float>(horizontal_sum_int32(
                        accumulators[output_lane])) *
                            combined_scale[output_lane] +
                        layer.b[out + output_lane]);
            }
        }
    }
    for (; out < layer.out_ch; ++out) {
        const int8_t* weight = weights.data() +
                               static_cast<size_t>(out) * padded_depth;
        const float combined_scale = input_scale * weight_scales[out];
        float* destination = output.data() + static_cast<size_t>(out) * cells;
        for (int cell = 0; cell < cells; ++cell) {
            const uint8_t* values = columns.data() +
                static_cast<size_t>(cell) * padded_depth;
            __m256i accumulator = _mm256_setzero_si256();
            for (int k = 0; k < padded_depth; k += 32) {
                accumulator = _mm256_dpbusd_epi32(
                    accumulator,
                    _mm256_loadu_si256(reinterpret_cast<const __m256i*>(
                        values + k)),
                    _mm256_loadu_si256(reinterpret_cast<const __m256i*>(
                        weight + k)));
            }
            destination[cell] = std::max(
                0.0f,
                static_cast<float>(horizontal_sum_int32(accumulator)) *
                    combined_scale + layer.b[out]);
        }
    }
}

__attribute__((target("avx2,fma")))
void fast_conv1x1_relu(const ConvLayer& layer,
                       const std::vector<float>& input,
                       std::vector<float>& output) {
    constexpr int cells = 225;
    output.resize(static_cast<size_t>(layer.out_ch) * cells);
    const __m256 zero = _mm256_setzero_ps();
    for (int out = 0; out < layer.out_ch; ++out) {
        int cell = 0;
        for (; cell + 8 <= cells; cell += 8) {
            __m256 sum = _mm256_set1_ps(layer.b[out]);
            for (int in = 0; in < layer.in_ch; ++in) {
                const __m256 x = _mm256_loadu_ps(
                    input.data() + static_cast<size_t>(in) * cells + cell);
                sum = _mm256_fmadd_ps(_mm256_set1_ps(
                    layer.w[static_cast<size_t>(out) * layer.in_ch + in]),
                    x, sum);
            }
            _mm256_storeu_ps(output.data() + static_cast<size_t>(out) * cells + cell,
                             _mm256_max_ps(sum, zero));
        }
        for (; cell < cells; ++cell) {
            float sum = layer.b[out];
            for (int in = 0; in < layer.in_ch; ++in)
                sum += input[static_cast<size_t>(in) * cells + cell] *
                       layer.w[static_cast<size_t>(out) * layer.in_ch + in];
            output[static_cast<size_t>(out) * cells + cell] =
                std::max(0.0f, sum);
        }
    }
}

__attribute__((target("avx2,fma")))
float horizontal_sum(__m256 value) {
    const __m128 low = _mm256_castps256_ps128(value);
    const __m128 high = _mm256_extractf128_ps(value, 1);
    __m128 sum = _mm_add_ps(low, high);
    sum = _mm_hadd_ps(sum, sum);
    sum = _mm_hadd_ps(sum, sum);
    return _mm_cvtss_f32(sum);
}

__attribute__((target("avx2,fma")))
void fast_fc(const FCLayer& layer, const std::vector<float>& input,
             std::vector<float>& output, bool relu) {
    output.resize(layer.out_n);
    int out = 0;
    for (; out + 4 <= layer.out_n; out += 4) {
        __m256 a0 = _mm256_setzero_ps();
        __m256 a1 = _mm256_setzero_ps();
        __m256 a2 = _mm256_setzero_ps();
        __m256 a3 = _mm256_setzero_ps();
        int in = 0;
        for (; in + 8 <= layer.in_n; in += 8) {
            const __m256 x = _mm256_loadu_ps(input.data() + in);
            a0 = _mm256_fmadd_ps(x, _mm256_loadu_ps(
                layer.w.data() + static_cast<size_t>(out + 0) * layer.in_n + in), a0);
            a1 = _mm256_fmadd_ps(x, _mm256_loadu_ps(
                layer.w.data() + static_cast<size_t>(out + 1) * layer.in_n + in), a1);
            a2 = _mm256_fmadd_ps(x, _mm256_loadu_ps(
                layer.w.data() + static_cast<size_t>(out + 2) * layer.in_n + in), a2);
            a3 = _mm256_fmadd_ps(x, _mm256_loadu_ps(
                layer.w.data() + static_cast<size_t>(out + 3) * layer.in_n + in), a3);
        }
        float sums[4] = {horizontal_sum(a0) + layer.b[out + 0],
                         horizontal_sum(a1) + layer.b[out + 1],
                         horizontal_sum(a2) + layer.b[out + 2],
                         horizontal_sum(a3) + layer.b[out + 3]};
        for (int lane = 0; lane < 4; ++lane) {
            for (int tail = in; tail < layer.in_n; ++tail)
                sums[lane] += input[tail] * layer.w[
                    static_cast<size_t>(out + lane) * layer.in_n + tail];
            output[out + lane] = relu ? std::max(0.0f, sums[lane]) : sums[lane];
        }
    }
    for (; out < layer.out_n; ++out) {
        __m256 accumulator = _mm256_setzero_ps();
        int in = 0;
        for (; in + 8 <= layer.in_n; in += 8) {
            accumulator = _mm256_fmadd_ps(
                _mm256_loadu_ps(input.data() + in),
                _mm256_loadu_ps(layer.w.data() +
                    static_cast<size_t>(out) * layer.in_n + in), accumulator);
        }
        float sum = horizontal_sum(accumulator) + layer.b[out];
        for (; in < layer.in_n; ++in)
            sum += input[in] * layer.w[static_cast<size_t>(out) * layer.in_n + in];
        output[out] = relu ? std::max(0.0f, sum) : sum;
    }
}

bool has_fast_inference() {
    static const bool supported = __builtin_cpu_supports("avx2") &&
                                  __builtin_cpu_supports("fma");
    return supported;
}

bool has_quantized_inference() {
    static const bool supported = __builtin_cpu_supports("avx2") &&
                                  __builtin_cpu_supports("avxvnni");
    return supported;
}
#else
bool has_fast_inference() { return false; }
bool has_quantized_inference() { return false; }
#endif
}  // namespace

class PureNet::InferencePool {
public:
    InferencePool(const PureNet& net, int threads)
        : net_(net), thread_count_(threads) {
        workers_.reserve(thread_count_ - 1);
        for (int worker = 1; worker < thread_count_; ++worker)
            workers_.emplace_back(&InferencePool::worker_loop, this, worker);
    }

    ~InferencePool() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stopping_ = true;
            ++generation_;
        }
        task_cv_.notify_all();
        for (std::thread& worker : workers_) worker.join();
    }

    void forward(const float* states, int batch, float* policy, float* value,
                 bool probabilities) {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            states_ = states;
            policy_ = policy;
            value_ = value;
            batch_ = batch;
            probabilities_ = probabilities;
            active_threads_ = std::min(batch, thread_count_);
            remaining_ = active_threads_ - 1;
            ++generation_;
        }
        task_cv_.notify_all();

        pin_inference_worker(0);
        process_partition(0, active_threads_, states, batch, policy, value,
                          probabilities);

        std::unique_lock<std::mutex> lock(mutex_);
        done_cv_.wait(lock, [this] { return remaining_ == 0; });
    }

private:
    void process_partition(int worker, int active_threads,
                           const float* states, int batch,
                           float* policy, float* value,
                           bool probabilities) const {
        for (int item = worker; item < batch; item += active_threads) {
            net_.forward_impl(states + static_cast<size_t>(item) * 4 * 225,
                              policy + static_cast<size_t>(item) * 225,
                              value[item], probabilities);
        }
    }

    void worker_loop(int worker) {
        pin_inference_worker(worker);
        uint64_t observed_generation = 0;
        while (true) {
            const float* states;
            float* policy;
            float* value;
            int batch;
            int active_threads;
            bool probabilities;
            {
                std::unique_lock<std::mutex> lock(mutex_);
                task_cv_.wait(lock, [this, &observed_generation] {
                    return stopping_ || generation_ != observed_generation;
                });
                if (stopping_) return;
                observed_generation = generation_;
                states = states_;
                policy = policy_;
                value = value_;
                batch = batch_;
                active_threads = active_threads_;
                probabilities = probabilities_;
            }
            if (worker >= active_threads) continue;
            process_partition(worker, active_threads, states, batch, policy,
                              value, probabilities);
            {
                std::lock_guard<std::mutex> lock(mutex_);
                if (--remaining_ == 0) done_cv_.notify_one();
            }
        }
    }

    const PureNet& net_;
    int thread_count_;
    std::vector<std::thread> workers_;
    std::mutex mutex_;
    std::condition_variable task_cv_;
    std::condition_variable done_cv_;
    bool stopping_ = false;
    uint64_t generation_ = 0;
    int remaining_ = 0;
    const float* states_ = nullptr;
    float* policy_ = nullptr;
    float* value_ = nullptr;
    int batch_ = 0;
    int active_threads_ = 1;
    bool probabilities_ = false;
};

static bool load_bin_file(const std::string& path, std::vector<float>& out,
                          size_t expected) {
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) return false;
    uint32_t count = 0;
    if (!f.read(reinterpret_cast<char*>(&count), sizeof(count)) ||
        count != expected) return false;
    std::vector<float> data(count);
    if (!f.read(reinterpret_cast<char*>(data.data()),
                static_cast<std::streamsize>(count * sizeof(float)))) return false;
    out = std::move(data);
    return true;
}

static bool save_bin_file(const std::string& path, const std::vector<float>& data) {
    if (data.size() > std::numeric_limits<uint32_t>::max()) return false;
    std::ofstream f(path, std::ios::binary);
    if (!f.is_open()) return false;
    uint32_t count = static_cast<uint32_t>(data.size());
    f.write(reinterpret_cast<const char*>(&count), sizeof(count));
    f.write(reinterpret_cast<const char*>(data.data()),
            static_cast<std::streamsize>(count * sizeof(float)));
    return static_cast<bool>(f);
}

PureNet::PureNet(double lr) : adam_(lr) {
    init_params();
    cuda_backend_ = std::make_unique<CudaNetworkBackend>(*this, lr);
    cuda_enabled_ = cuda_backend_->available();
    if (cuda_enabled_) {
        cuda_backend_->sync_from_cpu();
        std::fprintf(stderr, "[net] CUDA backend enabled\n");
    } else {
        std::fprintf(stderr, "[net] CUDA unavailable, using CPU backend\n");
    }
}

PureNet::~PureNet() = default;

void PureNet::init_params() {
    // layer sizes are fixed by the layer structs; just ensure buffers exist
    auto ensure = [](std::vector<float>& w, size_t n) { w.resize(n); };
    ensure(c1_.w, 32 * 4 * 9);  ensure(c1_.b, 32);
    ensure(c2_.w, 64 * 32 * 9); ensure(c2_.b, 64);
    ensure(c3_.w, 128 * 64 * 9); ensure(c3_.b, 128);
    ensure(act_c_.w, 4 * 128);  ensure(act_c_.b, 4);
    ensure(act_fc_.w, 225 * 900); ensure(act_fc_.b, 225);
    ensure(val_c_.w, 2 * 128);  ensure(val_c_.b, 2);
    ensure(val_fc1_.w, 64 * 450); ensure(val_fc1_.b, 64);
    ensure(val_fc2_.w, 1 * 64);  ensure(val_fc2_.b, 1);

    std::mt19937 rng(static_cast<uint32_t>(
        std::chrono::steady_clock::now().time_since_epoch().count()));
    auto init_weights = [&rng](std::vector<float>& weights, int fan_in,
                               int fan_out) {
        const float bound = std::sqrt(6.0f / (fan_in + fan_out));
        std::uniform_real_distribution<float> distribution(-bound, bound);
        for (float& weight : weights) weight = distribution(rng);
    };
    init_weights(c1_.w, 4 * 9, 32 * 9);
    init_weights(c2_.w, 32 * 9, 64 * 9);
    init_weights(c3_.w, 64 * 9, 128 * 9);
    init_weights(act_c_.w, 128, 4);
    init_weights(act_fc_.w, 900, 225);
    init_weights(val_c_.w, 128, 2);
    init_weights(val_fc1_.w, 450, 64);
    init_weights(val_fc2_.w, 64, 1);
    std::fill(c1_.b.begin(), c1_.b.end(), 0.0f);
    std::fill(c2_.b.begin(), c2_.b.end(), 0.0f);
    std::fill(c3_.b.begin(), c3_.b.end(), 0.0f);
    std::fill(act_c_.b.begin(), act_c_.b.end(), 0.0f);
    std::fill(act_fc_.b.begin(), act_fc_.b.end(), 0.0f);
    std::fill(val_c_.b.begin(), val_c_.b.end(), 0.0f);
    std::fill(val_fc1_.b.begin(), val_fc1_.b.end(), 0.0f);
    std::fill(val_fc2_.b.begin(), val_fc2_.b.end(), 0.0f);
    loaded_ = false;
    rebuild_quantized_weights();
    std::fprintf(stderr, "[net] init: c1.w=%zu c2.w=%zu c3.w=%zu act_fc.w=%zu\n",
                 c1_.w.size(), c2_.w.size(), c3_.w.size(), act_fc_.w.size());
}

bool PureNet::load(const std::string& dir) {
    struct Entry { const char* name; std::vector<float>* w; std::vector<float>* b; };
    Entry entries[] = {
        {"conv1", &c1_.w, &c1_.b}, {"conv2", &c2_.w, &c2_.b},
        {"conv3", &c3_.w, &c3_.b},
        {"act_conv1", &act_c_.w, &act_c_.b},
        {"act_fc1", &act_fc_.w, &act_fc_.b},
        {"val_conv1", &val_c_.w, &val_c_.b},
        {"val_fc1", &val_fc1_.w, &val_fc1_.b},
        {"val_fc2", &val_fc2_.w, &val_fc2_.b},
    };
    for (auto& e : entries) {
        std::string wn = dir + "/" + e.name + "_weight.bin";
        std::string bn = dir + "/" + e.name + "_bias.bin";
        if (!load_bin_file(wn, *e.w, e.w->size())) return false;
        if (!load_bin_file(bn, *e.b, e.b->size())) return false;
    }
    loaded_ = true;
    rebuild_quantized_weights();
    if (cuda_enabled_) cuda_backend_->sync_from_cpu();
    std::fprintf(stderr, "[net] loaded: c1.w=%zu act_fc.w=%zu\n", c1_.w.size(), act_fc_.w.size());
    return true;
}

void PureNet::rebuild_quantized_weights() {
#if (defined(__GNUC__) || defined(__clang__)) && defined(__x86_64__)
    quantize_conv3x3_weights(c2_, quantized_c2_, quantized_c2_scales_);
    quantize_conv3x3_weights(c3_, quantized_c3_, quantized_c3_scales_);
#endif
}

bool PureNet::save(const std::string& dir) const {
    if (cuda_enabled_) return cuda_backend_->save(dir);
    struct Entry { const char* name; const std::vector<float>* w; const std::vector<float>* b; };
    Entry entries[] = {
        {"conv1", &c1_.w, &c1_.b}, {"conv2", &c2_.w, &c2_.b},
        {"conv3", &c3_.w, &c3_.b},
        {"act_conv1", &act_c_.w, &act_c_.b},
        {"act_fc1", &act_fc_.w, &act_fc_.b},
        {"val_conv1", &val_c_.w, &val_c_.b},
        {"val_fc1", &val_fc1_.w, &val_fc1_.b},
        {"val_fc2", &val_fc2_.w, &val_fc2_.b},
    };
    for (auto& e : entries) {
        std::string wn = dir + "/" + e.name + "_weight.bin";
        std::string bn = dir + "/" + e.name + "_bias.bin";
        if (!save_bin_file(wn, *e.w)) return false;
        if (!save_bin_file(bn, *e.b)) return false;
    }
    return true;
}

void PureNet::forward_impl(const float* state, float* policy,
                           float& value, bool probabilities) const {
    // Each calling thread owns one workspace. This removes all allocations
    // from the MCTS hot loop while keeping concurrent callers independent.
    struct Workspace {
        std::vector<float> c1, c2, c3, pa, plogit, va, vh, vlogit;
        std::vector<float> columns;
        std::vector<uint8_t> quantized_input, quantized_columns;
    };
    thread_local Workspace ws;

    const bool fast = fast_inference_enabled_ && has_fast_inference();
    const bool quantized = quantized_inference_enabled_ &&
                           has_quantized_inference();
    if (quantized) {
#if (defined(__GNUC__) || defined(__clang__)) && defined(__x86_64__)
        fast_conv3x3_relu(c1_, state, ws.c1, ws.columns);
        quantized_conv3x3_relu(c2_, quantized_c2_, quantized_c2_scales_,
                               ws.c1.data(), ws.c2, ws.quantized_input,
                               ws.quantized_columns);
        quantized_conv3x3_relu(c3_, quantized_c3_, quantized_c3_scales_,
                               ws.c2.data(), ws.c3, ws.quantized_input,
                               ws.quantized_columns);
#endif
    } else if (fast) {
#if (defined(__GNUC__) || defined(__clang__)) && defined(__x86_64__)
        fast_conv3x3_relu(c1_, state, ws.c1, ws.columns);
        fast_conv3x3_relu(c2_, ws.c1.data(), ws.c2, ws.columns);
        fast_conv3x3_relu(c3_, ws.c2.data(), ws.c3, ws.columns);
#endif
    } else {
        conv3x3_forward(c1_, state, ws.c1); relu_forward(ws.c1);
        conv3x3_forward(c2_, ws.c1, ws.c2); relu_forward(ws.c2);
        conv3x3_forward(c3_, ws.c2, ws.c3); relu_forward(ws.c3);
    }

    if (fast) {
#if (defined(__GNUC__) || defined(__clang__)) && defined(__x86_64__)
        fast_conv1x1_relu(act_c_, ws.c3, ws.pa);
        fast_fc(act_fc_, ws.pa, ws.plogit, false);
#endif
    } else {
        conv1x1_forward(act_c_, ws.c3, ws.pa); relu_forward(ws.pa);
        fc_forward(act_fc_, ws.pa, ws.plogit);
    }
    float mx = *std::max_element(ws.plogit.begin(), ws.plogit.end());
    float sum = 0.0f;
    if (probabilities) {
        for (int i = 0; i < 225; ++i) {
            policy[i] = std::exp(ws.plogit[i] - mx);
            sum += policy[i];
        }
        const float inverse_sum = 1.0f / sum;
        for (int i = 0; i < 225; ++i) policy[i] *= inverse_sum;
    } else {
        for (float v : ws.plogit) sum += std::exp(v - mx);
        const float lse = mx + std::log(sum);
        for (int i = 0; i < 225; ++i) policy[i] = ws.plogit[i] - lse;
    }

    if (fast) {
#if (defined(__GNUC__) || defined(__clang__)) && defined(__x86_64__)
        fast_conv1x1_relu(val_c_, ws.c3, ws.va);
        fast_fc(val_fc1_, ws.va, ws.vh, true);
        fast_fc(val_fc2_, ws.vh, ws.vlogit, false);
#endif
    } else {
        conv1x1_forward(val_c_, ws.c3, ws.va); relu_forward(ws.va);
        fc_forward(val_fc1_, ws.va, ws.vh); relu_forward(ws.vh);
        fc_forward(val_fc2_, ws.vh, ws.vlogit);
    }
    value = std::tanh(ws.vlogit[0]);
}

// ---- forward for one state ----
void PureNet::forward_one(const std::vector<float>& state,
                          std::vector<float>& log_policy, float& value) const {
    if (state.size() != 4 * 225)
        throw std::invalid_argument("state must contain 4*15*15 floats");
    log_policy.resize(225);
    forward_batch_impl(state.data(), 1, log_policy.data(), &value, false);
}

void PureNet::forward_batch(const float* states, int B,
                            float* log_policy, float* value) const {
    forward_batch_impl(states, B, log_policy, value, false);
}

void PureNet::forward_batch_policy(const float* states, int B,
                                   float* policy, float* value) const {
    forward_batch_impl(states, B, policy, value, true);
}

void PureNet::forward_batch_impl(const float* states, int B,
                                 float* policy, float* value,
                                 bool probabilities) const {
    if (B < 0) throw std::invalid_argument("batch size must not be negative");
    if (B == 0) return;
    if (cuda_enabled_) {
        cuda_backend_->forward(states, B, policy, value, probabilities);
        return;
    }
    const int threads = std::max(1, std::min(B, inference_threads_));
    if (threads > 1 && inference_pool_) {
        inference_pool_->forward(states, B, policy, value, probabilities);
    } else {
        for (int b = 0; b < B; ++b)
            forward_impl(states + static_cast<size_t>(b) * 4 * 225,
                         policy + static_cast<size_t>(b) * 225, value[b],
                         probabilities);
    }
}

void PureNet::set_inference_threads(int threads) {
    const int requested = std::max(1, threads);
    if (requested == inference_threads_) return;
    inference_pool_.reset();
    inference_threads_ = requested;
    if (!cuda_enabled_ && inference_threads_ > 1)
        inference_pool_ = std::make_unique<InferencePool>(*this,
                                                          inference_threads_);
}

bool PureNet::cuda_available() const {
    return cuda_backend_ && cuda_backend_->available();
}

void PureNet::set_cuda_enabled(bool enabled) {
    const bool requested = enabled && cuda_available();
    if (requested == cuda_enabled_) return;
    if (!requested && cuda_enabled_) {
        cuda_backend_->sync_to_cpu();
        cuda_enabled_ = false;
        if (inference_threads_ > 1)
            inference_pool_ = std::make_unique<InferencePool>(
                *this, inference_threads_);
        std::fprintf(stderr, "[net] switched to CPU backend\n");
        return;
    }
    inference_pool_.reset();
    cuda_backend_->sync_from_cpu();
    cuda_enabled_ = true;
    std::fprintf(stderr, "[net] switched to CUDA backend\n");
}

// ---- training step (batch) ----
PureNet::TrainStats PureNet::train_step(
    const std::vector<float>& states, int B,
    const std::vector<float>& probs, const std::vector<float>& winners,
    double lr) {
    const int HW = 225;
    if (B <= 0 || states.size() != static_cast<size_t>(B) * 4 * HW ||
        probs.size() != static_cast<size_t>(B) * HW ||
        winners.size() != static_cast<size_t>(B))
        throw std::invalid_argument("invalid training batch dimensions");
    if (cuda_enabled_)
        return cuda_backend_->train_step(states, B, probs, winners, lr);
    // --- forward per sample, accumulate loss + gradients ---
    // (network is tiny; per-sample forward/backward with gradient sum is
    //  simpler than batched tensor ops and equally correct for the loss)
    adam_.set_lr(lr);
    // zero gradients
    c1_.gw.assign(c1_.w.size(), 0); c1_.gb.assign(c1_.b.size(), 0);
    c2_.gw.assign(c2_.w.size(), 0); c2_.gb.assign(c2_.b.size(), 0);
    c3_.gw.assign(c3_.w.size(), 0); c3_.gb.assign(c3_.b.size(), 0);
    act_c_.gw.assign(act_c_.w.size(), 0); act_c_.gb.assign(act_c_.b.size(), 0);
    act_fc_.gw.assign(act_fc_.w.size(), 0); act_fc_.gb.assign(act_fc_.b.size(), 0);
    val_c_.gw.assign(val_c_.w.size(), 0); val_c_.gb.assign(val_c_.b.size(), 0);
    val_fc1_.gw.assign(val_fc1_.w.size(), 0); val_fc1_.gb.assign(val_fc1_.b.size(), 0);
    val_fc2_.gw.assign(val_fc2_.w.size(), 0); val_fc2_.gb.assign(val_fc2_.b.size(), 0);

    struct LocalTraining {
        ConvLayer c1, c2, c3, act_c, val_c;
        FCLayer act_fc, val_fc1, val_fc2;
        double loss = 0.0;
        double entropy = 0.0;
        double policy_loss = 0.0;
        double value_loss = 0.0;

        std::vector<float> in, c1_out, c2_out, c3_out;
        std::vector<float> pa, plogit, logp, va, vh, vlogit;
        std::vector<float> g_vh, g_va, g_c3_value, g_pa, g_c3_policy;
        std::vector<float> g_c3, g_c2, g_c1, g_input, gp;

        LocalTraining(const ConvLayer& source_c1,
                      const ConvLayer& source_c2,
                      const ConvLayer& source_c3,
                      const ConvLayer& source_act_c,
                      const FCLayer& source_act_fc,
                      const ConvLayer& source_val_c,
                      const FCLayer& source_val_fc1,
                      const FCLayer& source_val_fc2)
            : c1(source_c1), c2(source_c2), c3(source_c3),
              act_c(source_act_c), val_c(source_val_c),
              act_fc(source_act_fc), val_fc1(source_val_fc1),
              val_fc2(source_val_fc2),
              in(4 * HW), c1_out(32 * HW), c2_out(64 * HW),
              c3_out(128 * HW), pa(4 * HW), plogit(HW), logp(HW),
              va(2 * HW), vh(64), vlogit(1), gp(HW) {
            auto zero_conv = [](ConvLayer& layer) {
                layer.gw.assign(layer.w.size(), 0.0f);
                layer.gb.assign(layer.b.size(), 0.0f);
            };
            auto zero_fc = [](FCLayer& layer) {
                layer.gw.assign(layer.w.size(), 0.0f);
                layer.gb.assign(layer.b.size(), 0.0f);
            };
            zero_conv(c1); zero_conv(c2); zero_conv(c3);
            zero_conv(act_c); zero_fc(act_fc);
            zero_conv(val_c); zero_fc(val_fc1); zero_fc(val_fc2);
        }
    };

    const int training_threads = std::max(
        1, std::min(B, inference_threads_));
    std::vector<std::unique_ptr<LocalTraining>> local;
    local.reserve(training_threads);
    for (int worker = 0; worker < training_threads; ++worker) {
        local.push_back(std::make_unique<LocalTraining>(
            c1_, c2_, c3_, act_c_, act_fc_, val_c_, val_fc1_, val_fc2_));
    }

    auto train_partition = [&](int worker) {
        pin_inference_worker(worker);
        LocalTraining& work = *local[worker];
        for (int b = worker; b < B; b += training_threads) {
            std::memcpy(work.in.data(),
                        states.data() + static_cast<size_t>(b) * 4 * HW,
                        static_cast<size_t>(4 * HW) * sizeof(float));
            const float* target_policy =
                probs.data() + static_cast<size_t>(b) * HW;
            const float target_value = winners[b];

            conv3x3_forward(work.c1, work.in, work.c1_out);
            relu_forward(work.c1_out);
            conv3x3_forward(work.c2, work.c1_out, work.c2_out);
            relu_forward(work.c2_out);
            conv3x3_forward(work.c3, work.c2_out, work.c3_out);
            relu_forward(work.c3_out);
            conv1x1_forward(work.act_c, work.c3_out, work.pa);
            relu_forward(work.pa);
            fc_forward(work.act_fc, work.pa, work.plogit);

            const float maximum = *std::max_element(
                work.plogit.begin(), work.plogit.end());
            float exponential_sum = 0.0f;
            for (float logit : work.plogit)
                exponential_sum += std::exp(logit - maximum);
            const float log_sum_exp =
                maximum + std::log(exponential_sum);
            float policy_loss = 0.0f;
            for (int move = 0; move < HW; ++move) {
                work.logp[move] = work.plogit[move] - log_sum_exp;
                policy_loss -= target_policy[move] * work.logp[move];
            }

            conv1x1_forward(work.val_c, work.c3_out, work.va);
            relu_forward(work.va);
            fc_forward(work.val_fc1, work.va, work.vh);
            relu_forward(work.vh);
            fc_forward(work.val_fc2, work.vh, work.vlogit);
            const float value = std::tanh(work.vlogit[0]);
            const float value_loss =
                (value - target_value) * (value - target_value);
            work.loss += policy_loss + value_loss;
            work.policy_loss += policy_loss;
            work.value_loss += value_loss;

            float entropy = 0.0f;
            for (int move = 0; move < HW; ++move)
                entropy -= std::exp(work.logp[move]) * work.logp[move];
            work.entropy += entropy;

            const std::vector<float> value_gradient{
                2.0f * (value - target_value) * (1.0f - value * value)};
            fc_backward(work.val_fc2, work.vh, value_gradient, work.g_vh);
            relu_backward(work.vh, work.g_vh);
            fc_backward(work.val_fc1, work.va, work.g_vh, work.g_va);
            relu_backward(work.va, work.g_va);
            conv1x1_backward(work.val_c, work.c3_out, work.g_va,
                             work.g_c3_value);

            for (int move = 0; move < HW; ++move) {
                work.gp[move] =
                    std::exp(work.logp[move]) - target_policy[move];
            }
            fc_backward(work.act_fc, work.pa, work.gp, work.g_pa);
            relu_backward(work.pa, work.g_pa);
            conv1x1_backward(work.act_c, work.c3_out, work.g_pa,
                             work.g_c3_policy);

            work.g_c3.resize(work.g_c3_policy.size());
            for (size_t i = 0; i < work.g_c3.size(); ++i)
                work.g_c3[i] =
                    work.g_c3_policy[i] + work.g_c3_value[i];
            relu_backward(work.c3_out, work.g_c3);
            conv3x3_backward(work.c3, work.c2_out, work.g_c3, work.g_c2);
            relu_backward(work.c2_out, work.g_c2);
            conv3x3_backward(work.c2, work.c1_out, work.g_c2, work.g_c1);
            relu_backward(work.c1_out, work.g_c1);
            conv3x3_backward(work.c1, work.in, work.g_c1, work.g_input);
        }
    };

    std::vector<std::thread> training_workers;
    training_workers.reserve(training_threads - 1);
    for (int worker = 1; worker < training_threads; ++worker)
        training_workers.emplace_back(train_partition, worker);
    train_partition(0);
    for (std::thread& worker : training_workers) worker.join();

    double loss_sum = 0.0;
    double entropy_sum = 0.0;
    double policy_loss_sum = 0.0;
    double value_loss_sum = 0.0;
    for (const auto& work : local) {
        loss_sum += work->loss;
        entropy_sum += work->entropy;
        policy_loss_sum += work->policy_loss;
        value_loss_sum += work->value_loss;
    }
    float total_loss = static_cast<float>(loss_sum);
    float total_entropy = static_cast<float>(entropy_sum);
    float total_ploss = static_cast<float>(policy_loss_sum);
    float total_vloss = static_cast<float>(value_loss_sum);

    auto reduce = [&local](std::vector<float>& destination,
                           const auto& select) {
        std::fill(destination.begin(), destination.end(), 0.0f);
        for (const auto& work : local) {
            const std::vector<float>& source = select(*work);
            for (size_t i = 0; i < destination.size(); ++i)
                destination[i] += source[i];
        }
    };
    reduce(c1_.gw, [](const LocalTraining& w) -> const auto& {
        return w.c1.gw;
    });
    reduce(c1_.gb, [](const LocalTraining& w) -> const auto& {
        return w.c1.gb;
    });
    reduce(c2_.gw, [](const LocalTraining& w) -> const auto& {
        return w.c2.gw;
    });
    reduce(c2_.gb, [](const LocalTraining& w) -> const auto& {
        return w.c2.gb;
    });
    reduce(c3_.gw, [](const LocalTraining& w) -> const auto& {
        return w.c3.gw;
    });
    reduce(c3_.gb, [](const LocalTraining& w) -> const auto& {
        return w.c3.gb;
    });
    reduce(act_c_.gw, [](const LocalTraining& w) -> const auto& {
        return w.act_c.gw;
    });
    reduce(act_c_.gb, [](const LocalTraining& w) -> const auto& {
        return w.act_c.gb;
    });
    reduce(act_fc_.gw, [](const LocalTraining& w) -> const auto& {
        return w.act_fc.gw;
    });
    reduce(act_fc_.gb, [](const LocalTraining& w) -> const auto& {
        return w.act_fc.gb;
    });
    reduce(val_c_.gw, [](const LocalTraining& w) -> const auto& {
        return w.val_c.gw;
    });
    reduce(val_c_.gb, [](const LocalTraining& w) -> const auto& {
        return w.val_c.gb;
    });
    reduce(val_fc1_.gw, [](const LocalTraining& w) -> const auto& {
        return w.val_fc1.gw;
    });
    reduce(val_fc1_.gb, [](const LocalTraining& w) -> const auto& {
        return w.val_fc1.gb;
    });
    reduce(val_fc2_.gw, [](const LocalTraining& w) -> const auto& {
        return w.val_fc2.gw;
    });
    reduce(val_fc2_.gb, [](const LocalTraining& w) -> const auto& {
        return w.val_fc2.gb;
    });
     // average losses
    // average losses
    total_loss /= B; total_ploss /= B; total_vloss /= B; total_entropy /= B;

    // average gradients
    float invB = 1.0f / B;
    auto scale = [invB](std::vector<float>& g) {
        for (auto& v : g) v *= invB;
    };
    scale(c1_.gw); scale(c1_.gb); scale(c2_.gw); scale(c2_.gb);
    scale(c3_.gw); scale(c3_.gb); scale(act_c_.gw); scale(act_c_.gb);
    scale(act_fc_.gw); scale(act_fc_.gb); scale(val_c_.gw); scale(val_c_.gb);
    scale(val_fc1_.gw); scale(val_fc1_.gb); scale(val_fc2_.gw); scale(val_fc2_.gb);

    // collect all (param, grad) pairs into Adam (registered once)
    if (!adam_registered_) {
        register_all_params();
        adam_registered_ = true;
    }
    adam_.clip_grad_norm(10.0f);
    adam_.step();
    rebuild_quantized_weights();

    return {total_loss, total_entropy, total_ploss, total_vloss};
}

void PureNet::register_all_params() {
    auto reg = [this](std::vector<float>& w, std::vector<float>& gw,
                      std::vector<float>& b, std::vector<float>& gb) {
        adam_.add_param(w); adam_.add_grad(gw);
        adam_.add_param(b); adam_.add_grad(gb);
    };
    reg(c1_.w, c1_.gw, c1_.b, c1_.gb);
    reg(c2_.w, c2_.gw, c2_.b, c2_.gb);
    reg(c3_.w, c3_.gw, c3_.b, c3_.gb);
    reg(act_c_.w, act_c_.gw, act_c_.b, act_c_.gb);
    reg(act_fc_.w, act_fc_.gw, act_fc_.b, act_fc_.gb);
    reg(val_c_.w, val_c_.gw, val_c_.b, val_c_.gb);
    reg(val_fc1_.w, val_fc1_.gw, val_fc1_.b, val_fc1_.gb);
    reg(val_fc2_.w, val_fc2_.gw, val_fc2_.b, val_fc2_.gb);
}

}  // namespace gomoku
