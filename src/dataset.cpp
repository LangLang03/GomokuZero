#include "dataset.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <unordered_set>
#include <unordered_map>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace gomoku {
namespace {
// A small strict JSON reader avoids a heavyweight runtime dependency for four
// metadata scalars. Unknown arrays/objects are still parsed, never regex-skipped.
struct JsonValue {
    enum Kind { Null, Boolean, Number, String, Composite } kind = Null;
    std::string text;
};
class MetadataReader {
public:
    explicit MetadataReader(std::string source) : source_(std::move(source)) {}
    std::map<std::string, JsonValue> read() {
        std::map<std::string, JsonValue> fields;
        expect('{');
        if (!consume('}')) {
            do {
                const auto key = string();
                expect(':');
                if (!fields.emplace(key, value(1)).second) fail();
            } while (consume(','));
            expect('}');
        }
        whitespace();
        if (pos_ != source_.size()) fail();
        return fields;
    }
private:
    std::string source_;
    size_t pos_ = 0;
    [[noreturn]] void fail() const { throw std::runtime_error("invalid dataset metadata JSON"); }
    void whitespace() {
        while (pos_ < source_.size() && (source_[pos_] == ' ' || source_[pos_] == '\n' ||
               source_[pos_] == '\r' || source_[pos_] == '\t')) ++pos_;
    }
    bool consume(char c) {
        whitespace();
        if (pos_ < source_.size() && source_[pos_] == c) { ++pos_; return true; }
        return false;
    }
    void expect(char c) { if (!consume(c)) fail(); }
    unsigned hex4() {
        unsigned code = 0;
        for (int i = 0; i < 4; ++i) {
            if (pos_ == source_.size()) fail();
            char c = source_[pos_++];
            int v = c >= '0' && c <= '9' ? c-'0' : c >= 'a' && c <= 'f' ? c-'a'+10 :
                    c >= 'A' && c <= 'F' ? c-'A'+10 : -1;
            if (v < 0) fail();
            code = code * 16 + v;
        }
        return code;
    }
    std::string string() {
        expect('"');
        std::string out;
        while (pos_ < source_.size()) {
            unsigned char c = source_[pos_++];
            if (c == '"') return out;
            if (c < 32) fail();
            if (c != '\\') { out += c; continue; }
            if (pos_ == source_.size()) fail();
            c = source_[pos_++];
            if (c == '"' || c == '\\' || c == '/') out += c;
            else if (c == 'b') out += '\b';
            else if (c == 'f') out += '\f';
            else if (c == 'n') out += '\n';
            else if (c == 'r') out += '\r';
            else if (c == 't') out += '\t';
            else if (c == 'u') {
                unsigned code = hex4();
                if (code >= 0xd800 && code <= 0xdbff) {
                    if (pos_ + 2 > source_.size() || source_[pos_++] != '\\' || source_[pos_++] != 'u') fail();
                    unsigned low = hex4();
                    if (low < 0xdc00 || low > 0xdfff) fail();
                    code = 0x10000 + ((code-0xd800)<<10) + low-0xdc00;
                } else if (code >= 0xdc00 && code <= 0xdfff) fail();
                if (code < 0x80) out += static_cast<char>(code);
                else {
                    if (code < 0x800) out += static_cast<char>(0xc0 | (code>>6));
                    else {
                        if (code < 0x10000) out += static_cast<char>(0xe0 | (code>>12));
                        else { out += static_cast<char>(0xf0 | (code>>18)); out += static_cast<char>(0x80 | ((code>>12)&63)); }
                        out += static_cast<char>(0x80 | ((code>>6)&63));
                    }
                    out += static_cast<char>(0x80 | (code&63));
                }
            } else fail();
        }
        fail();
    }
    JsonValue value(int depth) {
        if (depth > 64) fail();
        whitespace();
        if (pos_ == source_.size()) fail();
        if (source_[pos_] == '"') return {JsonValue::String, string()};
        if (consume('{')) {
            std::unordered_set<std::string> keys;
            if (!consume('}')) {
                do { if (!keys.insert(string()).second) fail(); expect(':'); value(depth+1); } while (consume(','));
                expect('}');
            }
            return {JsonValue::Composite, {}};
        }
        if (consume('[')) {
            if (!consume(']')) { do { value(depth+1); } while (consume(',')); expect(']'); }
            return {JsonValue::Composite, {}};
        }
        for (const auto* literal : {"null", "true", "false"}) {
            if (source_.compare(pos_, std::strlen(literal), literal) == 0) {
                pos_ += std::strlen(literal);
                return {literal[0] == 'n' ? JsonValue::Null : JsonValue::Boolean, literal};
            }
        }
        const size_t begin = pos_;
        auto digit = [&] { return pos_ < source_.size() && source_[pos_] >= '0' && source_[pos_] <= '9'; };
        if (source_[pos_] == '-') ++pos_;
        if (!digit()) fail();
        if (source_[pos_] == '0') ++pos_;
        else while (digit()) ++pos_;
        if (pos_ < source_.size() && source_[pos_] == '.') { ++pos_; if (!digit()) fail(); while (digit()) ++pos_; }
        if (pos_ < source_.size() && (source_[pos_] == 'e' || source_[pos_] == 'E')) {
            ++pos_;
            if (pos_ < source_.size() && (source_[pos_] == '+' || source_[pos_] == '-')) ++pos_;
            if (!digit()) fail();
            while (digit()) ++pos_;
        }
        return {JsonValue::Number, source_.substr(begin, pos_-begin)};
    }
};

bool has_five(const float* s) {
    constexpr int directions[4][2] = {{0,1}, {1,0}, {1,1}, {1,-1}};
    for (int player = 0; player < 2; ++player) {
        const auto* plane = s + player * BOARD_CELLS;
        for (int r = 0; r < BOARD_SIZE; ++r) for (int c = 0; c < BOARD_SIZE; ++c) {
            if (!plane[r*BOARD_SIZE+c]) continue;
            for (const auto& d : directions) {
                int er = r + 4*d[0], ec = c + 4*d[1];
                if (er >= BOARD_SIZE || ec < 0 || ec >= BOARD_SIZE) continue;
                int length = 1;
                while (length < 5 && plane[(r+length*d[0])*BOARD_SIZE+c+length*d[1]]) ++length;
                if (length == 5) return true;
            }
        }
    }
    return false;
}
using Maps = std::array<std::array<int, BOARD_CELLS>, 8>;
const Maps& maps() {
    static const Maps result = [] {
        Maps m{};
        for (int t = 0; t < 8; ++t) {
            for (int i = 0; i < BOARD_CELLS; ++i) {
                int r = i / BOARD_SIZE, c = i % BOARD_SIZE;
                for (int k = 0; k < (t & 3); ++k) {
                    int old_r = r;
                    r = c;
                    c = BOARD_SIZE - 1 - old_r;
                }
                if (t & 4) c = BOARD_SIZE - 1 - c;
                m[t][i] = r * BOARD_SIZE + c;
            }
        }
        return m;
    }();
    return result;
}
uint64_t mix64(uint64_t x) {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}
int validate_record(const float* s, const float* p, float z, size_t index) {
    auto fail = [index](const char* message) {
        throw std::runtime_error("dataset record " + std::to_string(index) + ": " + message);
    };
    int own = 0, other = 0, last = 0;
    double mass = 0;
    for (int i = 0; i < 4 * BOARD_CELLS; ++i)
        if (s[i] != 0.0f && s[i] != 1.0f) fail("state planes must be binary and finite");
    for (int i = 0; i < BOARD_CELLS; ++i) {
        own += static_cast<int>(s[i]);
        other += static_cast<int>(s[BOARD_CELLS + i]);
        last += static_cast<int>(s[2 * BOARD_CELLS + i]);
        if (s[i] && s[BOARD_CELLS + i]) fail("overlapping stones");
        if (s[2 * BOARD_CELLS + i] && !s[BOARD_CELLS + i]) fail("last move must belong to opponent");
        if (s[3 * BOARD_CELLS + i] != s[3 * BOARD_CELLS]) fail("side-to-move plane is not constant");
        if (!std::isfinite(p[i]) || p[i] < 0) fail("negative or non-finite policy");
        if ((s[i] || s[BOARD_CELLS + i]) && p[i] > 1e-7f) fail("policy targets an occupied cell");
        mass += p[i];
    }
    if ((s[3 * BOARD_CELLS] && own != other) ||
        (!s[3 * BOARD_CELLS] && other != own + 1)) fail("stone counts disagree with side to move");
    if (last != (own + other > 0 ? 1 : 0)) fail("invalid last-move plane");
    if (std::abs(mass - 1.0) > 1e-3) fail("policy mass is not one");
    if (!std::isfinite(z) || z < -1 || z > 1) fail("outcome outside [-1,1]");
    if (has_five(s)) fail("policy sample is already terminal under freestyle rules");
    return own + other;
}
bool consecutive(const float* prev, const float* cur) {
    // After one legal move the relative player planes swap, and the new
    // opponent plane gains exactly the cell marked as the last move.
    if (prev[3 * BOARD_CELLS] == cur[3 * BOARD_CELLS]) return false;
    for (int c = 0; c < BOARD_CELLS; ++c) {
        if (cur[c] != prev[BOARD_CELLS + c]) return false;
        if (cur[BOARD_CELLS + c] != prev[c] + cur[2 * BOARD_CELLS + c]) return false;
    }
    return true;
}
}  // namespace

int validate_training_record(const float* state, const float* policy, float outcome, size_t index) {
    return validate_record(state, policy, outcome, index);
}

MappedArray::~MappedArray() {
    if (mapping_) ::munmap(mapping_, bytes_);
}
void MappedArray::open(const std::string& path, size_t element_bytes) {
    if (!element_bytes) throw std::invalid_argument("dataset element size must be positive");
    if (mapping_) throw std::logic_error("dataset array already open");
    const uint16_t endian = 1;
    if (*reinterpret_cast<const unsigned char*>(&endian) != 1)
        throw std::runtime_error("dataset requires little-endian host");
    int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) throw std::runtime_error("cannot open dataset: " + path);
    struct stat st{};
    uint64_t count = 0;
    const bool header_ok = ::fstat(fd, &st) == 0 && st.st_size >= 8 &&
                          ::pread(fd, &count, 8, 0) == 8;
    if (!header_ok || count > (std::numeric_limits<size_t>::max() - 8) / element_bytes ||
        static_cast<uint64_t>(st.st_size) != 8 + count * element_bytes) {
        ::close(fd);
        throw std::runtime_error("invalid dataset size/header: " + path);
    }
    bytes_ = static_cast<size_t>(st.st_size);
    mapping_ = ::mmap(nullptr, bytes_, PROT_READ, MAP_PRIVATE, fd, 0);
    ::close(fd);
    if (mapping_ == MAP_FAILED) {
        mapping_ = nullptr;
        throw std::runtime_error("cannot map dataset: " + path);
    }
    count_ = static_cast<size_t>(count);
}
const void* MappedArray::data() const {
    return static_cast<const char*>(mapping_) + 8;
}
const float* TrainingData::state(size_t i) const {
    return static_cast<const float*>(states_.data()) + i * 4 * BOARD_CELLS;
}
const float* TrainingData::policy(size_t i) const {
    return static_cast<const float*>(policies_.data()) + i * BOARD_CELLS;
}
float TrainingData::outcome(size_t i) const {
    return static_cast<const float*>(winners_.data())[i];
}

uint64_t canonical_state_hash(const float* s) {
    static const auto keys = [] {
        std::array<std::array<std::array<uint64_t, 8>, BOARD_CELLS>, 3> k{};
        for (int ch = 0; ch < 3; ++ch)
            for (int c = 0; c < BOARD_CELLS; ++c)
                for (int t = 0; t < 8; ++t)
                    k[ch][c][t] = mix64(1 + ch * BOARD_CELLS + maps()[t][c]);
        return k;
    }();
    std::array<uint64_t, 8> h{};
    h.fill(mix64(s[3 * BOARD_CELLS] ? 2001 : 2000));
    for (int ch = 0; ch < 3; ++ch)
        for (int c = 0; c < BOARD_CELLS; ++c)
            if (s[ch * BOARD_CELLS + c] != 0)
                for (int t = 0; t < 8; ++t) h[t] ^= keys[ch][c][t];
    return *std::min_element(h.begin(), h.end());
}

void TrainingData::load(const std::string& prefix, double val_fraction,
                        uint64_t seed, bool infer_legacy_groups) {
    if (!std::isfinite(val_fraction) || val_fraction < 0 || val_fraction >= 1)
        throw std::invalid_argument("validation fraction must be in [0,1)");
    const std::string metadata_path = prefix + "_meta.json";
    metadata_present_ = std::filesystem::exists(metadata_path);
    if (metadata_present_) {
        if (std::filesystem::file_size(metadata_path) > 1024*1024)
            throw std::runtime_error("dataset metadata exceeds 1 MiB");
        std::ifstream input(metadata_path);
        if (!input) throw std::runtime_error("cannot open dataset metadata");
        const auto fields = MetadataReader(std::string(std::istreambuf_iterator<char>(input), {})).read();
        auto rules = fields.find("target_rules");
        if (rules != fields.end() && (rules->second.kind != JsonValue::String || rules->second.text != "freestyle"))
            throw std::runtime_error("dataset target_rules must be freestyle");
        auto weight = fields.find("value_weight");
        if (weight != fields.end()) {
            if (weight->second.kind != JsonValue::Number) throw std::runtime_error("dataset value_weight must be numeric");
            double value = std::stod(weight->second.text);
            if (!std::isfinite(value) || value < 0 || value > 1)
                throw std::runtime_error("dataset value_weight must be in [0,1]");
            value_weight_ = static_cast<float>(value);
        }
        auto supervision = fields.find("value_supervision");
        if (supervision != fields.end() && supervision->second.kind != JsonValue::String)
            throw std::runtime_error("dataset value_supervision must be a string");
        if (supervision != fields.end() && supervision->second.text == "disabled" && value_weight_ > 0)
            throw std::runtime_error("disabled dataset value supervision conflicts with positive value_weight");
        if (value_weight_ > 0 && rules == fields.end())
            throw std::runtime_error("positive dataset value_weight requires explicit freestyle target_rules");
    }
    states_.open(prefix + "_states.bin", sizeof(float));
    policies_.open(prefix + "_probs.bin", sizeof(float));
    winners_.open(prefix + "_winners.bin", sizeof(float));
    const size_t n = size();
    if (!n || states_.size() / (4 * BOARD_CELLS) != n ||
        states_.size() % (4 * BOARD_CELLS) || policies_.size() / BOARD_CELLS != n ||
        policies_.size() % BOARD_CELLS)
        throw std::runtime_error("dataset shapes disagree or dataset is empty: " + prefix);
    const bool explicit_groups = std::filesystem::exists(prefix + "_game_ids.bin");
    if (explicit_groups) {
        game_ids_.open(prefix + "_game_ids.bin", sizeof(uint64_t));
        if (game_ids_.size() != n) throw std::runtime_error("game_ids count disagrees with samples");
    } else if (!infer_legacy_groups && val_fraction > 0) {
        throw std::runtime_error("game_ids missing; regenerate data or use --infer-game-groups 1 for ordered legacy records");
    }
    inferred_ = !explicit_groups;
    std::vector<uint64_t> ids(n), hashes(n);
    std::unordered_map<uint64_t, float> group_outcomes;
    uint64_t group = 0;
    int prev_count = -1;
    for (size_t i = 0; i < n; ++i) {
        const int stones = validate_training_record(state(i), policy(i), outcome(i), i);
        if (explicit_groups) ids[i] = static_cast<const uint64_t*>(game_ids_.data())[i];
        else {
            if (i == 0 && val_fraction > 0 && stones > 1)
                throw std::runtime_error("legacy data does not start at a game boundary; provide explicit game_ids");
            if (i && stones == prev_count + 1 && !consecutive(state(i - 1), state(i)))
                throw std::runtime_error("legacy samples are not consecutive; provide explicit game_ids");
            if (i && stones != prev_count + 1) {
                if (val_fraction > 0 && stones > 1)
                    throw std::runtime_error("cannot infer game boundary in legacy data; regenerate with game_ids");
                ++group;
            }
            ids[i] = group;
        }
        // Winner targets are from the player-to-move perspective. A game
        // therefore has one invariant result when expressed for Black.
        const float black_outcome = outcome(i) * (state(i)[3*BOARD_CELLS] ? 1.0f : -1.0f);
        const auto inserted = group_outcomes.emplace(ids[i], black_outcome);
        if (!inserted.second && std::abs(inserted.first->second-black_outcome) > 1e-5f)
            throw std::runtime_error("dataset outcomes disagree within game " + std::to_string(ids[i]) +
                                     "; winners must use the current-player perspective");
        prev_count = stones;
        hashes[i] = canonical_state_hash(state(i));
    }
    std::vector<uint64_t> groups = ids;
    std::sort(groups.begin(), groups.end());
    groups.erase(std::unique(groups.begin(), groups.end()), groups.end());
    groups_count_ = groups.size();
    if (val_fraction > 0 && groups.size() < 2)
        throw std::runtime_error("validation requires at least two games; use --val-fraction 0 only for diagnostics");
    // Fixed uint64 arithmetic produces the same split in Python and C++ and
    // across standard library implementations (unlike std::shuffle).
    std::sort(groups.begin(), groups.end(), [seed](uint64_t a, uint64_t b) {
        const uint64_t ka = mix64(a ^ seed), kb = mix64(b ^ seed);
        return ka == kb ? a < b : ka < kb;
    });
    const size_t nval_groups = val_fraction > 0 ?
        std::min(groups.size() - 1, std::max<size_t>(1, std::llround(groups.size() * val_fraction))) : 0;
    std::unordered_set<uint64_t> val_groups(groups.begin(), groups.begin() + nval_groups);
    std::unordered_set<uint64_t> seen_train;
    seen_train.reserve(n);
    train_.reserve(n);
    for (size_t i = 0; i < n; ++i) {
        if (!val_groups.count(ids[i])) {
            train_.push_back(i);
            seen_train.insert(hashes[i]);
        }
    }
    for (size_t i = 0; i < n; ++i) {
        if (!val_groups.count(ids[i])) continue;
        if (seen_train.count(hashes[i])) ++purged_;
        else validation_.push_back(i);
    }
    if (val_fraction > 0 && validation_.empty())
        throw std::runtime_error("all held-out positions overlap training after D4 symmetry; independent validation unavailable");
}

void transform_sample(const float* s, const float* p, float* os, float* op,
                      int symmetry, float smoothing) {
    if (symmetry < 0 || symmetry >= 8 || !std::isfinite(smoothing) || smoothing < 0 || smoothing >= 1)
        throw std::invalid_argument("invalid augmentation parameters");
    if (symmetry == 0) {
        std::memcpy(os, s, 4 * BOARD_CELLS * sizeof(float));
        std::memcpy(op, p, BOARD_CELLS * sizeof(float));
    } else {
        const auto& m = maps()[symmetry];
        for (int ch = 0; ch < 4; ++ch)
            for (int c = 0; c < BOARD_CELLS; ++c) os[ch * BOARD_CELLS + m[c]] = s[ch * BOARD_CELLS + c];
        for (int c = 0; c < BOARD_CELLS; ++c) op[m[c]] = p[c];
    }
    if (smoothing > 0) {
        int legal = 0;
        for (int c = 0; c < BOARD_CELLS; ++c) legal += !(os[c] || os[BOARD_CELLS + c]);
        if (!legal) throw std::runtime_error("no legal move in training sample");
        for (int c = 0; c < BOARD_CELLS; ++c)
            op[c] = (os[c] || os[BOARD_CELLS + c]) ? 0.0f :
                    (1 - smoothing) * op[c] + smoothing / legal;
    }
}
}  // namespace gomoku
