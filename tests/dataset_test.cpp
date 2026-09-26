#include "dataset.h"
#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>
#include <unistd.h>

using namespace gomoku;
namespace {
void require(bool ok, const char* message) { if (!ok) throw std::runtime_error(message); }
template<class F> void rejected(F f, const char* message) {
    bool caught = false;
    try { f(); } catch (const std::exception&) { caught = true; }
    require(caught, message);
}
struct Fixture {
    std::filesystem::path root;
    std::vector<float> states, policies, values;
    std::vector<uint64_t> groups;
    Fixture() {
        char temporary[] = "/tmp/gomoku-dataset-XXXXXX";
        const char* result = ::mkdtemp(temporary);
        if (!result) throw std::runtime_error("mkdtemp failed");
        root = result;
    }
    ~Fixture() { std::filesystem::remove_all(root); }
    std::string prefix() const { return (root / "data").string(); }
    template<class T> void write(const std::string& suffix, const std::vector<T>& array) {
        std::ofstream file(prefix()+suffix, std::ios::binary);
        const uint64_t count = array.size();
        file.write(reinterpret_cast<const char*>(&count), sizeof(count));
        file.write(reinterpret_cast<const char*>(array.data()), array.size()*sizeof(T));
    }
    void save(bool with_groups = true) {
        write("_states.bin", states); write("_probs.bin", policies); write("_winners.bin", values);
        if (with_groups) write("_game_ids.bin", groups);
    }
    void metadata(const std::string& json) { std::ofstream(prefix()+"_meta.json") << json; }
    void add(const Board& board, int target, uint64_t group) {
        size_t offset = states.size(); states.resize(offset+4*BOARD_CELLS);
        board.encode_state(states.data()+offset);
        offset = policies.size(); policies.resize(offset+BOARD_CELLS, 0);
        policies[offset+target] = 1;
        values.push_back(0); groups.push_back(group);
    }
    void games() {
        const std::array<std::array<int, 8>, 3> moves{{
            {{112, 113, 127, 100, 145, 90, 42, 199}},
            {{112, 97, 129, 110, 43, 165, 32, 203}},
            {{112, 95, 170, 34, 46, 87, 26, 198}}}};
        for (size_t game = 0; game < moves.size(); ++game) {
            Board b;
            for (int move : moves[game]) { add(b, move, game+17); b.play(move); }
        }
    }
};
void split_and_augmentation() {
    Fixture f; f.games(); f.save();
    TrainingData data; data.load(f.prefix(), 0.34, 17, false);
    require(data.groups() == 3 && data.size() == 24, "dataset counts");
    require(data.value_weight() == 0 && !data.metadata_present(), "legacy value default must be zero");
    require(data.purged_validation() >= 2, "shared openings must be removed from held-out positions");
    // Golden cross-language split: splitmix64(18 ^ 17) sorts ahead of 17/19.
    for (size_t v : data.validation_indices()) require(f.groups[v] == 18, "cross-language split changed");
    for (size_t t : data.train_indices()) for (size_t v : data.validation_indices()) {
        require(f.groups[t] != f.groups[v], "same game leaks into train and validation");
        require(canonical_state_hash(data.state(t)) != canonical_state_hash(data.state(v)), "D4-equivalent position leakage");
    }
    TrainingData again; again.load(f.prefix(), 0.34, 17, false);
    require(again.validation_indices() == data.validation_indices(), "seeded split is not repeatable");
    std::array<float, 4*BOARD_CELLS> s{};
    std::array<float, BOARD_CELLS> p{};
    for (int symmetry = 0; symmetry < 8; ++symmetry) {
        transform_sample(data.state(7), data.policy(7), s.data(), p.data(), symmetry, 0.1f);
        require(canonical_state_hash(s.data()) == canonical_state_hash(data.state(7)), "D4 hash invariant");
        double total = 0;
        for (int c = 0; c < BOARD_CELLS; ++c) {
            total += p[c];
            if (s[c] || s[BOARD_CELLS+c]) require(p[c] == 0, "smoothing gave illegal move probability");
        }
        require(std::abs(total-1) < 1e-6, "smoothing mass is not one");
    }
}
void metadata_validation() {
    Fixture f; f.games(); f.save();
    f.metadata(R"({"value_weight":0.25,"target_rules":"freestyle","unknown":{"array":[true,null,"\u4e94\u5b50\u68cb"]}})");
    TrainingData good; good.load(f.prefix(), 0, 1, false);
    require(good.metadata_present() && good.value_weight() == 0.25f, "metadata value weight not loaded");
    const std::vector<std::string> bad = {
        R"({"target_rules":"renju"})", R"({"value_weight":1.5})", R"({"value_weight":1})",
        R"({"value_weight":"0"})", R"({"value_weight":NaN})", R"({"value_weight":01})",
        R"({"target_rules":"freestyle","value_supervision":"disabled","value_weight":1})",
        R"({"target_rules":"freestyle","target_rules":"renju"})", R"({"other":[1,]})",
        "{\"other\":\"unterminated}"
    };
    for (const auto& json : bad) {
        f.metadata(json);
        rejected([&] { TrainingData d; d.load(f.prefix(), 0, 1, false); }, "bad metadata accepted");
    }
}
void corrupt_and_terminal_data() {
    Fixture f; f.games(); f.save();
    f.policies[BOARD_CELLS+112] = 1; // position 1 already occupies the center.
    f.write("_probs.bin", f.policies);
    rejected([&] { TrainingData d; d.load(f.prefix(), 0, 1, false); }, "illegal policy accepted");
    f.policies[BOARD_CELLS+112] = 0; f.save();
    f.values[1] = 1; f.write("_winners.bin", f.values);
    rejected([&] { TrainingData d; d.load(f.prefix(), 0, 1, false); }, "inconsistent game outcomes accepted");
    f.values[1] = 0; f.save();
    std::filesystem::resize_file(f.prefix()+"_states.bin", 9);
    rejected([&] { TrainingData d; d.load(f.prefix(), 0, 1, false); }, "truncated tensor accepted");
    Fixture terminal; Board b;
    for (int move : {0,30,1,32,2,34,3,36,4}) b.play(move);
    terminal.add(b, 100, 1); terminal.save();
    rejected([&] { TrainingData d; d.load(terminal.prefix(), 0, 1, false); }, "terminal policy sample accepted");
}
void legacy_groups_and_full_overlap() {
    Fixture f; f.games(); f.save(false);
    rejected([&] { TrainingData d; d.load(f.prefix(), .2, 1, false); }, "missing game IDs accepted implicitly");
    TrainingData inferred; inferred.load(f.prefix(), .2, 1, true);
    require(inferred.inferred_groups() && inferred.groups() == 3, "legacy grouping from empty board");
    Fixture same; Board b;
    for (int g = 0; g < 2; ++g) same.add(b, 112, g);
    same.save();
    rejected([&] { TrainingData d; d.load(same.prefix(), .5, 1, false); }, "fully overlapping holdout accepted");
}
}
int main() {
    try {
        split_and_augmentation(); metadata_validation(); corrupt_and_terminal_data(); legacy_groups_and_full_overlap();
        std::cout << "dataset tests passed\n";
        return 0;
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
