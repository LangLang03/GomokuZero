#pragma once
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>
#include "game.h"

namespace gomoku {

// Read-only [u64 count][elements] storage. The OS pages records in on demand;
// train/validation partitions contain indices, never a second copy of the data.
class MappedArray {
public:
    MappedArray() = default;
    ~MappedArray();
    MappedArray(const MappedArray&) = delete;
    MappedArray& operator=(const MappedArray&) = delete;
    void open(const std::string& path, size_t element_bytes);
    const void* data() const;
    size_t size() const { return count_; }
private:
    void* mapping_ = nullptr;
    size_t bytes_ = 0, count_ = 0;
};

class TrainingData {
public:
    void load(const std::string& prefix, double validation_fraction,
              uint64_t split_seed, bool infer_legacy_groups);
    const float* state(size_t i) const;
    const float* policy(size_t i) const;
    float outcome(size_t i) const;
    size_t size() const { return winners_.size(); }
    const std::vector<size_t>& train_indices() const { return train_; }
    const std::vector<size_t>& validation_indices() const { return validation_; }
    size_t groups() const { return groups_count_; }
    size_t purged_validation() const { return purged_; }
    bool inferred_groups() const { return inferred_; }
    // Missing metadata deliberately disables external value supervision.
    float value_weight() const { return value_weight_; }
    bool metadata_present() const { return metadata_present_; }
private:
    MappedArray states_, policies_, winners_, game_ids_;
    std::vector<size_t> train_, validation_;
    size_t groups_count_ = 0, purged_ = 0;
    bool inferred_ = false;
    float value_weight_ = 0.0f;
    bool metadata_present_ = false;
};

// Validate a [4][15][15] state, [225] policy and current-player outcome.
// Returns the stone count; malformed or already-terminal samples throw.
// Shared by external dataset loading and checkpoint replay restoration.
int validate_training_record(const float* state, const float* policy, float outcome,
                             size_t index = 0);

// The same D4 transform is applied to both the state and policy target.
void transform_sample(const float* state, const float* policy, float* out_state,
                      float* out_policy, int symmetry, float smoothing = 0.0f);
uint64_t canonical_state_hash(const float* state);

// Write a self-play dataset in the exact [u64 count][elements] layout the
// loader reads. Every record is re-validated, an existing prefix is refused,
// and the metadata enables value supervision because z is a real freestyle
// game outcome produced by search rather than a human record under other rules.
void write_selfplay_dataset(const std::string& prefix,
                            const std::vector<float>& states,
                            const std::vector<float>& policies,
                            const std::vector<float>& outcomes,
                            const std::vector<uint64_t>& game_ids,
                            const std::string& source);

}  // namespace gomoku
