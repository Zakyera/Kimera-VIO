#include "kimera-vio/backend/CbsLocalBeliefCovariance.h"

#include <cbs/bpsam/bpsam.h>
#include <glog/logging.h>
#include <gtsam/inference/Symbol.h>

#include <algorithm>
#include <optional>
#include <set>
#include <utility>

namespace VIO {
namespace {

bool keyToFrameId(const gtsam::Key& key, size_t* frame_id) {
  const gtsam::Symbol symbol(key);
  const char chr = symbol.chr();
  if (chr == 'x' || chr == 'v' || chr == 'b') {
    *frame_id = symbol.index();
    return true;
  }
  return false;
}

void recursiveMarkAffectedKeys(
    const gtsam::Key& key,
    const gtsam::ISAM2Clique::shared_ptr& clique,
    std::set<gtsam::Key>* additional_keys) {
  if (!clique || !additional_keys) {
    return;
  }

  const auto& conditional = clique->conditional();
  if (std::find(
          conditional->beginParents(), conditional->endParents(), key) ==
      conditional->endParents()) {
    return;
  }

  for (const gtsam::Key frontal : conditional->frontals()) {
    additional_keys->insert(frontal);
  }

  for (const gtsam::ISAM2Clique::shared_ptr& child : clique->children) {
    recursiveMarkAffectedKeys(key, child, additional_keys);
  }
}

}  // namespace

std::optional<gtsam::Matrix> computePoseBeliefCovarianceWithBpsamSnapshot(
    const gtsam::NonlinearFactorGraph& local_graph,
    const gtsam::Values& local_values,
    gtsam::Key pose_key,
    const gtsam::ISAM2Params& isam2_params,
    char robot_id) {
  if (local_graph.empty() || !local_values.exists(pose_key)) {
    return std::nullopt;
  }

  cbs::BPSAM::Params bpsam_params;
  bpsam_params.robot_id = static_cast<cbs::AgentId>(robot_id);
  bpsam_params.sam_params_ = isam2_params;
  bpsam_params.enable_belief_dcs = false;

  try {
    cbs::BPSAM bpsam_snapshot(bpsam_params);
    cbs::BPSAM::UpdateParams update_params;
    bpsam_snapshot.update(local_graph, local_values, update_params);
    bpsam_snapshot.setMarginalizationGraph(
        cbs::BPSAM::MarginalizationType::LOCAL);
    const gtsam::Matrix pose_cov = bpsam_snapshot.marginalCovariance(pose_key);
    if (pose_cov.rows() != 6 || pose_cov.cols() != 6 || !pose_cov.allFinite()) {
      return std::nullopt;
    }
    return pose_cov;
  } catch (...) {
    return std::nullopt;
  }
}

struct PersistentBpsamLocalCovarianceSidecar::Impl {
  struct StateFrameCoverage {
    size_t frames_total = 0u;
    size_t complete_frames = 0u;
    size_t incomplete_frames = 0u;
    size_t oldest_frame = 0u;
    size_t newest_frame = 0u;
    bool has_any = false;
  };

  explicit Impl(const gtsam::ISAM2Params& isam2_params,
                char robot_id,
                size_t max_window_size)
      : robot_id_(robot_id),
        max_window_size_(std::max<size_t>(1u, max_window_size)),
        bpsam_([&]() {
          cbs::BPSAM::Params params;
          params.robot_id = static_cast<cbs::AgentId>(robot_id);
          params.sam_params_ = isam2_params;
          params.enable_belief_dcs = false;
          params.belief_similarity_threshold = 0.0;
          return params;
        }()) {}

  void logDiagnostics(size_t new_factor_count,
                      size_t removed_factor_slots_count,
                      size_t marginalizable_candidates_count,
                      size_t marginalized_now_count,
                      size_t stale_tracked_keys_after_marginalization,
                      const StateFrameCoverage& frame_coverage) {
    ++update_count_;
    const size_t active_key_count = bpsam_.getVariableIndex().size();
    const size_t tracked_key_count = key_timestamp_map_.size();

    max_active_key_count_ = std::max(max_active_key_count_, active_key_count);
    max_tracked_key_count_ = std::max(max_tracked_key_count_, tracked_key_count);

    const bool should_log_periodic = (update_count_ % kDiagLogPeriod) == 0u;
    const bool should_log_on_marginalization = marginalized_now_count > 0u;
    if (!should_log_periodic && !should_log_on_marginalization) {
      return;
    }

    LOG(INFO) << "Persistent BPSAM sidecar [" << robot_id_
              << "] update=" << update_count_
              << " active_keys=" << active_key_count
              << " tracked_keys=" << tracked_key_count
              << " max_active_keys=" << max_active_key_count_
              << " max_tracked_keys=" << max_tracked_key_count_
              << " new_factors=" << new_factor_count
              << " removed_slots=" << removed_factor_slots_count
              << " marginalizable_candidates=" << marginalizable_candidates_count
              << " marginalized_now=" << marginalized_now_count
              << " stale_after=" << stale_tracked_keys_after_marginalization
              << " frames_total=" << frame_coverage.frames_total
              << " frames_complete=" << frame_coverage.complete_frames
              << " frames_incomplete=" << frame_coverage.incomplete_frames
              << " frame_oldest="
              << (frame_coverage.has_any
                      ? static_cast<long>(frame_coverage.oldest_frame)
                      : -1L)
              << " frame_newest="
              << (frame_coverage.has_any
                      ? static_cast<long>(frame_coverage.newest_frame)
                      : -1L);

    if (stale_tracked_keys_after_marginalization > 0u ||
        frame_coverage.incomplete_frames > 0u) {
      LOG(WARNING) << "Persistent BPSAM sidecar [" << robot_id_
                   << "] fixed-lag integrity warning: stale_after="
                   << stale_tracked_keys_after_marginalization
                   << " incomplete_frames=" << frame_coverage.incomplete_frames;
    }
  }

  gtsam::FactorIndices filterExistingFactorSlots(
      const gtsam::FactorIndices& slots) const {
    gtsam::FactorIndices filtered;
    filtered.reserve(slots.size());
    for (const size_t slot : slots) {
      if (!bpsam_.getFactor(slot)) {
        continue;
      }
      filtered.push_back(slot);
    }
    std::sort(filtered.begin(), filtered.end());
    filtered.erase(std::unique(filtered.begin(), filtered.end()), filtered.end());
    return filtered;
  }

  void eraseTimestampKeyMapEntry(double timestamp, gtsam::Key key) {
    const auto range = timestamp_key_map_.equal_range(timestamp);
    for (auto it = range.first; it != range.second; ++it) {
      if (it->second == key) {
        timestamp_key_map_.erase(it);
        return;
      }
    }
  }

  void updateKeyTimestampMap(const std::map<gtsam::Key, double>& timestamps) {
    for (const auto& [key, timestamp] : timestamps) {
      size_t frame_id = 0u;
      if (!keyToFrameId(key, &frame_id)) {
        continue;
      }
      (void)frame_id;

      const auto key_it = key_timestamp_map_.find(key);
      if (key_it != key_timestamp_map_.end()) {
        eraseTimestampKeyMapEntry(key_it->second, key);
      }
      key_timestamp_map_[key] = timestamp;
      timestamp_key_map_.emplace(timestamp, key);
    }
  }

  void eraseKeyTimestampMap(const gtsam::KeyVector& keys) {
    for (const gtsam::Key key : keys) {
      const auto key_it = key_timestamp_map_.find(key);
      if (key_it == key_timestamp_map_.end()) {
        continue;
      }
      eraseTimestampKeyMapEntry(key_it->second, key);
      key_timestamp_map_.erase(key_it);
    }
  }

  void pruneMissingKeysFromTimestampMap() {
    gtsam::KeyVector keys_to_drop;
    keys_to_drop.reserve(key_timestamp_map_.size());
    for (const auto& [key, _] : key_timestamp_map_) {
      if (!bpsam_.valueExists(key)) {
        keys_to_drop.push_back(key);
      }
    }
    eraseKeyTimestampMap(keys_to_drop);
  }

  double getCurrentTimestamp() const {
    if (timestamp_key_map_.empty()) {
      return 0.0;
    }
    return timestamp_key_map_.rbegin()->first;
  }

  gtsam::KeyVector findKeysBefore(double timestamp) const {
    gtsam::KeyVector keys;
    const auto end_it = timestamp_key_map_.lower_bound(timestamp);
    for (auto it = timestamp_key_map_.begin(); it != end_it; ++it) {
      keys.push_back(it->second);
    }
    std::sort(keys.begin(), keys.end());
    keys.erase(std::unique(keys.begin(), keys.end()), keys.end());
    return keys;
  }

  size_t countTrackedKeysBefore(double timestamp) const {
    const auto end_it = timestamp_key_map_.lower_bound(timestamp);
    return static_cast<size_t>(std::distance(timestamp_key_map_.begin(), end_it));
  }

  StateFrameCoverage computeStateFrameCoverage() const {
    std::map<size_t, unsigned int> frame_masks;
    frame_masks.clear();

    for (const auto& [key, _] : key_timestamp_map_) {
      const gtsam::Symbol symbol(key);
      unsigned int bit = 0u;
      switch (symbol.chr()) {
        case 'x':
          bit = 1u;
          break;
        case 'v':
          bit = 2u;
          break;
        case 'b':
          bit = 4u;
          break;
        default:
          continue;
      }
      frame_masks[symbol.index()] |= bit;
    }

    StateFrameCoverage coverage;
    coverage.frames_total = frame_masks.size();
    if (frame_masks.empty()) {
      return coverage;
    }

    coverage.has_any = true;
    coverage.oldest_frame = frame_masks.begin()->first;
    coverage.newest_frame = frame_masks.rbegin()->first;
    for (const auto& [_, mask] : frame_masks) {
      if (mask == 7u) {
        ++coverage.complete_frames;
      } else {
        ++coverage.incomplete_frames;
      }
    }
    return coverage;
  }

  std::optional<gtsam::FastMap<gtsam::Key, int>> createOrderingConstraints(
      const gtsam::KeyVector& marginalizable_keys) const {
    if (marginalizable_keys.empty()) {
      return std::nullopt;
    }

    gtsam::FastMap<gtsam::Key, int> constrained_keys;
    for (const auto& [key, _] : key_timestamp_map_) {
      constrained_keys[key] = 1;
    }
    for (const gtsam::Key key : marginalizable_keys) {
      constrained_keys[key] = 0;
    }
    return constrained_keys;
  }

  std::optional<gtsam::FastList<gtsam::Key>> createAdditionalMarkedKeys(
      const gtsam::KeyVector& marginalizable_keys) const {
    if (marginalizable_keys.empty()) {
      return std::nullopt;
    }

    std::set<gtsam::Key> additional_keys;
    for (const gtsam::Key key : marginalizable_keys) {
      if (!bpsam_.valueExists(key)) {
        continue;
      }
      const gtsam::ISAM2Clique::shared_ptr clique = bpsam_[key];
      if (!clique) {
        continue;
      }
      for (const gtsam::ISAM2Clique::shared_ptr& child : clique->children) {
        recursiveMarkAffectedKeys(key, child, &additional_keys);
      }
    }

    if (additional_keys.empty()) {
      return std::nullopt;
    }
    return gtsam::FastList<gtsam::Key>(additional_keys.begin(),
                                       additional_keys.end());
  }

  bool update(const gtsam::NonlinearFactorGraph& new_factors,
              const gtsam::Values& new_values,
              const std::map<gtsam::Key, double>& timestamps,
              const gtsam::FactorIndices& delete_slots,
              size_t num_smart_factors,
              std::vector<size_t>* smart_factor_slots_out) {
    updateKeyTimestampMap(timestamps);

    const double current_timestamp = getCurrentTimestamp();
    const double keep_from_timestamp =
        current_timestamp - static_cast<double>(max_window_size_) + 1.0;
    const gtsam::KeyVector marginalizable_keys =
        findKeysBefore(keep_from_timestamp);

    cbs::BPSAM::UpdateParams update_params;
    update_params.removeFactorIndices = filterExistingFactorSlots(delete_slots);

    std::vector<cbs::BPSAM::ActiveWindowCbsOdomFactorRemoval>
        active_window_cbs_removals;
    if (bpsam_.usesActiveWindowTemporaryCbsOdomFactors()) {
      gtsam::KeySet active_keys_after_update;
      for (const auto& [key, timestamp] : key_timestamp_map_) {
        if (timestamp >= keep_from_timestamp) {
          active_keys_after_update.insert(key);
        }
      }
      for (const gtsam::Key key : marginalizable_keys) {
        active_keys_after_update.erase(key);
      }

      active_window_cbs_removals =
          bpsam_.activeWindowTemporaryCbsOdomFactorRemovalsOutsideKeys(
              active_keys_after_update);
      const gtsam::FactorIndices cbs_slots_to_remove =
          bpsam_.activeWindowTemporaryCbsOdomFactorRemovalSlots(
              active_window_cbs_removals);
      update_params.removeFactorIndices.insert(
          update_params.removeFactorIndices.end(),
          cbs_slots_to_remove.begin(),
          cbs_slots_to_remove.end());
      update_params.removeFactorIndices =
          filterExistingFactorSlots(update_params.removeFactorIndices);

      if (!active_window_cbs_removals.empty()) {
        LOG(INFO) << "Persistent BPSAM sidecar [" << robot_id_
                  << "] scheduling "
                  << active_window_cbs_removals.size()
                  << " active-window CBS odometry removals before "
                     "marginalization.";
      }
    }

    const auto constrained_keys = createOrderingConstraints(marginalizable_keys);
    if (constrained_keys) {
      update_params.constrainedKeys = constrained_keys;
    }

    const auto additional_marked_keys =
        createAdditionalMarkedKeys(marginalizable_keys);
    if (additional_marked_keys) {
      update_params.extraReelimKeys = additional_marked_keys;
    }

    const gtsam::ISAM2Result result =
        bpsam_.update(new_factors, new_values, update_params);
    last_update_result_ = result;

    if (!active_window_cbs_removals.empty()) {
      bpsam_.commitActiveWindowTemporaryCbsOdomFactorRemovals(
          active_window_cbs_removals);
    }

    if (smart_factor_slots_out) {
      smart_factor_slots_out->clear();
      const size_t n_smart =
          std::min(num_smart_factors, result.newFactorsIndices.size());
      smart_factor_slots_out->reserve(n_smart);
      for (size_t i = 0u; i < n_smart; ++i) {
        smart_factor_slots_out->push_back(result.newFactorsIndices.at(i));
      }
    }

    pruneMissingKeysFromTimestampMap();

    size_t marginalized_now_count = 0u;
    if (!marginalizable_keys.empty()) {
      gtsam::FastList<gtsam::Key> leaf_keys;
      for (const gtsam::Key key : marginalizable_keys) {
        if (bpsam_.valueExists(key)) {
          leaf_keys.push_back(key);
        }
      }
      if (!leaf_keys.empty()) {
        bpsam_.marginalizeLeaves(leaf_keys);
        marginalized_now_count = leaf_keys.size();
        eraseKeyTimestampMap(
            gtsam::KeyVector(leaf_keys.begin(), leaf_keys.end()));
        bpsam_.pruneTrackedCbsFactors();
      }
    }

    const size_t stale_tracked_keys_after_marginalization =
        countTrackedKeysBefore(keep_from_timestamp);
    const StateFrameCoverage frame_coverage = computeStateFrameCoverage();

    logDiagnostics(new_factors.size(),
                   update_params.removeFactorIndices.size(),
                   marginalizable_keys.size(),
                   marginalized_now_count,
                   stale_tracked_keys_after_marginalization,
                   frame_coverage);
    return true;
  }

  std::optional<gtsam::Matrix> computePoseCovariance(gtsam::Key pose_key) {
    const gtsam::Values values = bpsam_.calculateEstimate();
    if (!values.exists(pose_key)) {
      return std::nullopt;
    }

    bpsam_.setMarginalizationGraph(cbs::BPSAM::MarginalizationType::LOCAL);
    const gtsam::Matrix pose_cov = bpsam_.marginalCovariance(pose_key);
    if (pose_cov.rows() != 6 || pose_cov.cols() != 6 || !pose_cov.allFinite()) {
      return std::nullopt;
    }
    return pose_cov;
  }

  gtsam::Values calculateEstimate() const { return bpsam_.calculateEstimate(); }

  const gtsam::NonlinearFactorGraph& factors() const {
    return bpsam_.getFactorsUnsafe();
  }

  bool factorExists(size_t slot) const { return static_cast<bool>(bpsam_.getFactor(slot)); }

  gtsam::NonlinearFactor::shared_ptr factorAt(size_t slot) const {
    return bpsam_.getFactor(slot);
  }

  const gtsam::ISAM2Result& lastUpdateResult() const {
    return last_update_result_;
  }

  static constexpr size_t kDiagLogPeriod = 25u;

  char robot_id_ = 'k';
  size_t max_window_size_ = 1u;
  size_t update_count_ = 0u;
  size_t max_active_key_count_ = 0u;
  size_t max_tracked_key_count_ = 0u;
  cbs::BPSAM bpsam_;
  gtsam::ISAM2Result last_update_result_;
  std::map<gtsam::Key, double> key_timestamp_map_;
  std::multimap<double, gtsam::Key> timestamp_key_map_;
};

PersistentBpsamLocalCovarianceSidecar::PersistentBpsamLocalCovarianceSidecar(
    const gtsam::ISAM2Params& isam2_params,
    char robot_id,
    size_t max_window_size)
    : impl_(std::make_unique<Impl>(isam2_params, robot_id, max_window_size)) {}

PersistentBpsamLocalCovarianceSidecar::~PersistentBpsamLocalCovarianceSidecar() =
    default;

bool PersistentBpsamLocalCovarianceSidecar::update(
    const gtsam::NonlinearFactorGraph& new_factors,
    const gtsam::Values& new_values,
    const std::map<gtsam::Key, double>& timestamps,
    const gtsam::FactorIndices& delete_slots,
    size_t num_smart_factors,
    std::vector<size_t>* smart_factor_slots_out) {
  try {
    return impl_->update(new_factors,
                         new_values,
                         timestamps,
                         delete_slots,
                         num_smart_factors,
                         smart_factor_slots_out);
  } catch (...) {
    return false;
  }
}

std::optional<gtsam::Matrix> PersistentBpsamLocalCovarianceSidecar::
    computePoseCovariance(gtsam::Key pose_key) {
  try {
    return impl_->computePoseCovariance(pose_key);
  } catch (...) {
    return std::nullopt;
  }
}

gtsam::Values PersistentBpsamLocalCovarianceSidecar::calculateEstimate() const {
  return impl_->calculateEstimate();
}

const gtsam::NonlinearFactorGraph& PersistentBpsamLocalCovarianceSidecar::
    factors() const {
  return impl_->factors();
}

bool PersistentBpsamLocalCovarianceSidecar::factorExists(size_t slot) const {
  return impl_->factorExists(slot);
}

gtsam::NonlinearFactor::shared_ptr PersistentBpsamLocalCovarianceSidecar::
    factorAt(size_t slot) const {
  return impl_->factorAt(slot);
}

const gtsam::ISAM2Result& PersistentBpsamLocalCovarianceSidecar::
    lastUpdateResult() const {
  return impl_->lastUpdateResult();
}

}  // namespace VIO
