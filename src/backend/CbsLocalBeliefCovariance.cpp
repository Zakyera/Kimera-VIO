#include "kimera-vio/backend/CbsLocalBeliefCovariance.h"

#include <cbs/bpsam/bpsam.h>
#include <glog/logging.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/nonlinear/LinearContainerFactor.h>

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <limits>
#include <map>
#include <optional>
#include <set>
#include <sstream>
#include <string>
#include <typeinfo>
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
  struct FactorErrorSummary {
    size_t count = 0u;
    size_t before_valid = 0u;
    size_t after_valid = 0u;
    double before_error = 0.0;
    double after_error = 0.0;
  };

  struct PoseMovementSummary {
    size_t pose_count = 0u;
    double translation_mean_m = 0.0;
    double translation_max_m = 0.0;
    double rotation_mean_deg = 0.0;
    double rotation_max_deg = 0.0;
  };

  struct CandidateAudit {
    bool valid = false;
    double timestamp = 0.0;
    size_t latest_frame = 0u;
    double objective_before = std::numeric_limits<double>::quiet_NaN();
    double objective_after = std::numeric_limits<double>::quiet_NaN();
    PoseMovementSummary movement;
    size_t variables_relinearized = 0u;
    size_t variables_reeliminated = 0u;
    std::map<std::string, FactorErrorSummary> factor_errors;
    gtsam::Values candidate_values;
  };

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

  static std::string factorAuditCategory(
      const gtsam::NonlinearFactor::shared_ptr& factor) {
    if (!factor) {
      return "null";
    }
    if (dynamic_cast<const gtsam::LinearContainerFactor*>(factor.get())) {
      return "marginal_prior";
    }

    const std::string type_name = typeid(*factor).name();
    if (type_name.find("SmartStereoProjection") != std::string::npos) {
      return "smart_stereo";
    }
    if (type_name.find("ImuFactor") != std::string::npos) {
      return "imu";
    }
    if (type_name.find("PriorFactor") != std::string::npos) {
      return "prior";
    }
    if (type_name.find("BetweenFactor") != std::string::npos) {
      bool has_pose = false;
      bool has_bias = false;
      for (const gtsam::Key key : factor->keys()) {
        const char symbol = gtsam::Symbol(key).chr();
        has_pose = has_pose || symbol == 'x';
        has_bias = has_bias || symbol == 'b';
      }
      if (has_bias) {
        return "bias_between";
      }
      if (has_pose) {
        return "pose_between";
      }
      return "between_other";
    }
    return "other";
  }

  static double factorErrorOrNan(
      const gtsam::NonlinearFactor::shared_ptr& factor,
      const gtsam::Values& values) {
    try {
      return factor ? factor->error(values)
                    : std::numeric_limits<double>::quiet_NaN();
    } catch (const std::exception&) {
      return std::numeric_limits<double>::quiet_NaN();
    }
  }

  static gtsam::Values initializedValues(
      const gtsam::Values& before,
      const gtsam::Values& new_values) {
    gtsam::Values initialized(before);
    for (const auto& key_value : new_values) {
      if (!initialized.exists(key_value.key)) {
        initialized.insert(key_value.key, key_value.value);
      }
    }
    return initialized;
  }

  static PoseMovementSummary summarizePoseMovement(
      const gtsam::Values& before,
      const gtsam::Values& after) {
    PoseMovementSummary summary;
    double translation_sum = 0.0;
    double rotation_sum_deg = 0.0;
    for (const gtsam::Key key : after.keys()) {
      if (gtsam::Symbol(key).chr() != 'x' || !before.exists(key)) {
        continue;
      }
      try {
        const gtsam::Pose3 before_pose = before.at<gtsam::Pose3>(key);
        const gtsam::Pose3 after_pose = after.at<gtsam::Pose3>(key);
        const gtsam::Pose3 delta = before_pose.between(after_pose);
        const double translation_m = delta.translation().norm();
        const double rotation_deg =
            gtsam::Rot3::Logmap(delta.rotation()).norm() * 180.0 / M_PI;
        ++summary.pose_count;
        translation_sum += translation_m;
        rotation_sum_deg += rotation_deg;
        summary.translation_max_m =
            std::max(summary.translation_max_m, translation_m);
        summary.rotation_max_deg =
            std::max(summary.rotation_max_deg, rotation_deg);
      } catch (const std::exception&) {
      }
    }
    if (summary.pose_count > 0u) {
      summary.translation_mean_m =
          translation_sum / static_cast<double>(summary.pose_count);
      summary.rotation_mean_deg =
          rotation_sum_deg / static_cast<double>(summary.pose_count);
    }
    return summary;
  }

  static CandidateAudit captureCandidateAudit(
      const double timestamp,
      const gtsam::NonlinearFactorGraph& graph,
      const gtsam::Values& before,
      const gtsam::Values& after,
      const gtsam::ISAM2Result& result) {
    CandidateAudit audit;
    audit.timestamp = timestamp;
    audit.candidate_values = after;
    audit.movement = summarizePoseMovement(before, after);
    audit.variables_relinearized = result.variablesRelinearized;
    audit.variables_reeliminated = result.variablesReeliminated;

    for (const gtsam::Key key : after.keys()) {
      const gtsam::Symbol symbol(key);
      if (symbol.chr() == 'x') {
        audit.latest_frame = std::max(audit.latest_frame, symbol.index());
      }
    }

    double before_total = 0.0;
    double after_total = 0.0;
    bool all_before_valid = true;
    bool all_after_valid = true;
    for (const auto& factor : graph) {
      if (!factor) {
        continue;
      }
      FactorErrorSummary& summary =
          audit.factor_errors[factorAuditCategory(factor)];
      ++summary.count;

      const double before_error = factorErrorOrNan(factor, before);
      if (std::isfinite(before_error)) {
        ++summary.before_valid;
        summary.before_error += before_error;
        before_total += before_error;
      } else {
        all_before_valid = false;
      }

      const double after_error = factorErrorOrNan(factor, after);
      if (std::isfinite(after_error)) {
        ++summary.after_valid;
        summary.after_error += after_error;
        after_total += after_error;
      } else {
        all_after_valid = false;
      }
    }

    audit.objective_before =
        all_before_valid ? before_total
                         : std::numeric_limits<double>::quiet_NaN();
    audit.objective_after =
        all_after_valid ? after_total
                        : std::numeric_limits<double>::quiet_NaN();
    audit.valid = all_before_valid && all_after_valid &&
                  std::isfinite(audit.objective_before) &&
                  std::isfinite(audit.objective_after);
    return audit;
  }

  static double objectiveRatio(const CandidateAudit& audit) {
    if (!audit.valid || std::abs(audit.objective_before) < 1e-15) {
      return std::numeric_limits<double>::quiet_NaN();
    }
    return audit.objective_after / audit.objective_before;
  }

  static void logCandidateFactorErrors(const char* candidate,
                                       const CandidateAudit& audit) {
    for (const auto& [category, summary] : audit.factor_errors) {
      LOG(INFO) << std::setprecision(17)
                << "KIMERA_SHADOW_ACCEPTANCE_FACTOR_ROW,"
                << audit.timestamp << "," << audit.latest_frame << ","
                << candidate << "," << category << "," << summary.count
                << "," << summary.before_valid << "," << summary.after_valid
                << "," << summary.before_error << "," << summary.after_error;
    }
  }

  static void logShadowAcceptanceAudit(const CandidateAudit& normal,
                                       const CandidateAudit& shadow,
                                       const std::string& status) {
    const PoseMovementSummary normal_vs_shadow =
        summarizePoseMovement(normal.candidate_values, shadow.candidate_values);
    LOG(INFO) << std::setprecision(17)
              << "KIMERA_SHADOW_ACCEPTANCE_ROW," << normal.timestamp << ","
              << normal.latest_frame << "," << status << ","
              << normal.valid << "," << normal.objective_before << ","
              << normal.objective_after << "," << objectiveRatio(normal) << ","
              << normal.movement.pose_count << ","
              << normal.movement.translation_mean_m << ","
              << normal.movement.translation_max_m << ","
              << normal.movement.rotation_mean_deg << ","
              << normal.movement.rotation_max_deg << ","
              << normal.variables_relinearized << ","
              << normal.variables_reeliminated << "," << shadow.valid << ","
              << shadow.objective_before << "," << shadow.objective_after << ","
              << objectiveRatio(shadow) << "," << shadow.movement.pose_count
              << "," << shadow.movement.translation_mean_m << ","
              << shadow.movement.translation_max_m << ","
              << shadow.movement.rotation_mean_deg << ","
              << shadow.movement.rotation_max_deg << ","
              << shadow.variables_relinearized << ","
              << shadow.variables_reeliminated << ","
              << normal_vs_shadow.pose_count << ","
              << normal_vs_shadow.translation_mean_m << ","
              << normal_vs_shadow.translation_max_m << ","
              << normal_vs_shadow.rotation_mean_deg << ","
              << normal_vs_shadow.rotation_max_deg;
    logCandidateFactorErrors("normal", normal);
    logCandidateFactorErrors("shadow_forced_full", shadow);
  }

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
              std::vector<size_t>* smart_factor_slots_out,
              bool force_full_solve,
              bool emit_diagnostics,
              CandidateAudit* candidate_audit_out) {
    std::optional<gtsam::Values> initialized_values;
    if (candidate_audit_out) {
      initialized_values =
          initializedValues(bpsam_.calculateEstimate(), new_values);
    }
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
    update_params.force_relinearize = force_full_solve;
    update_params.forceFullSolve = force_full_solve;

    const gtsam::ISAM2Result result =
        bpsam_.update(new_factors, new_values, update_params);
    last_update_result_ = result;
    if (candidate_audit_out && initialized_values) {
      const gtsam::NonlinearFactorGraph audit_graph =
          bpsam_.getFactorsUnsafe().clone();
      *candidate_audit_out = captureCandidateAudit(
          current_timestamp,
          audit_graph,
          *initialized_values,
          bpsam_.calculateEstimate(),
          result);
    }

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

    if (emit_diagnostics) {
      logDiagnostics(new_factors.size(),
                     update_params.removeFactorIndices.size(),
                     marginalizable_keys.size(),
                     marginalized_now_count,
                     stale_tracked_keys_after_marginalization,
                     frame_coverage);
    }
    return true;
  }

  void isolateForShadowAudit() {
    bpsam_.isolateNonlinearFactorsForAudit();
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
    std::vector<size_t>* smart_factor_slots_out,
    bool shadow_acceptance_audit_enable) {
  std::unique_ptr<Impl> shadow_impl;
  gtsam::NonlinearFactorGraph shadow_new_factors;
  std::string shadow_status = "disabled";
  if (shadow_acceptance_audit_enable) {
    try {
      shadow_impl = std::make_unique<Impl>(*impl_);
      shadow_impl->isolateForShadowAudit();
      shadow_new_factors = new_factors.clone();
      shadow_status = "prepared";
    } catch (const std::exception& e) {
      shadow_impl.reset();
      shadow_status = std::string("prepare_exception_") + e.what();
    } catch (...) {
      shadow_impl.reset();
      shadow_status = "prepare_exception_unknown";
    }
  }

  Impl::CandidateAudit normal_audit;
  try {
    const bool normal_ok = impl_->update(new_factors,
                                        new_values,
                                        timestamps,
                                        delete_slots,
                                        num_smart_factors,
                                        smart_factor_slots_out,
                                        false,
                                        true,
                                        shadow_acceptance_audit_enable
                                            ? &normal_audit
                                            : nullptr);
    if (!normal_ok || !shadow_acceptance_audit_enable) {
      return normal_ok;
    }
  } catch (...) {
    return false;
  }

  Impl::CandidateAudit shadow_audit;
  if (shadow_impl) {
    try {
      const bool shadow_ok = shadow_impl->update(shadow_new_factors,
                                                new_values,
                                                timestamps,
                                                delete_slots,
                                                num_smart_factors,
                                                nullptr,
                                                true,
                                                false,
                                                &shadow_audit);
      shadow_status = shadow_ok ? "ok" : "shadow_update_failed";
    } catch (const std::exception& e) {
      shadow_status = std::string("shadow_exception_") + e.what();
    } catch (...) {
      shadow_status = "shadow_exception_unknown";
    }
  }
  Impl::logShadowAcceptanceAudit(normal_audit, shadow_audit, shadow_status);
  return true;
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
