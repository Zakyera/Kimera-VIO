#include "kimera-vio/backend/CbsFixedLagBpsamHeart.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <queue>
#include <sstream>
#include <unordered_set>

#include <gtsam/inference/LabeledSymbol.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/nonlinear/LinearContainerFactor.h>
#include <gtsam/linear/NoiseModel.h>
#include <gtsam/navigation/ImuBias.h>
#include <gtsam/slam/PriorFactor.h>

namespace VIO {

#ifdef KIMERA_USE_CBS

namespace {

inline FrameId frameIdFromKey(const gtsam::Key key) {
  return static_cast<FrameId>(gtsam::Symbol(key).index());
}

inline bool factorExists(const gtsam::NonlinearFactorGraph& graph,
                         const gtsam::FactorIndex slot) {
  return slot < graph.size() && graph.exists(slot) && static_cast<bool>(graph.at(slot));
}

inline bool extractSymmetricCovarianceBlock(const gtsam::Matrix& raw,
                                            const int dim,
                                            const double min_var,
                                            gtsam::Matrix* out) {
  if (!out || raw.rows() < dim || raw.cols() < dim || !raw.allFinite()) {
    return false;
  }
  gtsam::Matrix cov = raw.block(0, 0, dim, dim);
  cov = 0.5 * (cov + cov.transpose());
  for (int i = 0; i < dim; ++i) {
    if (!std::isfinite(cov(i, i)) || cov(i, i) < min_var) {
      cov(i, i) = min_var;
    }
  }
  *out = cov;
  return true;
}

template <typename ClockT = std::chrono::steady_clock>
double elapsedMs(const typename ClockT::time_point& start,
                 const typename ClockT::time_point& end) {
  return std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(
             end - start)
      .count();
}

}  // namespace

size_t CbsFixedLagBpsamHeart::LocalFrameOwnedKeys::totalPoseKeys() const {
  size_t total = 0u;
  for (const auto& kv : pose_keys_by_frame) {
    total += kv.second.size();
  }
  return total;
}

size_t CbsFixedLagBpsamHeart::LocalFrameOwnedKeys::totalVelKeys() const {
  size_t total = 0u;
  for (const auto& kv : vel_keys_by_frame) {
    total += kv.second.size();
  }
  return total;
}

size_t CbsFixedLagBpsamHeart::LocalFrameOwnedKeys::totalBiasKeys() const {
  size_t total = 0u;
  for (const auto& kv : bias_keys_by_frame) {
    total += kv.second.size();
  }
  return total;
}

CbsFixedLagBpsamHeart::CbsFixedLagBpsamHeart(cbs::BPSAM::Ptr bpsam,
                                             const size_t lag_states)
    : bpsam_(std::move(bpsam)), lag_states_(lag_states) {
  resetIncrementalLagState();
}

void CbsFixedLagBpsamHeart::setBpsam(cbs::BPSAM::Ptr bpsam) {
  if (bpsam_ != bpsam) {
    resetIncrementalLagState();
  }
  bpsam_ = std::move(bpsam);
  belief_ownership_dirty_ = true;
}

void CbsFixedLagBpsamHeart::setLagStates(const size_t lag_states) {
  lag_states_ = lag_states;
}

void CbsFixedLagBpsamHeart::resetIncrementalLagState() {
  incremental_key_timestamps_.clear();
  incremental_local_ownership_ = LocalFrameOwnedKeys{};
  incremental_belief_ownership_ = BeliefOwnedState{};
  incremental_newest_local_frame_id_ = 0u;
  incremental_oldest_active_frame_id_ = 0u;
  incremental_boundary_candidates_ = BoundarySeparatorCandidateSet{};
  incremental_summary_insertion_ = SummaryInsertionPlan{};
  incremental_plan_cache_ = LagWindowPlan{};
  pending_prune_plan_ = LagWindowPlan{};
  last_incremental_diagnostics_ = Diagnostics{};
  incremental_plan_valid_ = false;
  pending_prune_plan_valid_ = false;
  incremental_state_seeded_from_full_recompute_ = false;
  belief_ownership_dirty_ = true;
}

void CbsFixedLagBpsamHeart::ingestLocalKimeraPacket(
    const LocalIngestPacket& packet) {
  if (!bpsam_) {
    return;
  }

  incremental_newest_local_frame_id_ =
      std::max(incremental_newest_local_frame_id_, packet.newest_local_frame_id);

  for (const auto& kv : packet.local_key_timestamps) {
    incremental_key_timestamps_[kv.first] = kv.second;
    const gtsam::Symbol symbol(kv.first);
    incremental_newest_local_frame_id_ = std::max(
        incremental_newest_local_frame_id_,
        static_cast<FrameId>(symbol.index()));
  }

  for (const auto& kv : packet.local_values) {
    const Key key = kv.key;
    if (!isLocalStateKey(key)) {
      continue;
    }
    const FrameId frame_id = frameIdFromKey(key);
    incremental_local_ownership_.all_local_state_keys.insert(key);
    const gtsam::Symbol symbol(key);
    if (symbol.chr() == kPoseSymbolChar) {
      incremental_local_ownership_.pose_keys_by_frame[frame_id].insert(key);
    } else if (symbol.chr() == kVelocitySymbolChar) {
      incremental_local_ownership_.vel_keys_by_frame[frame_id].insert(key);
    } else if (symbol.chr() == kImuBiasSymbolChar) {
      incremental_local_ownership_.bias_keys_by_frame[frame_id].insert(key);
    }
    if (incremental_key_timestamps_.count(key) == 0u) {
      incremental_key_timestamps_[key] = static_cast<double>(frame_id);
    }
    incremental_newest_local_frame_id_ =
        std::max(incremental_newest_local_frame_id_, frame_id);
  }

  last_local_ownership_ = incremental_local_ownership_;
  incremental_plan_valid_ = false;
}

size_t CbsFixedLagBpsamHeart::ingestIncomingBeliefs(
    const std::map<Key, std::map<cbs::AgentId, gbp::Gaussian>>& beliefs) {
  if (!bpsam_ || beliefs.empty()) {
    return 0u;
  }
  // BPSAM::addBeliefs routes through beliefToFactor(...), which currently
  // materializes CBS belief exchange with BetweenFactor<Pose3> semantics.
  const size_t inserted = static_cast<size_t>(bpsam_->addBeliefs(beliefs));
  if (inserted > 0u) {
    belief_ownership_dirty_ = true;
    incremental_plan_valid_ = false;
  }
  return inserted;
}

const CbsFixedLagBpsamHeart::LagWindowPlan&
CbsFixedLagBpsamHeart::advanceLagWindowIncremental(
    const FrameId fallback_newest_frame_id) {
  if (!bpsam_) {
    incremental_plan_cache_ = LagWindowPlan{};
    incremental_plan_valid_ = false;
    pending_prune_plan_ = LagWindowPlan{};
    pending_prune_plan_valid_ = false;
    last_incremental_diagnostics_ = Diagnostics{};
    return incremental_plan_cache_;
  }

  const auto t0 = std::chrono::steady_clock::now();
  bool used_full_recompute_fallback = false;

  if (!incremental_state_seeded_from_full_recompute_) {
    // One-time safety seed from active graph, then continue incrementally.
    incremental_plan_cache_ =
        recomputeLagWindow(incremental_key_timestamps_, fallback_newest_frame_id);
    incremental_local_ownership_ = incremental_plan_cache_.local_ownership;
    incremental_belief_ownership_ = incremental_plan_cache_.belief_ownership;
    incremental_boundary_candidates_ = incremental_plan_cache_.boundary_candidates;
    incremental_summary_insertion_ =
        incremental_plan_cache_.future_summary_insertion;
    incremental_oldest_active_frame_id_ =
        incremental_plan_cache_.stale_eviction.oldest_active_frame_id;
    incremental_newest_local_frame_id_ = std::max(
        incremental_newest_local_frame_id_,
        incremental_plan_cache_.stale_eviction.newest_frame_id);
    incremental_state_seeded_from_full_recompute_ = true;
    belief_ownership_dirty_ = false;
    used_full_recompute_fallback = true;
  } else {
    refreshBeliefOwnershipFromActiveGraphIfNeeded(false);

    LagWindowPlan plan;
    plan.local_ownership = incremental_local_ownership_;
    plan.belief_ownership = incremental_belief_ownership_;
    plan.stale_eviction = computeStaleFrameEviction(
        incremental_key_timestamps_,
        std::max(fallback_newest_frame_id, incremental_newest_local_frame_id_),
        plan.local_ownership);
    plan.orphan_prune = computeOrphanPrune(
        plan.stale_eviction, plan.belief_ownership, plan.local_ownership);
    plan.boundary_candidates =
        computeBoundaryCandidates(plan.stale_eviction, plan.orphan_prune);
    plan.future_summary_insertion = buildFutureSummaryInsertionPlan(
        plan.boundary_candidates, plan.stale_eviction);
    plan.remove_factor_indices =
        buildRemoveFactorIndices(plan.stale_eviction, plan.orphan_prune);

    incremental_plan_cache_ = std::move(plan);
    incremental_oldest_active_frame_id_ =
        incremental_plan_cache_.stale_eviction.oldest_active_frame_id;
    incremental_boundary_candidates_ =
        incremental_plan_cache_.boundary_candidates;
    incremental_summary_insertion_ =
        incremental_plan_cache_.future_summary_insertion;
  }

  incremental_plan_valid_ = true;
  pending_prune_plan_ = incremental_plan_cache_;
  pending_prune_plan_valid_ = true;
  last_local_ownership_ = incremental_local_ownership_;
  last_belief_ownership_ = incremental_belief_ownership_;
  const auto t1 = std::chrono::steady_clock::now();
  const double lag_ms = elapsedMs(t0, t1);
  updateIncrementalDiagnosticsFromPlan(incremental_plan_cache_,
                                       used_full_recompute_fallback,
                                       lag_ms);

  return incremental_plan_cache_;
}

const CbsFixedLagBpsamHeart::LagWindowPlan&
CbsFixedLagBpsamHeart::getIncrementalLagPlan() const {
  return incremental_plan_cache_;
}

const gtsam::FactorIndices&
CbsFixedLagBpsamHeart::getIncrementalRemoveFactorIndices() const {
  return incremental_plan_cache_.remove_factor_indices;
}

const CbsFixedLagBpsamHeart::Diagnostics&
CbsFixedLagBpsamHeart::getLastIncrementalDiagnostics() const {
  return last_incremental_diagnostics_;
}

bool CbsFixedLagBpsamHeart::hasPendingPrunePlan() const {
  return pending_prune_plan_valid_;
}

bool CbsFixedLagBpsamHeart::commitPendingPrunePlan() {
  if (!pending_prune_plan_valid_) {
    return false;
  }
  pruneIncrementalStateWithPlan(pending_prune_plan_);
  pending_prune_plan_ = LagWindowPlan{};
  pending_prune_plan_valid_ = false;
  last_local_ownership_ = incremental_local_ownership_;
  last_belief_ownership_ = incremental_belief_ownership_;
  return true;
}

void CbsFixedLagBpsamHeart::clearPendingPrunePlan() {
  pending_prune_plan_ = LagWindowPlan{};
  pending_prune_plan_valid_ = false;
}

CbsFixedLagBpsamHeart::LagWindowPlan CbsFixedLagBpsamHeart::recomputeLagWindow(
    const KeyTimestampMap& timestamps,
    const FrameId fallback_newest_frame_id) const {
  LagWindowPlan plan;
  if (!bpsam_) {
    return plan;
  }

  plan.local_ownership = buildLocalOwnershipFromVariableIndex();
  plan.belief_ownership = buildBeliefOwnershipFromActiveGraph();
  plan.stale_eviction =
      computeStaleFrameEviction(timestamps, fallback_newest_frame_id, plan.local_ownership);
  plan.orphan_prune =
      computeOrphanPrune(plan.stale_eviction, plan.belief_ownership, plan.local_ownership);
  plan.boundary_candidates =
      computeBoundaryCandidates(plan.stale_eviction, plan.orphan_prune);
  plan.future_summary_insertion = buildFutureSummaryInsertionPlan(
      plan.boundary_candidates, plan.stale_eviction);
  plan.remove_factor_indices =
      buildRemoveFactorIndices(plan.stale_eviction, plan.orphan_prune);
  return plan;
}

CbsFixedLagBpsamHeart::Diagnostics CbsFixedLagBpsamHeart::makeDiagnostics(
    const LagWindowPlan& plan) const {
  Diagnostics d;

  d.stale_local_key_count = plan.stale_eviction.stale_local_keys.size();
  d.stale_belief_factor_count =
      plan.stale_eviction.stale_belief_factor_slots.size();
  d.orphan_key_count = plan.orphan_prune.orphan_robot_keys.size() +
                       plan.orphan_prune.orphan_gbp_keys.size() +
                       plan.orphan_prune.orphan_consensus_keys.size();
  d.boundary_candidate_count = plan.boundary_candidates.separator_keys.size();

  const size_t active_local =
      plan.local_ownership.all_local_state_keys.size() > d.stale_local_key_count
          ? plan.local_ownership.all_local_state_keys.size() -
                d.stale_local_key_count
          : 0u;
  d.active_local_key_count = active_local;

  const size_t total_belief_factors =
      plan.belief_ownership.belief_factor_by_slot.size();
  const size_t removed_belief_factors =
      plan.stale_eviction.stale_belief_factor_slots.size() +
      plan.orphan_prune.orphan_belief_factor_slots.size();
  d.active_belief_factor_count =
      total_belief_factors > removed_belief_factors
          ? total_belief_factors - removed_belief_factors
          : 0u;

  const size_t total_robot_gbp =
      plan.belief_ownership.robot_keys.size() + plan.belief_ownership.gbp_keys.size();
  const size_t orphan_robot_gbp =
      plan.orphan_prune.orphan_robot_keys.size() +
      plan.orphan_prune.orphan_gbp_keys.size();
  d.active_robot_gbp_key_count =
      total_robot_gbp > orphan_robot_gbp ? total_robot_gbp - orphan_robot_gbp : 0u;

  return d;
}

CbsFixedLagBpsamHeart::BoundarySummaryInput
CbsFixedLagBpsamHeart::buildBoundarySummaryInput(
    const LagWindowPlan& plan) const {
  BoundarySummaryInput input;
  input.newest_frame_id = plan.stale_eviction.newest_frame_id;
  input.oldest_active_frame_id = plan.stale_eviction.oldest_active_frame_id;
  input.separator_keys = plan.boundary_candidates.separator_keys;
  input.local_separator_keys = plan.boundary_candidates.local_separator_keys;
  input.belief_separator_keys = plan.boundary_candidates.belief_separator_keys;
  input.crossing_factor_slots = plan.boundary_candidates.crossing_factor_slots;
  input.summary_target_keys = plan.future_summary_insertion.summary_target_keys;
  input.has_candidates = !input.summary_target_keys.empty();
  return input;
}

CbsFixedLagBpsamHeart::BoundarySummaryInput
CbsFixedLagBpsamHeart::buildBoundarySummaryInputFromIncrementalState() const {
  if (incremental_plan_valid_) {
    return buildBoundarySummaryInput(incremental_plan_cache_);
  }
  return BoundarySummaryInput{};
}

bool CbsFixedLagBpsamHeart::appendTemporarySummaryPriors(
    const LagWindowPlan& plan,
    gtsam::NonlinearFactorGraph* out_factors,
    SummaryPriorBridgeStats* stats) const {
  if (stats) {
    *stats = SummaryPriorBridgeStats{};
  }
  auto noteFailure = [&](const std::string& reason) {
    if (stats && stats->first_failure_reason.empty()) {
      stats->first_failure_reason = reason;
    }
  };

  if (!out_factors) {
    noteFailure("null_output_factor_graph");
    return false;
  }
  if (!bpsam_) {
    noteFailure("bpsam_not_set");
    return false;
  }

  gtsam::Values estimate;
  try {
    estimate = bpsam_->calculateEstimate();
  } catch (const std::exception& e) {
    noteFailure(std::string("calculateEstimate_failed: ") + e.what());
    return false;
  }

  std::vector<Key> targets(plan.future_summary_insertion.summary_target_keys.begin(),
                           plan.future_summary_insertion.summary_target_keys.end());
  std::sort(targets.begin(), targets.end());
  size_t emitted_total = 0u;

  for (const Key key : targets) {
    if (stats) {
      ++stats->candidate_keys_total;
    }
    if (!isLocalStateKey(key)) {
      if (stats) {
        ++stats->skipped_non_local_keys;
      }
      continue;
    }
    if (stats) {
      ++stats->candidate_local_keys;
    }
    if (!estimate.exists(key)) {
      if (stats) {
        ++stats->skipped_missing_estimate;
      }
      noteFailure("missing_estimate_for_separator_key");
      continue;
    }

    const gtsam::Symbol symbol(key);
    const unsigned char symbol_char = static_cast<unsigned char>(symbol.chr());

    auto markCovFailure = [&](const std::string& reason) {
      if (stats) {
        ++stats->skipped_covariance_failure;
        stats->used_fallback_sigmas = true;
      }
      noteFailure(reason);
    };

    if (symbol_char == kPoseSymbolChar) {
      gtsam::SharedNoiseModel noise;
      bool cov_ok = false;
      bool cov_failure_marked = false;
      auto markCovFailureOnce = [&](const std::string& reason) {
        if (!cov_failure_marked) {
          markCovFailure(reason);
          cov_failure_marked = true;
        }
      };
      try {
        if (stats) {
          ++stats->covariance_queries_performed;
        }
        gtsam::Matrix pose_cov;
        const gtsam::Matrix pose_cov_raw =
            bpsam_->marginalCovariance(key, cbs::BPSAM::MarginalizationType::LOCAL);
        cov_ok = extractSymmetricCovarianceBlock(pose_cov_raw, 6, 1e-8, &pose_cov);
        if (cov_ok) {
          noise = gtsam::noiseModel::Gaussian::Covariance(pose_cov);
        }
      } catch (const std::exception& e) {
        std::ostringstream oss;
        oss << "pose_covariance_query_failed: " << e.what();
        markCovFailureOnce(oss.str());
      }
      if (!cov_ok) {
        markCovFailureOnce("pose_covariance_invalid_or_nonfinite");
        gtsam::Vector6 sigmas;
        sigmas.head<3>().setConstant(0.05);
        sigmas.tail<3>().setConstant(0.25);
        noise = gtsam::noiseModel::Diagonal::Sigmas(sigmas);
      }
      out_factors->emplace_shared<gtsam::PriorFactor<gtsam::Pose3>>(
          key, estimate.at<gtsam::Pose3>(key), noise);
      ++emitted_total;
      if (stats) {
        ++stats->emitted_pose_priors;
      }
      continue;
    }

    if (symbol_char == kVelocitySymbolChar) {
      gtsam::SharedNoiseModel noise;
      bool cov_ok = false;
      bool cov_failure_marked = false;
      auto markCovFailureOnce = [&](const std::string& reason) {
        if (!cov_failure_marked) {
          markCovFailure(reason);
          cov_failure_marked = true;
        }
      };
      try {
        if (stats) {
          ++stats->covariance_queries_performed;
        }
        gtsam::Matrix vel_cov;
        const gtsam::Matrix vel_cov_raw =
            bpsam_->marginalCovariance(key, cbs::BPSAM::MarginalizationType::LOCAL);
        cov_ok = extractSymmetricCovarianceBlock(vel_cov_raw, 3, 1e-8, &vel_cov);
        if (cov_ok) {
          noise = gtsam::noiseModel::Gaussian::Covariance(vel_cov);
        }
      } catch (const std::exception& e) {
        std::ostringstream oss;
        oss << "velocity_covariance_query_failed: " << e.what();
        markCovFailureOnce(oss.str());
      }
      if (!cov_ok) {
        markCovFailureOnce("velocity_covariance_invalid_or_nonfinite");
        noise = gtsam::noiseModel::Diagonal::Sigmas(
            (gtsam::Vector3() << 0.25, 0.25, 0.25).finished());
      }
      out_factors->emplace_shared<gtsam::PriorFactor<gtsam::Vector3>>(
          key, estimate.at<gtsam::Vector3>(key), noise);
      ++emitted_total;
      if (stats) {
        ++stats->emitted_vel_priors;
      }
      continue;
    }

    if (symbol_char == kImuBiasSymbolChar) {
      gtsam::SharedNoiseModel noise;
      bool cov_ok = false;
      bool cov_failure_marked = false;
      auto markCovFailureOnce = [&](const std::string& reason) {
        if (!cov_failure_marked) {
          markCovFailure(reason);
          cov_failure_marked = true;
        }
      };
      try {
        if (stats) {
          ++stats->covariance_queries_performed;
        }
        gtsam::Matrix bias_cov;
        const gtsam::Matrix bias_cov_raw =
            bpsam_->marginalCovariance(key, cbs::BPSAM::MarginalizationType::LOCAL);
        cov_ok = extractSymmetricCovarianceBlock(bias_cov_raw, 6, 1e-10, &bias_cov);
        if (cov_ok) {
          noise = gtsam::noiseModel::Gaussian::Covariance(bias_cov);
        }
      } catch (const std::exception& e) {
        std::ostringstream oss;
        oss << "bias_covariance_query_failed: " << e.what();
        markCovFailureOnce(oss.str());
      }
      if (!cov_ok) {
        markCovFailureOnce("bias_covariance_invalid_or_nonfinite");
        gtsam::Vector6 sigmas;
        sigmas.head<3>().setConstant(0.02);
        sigmas.tail<3>().setConstant(0.002);
        noise = gtsam::noiseModel::Diagonal::Sigmas(sigmas);
      }
      out_factors->emplace_shared<gtsam::PriorFactor<gtsam::imuBias::ConstantBias>>(
          key, estimate.at<gtsam::imuBias::ConstantBias>(key), noise);
      ++emitted_total;
      if (stats) {
        ++stats->emitted_bias_priors;
      }
      continue;
    }
  }

  if (stats) {
    stats->emitted_total_priors = emitted_total;
  }
  last_incremental_diagnostics_.covariance_queries_performed =
      stats ? stats->covariance_queries_performed : 0u;
  last_incremental_diagnostics_.temporary_priors_emitted = emitted_total;

  return emitted_total > 0u;
}

bool CbsFixedLagBpsamHeart::queryLocalOnlyCovariance(
    const Key& key,
    gtsam::Matrix* covariance,
    std::string* reason) const {
  if (reason) {
    *reason = "none";
  }
  if (!covariance) {
    if (reason) {
      *reason = "null_covariance_output";
    }
    return false;
  }
  if (!bpsam_) {
    if (reason) {
      *reason = "bpsam_not_set";
    }
    return false;
  }
  if (!bpsam_->valueExists(key)) {
    if (reason) {
      *reason = "key_not_present_in_bpsam";
    }
    return false;
  }
  try {
    *covariance = bpsam_->marginalCovariance(
        key, cbs::BPSAM::MarginalizationType::LOCAL);
    return true;
  } catch (const std::exception& e) {
    if (reason) {
      *reason = std::string("local_covariance_query_failed: ") + e.what();
    }
    return false;
  }
}

FrameId CbsFixedLagBpsamHeart::computeOldestActiveFrame(
    const FrameId newest_frame_id,
    const size_t lag_states) {
  if (lag_states == 0u || newest_frame_id + 1u <= lag_states) {
    return 0u;
  }
  return newest_frame_id - lag_states + 1u;
}

CbsFixedLagBpsamHeart::LocalFrameOwnedKeys
CbsFixedLagBpsamHeart::buildLocalOwnershipFromVariableIndex() const {
  LocalFrameOwnedKeys ownership;
  if (!bpsam_) {
    return ownership;
  }

  const auto& variable_index = bpsam_->getVariableIndex();
  for (const auto& kv : variable_index) {
    const Key key = kv.first;
    if (!isLocalStateKey(key)) {
      continue;
    }

    ownership.all_local_state_keys.insert(key);
    const FrameId frame_id = frameIdFromKey(key);
    const gtsam::Symbol symbol(key);
    if (symbol.chr() == kPoseSymbolChar) {
      ownership.pose_keys_by_frame[frame_id].insert(key);
    } else if (symbol.chr() == kVelocitySymbolChar) {
      ownership.vel_keys_by_frame[frame_id].insert(key);
    } else if (symbol.chr() == kImuBiasSymbolChar) {
      ownership.bias_keys_by_frame[frame_id].insert(key);
    }
  }
  return ownership;
}

CbsFixedLagBpsamHeart::BeliefOwnedState
CbsFixedLagBpsamHeart::buildBeliefOwnershipFromActiveGraph() const {
  BeliefOwnedState ownership;
  if (!bpsam_) {
    return ownership;
  }

  const gtsam::NonlinearFactorGraph& graph = bpsam_->getFactorsUnsafe();
  for (size_t slot = 0u; slot < graph.size(); ++slot) {
    if (!factorExists(graph, slot)) {
      continue;
    }
    const auto factor = graph.at(slot);
    if (!isBeliefFactor(factor)) {
      continue;
    }

    BeliefFactorOwnership fown;
    fown.slot = slot;
    fown.keys.reserve(factor->keys().size());
    fown.is_pose_belief_factor =
        cbs::isPoseBeliefFactor(bpsam_->getParams().robot_id, factor);
    fown.is_anchor_belief_factor = cbs::isAnchorBeliefFactor(factor);
    for (const Key key : factor->keys()) {
      fown.keys.push_back(key);
      if (isLocalStateKey(key)) {
        fown.touches_local_state = true;
      }
      if (isLocalPoseKey(key)) {
        fown.touches_local_pose = true;
      }
      if (isRobotKey(key)) {
        ownership.robot_keys.insert(key);
      } else if (isConsensusKey(key)) {
        ownership.consensus_keys.insert(key);
      } else if (!isLocalStateKey(key)) {
        ownership.gbp_keys.insert(key);
      }
    }
    ownership.belief_factor_by_slot.insert({slot, std::move(fown)});
  }

  return ownership;
}

CbsFixedLagBpsamHeart::StaleFrameEvictionResult
CbsFixedLagBpsamHeart::computeStaleFrameEviction(
    const KeyTimestampMap& timestamps,
    const FrameId fallback_newest_frame_id,
    const LocalFrameOwnedKeys& local_ownership) const {
  (void)local_ownership;
  StaleFrameEvictionResult result;
  result.newest_frame_id = fallback_newest_frame_id;

  for (const auto& key_ts : timestamps) {
    result.newest_frame_id = std::max(
        result.newest_frame_id,
        static_cast<FrameId>(std::llround(key_ts.second)));
  }

  result.oldest_active_frame_id =
      computeOldestActiveFrame(result.newest_frame_id, lag_states_);
  result.eviction_start_frame_id = 0u;
  result.eviction_end_frame_id = result.oldest_active_frame_id;
  result.full_rescan = true;

  if (!bpsam_ || result.eviction_end_frame_id <= result.eviction_start_frame_id) {
    return result;
  }

  const auto& variable_index = bpsam_->getVariableIndex();
  std::unordered_set<gtsam::FactorIndex> stale_factor_slots_set;
  std::unordered_set<gtsam::FactorIndex> stale_belief_slots_set;
  const gtsam::NonlinearFactorGraph& graph = bpsam_->getFactorsUnsafe();

  for (const auto& kv : variable_index) {
    const Key key = kv.first;
    if (!isLocalStateKey(key)) {
      continue;
    }
    const FrameId key_frame = frameIdFromKey(key);
    if (key_frame < result.eviction_start_frame_id ||
        key_frame >= result.eviction_end_frame_id) {
      continue;
    }

    result.stale_local_keys.insert(key);
    for (const gtsam::FactorIndex slot : kv.second) {
      stale_factor_slots_set.insert(slot);
      if (factorExists(graph, slot) && isBeliefFactor(graph.at(slot))) {
        stale_belief_slots_set.insert(slot);
      }
    }
  }

  result.stale_local_factor_slots.assign(
      stale_factor_slots_set.begin(), stale_factor_slots_set.end());
  std::sort(result.stale_local_factor_slots.begin(),
            result.stale_local_factor_slots.end());

  result.stale_belief_factor_slots.assign(
      stale_belief_slots_set.begin(), stale_belief_slots_set.end());
  std::sort(result.stale_belief_factor_slots.begin(),
            result.stale_belief_factor_slots.end());

  return result;
}

CbsFixedLagBpsamHeart::OrphanPruneResult CbsFixedLagBpsamHeart::computeOrphanPrune(
    const StaleFrameEvictionResult& stale_eviction,
    const BeliefOwnedState& belief_ownership,
    const LocalFrameOwnedKeys& local_ownership) const {
  OrphanPruneResult result;
  if (!bpsam_) {
    return result;
  }

  const gtsam::NonlinearFactorGraph& graph = bpsam_->getFactorsUnsafe();
  const auto& variable_index = bpsam_->getVariableIndex();

  std::unordered_set<gtsam::FactorIndex> removed_slots(
      stale_eviction.stale_local_factor_slots.begin(),
      stale_eviction.stale_local_factor_slots.end());
  removed_slots.insert(stale_eviction.stale_belief_factor_slots.begin(),
                       stale_eviction.stale_belief_factor_slots.end());

  std::unordered_map<Key, std::vector<gtsam::FactorIndex>> belief_key_to_slots;
  std::unordered_set<gtsam::FactorIndex> remaining_belief_slots;
  for (const auto& kv : belief_ownership.belief_factor_by_slot) {
    const gtsam::FactorIndex slot = kv.first;
    if (removed_slots.count(slot) > 0u || !factorExists(graph, slot)) {
      continue;
    }
    remaining_belief_slots.insert(slot);
    for (const Key key : kv.second.keys) {
      belief_key_to_slots[key].push_back(slot);
    }
  }

  std::queue<Key> bfs;
  std::unordered_set<Key> reachable_keys;
  std::unordered_set<gtsam::FactorIndex> reachable_slots;

  for (const Key key : local_ownership.all_local_state_keys) {
    if (!isLocalPoseKey(key)) {
      continue;
    }
    if (stale_eviction.stale_local_keys.count(key) > 0u) {
      continue;
    }
    if (belief_key_to_slots.count(key) == 0u) {
      continue;
    }
    if (reachable_keys.insert(key).second) {
      bfs.push(key);
    }
  }

  while (!bfs.empty()) {
    const Key key = bfs.front();
    bfs.pop();
    const auto it = belief_key_to_slots.find(key);
    if (it == belief_key_to_slots.end()) {
      continue;
    }
    for (const gtsam::FactorIndex slot : it->second) {
      if (!reachable_slots.insert(slot).second || !factorExists(graph, slot)) {
        continue;
      }
      const auto factor = graph.at(slot);
      for (const Key other_key : factor->keys()) {
        if (reachable_keys.insert(other_key).second) {
          bfs.push(other_key);
        }
      }
    }
  }

  std::unordered_set<gtsam::FactorIndex> orphan_belief_slots_set;
  for (const gtsam::FactorIndex slot : remaining_belief_slots) {
    if (reachable_slots.count(slot) == 0u) {
      orphan_belief_slots_set.insert(slot);
    }
  }

  result.orphan_belief_factor_slots.assign(orphan_belief_slots_set.begin(),
                                           orphan_belief_slots_set.end());
  std::sort(result.orphan_belief_factor_slots.begin(),
            result.orphan_belief_factor_slots.end());

  std::unordered_set<gtsam::FactorIndex> final_removed_slots = removed_slots;
  final_removed_slots.insert(orphan_belief_slots_set.begin(),
                             orphan_belief_slots_set.end());

  auto mark_orphan_if_needed = [&](const std::unordered_set<Key>& keys,
                                   std::unordered_set<Key>* orphan_out) {
    for (const Key key : keys) {
      const auto key_it = variable_index.find(key);
      if (key_it == variable_index.end()) {
        continue;
      }
      if (!keyHasRemainingIncidentFactor(key, final_removed_slots)) {
        orphan_out->insert(key);
      }
    }
  };

  mark_orphan_if_needed(belief_ownership.robot_keys, &result.orphan_robot_keys);
  mark_orphan_if_needed(belief_ownership.gbp_keys, &result.orphan_gbp_keys);
  mark_orphan_if_needed(belief_ownership.consensus_keys,
                        &result.orphan_consensus_keys);

  return result;
}

CbsFixedLagBpsamHeart::BoundarySeparatorCandidateSet
CbsFixedLagBpsamHeart::computeBoundaryCandidates(
    const StaleFrameEvictionResult& stale_eviction,
    const OrphanPruneResult& orphan_prune) const {
  BoundarySeparatorCandidateSet boundary;
  if (!bpsam_) {
    return boundary;
  }

  const gtsam::NonlinearFactorGraph& graph = bpsam_->getFactorsUnsafe();

  std::unordered_set<gtsam::FactorIndex> removed_slots(
      stale_eviction.stale_local_factor_slots.begin(),
      stale_eviction.stale_local_factor_slots.end());
  removed_slots.insert(orphan_prune.orphan_belief_factor_slots.begin(),
                       orphan_prune.orphan_belief_factor_slots.end());

  for (const gtsam::FactorIndex slot : stale_eviction.stale_local_factor_slots) {
    if (!factorExists(graph, slot)) {
      continue;
    }
    const auto factor = graph.at(slot);

    bool touches_stale = false;
    bool touches_active = false;
    for (const Key key : factor->keys()) {
      const bool key_stale = stale_eviction.stale_local_keys.count(key) > 0u;
      touches_stale = touches_stale || key_stale;
      touches_active = touches_active || !key_stale;
    }

    if (!(touches_stale && touches_active)) {
      continue;
    }

    boundary.crossing_factor_slots.insert(slot);
    for (const Key key : factor->keys()) {
      if (stale_eviction.stale_local_keys.count(key) > 0u) {
        continue;
      }
      boundary.separator_keys.insert(key);
      if (isLocalStateKey(key)) {
        boundary.local_separator_keys.insert(key);
      } else {
        boundary.belief_separator_keys.insert(key);
      }
    }
  }

  // Include crossing orphan-pruned belief factors as future summary boundary
  // signals for expanded-graph lag handling.
  for (const gtsam::FactorIndex slot : orphan_prune.orphan_belief_factor_slots) {
    if (!factorExists(graph, slot) || removed_slots.count(slot) == 0u) {
      continue;
    }
    const auto factor = graph.at(slot);
    for (const Key key : factor->keys()) {
      if (stale_eviction.stale_local_keys.count(key) > 0u) {
        continue;
      }
      boundary.separator_keys.insert(key);
      if (isLocalStateKey(key)) {
        boundary.local_separator_keys.insert(key);
      } else {
        boundary.belief_separator_keys.insert(key);
      }
    }
  }

  return boundary;
}

CbsFixedLagBpsamHeart::SummaryInsertionPlan
CbsFixedLagBpsamHeart::buildFutureSummaryInsertionPlan(
    const BoundarySeparatorCandidateSet& boundary_candidates,
    const StaleFrameEvictionResult& stale_eviction) const {
  (void)stale_eviction;
  SummaryInsertionPlan plan;
  plan.boundary = boundary_candidates;
  // Real summary target set: all LOCAL kept separator keys, deterministically.
  for (const Key key : boundary_candidates.local_separator_keys) {
    if (isLocalStateKey(key)) {
      plan.summary_target_keys.insert(key);
    }
  }
  plan.implemented = false;
  plan.mode = "pending_exact_schur";
  if (plan.summary_target_keys.empty()) {
    plan.pending_reason = "no_local_separator_targets";
  } else {
    plan.pending_reason = "pending_summary_build";
  }
  return plan;
}

bool CbsFixedLagBpsamHeart::appendLagEdgeSummaryFactors(
    LagWindowPlan* plan,
    gtsam::NonlinearFactorGraph* out_factors,
    SummaryBuildStats* stats,
    const std::unordered_set<gtsam::FactorIndex>*
        augmented_required_crossing_slots) const {
  const auto start = std::chrono::steady_clock::now();
  bool target_coherence_ok = false;
  std::unordered_set<gtsam::FactorIndex> summary_required_crossing_slots;
  std::unordered_set<gtsam::FactorIndex> summary_covered_crossing_slots;
  std::unordered_set<gtsam::FactorIndex>
      directly_removable_local_internal_candidate_slots;
  std::unordered_set<gtsam::FactorIndex> directly_removable_local_internal_slots;
  std::unordered_set<gtsam::FactorIndex> retained_protected_local_support_slots;
  std::unordered_set<gtsam::FactorIndex> unsupported_deferred_lag_slots;
  std::unordered_set<gtsam::FactorIndex> requested_lag_slot_set;
  std::unordered_set<gtsam::FactorIndex> stale_local_factor_slot_set;
  std::unordered_set<gtsam::FactorIndex> stale_belief_factor_slot_set;
  std::unordered_set<gtsam::FactorIndex> orphan_belief_factor_slot_set;
  std::unordered_set<Key> stale_local_key_set;
  std::unordered_set<gtsam::FactorIndex> summary_selected_slots_for_coverage;
  size_t strict_crossing_candidate_slots_count = 0u;
  size_t strict_crossing_selected_slots_count = 0u;
  size_t strict_crossing_dropped_slots_count = 0u;
  std::unordered_set<gtsam::FactorIndex> boundary_crossing_factor_slots_set;
  std::unordered_set<gtsam::FactorIndex> local_crossing_factor_slots_set;
  std::unordered_set<gtsam::FactorIndex>
      nonbelief_local_crossing_factor_slots_set;
  std::unordered_set<gtsam::FactorIndex>
      selected_local_nonbelief_crossing_factor_slots_set;
  size_t stale_state_keys_count = 0u;
  size_t kept_state_keys_count = 0u;
  size_t stale_pose_keys_count = 0u;
  size_t kept_pose_keys_count = 0u;
  size_t stale_local_factor_slots_count = 0u;
  size_t candidate_factors_touching_stale_count = 0u;
  size_t candidate_factors_touching_kept_count = 0u;
  size_t candidate_factors_touching_both_stale_and_kept_count = 0u;
  std::unordered_set<gtsam::FactorIndex>
      boundary_candidate_crossing_factor_slots_raw_set;
  size_t rejected_candidate_due_to_no_stale_incidence_count = 0u;
  size_t rejected_candidate_due_to_no_kept_incidence_count = 0u;
  size_t rejected_candidate_due_to_nonlocal_ownership_count = 0u;
  size_t rejected_candidate_due_to_belief_factor_count = 0u;
  size_t rejected_candidate_due_to_missing_factor_ptr_count = 0u;
  long long first_boundary_candidate_rejected_slot = -1;
  std::string first_boundary_candidate_rejected_slot_class = "none";
  std::string first_boundary_candidate_rejected_slot_keys = "none";
  std::string first_boundary_candidate_rejected_reason = "none";
  size_t rejected_crossing_due_to_nonlocal_count = 0u;
  size_t rejected_crossing_due_to_belief_count = 0u;
  size_t rejected_crossing_due_to_missing_estimate_count = 0u;
  size_t rejected_crossing_due_to_target_mismatch_count = 0u;
  long long strict_first_dropped_crossing_slot = -1;
  std::string strict_first_dropped_crossing_slot_class = "none";
  std::string strict_first_dropped_crossing_slot_keys = "none";
  std::string strict_first_dropped_crossing_slot_reason = "none";
  long long strict_first_selected_crossing_slot = -1;
  std::string strict_first_selected_crossing_slot_class = "none";
  std::string strict_first_selected_crossing_slot_keys = "none";
  long long strict_first_covered_crossing_slot = -1;
  std::string strict_first_covered_crossing_slot_class = "none";
  std::string strict_first_covered_crossing_slot_keys = "none";
  size_t
      direct_local_internal_rejected_due_to_incomplete_component_count = 0u;
  long long first_direct_local_internal_rejected_slot = -1;
  std::string first_direct_local_internal_rejected_slot_class = "none";
  std::string first_direct_local_internal_rejected_slot_keys = "none";
  std::string first_direct_local_internal_rejected_reason = "none";
  std::string first_direct_local_internal_rejected_blocking_key = "none";
  long long first_direct_local_internal_rejected_blocking_slot = -1;
  std::string first_direct_local_internal_rejected_blocking_slot_class = "none";
  std::string first_direct_local_internal_rejected_blocking_slot_keys = "none";
  long long first_direct_local_internal_rejected_due_to_retained_support_slot =
      -1;
  std::string first_direct_local_internal_rejected_due_to_retained_support_reason =
      "none";
  std::unordered_set<gtsam::FactorIndex>
      direct_local_internal_rejected_due_to_retained_support_slots;
  struct RetainedSupportIncidentGroup {
    std::unordered_set<gtsam::FactorIndex> member_slots;
    std::string blocking_key = "none";
    long long blocking_support_slot = -1;
    std::string blocking_support_slot_class = "none";
    std::string blocking_support_slot_keys = "none";
  };
  std::unordered_map<std::string, RetainedSupportIncidentGroup>
      direct_local_retained_support_incidents;
  std::string first_direct_local_internal_rejected_retained_support_incident_id =
      "none";
  std::string first_retained_protected_local_support_slot_class = "none";
  std::string first_retained_protected_local_support_slot_keys = "none";
  size_t unclassified_requested_lag_slots_count = 0u;
  std::string unclassified_requested_lag_slots_first_few = "none";
  long long first_unclassified_requested_lag_slot = -1;
  std::string first_unclassified_requested_lag_slot_class = "none";
  std::string first_unclassified_requested_lag_slot_keys = "none";
  std::string first_unclassified_requested_lag_slot_reason = "none";
  bool lag_requested_partition_coherence_ok = true;
  if (stats) {
    *stats = SummaryBuildStats{};
  }
  auto format_slots_set_preview =
      [](const std::unordered_set<gtsam::FactorIndex>& slots,
         const size_t max_items) {
        if (slots.empty()) {
          return std::string("none");
        }
        std::vector<gtsam::FactorIndex> ordered(slots.begin(), slots.end());
        std::sort(ordered.begin(), ordered.end());
        std::ostringstream oss;
        const size_t limit = std::min(max_items, ordered.size());
        for (size_t i = 0; i < limit; ++i) {
          if (i > 0u) {
            oss << ",";
          }
          oss << ordered[i];
        }
        if (ordered.size() > limit) {
          oss << ",...";
        }
        return oss.str();
      };
  const gtsam::NonlinearFactorGraph* graph_ptr = nullptr;
  auto format_factor_keys_for_diag_early =
      [](const gtsam::NonlinearFactor::shared_ptr& factor) {
        if (!factor) {
          return std::string("none");
        }
        std::ostringstream oss;
        for (size_t i = 0u; i < factor->keys().size(); ++i) {
          if (i > 0u) {
            oss << "|";
          }
          oss << gtsam::DefaultKeyFormatter(factor->keys()[i]);
        }
        return oss.str().empty() ? std::string("none") : oss.str();
      };
  auto classify_factor_class_for_diag_early =
      [&](const gtsam::NonlinearFactor::shared_ptr& factor) {
        if (!factor) {
          return std::string("missing_slot");
        }
        if (isBeliefFactor(factor)) {
          return std::string("belief");
        }
        const size_t arity = factor->keys().size();
        if (arity == 1u) {
          return std::string("unary");
        }
        if (arity == 2u) {
          return std::string("binary");
        }
        if (arity >= 5u) {
          return std::string("imu_like");
        }
        return std::string("other");
      };
  auto is_frame0_bootstrap_state_key_early = [](const Key key) {
    const gtsam::Symbol sym(key);
    return sym.index() == 0u &&
           (sym.chr() == kPoseSymbolChar || sym.chr() == kVelocitySymbolChar ||
            sym.chr() == kImuBiasSymbolChar);
  };
  auto updateSummaryCoverageStats = [&]() {
    if (!stats) {
      return;
    }

    summary_covered_crossing_slots.clear();
    for (const gtsam::FactorIndex slot : summary_selected_slots_for_coverage) {
      if (summary_required_crossing_slots.count(slot) > 0u) {
        summary_covered_crossing_slots.insert(slot);
      }
    }

    // Bootstrap-connected crossing factors are only valid as crossing when
    // they are truly summary-covered this epoch. Otherwise classify them as
    // explicitly unsupported/deferred lag slots.
    std::vector<gtsam::FactorIndex> bootstrap_crossing_uncovered_to_unsupported;
    for (const gtsam::FactorIndex slot : summary_required_crossing_slots) {
      if (summary_covered_crossing_slots.count(slot) > 0u ||
          !(graph_ptr && factorExists(*graph_ptr, slot))) {
        continue;
      }
      const auto factor = graph_ptr->at(slot);
      if (!factor) {
        continue;
      }
      bool touches_bootstrap_key = false;
      for (const Key key : factor->keys()) {
        if (is_frame0_bootstrap_state_key_early(key)) {
          touches_bootstrap_key = true;
          break;
        }
      }
      if (touches_bootstrap_key) {
        bootstrap_crossing_uncovered_to_unsupported.push_back(slot);
      }
    }
    for (const gtsam::FactorIndex slot :
         bootstrap_crossing_uncovered_to_unsupported) {
      summary_required_crossing_slots.erase(slot);
      summary_covered_crossing_slots.erase(slot);
      unsupported_deferred_lag_slots.insert(slot);
    }

    std::unordered_set<gtsam::FactorIndex> uncovered_crossing_slots =
        summary_required_crossing_slots;
    for (const gtsam::FactorIndex slot : summary_covered_crossing_slots) {
      uncovered_crossing_slots.erase(slot);
    }
    std::unordered_set<gtsam::FactorIndex>
        bootstrap_crossing_uncovered_to_unsupported_set(
            bootstrap_crossing_uncovered_to_unsupported.begin(),
            bootstrap_crossing_uncovered_to_unsupported.end());
    std::unordered_set<gtsam::FactorIndex>
        caller_requested_crossing_slots_for_this_call;
    if (augmented_required_crossing_slots) {
      caller_requested_crossing_slots_for_this_call.insert(
          augmented_required_crossing_slots->begin(),
          augmented_required_crossing_slots->end());
    } else {
      caller_requested_crossing_slots_for_this_call =
          summary_required_crossing_slots;
    }
    const std::unordered_set<gtsam::FactorIndex>
        internally_classified_crossing_candidates =
            summary_required_crossing_slots;
    const std::unordered_set<gtsam::FactorIndex> exact_covered_slots_for_this_call =
        summary_covered_crossing_slots;
    std::unordered_set<gtsam::FactorIndex>
        caller_requested_supported_crossing_slots_exact =
            caller_requested_crossing_slots_for_this_call;
    for (auto it = caller_requested_supported_crossing_slots_exact.begin();
         it != caller_requested_supported_crossing_slots_exact.end();) {
      if (exact_covered_slots_for_this_call.count(*it) == 0u) {
        it = caller_requested_supported_crossing_slots_exact.erase(it);
      } else {
        ++it;
      }
    }
    bool caller_requested_domain_coherence_ok = true;
    long long
        first_caller_requested_slot_missing_from_internal_crossing_domain = -1;
    std::string
        first_caller_requested_slot_missing_from_internal_crossing_domain_class =
            "none";
    std::string
        first_caller_requested_slot_missing_from_internal_crossing_domain_keys =
            "none";
    size_t caller_requested_missing_from_internal_domain_count = 0u;
    std::unordered_set<gtsam::FactorIndex>
        caller_requested_missing_from_internal_domain_set;
    std::string caller_requested_missing_from_internal_domain_first_few = "none";
    std::string caller_requested_slot_classification_first_few = "none";
    std::string first_caller_requested_slot_exclusion_reason = "none";
    if (!caller_requested_crossing_slots_for_this_call.empty()) {
      std::vector<gtsam::FactorIndex> ordered_caller_requested_slots(
          caller_requested_crossing_slots_for_this_call.begin(),
          caller_requested_crossing_slots_for_this_call.end());
      std::sort(ordered_caller_requested_slots.begin(),
                ordered_caller_requested_slots.end());
      std::vector<std::string> caller_requested_slot_classification_entries;
      caller_requested_slot_classification_entries.reserve(
          std::min<size_t>(8u, ordered_caller_requested_slots.size()));
      for (const gtsam::FactorIndex slot : ordered_caller_requested_slots) {
        const bool slot_exists = graph_ptr && factorExists(*graph_ptr, slot);
        const auto factor = slot_exists ? graph_ptr->at(slot) : nullptr;
        const std::string slot_factor_class =
            (slot_exists && factor)
                ? classify_factor_class_for_diag_early(factor)
                : std::string("missing_slot");
        const std::string slot_factor_keys =
            (slot_exists && factor)
                ? format_factor_keys_for_diag_early(factor)
                : std::string("none");

        std::string slot_classification = "present_in_internal_crossing_domain";
        std::string slot_exclusion_reason = "none";
        if (internally_classified_crossing_candidates.count(slot) == 0u) {
          if (!slot_exists || !factor) {
            slot_classification = "excluded_as_missing_slot";
            slot_exclusion_reason =
                slot_exists ? "requested_slot_null_factor"
                            : "requested_slot_missing_in_active_graph";
          } else if (stale_belief_factor_slot_set.count(slot) > 0u) {
            slot_classification = "excluded_as_stale_belief";
            slot_exclusion_reason = "slot_in_stale_belief_factor_set";
          } else if (orphan_belief_factor_slot_set.count(slot) > 0u) {
            slot_classification = "excluded_as_orphan_belief";
            slot_exclusion_reason = "slot_in_orphan_belief_factor_set";
          } else if (directly_removable_local_internal_slots.count(slot) > 0u ||
                     directly_removable_local_internal_candidate_slots.count(
                         slot) > 0u) {
            slot_classification = "excluded_as_direct_local_internal";
            slot_exclusion_reason =
                directly_removable_local_internal_slots.count(slot) > 0u
                    ? "slot_in_direct_local_internal_slots"
                    : "slot_in_direct_local_internal_candidate_slots";
          } else if (bootstrap_crossing_uncovered_to_unsupported_set.count(
                         slot) > 0u ||
                     retained_protected_local_support_slots.count(slot) > 0u) {
            slot_classification = "excluded_by_bootstrap_or_boundary_rule";
            if (bootstrap_crossing_uncovered_to_unsupported_set.count(slot) >
                0u) {
              slot_exclusion_reason =
                  "bootstrap_crossing_uncovered_to_unsupported";
            } else {
              slot_exclusion_reason = "retained_bootstrap_support_prior";
            }
          } else if (requested_lag_slot_set.count(slot) == 0u) {
            slot_classification = "excluded_other";
            slot_exclusion_reason = "slot_not_in_requested_lag_slot_set";
          } else {
            bool all_local = true;
            bool touches_stale = false;
            bool touches_kept = false;
            for (const Key key : factor->keys()) {
              const bool key_stale = stale_local_key_set.count(key) > 0u;
              touches_stale = touches_stale || key_stale;
              touches_kept = touches_kept || !key_stale;
              if (!isLocalStateKey(key)) {
                all_local = false;
              }
            }
            if (!all_local) {
              slot_classification = "excluded_as_non_crossing";
              slot_exclusion_reason = "factor_contains_nonlocal_key";
            } else if (!(touches_stale && touches_kept)) {
              slot_classification = "excluded_as_non_crossing";
              if (!touches_stale) {
                slot_exclusion_reason = "local_factor_has_no_stale_key";
              } else if (!touches_kept) {
                slot_exclusion_reason = "local_factor_has_no_kept_key";
              } else {
                slot_exclusion_reason = "local_factor_not_crossing";
              }
            } else if (unsupported_deferred_lag_slots.count(slot) > 0u) {
              slot_classification = "excluded_other";
              slot_exclusion_reason = "slot_in_unsupported_deferred_lag_slots";
            } else {
              slot_classification = "excluded_other";
              slot_exclusion_reason = "crossing_candidate_not_in_domain";
            }
          }
        }

        if (caller_requested_slot_classification_entries.size() < 8u) {
          std::ostringstream classification_entry;
          classification_entry
              << "slot=" << slot
              << "|classification=" << slot_classification
              << "|reason="
              << (slot_exclusion_reason.empty() ? "none"
                                                : slot_exclusion_reason)
              << "|class=" << slot_factor_class
              << "|keys=" << slot_factor_keys;
          caller_requested_slot_classification_entries.push_back(
              classification_entry.str());
        }

        if (slot_classification != "present_in_internal_crossing_domain") {
          caller_requested_domain_coherence_ok = false;
          caller_requested_missing_from_internal_domain_set.insert(slot);
          if (first_caller_requested_slot_missing_from_internal_crossing_domain <
              0) {
            first_caller_requested_slot_missing_from_internal_crossing_domain =
                static_cast<long long>(slot);
            first_caller_requested_slot_missing_from_internal_crossing_domain_class =
                slot_factor_class;
            first_caller_requested_slot_missing_from_internal_crossing_domain_keys =
                slot_factor_keys;
            first_caller_requested_slot_exclusion_reason =
                slot_exclusion_reason.empty() ? "none" : slot_exclusion_reason;
          }
        }
      }
      caller_requested_missing_from_internal_domain_count =
          caller_requested_missing_from_internal_domain_set.size();
      caller_requested_missing_from_internal_domain_first_few =
          format_slots_set_preview(
              caller_requested_missing_from_internal_domain_set, 8u);
      if (caller_requested_slot_classification_entries.empty()) {
        caller_requested_slot_classification_first_few = "none";
      } else {
        std::ostringstream classification_oss;
        for (size_t i = 0u; i < caller_requested_slot_classification_entries.size();
             ++i) {
          if (i > 0u) {
            classification_oss << ";";
          }
          classification_oss << caller_requested_slot_classification_entries[i];
        }
        if (ordered_caller_requested_slots.size() >
            caller_requested_slot_classification_entries.size()) {
          classification_oss << ";...";
        }
        caller_requested_slot_classification_first_few =
            classification_oss.str();
      }
    }

    auto infer_unclassified_reason = [&](const gtsam::FactorIndex slot) {
      if (stale_belief_factor_slot_set.count(slot) > 0u) {
        return std::string("stale_belief_slot_not_partitioned");
      }
      if (orphan_belief_factor_slot_set.count(slot) > 0u) {
        return std::string("orphan_belief_slot_not_partitioned");
      }
      if (!(graph_ptr && factorExists(*graph_ptr, slot))) {
        return std::string("requested_slot_missing_in_active_graph");
      }
      const auto factor = graph_ptr->at(slot);
      if (!factor) {
        return std::string("requested_slot_null_factor");
      }
      if (isBeliefFactor(factor)) {
        return std::string("belief_factor_unclassified");
      }
      if (!stale_local_factor_slot_set.count(slot)) {
        return std::string("requested_slot_not_in_stale_local_or_belief_sets");
      }
      bool all_local = true;
      bool touches_stale = false;
      bool touches_kept = false;
      bool all_stale = true;
      for (const Key key : factor->keys()) {
        const bool key_stale = stale_local_key_set.count(key) > 0u;
        touches_stale = touches_stale || key_stale;
        touches_kept = touches_kept || !key_stale;
        all_stale = all_stale && key_stale;
        if (!isLocalStateKey(key)) {
          all_local = false;
        }
      }
      if (!all_local) {
        return std::string("nonlocal_factor_unclassified");
      }
      if (retained_protected_local_support_slots.count(slot) > 0u) {
        return std::string("retained_protected_support_slot_unclassified");
      }
      if (touches_stale && touches_kept) {
        return std::string("crossing_slot_unclassified");
      }
      if (touches_stale && all_stale) {
        return std::string("all_stale_local_slot_unclassified");
      }
      if (!touches_stale) {
        return std::string("no_stale_local_key_membership");
      }
      return std::string("unknown_unclassified_reason");
    };

    std::unordered_set<gtsam::FactorIndex> unclassified_requested_slots;
    {
      std::unordered_set<gtsam::FactorIndex> partition_union =
          summary_required_crossing_slots;
      partition_union.insert(directly_removable_local_internal_slots.begin(),
                             directly_removable_local_internal_slots.end());
      partition_union.insert(unsupported_deferred_lag_slots.begin(),
                             unsupported_deferred_lag_slots.end());
      for (const gtsam::FactorIndex slot : requested_lag_slot_set) {
        if (partition_union.count(slot) == 0u) {
          unclassified_requested_slots.insert(slot);
        }
      }
    }
    // Total partition rule: every requested lag slot must end in one bucket.
    if (!unclassified_requested_slots.empty()) {
      unsupported_deferred_lag_slots.insert(unclassified_requested_slots.begin(),
                                            unclassified_requested_slots.end());
    }

    unclassified_requested_lag_slots_count = unclassified_requested_slots.size();
    unclassified_requested_lag_slots_first_few =
        format_slots_set_preview(unclassified_requested_slots, 8u);
    first_unclassified_requested_lag_slot = -1;
    first_unclassified_requested_lag_slot_class = "none";
    first_unclassified_requested_lag_slot_keys = "none";
    first_unclassified_requested_lag_slot_reason = "none";
    if (!unclassified_requested_slots.empty()) {
      std::vector<gtsam::FactorIndex> ordered_unclassified(
          unclassified_requested_slots.begin(), unclassified_requested_slots.end());
      std::sort(ordered_unclassified.begin(), ordered_unclassified.end());
      const gtsam::FactorIndex first_slot = ordered_unclassified.front();
      first_unclassified_requested_lag_slot =
          static_cast<long long>(first_slot);
      if (graph_ptr && factorExists(*graph_ptr, first_slot)) {
        const auto first_factor = graph_ptr->at(first_slot);
        first_unclassified_requested_lag_slot_class =
            classify_factor_class_for_diag_early(first_factor);
        first_unclassified_requested_lag_slot_keys =
            format_factor_keys_for_diag_early(first_factor);
      } else {
        first_unclassified_requested_lag_slot_class = "missing_slot";
      }
      first_unclassified_requested_lag_slot_reason =
          infer_unclassified_reason(first_slot);
    }

    bool partition_has_overlap = false;
    for (const gtsam::FactorIndex slot : summary_required_crossing_slots) {
      if (directly_removable_local_internal_slots.count(slot) > 0u ||
          unsupported_deferred_lag_slots.count(slot) > 0u) {
        partition_has_overlap = true;
        break;
      }
    }
    if (!partition_has_overlap) {
      for (const gtsam::FactorIndex slot :
           directly_removable_local_internal_slots) {
        if (unsupported_deferred_lag_slots.count(slot) > 0u) {
          partition_has_overlap = true;
          break;
        }
      }
    }
    const size_t partition_sum_count =
        summary_required_crossing_slots.size() +
        directly_removable_local_internal_slots.size() +
        unsupported_deferred_lag_slots.size();
    lag_requested_partition_coherence_ok =
        !partition_has_overlap && unclassified_requested_slots.empty() &&
        partition_sum_count == requested_lag_slot_set.size();

    const size_t requested_count = summary_required_crossing_slots.size();
    const size_t covered_count = summary_covered_crossing_slots.size();
    const size_t uncovered_count = uncovered_crossing_slots.size();

    stats->summary_required_crossing_slots_exact = summary_required_crossing_slots;
    stats->summary_required_crossing_slots_count =
        summary_required_crossing_slots.size();
    stats->summary_required_crossing_slots_first_few = format_slots_set_preview(
        stats->summary_required_crossing_slots_exact, 8u);
    stats->summary_covered_crossing_slots_exact = summary_covered_crossing_slots;
    stats->summary_covered_crossing_slots_count =
        summary_covered_crossing_slots.size();
    stats->summary_covered_crossing_slots_first_few = format_slots_set_preview(
        stats->summary_covered_crossing_slots_exact, 8u);
    stats->augmented_supported_requested_crossing_slots_exact =
        caller_requested_supported_crossing_slots_exact;
    stats->augmented_supported_requested_crossing_slots_count =
        caller_requested_supported_crossing_slots_exact.size();
    stats->augmented_supported_requested_crossing_slots_first_few =
        format_slots_set_preview(
            stats->augmented_supported_requested_crossing_slots_exact, 8u);
    stats->augmented_requested_crossing_slots_first_few =
        format_slots_set_preview(caller_requested_crossing_slots_for_this_call,
                                 8u);
    stats->augmented_requested_supported_coherence_ok =
        caller_requested_domain_coherence_ok;
    stats->caller_requested_crossing_slots_exact =
        caller_requested_crossing_slots_for_this_call;
    stats->caller_requested_crossing_slots_count =
        caller_requested_crossing_slots_for_this_call.size();
    stats->caller_requested_crossing_slots_first_few = format_slots_set_preview(
        stats->caller_requested_crossing_slots_exact, 8u);
    stats->caller_requested_supported_crossing_slots_exact =
        caller_requested_supported_crossing_slots_exact;
    stats->caller_requested_supported_crossing_slots_count =
        caller_requested_supported_crossing_slots_exact.size();
    stats->caller_requested_supported_crossing_slots_first_few =
        format_slots_set_preview(
            stats->caller_requested_supported_crossing_slots_exact, 8u);
    stats->caller_requested_domain_coherence_ok =
        caller_requested_domain_coherence_ok;
    stats->first_caller_requested_slot_missing_from_internal_crossing_domain =
        first_caller_requested_slot_missing_from_internal_crossing_domain;
    stats
        ->first_caller_requested_slot_missing_from_internal_crossing_domain_class =
        first_caller_requested_slot_missing_from_internal_crossing_domain_class;
    stats
        ->first_caller_requested_slot_missing_from_internal_crossing_domain_keys =
        first_caller_requested_slot_missing_from_internal_crossing_domain_keys;
    stats->caller_requested_missing_from_internal_domain_count =
        caller_requested_missing_from_internal_domain_count;
    stats->caller_requested_missing_from_internal_domain_first_few =
        caller_requested_missing_from_internal_domain_first_few;
    stats->caller_requested_slot_classification_first_few =
        caller_requested_slot_classification_first_few;
    stats->first_caller_requested_slot_exclusion_reason =
        first_caller_requested_slot_exclusion_reason;
    stats->directly_removable_local_internal_candidate_slots_exact =
        directly_removable_local_internal_candidate_slots;
    stats->directly_removable_local_internal_candidate_slots_count =
        directly_removable_local_internal_candidate_slots.size();
    stats->directly_removable_local_internal_candidate_slots_first_few =
        format_slots_set_preview(
            stats->directly_removable_local_internal_candidate_slots_exact, 8u);
    stats->directly_removable_local_internal_slots_exact =
        directly_removable_local_internal_slots;
    stats->directly_removable_local_internal_slots_count =
        directly_removable_local_internal_slots.size();
    stats->directly_removable_local_internal_slots_first_few =
        format_slots_set_preview(
            stats->directly_removable_local_internal_slots_exact, 8u);
    stats->retained_protected_local_support_slots_exact =
        retained_protected_local_support_slots;
    stats->retained_protected_local_support_slots_count =
        retained_protected_local_support_slots.size();
    stats->retained_protected_local_support_slots_first_few =
        format_slots_set_preview(
            stats->retained_protected_local_support_slots_exact, 8u);
    stats->first_retained_protected_local_support_slot_class =
        first_retained_protected_local_support_slot_class;
    stats->first_retained_protected_local_support_slot_keys =
        first_retained_protected_local_support_slot_keys;
    stats->direct_local_internal_rejected_due_to_incomplete_component_count =
        direct_local_internal_rejected_due_to_incomplete_component_count;
    stats->first_direct_local_internal_rejected_slot =
        first_direct_local_internal_rejected_slot;
    stats->first_direct_local_internal_rejected_slot_class =
        first_direct_local_internal_rejected_slot_class;
    stats->first_direct_local_internal_rejected_slot_keys =
        first_direct_local_internal_rejected_slot_keys;
    stats->first_direct_local_internal_rejected_reason =
        first_direct_local_internal_rejected_reason;
    stats->first_direct_local_internal_rejected_blocking_key =
        first_direct_local_internal_rejected_blocking_key;
    stats->first_direct_local_internal_rejected_blocking_slot =
        first_direct_local_internal_rejected_blocking_slot;
    stats->first_direct_local_internal_rejected_blocking_slot_class =
        first_direct_local_internal_rejected_blocking_slot_class;
    stats->first_direct_local_internal_rejected_blocking_slot_keys =
        first_direct_local_internal_rejected_blocking_slot_keys;
    stats->first_direct_local_internal_rejected_due_to_retained_support_slot =
        first_direct_local_internal_rejected_due_to_retained_support_slot;
    stats->first_direct_local_internal_rejected_due_to_retained_support_reason =
        first_direct_local_internal_rejected_due_to_retained_support_reason;
    stats->direct_local_internal_rejected_due_to_retained_support_slots_exact =
        direct_local_internal_rejected_due_to_retained_support_slots;
    stats->direct_local_internal_rejected_due_to_retained_support_slots_count =
        direct_local_internal_rejected_due_to_retained_support_slots.size();
    stats->direct_local_internal_rejected_due_to_retained_support_slots_first_few =
        format_slots_set_preview(
            stats
                ->direct_local_internal_rejected_due_to_retained_support_slots_exact,
            8u);
    std::unordered_set<gtsam::FactorIndex>
        root_bootstrap_blocked_component_member_slots_exact;
    std::unordered_set<gtsam::FactorIndex> root_bootstrap_seed_slots_exact;
    std::string root_bootstrap_blocking_key = "none";
    long long root_bootstrap_blocking_support_slot = -1;
    std::string root_bootstrap_blocking_support_slot_class = "none";
    std::string root_bootstrap_blocking_support_slot_keys = "none";
    std::string root_bootstrap_blocking_incident_id = "none";
    size_t root_bootstrap_blocked_component_count = 0u;
    size_t root_bootstrap_component_seed_count = 0u;
    std::string root_bootstrap_seed_slots_first_few = "none";
    std::string root_bootstrap_component_closure_mode = "none";
    if (!direct_local_retained_support_incidents.empty()) {
      const auto selected_it = std::max_element(
          direct_local_retained_support_incidents.begin(),
          direct_local_retained_support_incidents.end(),
          [](const auto& lhs, const auto& rhs) {
            const size_t lhs_size = lhs.second.member_slots.size();
            const size_t rhs_size = rhs.second.member_slots.size();
            if (lhs_size != rhs_size) {
              return lhs_size < rhs_size;
            }
            return lhs.first > rhs.first;
          });
      root_bootstrap_seed_slots_exact = selected_it->second.member_slots;
      root_bootstrap_blocking_key = selected_it->second.blocking_key;
      root_bootstrap_blocking_support_slot =
          selected_it->second.blocking_support_slot;
      root_bootstrap_blocking_support_slot_class =
          selected_it->second.blocking_support_slot_class;
      root_bootstrap_blocking_support_slot_keys =
          selected_it->second.blocking_support_slot_keys;
      root_bootstrap_blocking_incident_id = selected_it->first;
      root_bootstrap_component_closure_mode =
          "seeded_from_grouped_retained_support_incident";
    } else {
      root_bootstrap_seed_slots_exact =
          direct_local_internal_rejected_due_to_retained_support_slots;
      root_bootstrap_blocking_key =
          first_direct_local_internal_rejected_blocking_key;
      root_bootstrap_blocking_support_slot =
          first_direct_local_internal_rejected_blocking_slot;
      root_bootstrap_blocking_support_slot_class =
          first_direct_local_internal_rejected_blocking_slot_class;
      root_bootstrap_blocking_support_slot_keys =
          first_direct_local_internal_rejected_blocking_slot_keys;
      root_bootstrap_blocking_incident_id =
          first_direct_local_internal_rejected_retained_support_incident_id;
      root_bootstrap_component_closure_mode =
          "seeded_from_retained_support_rejected_slots_fallback";
    }

    root_bootstrap_component_seed_count = root_bootstrap_seed_slots_exact.size();
    root_bootstrap_seed_slots_first_few =
        format_slots_set_preview(root_bootstrap_seed_slots_exact, 8u);

    if (root_bootstrap_seed_slots_exact.empty()) {
      root_bootstrap_component_closure_mode = "no_seed_slots";
    } else {
      const auto& graph_for_closure = bpsam_->getFactorsUnsafe();
      std::unordered_map<Key, std::vector<gtsam::FactorIndex>>
          candidate_slots_by_local_key;
      candidate_slots_by_local_key.reserve(
          directly_removable_local_internal_candidate_slots.size());
      for (const gtsam::FactorIndex candidate_slot :
           directly_removable_local_internal_candidate_slots) {
        if (!factorExists(graph_for_closure, candidate_slot)) {
          continue;
        }
        const auto candidate_factor = graph_for_closure.at(candidate_slot);
        if (!candidate_factor) {
          continue;
        }
        for (const Key key : candidate_factor->keys()) {
          if (!isLocalStateKey(key) || stale_local_key_set.count(key) == 0u) {
            continue;
          }
          candidate_slots_by_local_key[key].push_back(candidate_slot);
        }
      }

      std::queue<gtsam::FactorIndex> closure_queue;
      for (const gtsam::FactorIndex seed_slot : root_bootstrap_seed_slots_exact) {
        if (!factorExists(graph_for_closure, seed_slot)) {
          continue;
        }
        if (directly_removable_local_internal_candidate_slots.count(seed_slot) ==
            0u) {
          continue;
        }
        if (root_bootstrap_blocked_component_member_slots_exact.insert(seed_slot)
                .second) {
          closure_queue.push(seed_slot);
        }
      }

      while (!closure_queue.empty()) {
        const gtsam::FactorIndex slot = closure_queue.front();
        closure_queue.pop();
        if (!factorExists(graph_for_closure, slot)) {
          continue;
        }
        const auto factor = graph_for_closure.at(slot);
        if (!factor) {
          continue;
        }
        for (const Key key : factor->keys()) {
          if (!isLocalStateKey(key) || stale_local_key_set.count(key) == 0u) {
            continue;
          }
          const auto key_it = candidate_slots_by_local_key.find(key);
          if (key_it == candidate_slots_by_local_key.end()) {
            continue;
          }
          for (const gtsam::FactorIndex neighbor_slot : key_it->second) {
            if (root_bootstrap_blocked_component_member_slots_exact
                    .insert(neighbor_slot)
                    .second) {
              closure_queue.push(neighbor_slot);
            }
          }
        }
      }

      if (root_bootstrap_blocked_component_member_slots_exact.empty()) {
        root_bootstrap_blocked_component_member_slots_exact =
            root_bootstrap_seed_slots_exact;
        root_bootstrap_component_closure_mode =
            "seed_slots_only_closure_fallback";
      } else {
        root_bootstrap_component_closure_mode =
            "direct_local_candidate_key_connectivity_closure";
      }
    }
    if (!root_bootstrap_blocked_component_member_slots_exact.empty()) {
      root_bootstrap_blocked_component_count = 1u;
    }
    stats->root_bootstrap_blocked_component_member_slots_exact =
        root_bootstrap_blocked_component_member_slots_exact;
    stats->root_bootstrap_blocked_component_member_count =
        root_bootstrap_blocked_component_member_slots_exact.size();
    stats->root_bootstrap_blocked_component_member_slots_first_few =
        format_slots_set_preview(
            stats->root_bootstrap_blocked_component_member_slots_exact, 8u);
    stats->root_bootstrap_blocked_component_count =
        root_bootstrap_blocked_component_count;
    stats->root_bootstrap_component_seed_count =
        root_bootstrap_component_seed_count;
    stats->root_bootstrap_seed_slots_first_few =
        root_bootstrap_seed_slots_first_few;
    stats->root_bootstrap_component_closure_mode =
        root_bootstrap_component_closure_mode;
    stats->root_bootstrap_blocking_key = root_bootstrap_blocking_key;
    stats->root_bootstrap_blocking_support_slot =
        root_bootstrap_blocking_support_slot;
    stats->root_bootstrap_blocking_support_slot_class =
        root_bootstrap_blocking_support_slot_class;
    stats->root_bootstrap_blocking_support_slot_keys =
        root_bootstrap_blocking_support_slot_keys;
    stats->root_bootstrap_blocking_incident_id =
        root_bootstrap_blocking_incident_id;
    stats->unsupported_deferred_lag_slots_exact = unsupported_deferred_lag_slots;
    stats->unsupported_deferred_lag_slots_count =
        unsupported_deferred_lag_slots.size();
    stats->unsupported_deferred_lag_slots_first_few =
        format_slots_set_preview(stats->unsupported_deferred_lag_slots_exact, 8u);
    stats->lag_requested_slot_count = requested_lag_slot_set.size();
    stats->lag_requested_partition_coherence_ok =
        lag_requested_partition_coherence_ok;
    stats->unclassified_requested_lag_slots_count =
        unclassified_requested_lag_slots_count;
    stats->unclassified_requested_lag_slots_first_few =
        unclassified_requested_lag_slots_first_few;
    stats->first_unclassified_requested_lag_slot =
        first_unclassified_requested_lag_slot;
    stats->first_unclassified_requested_lag_slot_class =
        first_unclassified_requested_lag_slot_class;
    stats->first_unclassified_requested_lag_slot_keys =
        first_unclassified_requested_lag_slot_keys;
    stats->first_unclassified_requested_lag_slot_reason =
        first_unclassified_requested_lag_slot_reason;

    stats->summary_remove_coverage_requested_count = requested_count;
    stats->summary_remove_coverage_covered_count = covered_count;
    stats->summary_remove_coverage_uncovered_count = uncovered_count;
    // Backward-compatible alias of split crossing coverage.
    stats->summary_covered_remove_slots_exact = summary_covered_crossing_slots;
    stats->summary_covered_remove_slots_count = covered_count;
    stats->summary_covered_remove_slots_first_few = format_slots_set_preview(
        stats->summary_covered_remove_slots_exact, 8u);
    stats->summary_uncovered_requested_remove_slots_count = uncovered_count;
    stats->summary_uncovered_requested_remove_slots_first_few =
        format_slots_set_preview(uncovered_crossing_slots, 8u);

    if (strict_first_selected_crossing_slot < 0 &&
        !summary_selected_slots_for_coverage.empty()) {
      std::vector<gtsam::FactorIndex> ordered_selected(
          summary_selected_slots_for_coverage.begin(),
          summary_selected_slots_for_coverage.end());
      std::sort(ordered_selected.begin(), ordered_selected.end());
      const auto selected_slot = ordered_selected.front();
      strict_first_selected_crossing_slot =
          static_cast<long long>(selected_slot);
    }
    if (!summary_covered_crossing_slots.empty()) {
      std::vector<gtsam::FactorIndex> ordered_covered(
          summary_covered_crossing_slots.begin(),
          summary_covered_crossing_slots.end());
      std::sort(ordered_covered.begin(), ordered_covered.end());
      const auto covered_slot = ordered_covered.front();
      strict_first_covered_crossing_slot = static_cast<long long>(covered_slot);
    }

    std::string expected_mode = "none";
    if (requested_count == 0u || covered_count == 0u) {
      expected_mode = "none";
    } else if (covered_count == requested_count) {
      expected_mode = "full";
    } else {
      expected_mode = "partial";
    }
    stats->summary_remove_coverage_mode = expected_mode;

    bool coherence_ok = true;
    std::string incoherence_reason = "none";
    if (covered_count > requested_count) {
      coherence_ok = false;
      incoherence_reason = "covered_count_exceeds_requested_count";
    } else if (covered_count > 0u &&
               !caller_requested_crossing_slots_for_this_call.empty() &&
               caller_requested_supported_crossing_slots_exact.empty()) {
      coherence_ok = false;
      incoherence_reason =
          "covered_nonzero_but_caller_requested_supported_intersection_empty";
    } else if (uncovered_count != (requested_count - covered_count)) {
      coherence_ok = false;
      incoherence_reason = "requested_minus_covered_mismatch_uncovered";
    } else if (expected_mode == "none" &&
               !(requested_count == 0u || covered_count == 0u)) {
      coherence_ok = false;
      incoherence_reason = "mode_none_inconsistent_with_counts";
    } else if (expected_mode == "full" &&
               !(requested_count > 0u && covered_count == requested_count)) {
      coherence_ok = false;
      incoherence_reason = "mode_full_inconsistent_with_counts";
    } else if (expected_mode == "partial" &&
               !(requested_count > 0u && covered_count > 0u &&
                 covered_count < requested_count)) {
      coherence_ok = false;
      incoherence_reason = "mode_partial_inconsistent_with_counts";
    }
    stats->summary_remove_coverage_coherence_ok = coherence_ok;
    stats->summary_remove_coverage_first_inconsistency_reason =
        incoherence_reason;
    stats->augmented_requested_supported_coherence_ok =
        caller_requested_domain_coherence_ok &&
        !(covered_count > 0u &&
          !caller_requested_crossing_slots_for_this_call.empty() &&
          caller_requested_supported_crossing_slots_exact.empty());
  };
  auto finishWith = [&](const bool ok,
                        const std::string& mode,
                        const std::string& reason) -> bool {
    if (plan) {
      plan->future_summary_insertion.implemented = ok;
      plan->future_summary_insertion.mode = mode;
      plan->future_summary_insertion.pending_reason = ok ? "none" : reason;
    }
    if (stats) {
      stats->summary_implemented_this_epoch = ok;
      stats->summary_target_coherence_ok = target_coherence_ok;
      stats->summary_mode = mode;
      stats->summary_crossing_candidate_slots_count =
          strict_crossing_candidate_slots_count;
      stats->summary_crossing_selected_slots_count =
          strict_crossing_selected_slots_count;
      stats->summary_crossing_dropped_slots_count =
          strict_crossing_dropped_slots_count;
      stats->boundary_crossing_factor_slots_count =
          boundary_crossing_factor_slots_set.size();
      stats->boundary_crossing_factor_slots_first_few = format_slots_set_preview(
          boundary_crossing_factor_slots_set, 8u);
      stats->local_crossing_factor_slots_count =
          local_crossing_factor_slots_set.size();
      stats->local_crossing_factor_slots_first_few = format_slots_set_preview(
          local_crossing_factor_slots_set, 8u);
      stats->nonbelief_local_crossing_factor_slots_count =
          nonbelief_local_crossing_factor_slots_set.size();
      stats->nonbelief_local_crossing_factor_slots_first_few =
          format_slots_set_preview(nonbelief_local_crossing_factor_slots_set,
                                   8u);
      stats->selected_local_nonbelief_crossing_factor_slots_count =
          selected_local_nonbelief_crossing_factor_slots_set.size();
      stats->selected_local_nonbelief_crossing_factor_slots_first_few =
          format_slots_set_preview(
              selected_local_nonbelief_crossing_factor_slots_set, 8u);
      stats->stale_state_keys_count = stale_state_keys_count;
      stats->kept_state_keys_count = kept_state_keys_count;
      stats->stale_pose_keys_count = stale_pose_keys_count;
      stats->kept_pose_keys_count = kept_pose_keys_count;
      stats->stale_local_factor_slots_count = stale_local_factor_slots_count;
      stats->candidate_factors_touching_stale_count =
          candidate_factors_touching_stale_count;
      stats->candidate_factors_touching_kept_count =
          candidate_factors_touching_kept_count;
      stats->candidate_factors_touching_both_stale_and_kept_count =
          candidate_factors_touching_both_stale_and_kept_count;
      stats->boundary_candidate_crossing_factor_slots_count_raw =
          boundary_candidate_crossing_factor_slots_raw_set.size();
      stats->boundary_candidate_crossing_factor_slots_first_few_raw =
          format_slots_set_preview(boundary_candidate_crossing_factor_slots_raw_set,
                                   8u);
      stats->rejected_candidate_due_to_no_stale_incidence_count =
          rejected_candidate_due_to_no_stale_incidence_count;
      stats->rejected_candidate_due_to_no_kept_incidence_count =
          rejected_candidate_due_to_no_kept_incidence_count;
      stats->rejected_candidate_due_to_nonlocal_ownership_count =
          rejected_candidate_due_to_nonlocal_ownership_count;
      stats->rejected_candidate_due_to_belief_factor_count =
          rejected_candidate_due_to_belief_factor_count;
      stats->rejected_candidate_due_to_missing_factor_ptr_count =
          rejected_candidate_due_to_missing_factor_ptr_count;
      stats->first_boundary_candidate_rejected_slot =
          first_boundary_candidate_rejected_slot;
      stats->first_boundary_candidate_rejected_slot_class =
          first_boundary_candidate_rejected_slot_class;
      stats->first_boundary_candidate_rejected_slot_keys =
          first_boundary_candidate_rejected_slot_keys;
      stats->first_boundary_candidate_rejected_reason =
          first_boundary_candidate_rejected_reason;
      stats->rejected_crossing_due_to_nonlocal_count =
          rejected_crossing_due_to_nonlocal_count;
      stats->rejected_crossing_due_to_belief_count =
          rejected_crossing_due_to_belief_count;
      stats->rejected_crossing_due_to_missing_estimate_count =
          rejected_crossing_due_to_missing_estimate_count;
      stats->rejected_crossing_due_to_target_mismatch_count =
          rejected_crossing_due_to_target_mismatch_count;
      stats->first_rejected_crossing_slot = strict_first_dropped_crossing_slot;
      stats->first_rejected_crossing_slot_class =
          strict_first_dropped_crossing_slot_class;
      stats->first_rejected_crossing_slot_keys =
          strict_first_dropped_crossing_slot_keys;
      stats->first_rejected_crossing_reason =
          strict_first_dropped_crossing_slot_reason;
      stats->summary_first_dropped_crossing_slot =
          strict_first_dropped_crossing_slot;
      stats->summary_first_dropped_crossing_slot_class =
          strict_first_dropped_crossing_slot_class;
      stats->summary_first_dropped_crossing_slot_keys =
          strict_first_dropped_crossing_slot_keys;
      stats->summary_first_dropped_crossing_slot_reason =
          strict_first_dropped_crossing_slot_reason;
      stats->summary_first_selected_crossing_slot =
          strict_first_selected_crossing_slot;
      stats->summary_first_selected_crossing_slot_class =
          strict_first_selected_crossing_slot_class;
      stats->summary_first_selected_crossing_slot_keys =
          strict_first_selected_crossing_slot_keys;
      stats->summary_first_covered_crossing_slot =
          strict_first_covered_crossing_slot;
      stats->summary_first_covered_crossing_slot_class =
          strict_first_covered_crossing_slot_class;
      stats->summary_first_covered_crossing_slot_keys =
          strict_first_covered_crossing_slot_keys;
      if (!ok && stats->first_summary_failure_reason.empty()) {
        stats->first_summary_failure_reason = reason;
      }
      updateSummaryCoverageStats();
      stats->summary_build_ms =
          elapsedMs(start, std::chrono::steady_clock::now());
    }
    return ok;
  };

  if (!plan) {
    return finishWith(false, "failed", "null_plan");
  }
  if (!out_factors) {
    return finishWith(false, "failed", "null_output_factor_graph");
  }
  if (!bpsam_) {
    return finishWith(false, "failed", "bpsam_not_set");
  }

  if (stats) {
    stats->summary_crossing_factor_count =
        plan->boundary_candidates.crossing_factor_slots.size();
    strict_crossing_candidate_slots_count =
        plan->boundary_candidates.crossing_factor_slots.size();
    stats->expanded_summary_crossing_factor_count =
        plan->boundary_candidates.crossing_factor_slots.size();
  }

  std::vector<Key> requested_targets(
      plan->future_summary_insertion.summary_target_keys.begin(),
      plan->future_summary_insertion.summary_target_keys.end());
  std::sort(requested_targets.begin(), requested_targets.end());
  if (stats) {
    stats->summary_requested_target_count = requested_targets.size();
  }

  std::vector<Key> expanded_requested_targets(
      plan->boundary_candidates.separator_keys.begin(),
      plan->boundary_candidates.separator_keys.end());
  std::sort(expanded_requested_targets.begin(), expanded_requested_targets.end());
  if (stats) {
    stats->expanded_summary_requested_target_count =
        expanded_requested_targets.size();
  }

  if (plan->stale_eviction.stale_local_keys.empty()) {
    return finishWith(false, "failed", "no_stale_local_keys");
  }

  gtsam::Values estimate;
  try {
    estimate = bpsam_->calculateEstimate();
  } catch (const std::exception& e) {
    return finishWith(false,
                      "failed",
                      std::string("calculateEstimate_failed:") + e.what());
  }

  const auto& graph = bpsam_->getFactorsUnsafe();
  graph_ptr = &graph;
  const Key invalid_key = std::numeric_limits<Key>::max();
  const gtsam::FactorIndex invalid_slot =
      std::numeric_limits<gtsam::FactorIndex>::max();
  bool expanded_vetoed = false;
  auto noteExpandedFailure = [&](const std::string& reason) {
    if (!stats) {
      return;
    }
    if (stats->first_expanded_summary_failure_reason.empty()) {
      stats->first_expanded_summary_failure_reason = reason;
    }
  };
  auto noteExpandedExceptionContext = [&](const std::string& context) {
    if (!stats || context.empty()) {
      return;
    }
    if (stats->first_expanded_exception_context.empty()) {
      stats->first_expanded_exception_context = context;
    }
  };
  auto noteExpandedVeto = [&](const std::string& reason,
                              const Key bad_key,
                              const gtsam::FactorIndex bad_slot,
                              const std::string& context) {
    if (!stats || reason.empty()) {
      return;
    }
    expanded_vetoed = true;
    if (stats->expanded_veto_count == 0u) {
      stats->expanded_veto_count = 1u;
      stats->expanded_veto_reason = reason;
      if (bad_key != invalid_key &&
          stats->first_expanded_bad_key == invalid_key) {
        stats->first_expanded_bad_key = bad_key;
      }
      if (bad_slot != invalid_slot &&
          stats->first_expanded_bad_factor_slot == invalid_slot) {
        stats->first_expanded_bad_factor_slot = bad_slot;
      }
      if (!context.empty() && stats->first_expanded_exception_context.empty()) {
        stats->first_expanded_exception_context = context;
      }
    }
  };
  auto formatSlotsPreview = [](const gtsam::FactorIndices& slots,
                               const size_t max_items) {
    std::ostringstream oss;
    const size_t limit = std::min(max_items, slots.size());
    for (size_t i = 0; i < limit; ++i) {
      if (i > 0u) {
        oss << ",";
      }
      oss << slots[i];
    }
    if (slots.size() > limit) {
      oss << ",...";
    }
    return oss.str();
  };
  auto formatKeySet = [](const std::unordered_set<Key>& keys,
                         const size_t max_items) {
    if (keys.empty()) {
      return std::string("none");
    }
    std::vector<Key> ordered(keys.begin(), keys.end());
    std::sort(ordered.begin(), ordered.end());
    std::ostringstream oss;
    const size_t limit = std::min(max_items, ordered.size());
    for (size_t i = 0; i < limit; ++i) {
      if (i > 0u) {
        oss << "|";
      }
      oss << gtsam::DefaultKeyFormatter(ordered[i]);
    }
    if (ordered.size() > limit) {
      oss << "|...";
    }
    return oss.str();
  };
  auto formatFactorKeysForDiag = [](const gtsam::NonlinearFactor::shared_ptr& factor) {
    if (!factor) {
      return std::string("none");
    }
    std::ostringstream oss;
    for (size_t i = 0u; i < factor->keys().size(); ++i) {
      if (i > 0u) {
        oss << "|";
      }
      oss << gtsam::DefaultKeyFormatter(factor->keys()[i]);
    }
    return oss.str().empty() ? std::string("none") : oss.str();
  };
  auto classifyFactorClassForDiag =
      [&](const gtsam::NonlinearFactor::shared_ptr& factor) {
        if (!factor) {
          return std::string("missing_slot");
        }
        if (isBeliefFactor(factor)) {
          return std::string("belief");
        }
        const size_t arity = factor->keys().size();
        if (arity == 1u) {
          return std::string("unary");
        }
        if (arity == 2u) {
          return std::string("binary");
        }
        if (arity >= 5u) {
          return std::string("imu_like");
        }
        return std::string("other");
      };
  auto make_retained_support_incident_id =
      [](const std::string& blocking_key,
         const gtsam::FactorIndex blocking_slot,
         const std::string& blocking_slot_class,
         const std::string& blocking_slot_keys) {
        std::ostringstream oss;
        oss << "key=" << (blocking_key.empty() ? "none" : blocking_key)
            << "|slot=";
        if (blocking_slot == std::numeric_limits<gtsam::FactorIndex>::max()) {
          oss << "none";
        } else {
          oss << blocking_slot;
        }
        oss << "|class="
            << (blocking_slot_class.empty() ? "none" : blocking_slot_class)
            << "|keys="
            << (blocking_slot_keys.empty() ? "none" : blocking_slot_keys);
        return oss.str();
      };
  auto isFrame0BootstrapStateKey = [](const Key key) {
    const gtsam::Symbol sym(key);
    return sym.index() == 0u &&
           (sym.chr() == kPoseSymbolChar || sym.chr() == kVelocitySymbolChar ||
            sym.chr() == kImuBiasSymbolChar);
  };
  auto isBootstrapSupportPriorFactor =
      [&](const gtsam::NonlinearFactor::shared_ptr& factor,
          Key* support_key_out) {
        if (support_key_out) {
          *support_key_out = invalid_key;
        }
        if (!factor || factor->keys().size() != 1u) {
          return false;
        }
        const Key key = factor->keys().front();
        if (!isFrame0BootstrapStateKey(key)) {
          return false;
        }
        const bool is_pose_prior =
            dynamic_cast<const gtsam::PriorFactor<gtsam::Pose3>*>(factor.get()) !=
            nullptr;
        const bool is_vel_prior =
            dynamic_cast<const gtsam::PriorFactor<gtsam::Vector3>*>(
                factor.get()) != nullptr;
        const bool is_bias_prior =
            dynamic_cast<const gtsam::PriorFactor<gtsam::imuBias::ConstantBias>*>(
                factor.get()) != nullptr;
        if (!(is_pose_prior || is_vel_prior || is_bias_prior)) {
          return false;
        }
        if (support_key_out) {
          *support_key_out = key;
        }
        return true;
      };
  auto noteBoundaryCandidateRejected =
      [&](const gtsam::FactorIndex slot,
          const gtsam::NonlinearFactor::shared_ptr& factor,
          const std::string& reason) {
        if (first_boundary_candidate_rejected_slot >= 0) {
          return;
        }
        first_boundary_candidate_rejected_slot = static_cast<long long>(slot);
        first_boundary_candidate_rejected_slot_class =
            classifyFactorClassForDiag(factor);
        first_boundary_candidate_rejected_slot_keys =
            formatFactorKeysForDiag(factor);
        first_boundary_candidate_rejected_reason = reason;
      };
  const std::unordered_set<Key> orphan_pruned_keys = [&]() {
    std::unordered_set<Key> keys;
    keys.insert(plan->orphan_prune.orphan_robot_keys.begin(),
                plan->orphan_prune.orphan_robot_keys.end());
    keys.insert(plan->orphan_prune.orphan_gbp_keys.begin(),
                plan->orphan_prune.orphan_gbp_keys.end());
    keys.insert(plan->orphan_prune.orphan_consensus_keys.begin(),
                plan->orphan_prune.orphan_consensus_keys.end());
    return keys;
  }();
  const std::unordered_set<Key> belief_owned_keys = [&]() {
    std::unordered_set<Key> keys;
    keys.insert(plan->belief_ownership.robot_keys.begin(),
                plan->belief_ownership.robot_keys.end());
    keys.insert(plan->belief_ownership.gbp_keys.begin(),
                plan->belief_ownership.gbp_keys.end());
    keys.insert(plan->belief_ownership.consensus_keys.begin(),
                plan->belief_ownership.consensus_keys.end());
    return keys;
  }();

  enum class FactorClass {
    kLocalNonBelief,
    kMixedNonBelief,
    kBelief
  };
  auto classifyFactor = [&](const gtsam::NonlinearFactor::shared_ptr& factor) {
    if (factor && isBeliefFactor(factor)) {
      return FactorClass::kBelief;
    }
    bool all_local = true;
    if (factor) {
      for (const Key key : factor->keys()) {
        if (!isLocalStateKey(key)) {
          all_local = false;
          break;
        }
      }
    }
    return all_local ? FactorClass::kLocalNonBelief : FactorClass::kMixedNonBelief;
  };
  auto markExpandedIncludedClass = [&](const FactorClass klass) {
    if (!stats) {
      return;
    }
    if (klass == FactorClass::kBelief) {
      ++stats->expanded_included_belief_factor_count;
    } else if (klass == FactorClass::kLocalNonBelief) {
      ++stats->expanded_included_local_nonbelief_factor_count;
    } else {
      ++stats->expanded_included_mixed_nonbelief_factor_count;
    }
  };
  auto markExpandedExcludedClass = [&](const FactorClass klass) {
    if (!stats) {
      return;
    }
    if (klass == FactorClass::kBelief) {
      ++stats->expanded_excluded_belief_factor_count;
    } else if (klass == FactorClass::kLocalNonBelief) {
      ++stats->expanded_excluded_local_nonbelief_factor_count;
    } else {
      ++stats->expanded_excluded_mixed_nonbelief_factor_count;
    }
  };

  // Strict first working coverage path:
  // LOCAL non-belief stale->kept crossing factors only.
  std::unordered_set<gtsam::FactorIndex> strict_crossing_candidate_set(
      plan->boundary_candidates.crossing_factor_slots.begin(),
      plan->boundary_candidates.crossing_factor_slots.end());
  if (augmented_required_crossing_slots) {
    strict_crossing_candidate_set.insert(
        augmented_required_crossing_slots->begin(),
        augmented_required_crossing_slots->end());
  }
  std::vector<gtsam::FactorIndex> strict_crossing_candidates(
      strict_crossing_candidate_set.begin(),
      strict_crossing_candidate_set.end());
  std::sort(strict_crossing_candidates.begin(), strict_crossing_candidates.end());
  strict_crossing_candidate_slots_count = strict_crossing_candidates.size();
  boundary_crossing_factor_slots_set.clear();
  boundary_crossing_factor_slots_set.insert(
      plan->boundary_candidates.crossing_factor_slots.begin(),
      plan->boundary_candidates.crossing_factor_slots.end());
  local_crossing_factor_slots_set.clear();
  nonbelief_local_crossing_factor_slots_set.clear();
  selected_local_nonbelief_crossing_factor_slots_set.clear();
  rejected_crossing_due_to_nonlocal_count = 0u;
  rejected_crossing_due_to_belief_count = 0u;
  rejected_crossing_due_to_missing_estimate_count = 0u;
  rejected_crossing_due_to_target_mismatch_count = 0u;
  stale_local_factor_slot_set.clear();
  stale_local_factor_slot_set.insert(
      plan->stale_eviction.stale_local_factor_slots.begin(),
      plan->stale_eviction.stale_local_factor_slots.end());
  stale_belief_factor_slot_set.clear();
  stale_belief_factor_slot_set.insert(
      plan->stale_eviction.stale_belief_factor_slots.begin(),
      plan->stale_eviction.stale_belief_factor_slots.end());
  orphan_belief_factor_slot_set.clear();
  orphan_belief_factor_slot_set.insert(
      plan->orphan_prune.orphan_belief_factor_slots.begin(),
      plan->orphan_prune.orphan_belief_factor_slots.end());
  stale_local_key_set.clear();
  stale_local_key_set.insert(plan->stale_eviction.stale_local_keys.begin(),
                             plan->stale_eviction.stale_local_keys.end());
  stale_state_keys_count = stale_local_key_set.size();
  stale_pose_keys_count = 0u;
  for (const Key key : stale_local_key_set) {
    if (isLocalPoseKey(key)) {
      ++stale_pose_keys_count;
    }
  }
  kept_state_keys_count = 0u;
  kept_pose_keys_count = 0u;
  for (const Key key : plan->local_ownership.all_local_state_keys) {
    if (stale_local_key_set.count(key) > 0u) {
      continue;
    }
    ++kept_state_keys_count;
    if (isLocalPoseKey(key)) {
      ++kept_pose_keys_count;
    }
  }
  stale_local_factor_slots_count = stale_local_factor_slot_set.size();
  candidate_factors_touching_stale_count = 0u;
  candidate_factors_touching_kept_count = 0u;
  candidate_factors_touching_both_stale_and_kept_count = 0u;
  boundary_candidate_crossing_factor_slots_raw_set.clear();
  rejected_candidate_due_to_no_stale_incidence_count = 0u;
  rejected_candidate_due_to_no_kept_incidence_count = 0u;
  rejected_candidate_due_to_nonlocal_ownership_count = 0u;
  rejected_candidate_due_to_belief_factor_count = 0u;
  rejected_candidate_due_to_missing_factor_ptr_count = 0u;
  for (const gtsam::FactorIndex slot : stale_local_factor_slot_set) {
    if (!factorExists(graph, slot)) {
      ++rejected_candidate_due_to_missing_factor_ptr_count;
      noteBoundaryCandidateRejected(slot, nullptr, "missing_factor_ptr");
      continue;
    }
    const auto factor = graph.at(slot);
    if (!factor) {
      ++rejected_candidate_due_to_missing_factor_ptr_count;
      noteBoundaryCandidateRejected(slot, factor, "missing_factor_ptr");
      continue;
    }

    bool touches_stale = false;
    bool touches_kept = false;
    bool all_local = true;
    for (const Key key : factor->keys()) {
      const bool key_stale = stale_local_key_set.count(key) > 0u;
      touches_stale = touches_stale || key_stale;
      touches_kept = touches_kept || !key_stale;
      if (!isLocalStateKey(key)) {
        all_local = false;
      }
    }
    if (touches_stale) {
      ++candidate_factors_touching_stale_count;
    }
    if (touches_kept) {
      ++candidate_factors_touching_kept_count;
    }
    if (touches_stale && touches_kept) {
      ++candidate_factors_touching_both_stale_and_kept_count;
      boundary_candidate_crossing_factor_slots_raw_set.insert(slot);
    } else if (!touches_stale) {
      ++rejected_candidate_due_to_no_stale_incidence_count;
      noteBoundaryCandidateRejected(slot, factor, "no_stale_incidence");
      continue;
    } else {
      ++rejected_candidate_due_to_no_kept_incidence_count;
      noteBoundaryCandidateRejected(slot, factor, "no_kept_incidence");
      continue;
    }

    if (isBeliefFactor(factor)) {
      ++rejected_candidate_due_to_belief_factor_count;
      noteBoundaryCandidateRejected(slot, factor, "belief_factor");
      continue;
    }
    if (!all_local) {
      ++rejected_candidate_due_to_nonlocal_ownership_count;
      noteBoundaryCandidateRejected(slot, factor, "nonlocal_ownership");
      continue;
    }
  }

  // Split lag-derived slots into:
  // 1) summary-required local crossing
  // 2) directly-removable local stale-internal
  // 3) unsupported/deferred.
  summary_required_crossing_slots.clear();
  directly_removable_local_internal_candidate_slots.clear();
  directly_removable_local_internal_slots.clear();
  retained_protected_local_support_slots.clear();
  unsupported_deferred_lag_slots.clear();
  requested_lag_slot_set.clear();
  requested_lag_slot_set.insert(plan->remove_factor_indices.begin(),
                                plan->remove_factor_indices.end());
  if (augmented_required_crossing_slots) {
    requested_lag_slot_set.insert(augmented_required_crossing_slots->begin(),
                                  augmented_required_crossing_slots->end());
  }
  unsupported_deferred_lag_slots.insert(stale_belief_factor_slot_set.begin(),
                                        stale_belief_factor_slot_set.end());
  unsupported_deferred_lag_slots.insert(orphan_belief_factor_slot_set.begin(),
                                        orphan_belief_factor_slot_set.end());

  for (const gtsam::FactorIndex slot : requested_lag_slot_set) {
    if (!factorExists(graph, slot)) {
      continue;
    }
    const auto factor = graph.at(slot);
    Key support_key = invalid_key;
    if (!isBootstrapSupportPriorFactor(factor, &support_key) ||
        stale_local_key_set.count(support_key) == 0u) {
      continue;
    }
    retained_protected_local_support_slots.insert(slot);
    if (first_retained_protected_local_support_slot_class == "none") {
      first_retained_protected_local_support_slot_class =
          classifyFactorClassForDiag(factor);
      first_retained_protected_local_support_slot_keys =
          formatFactorKeysForDiag(factor);
    }
  }

  for (const gtsam::FactorIndex slot : requested_lag_slot_set) {
    if (stale_belief_factor_slot_set.count(slot) > 0u ||
        orphan_belief_factor_slot_set.count(slot) > 0u) {
      unsupported_deferred_lag_slots.insert(slot);
      continue;
    }
    if (!factorExists(graph, slot)) {
      unsupported_deferred_lag_slots.insert(slot);
      continue;
    }
    const auto factor = graph.at(slot);
    if (!factor || isBeliefFactor(factor)) {
      unsupported_deferred_lag_slots.insert(slot);
      continue;
    }
    bool all_local = true;
    bool touches_stale = false;
    bool touches_kept = false;
    bool all_stale = true;
    for (const Key key : factor->keys()) {
      const bool key_stale = stale_local_key_set.count(key) > 0u;
      touches_stale = touches_stale || key_stale;
      touches_kept = touches_kept || !key_stale;
      all_stale = all_stale && key_stale;
      if (!isLocalStateKey(key)) {
        all_local = false;
      }
    }
    if (!all_local) {
      unsupported_deferred_lag_slots.insert(slot);
      continue;
    }
    if (retained_protected_local_support_slots.count(slot) > 0u) {
      unsupported_deferred_lag_slots.insert(slot);
      continue;
    }
    if (touches_stale && touches_kept) {
      summary_required_crossing_slots.insert(slot);
    } else if (touches_stale && all_stale) {
      directly_removable_local_internal_candidate_slots.insert(slot);
    } else {
      unsupported_deferred_lag_slots.insert(slot);
    }
  }

  auto noteDirectLocalInternalRejected =
      [&](const gtsam::FactorIndex candidate_slot,
          const gtsam::NonlinearFactor::shared_ptr& candidate_factor,
          const std::string& reason,
          const Key blocking_key,
          const gtsam::FactorIndex blocking_slot,
          const gtsam::NonlinearFactor::shared_ptr& blocking_factor) {
        ++direct_local_internal_rejected_due_to_incomplete_component_count;
        const bool retained_support_rejection =
            reason == "blocking_incident_retained_protected_support_factor" ||
            reason == "blocking_incident_protected_bootstrap_prior";
        if (retained_support_rejection) {
          direct_local_internal_rejected_due_to_retained_support_slots.insert(
              candidate_slot);
          const std::string blocking_key_token =
              blocking_key == invalid_key
                  ? std::string("none")
                  : gtsam::DefaultKeyFormatter(blocking_key);
          const std::string blocking_slot_class =
              classifyFactorClassForDiag(blocking_factor);
          const std::string blocking_slot_keys =
              formatFactorKeysForDiag(blocking_factor);
          const std::string incident_id = make_retained_support_incident_id(
              blocking_key_token,
              blocking_slot,
              blocking_slot_class,
              blocking_slot_keys);
          auto& incident =
              direct_local_retained_support_incidents[incident_id];
          incident.member_slots.insert(candidate_slot);
          incident.blocking_key = blocking_key_token;
          if (blocking_slot ==
              std::numeric_limits<gtsam::FactorIndex>::max()) {
            incident.blocking_support_slot = -1;
          } else {
            incident.blocking_support_slot =
                static_cast<long long>(blocking_slot);
          }
          incident.blocking_support_slot_class =
              blocking_slot_class.empty() ? "none" : blocking_slot_class;
          incident.blocking_support_slot_keys =
              blocking_slot_keys.empty() ? "none" : blocking_slot_keys;
          if (first_direct_local_internal_rejected_retained_support_incident_id ==
              "none") {
            first_direct_local_internal_rejected_retained_support_incident_id =
                incident_id;
          }
        }
        if (first_direct_local_internal_rejected_slot >= 0) {
          return;
        }
        first_direct_local_internal_rejected_slot =
            static_cast<long long>(candidate_slot);
        first_direct_local_internal_rejected_slot_class =
            classifyFactorClassForDiag(candidate_factor);
        first_direct_local_internal_rejected_slot_keys =
            formatFactorKeysForDiag(candidate_factor);
        first_direct_local_internal_rejected_reason = reason;
        if (blocking_key != invalid_key) {
          first_direct_local_internal_rejected_blocking_key =
              gtsam::DefaultKeyFormatter(blocking_key);
        }
        if (blocking_slot != std::numeric_limits<gtsam::FactorIndex>::max()) {
          first_direct_local_internal_rejected_blocking_slot =
              static_cast<long long>(blocking_slot);
          first_direct_local_internal_rejected_blocking_slot_class =
              classifyFactorClassForDiag(blocking_factor);
          first_direct_local_internal_rejected_blocking_slot_keys =
              formatFactorKeysForDiag(blocking_factor);
        }
        if (retained_support_rejection &&
            first_direct_local_internal_rejected_due_to_retained_support_slot <
                0) {
          first_direct_local_internal_rejected_due_to_retained_support_slot =
              static_cast<long long>(candidate_slot);
          first_direct_local_internal_rejected_due_to_retained_support_reason = reason;
        }
      };

  auto refreshDirectLocalInternalClassification = [&]() {
    directly_removable_local_internal_slots.clear();

    // A crossing slot is only safe as stale-key support if it is covered by
    // the summary emitted in this same epoch.
    std::unordered_set<gtsam::FactorIndex> summary_covered_crossing;
    summary_covered_crossing.reserve(summary_selected_slots_for_coverage.size());
    for (const gtsam::FactorIndex slot : summary_selected_slots_for_coverage) {
      if (summary_required_crossing_slots.count(slot) > 0u) {
        summary_covered_crossing.insert(slot);
      }
    }

    std::unordered_set<gtsam::FactorIndex> allowed_incident_slots =
        summary_covered_crossing;

    const auto& variable_index = bpsam_->getVariableIndex();
    std::vector<gtsam::FactorIndex> ordered_candidates(
        directly_removable_local_internal_candidate_slots.begin(),
        directly_removable_local_internal_candidate_slots.end());
    std::sort(ordered_candidates.begin(), ordered_candidates.end());
    std::unordered_set<gtsam::FactorIndex> tentative_direct =
        directly_removable_local_internal_candidate_slots;
    auto evaluateDirectCandidate =
        [&](const gtsam::FactorIndex candidate_slot,
            const std::unordered_set<gtsam::FactorIndex>& direct_membership,
            std::string* reason_out,
            Key* blocking_key_out,
            gtsam::FactorIndex* blocking_slot_out,
            gtsam::NonlinearFactor::shared_ptr* blocking_factor_out) {
          if (reason_out) {
            *reason_out = "none";
          }
          if (blocking_key_out) {
            *blocking_key_out = invalid_key;
          }
          if (blocking_slot_out) {
            *blocking_slot_out = std::numeric_limits<gtsam::FactorIndex>::max();
          }
          if (blocking_factor_out) {
            *blocking_factor_out = nullptr;
          }
          if (!factorExists(graph, candidate_slot)) {
            if (reason_out) {
              *reason_out = "candidate_factor_missing";
            }
            return false;
          }
          const auto candidate_factor = graph.at(candidate_slot);
          if (!candidate_factor) {
            if (reason_out) {
              *reason_out = "candidate_factor_null";
            }
            return false;
          }

          for (const Key key : candidate_factor->keys()) {
            if (!isLocalStateKey(key) || stale_local_key_set.count(key) == 0u) {
              continue;
            }

            const auto key_it = variable_index.find(key);
            if (key_it == variable_index.end()) {
              if (reason_out) {
                *reason_out = "blocking_key_missing_variable_index";
              }
              if (blocking_key_out) {
                *blocking_key_out = key;
              }
              return false;
            }

            for (const gtsam::FactorIndex incident_slot : key_it->second) {
              if (incident_slot == candidate_slot ||
                  !factorExists(graph, incident_slot)) {
                continue;
              }
              if (retained_protected_local_support_slots.count(incident_slot) >
                  0u) {
                if (reason_out) {
                  *reason_out =
                      "blocking_incident_retained_protected_support_factor";
                }
                if (blocking_key_out) {
                  *blocking_key_out = key;
                }
                if (blocking_slot_out) {
                  *blocking_slot_out = incident_slot;
                }
                if (blocking_factor_out) {
                  *blocking_factor_out = graph.at(incident_slot);
                }
                return false;
              }
              if (allowed_incident_slots.count(incident_slot) > 0u ||
                  direct_membership.count(incident_slot) > 0u) {
                continue;
              }

              const auto incident_factor = graph.at(incident_slot);
              std::string reason = "blocking_incident_retained_factor";
              if (!incident_factor) {
                reason = "blocking_incident_null_factor";
              } else if (isBeliefFactor(incident_factor)) {
                reason = "blocking_incident_belief_factor";
              } else if (summary_required_crossing_slots.count(incident_slot) > 0u &&
                         summary_covered_crossing.count(incident_slot) == 0u) {
                reason = "blocking_incident_crossing_uncovered";
              } else if (unsupported_deferred_lag_slots.count(incident_slot) > 0u) {
                reason = "blocking_incident_unsupported_lag_factor";
              } else if (directly_removable_local_internal_candidate_slots.count(
                             incident_slot) > 0u) {
                reason = "blocking_incident_candidate_not_component_complete";
              } else {
                Key support_key = invalid_key;
                if (isBootstrapSupportPriorFactor(incident_factor, &support_key)) {
                  reason = "blocking_incident_protected_bootstrap_prior";
                } else {
                  bool incident_all_local = true;
                  for (const Key incident_key : incident_factor->keys()) {
                    if (!isLocalStateKey(incident_key)) {
                      incident_all_local = false;
                      break;
                    }
                  }
                  if (!incident_all_local) {
                    reason = "blocking_incident_nonlocal_factor";
                  }
                }
              }

              if (reason_out) {
                *reason_out = reason;
              }
              if (blocking_key_out) {
                *blocking_key_out = key;
              }
              if (blocking_slot_out) {
                *blocking_slot_out = incident_slot;
              }
              if (blocking_factor_out) {
                *blocking_factor_out = incident_factor;
              }
              return false;
            }
          }
          return true;
        };

    bool changed = true;
    while (changed) {
      changed = false;
      for (const gtsam::FactorIndex candidate_slot : ordered_candidates) {
        if (tentative_direct.count(candidate_slot) == 0u) {
          continue;
        }
        std::string reason = "none";
        Key blocking_key = invalid_key;
        gtsam::FactorIndex blocking_slot =
            std::numeric_limits<gtsam::FactorIndex>::max();
        gtsam::NonlinearFactor::shared_ptr blocking_factor;
        const bool ok = evaluateDirectCandidate(candidate_slot,
                                                tentative_direct,
                                                &reason,
                                                &blocking_key,
                                                &blocking_slot,
                                                &blocking_factor);
        if (ok) {
          continue;
        }
        tentative_direct.erase(candidate_slot);
        changed = true;
        const auto candidate_factor =
            factorExists(graph, candidate_slot) ? graph.at(candidate_slot) : nullptr;
        noteDirectLocalInternalRejected(candidate_slot,
                                        candidate_factor,
                                        reason,
                                        blocking_key,
                                        blocking_slot,
                                        blocking_factor);
      }
    }

    directly_removable_local_internal_slots = std::move(tentative_direct);
    for (const gtsam::FactorIndex candidate_slot :
         directly_removable_local_internal_candidate_slots) {
      if (directly_removable_local_internal_slots.count(candidate_slot) == 0u) {
        unsupported_deferred_lag_slots.insert(candidate_slot);
      }
    }
  };

  std::vector<gtsam::FactorIndex> strict_selected_slots;
  strict_selected_slots.reserve(strict_crossing_candidates.size());
  std::unordered_set<Key> strict_selected_keys_set;
  gtsam::NonlinearFactorGraph strict_removed_graph;
  auto noteStrictDroppedCrossing =
      [&](const gtsam::FactorIndex slot,
          const gtsam::NonlinearFactor::shared_ptr& factor,
          const std::string& reason) {
        if (reason == "non_local_factor_not_supported") {
          ++rejected_crossing_due_to_nonlocal_count;
        } else if (reason == "belief_factor_not_supported") {
          ++rejected_crossing_due_to_belief_count;
        } else if (reason == "factor_missing_estimate_key") {
          ++rejected_crossing_due_to_missing_estimate_count;
        } else if (reason == "crossing_slot_not_stale_local" ||
                   reason == "crossing_slot_not_in_summary_required_set") {
          ++rejected_crossing_due_to_target_mismatch_count;
        }
        ++strict_crossing_dropped_slots_count;
        if (strict_first_dropped_crossing_slot < 0) {
          strict_first_dropped_crossing_slot = static_cast<long long>(slot);
          strict_first_dropped_crossing_slot_class =
              classifyFactorClassForDiag(factor);
          strict_first_dropped_crossing_slot_keys =
              formatFactorKeysForDiag(factor);
          strict_first_dropped_crossing_slot_reason = reason;
        }
      };

  for (const gtsam::FactorIndex slot : strict_crossing_candidates) {
    if (stale_local_factor_slot_set.count(slot) == 0u) {
      noteStrictDroppedCrossing(slot, nullptr, "crossing_slot_not_stale_local");
      continue;
    }
    if (summary_required_crossing_slots.count(slot) == 0u) {
      noteStrictDroppedCrossing(slot,
                                factorExists(graph, slot) ? graph.at(slot) : nullptr,
                                "crossing_slot_not_in_summary_required_set");
      continue;
    }
    if (!factorExists(graph, slot)) {
      noteStrictDroppedCrossing(slot, nullptr, "factor_missing");
      continue;
    }
    const auto factor = graph.at(slot);
    if (!factor) {
      noteStrictDroppedCrossing(slot, factor, "null_factor");
      continue;
    }
    if (isBeliefFactor(factor)) {
      noteStrictDroppedCrossing(slot, factor, "belief_factor_not_supported");
      continue;
    }

    bool all_local = true;
    bool all_keys_have_estimate = true;
    for (const Key key : factor->keys()) {
      if (!isLocalStateKey(key)) {
        all_local = false;
      }
      if (!estimate.exists(key)) {
        all_keys_have_estimate = false;
      }
    }
    if (all_local) {
      local_crossing_factor_slots_set.insert(slot);
      nonbelief_local_crossing_factor_slots_set.insert(slot);
    }
    if (!all_local) {
      noteStrictDroppedCrossing(slot, factor, "non_local_factor_not_supported");
      continue;
    }
    if (!all_keys_have_estimate) {
      noteStrictDroppedCrossing(slot, factor, "factor_missing_estimate_key");
      continue;
    }

    strict_selected_slots.push_back(slot);
    selected_local_nonbelief_crossing_factor_slots_set.insert(slot);
    strict_removed_graph.push_back(factor);
    for (const Key key : factor->keys()) {
      strict_selected_keys_set.insert(key);
    }
  }
  strict_crossing_selected_slots_count = strict_selected_slots.size();

  std::vector<Key> strict_realizable_targets;
  strict_realizable_targets.reserve(requested_targets.size());
  for (const Key key : requested_targets) {
    if (strict_selected_keys_set.count(key) == 0u) {
      continue;
    }
    if (!estimate.exists(key)) {
      continue;
    }
    strict_realizable_targets.push_back(key);
  }

  std::string strict_failed_reason = "none";
  if (strict_crossing_candidates.empty() && requested_targets.empty()) {
    // No crossing requested for this epoch: this is a valid no-op summary case,
    // not a materialization failure.
    target_coherence_ok = true;
    summary_selected_slots_for_coverage.clear();
    refreshDirectLocalInternalClassification();
    return finishWith(true, "no_crossing_requested", "none");
  } else if (strict_selected_slots.empty()) {
    strict_failed_reason = "no_local_nonbelief_crossing_factors_selected";
  } else if (strict_realizable_targets.empty()) {
    strict_failed_reason = "no_realizable_local_crossing_targets";
  } else {
    std::unordered_set<Key> strict_target_set(strict_realizable_targets.begin(),
                                              strict_realizable_targets.end());
    gtsam::KeyVector strict_eliminate_keys;
    strict_eliminate_keys.reserve(strict_selected_keys_set.size());
    for (const Key key : strict_selected_keys_set) {
      if (strict_target_set.count(key) == 0u) {
        strict_eliminate_keys.push_back(key);
      }
    }
    std::sort(strict_eliminate_keys.begin(), strict_eliminate_keys.end());
    if (strict_eliminate_keys.empty()) {
      strict_failed_reason = "no_local_crossing_keys_to_eliminate";
    } else {
      gtsam::Values strict_linearization_point;
      for (const Key key : strict_selected_keys_set) {
        if (!estimate.exists(key)) {
          strict_failed_reason = "local_crossing_selected_key_missing_estimate";
          strict_eliminate_keys.clear();
          break;
        }
        strict_linearization_point.insert(key, estimate.at(key));
      }
      if (!strict_eliminate_keys.empty()) {
        try {
          auto strict_gaussian_subgraph =
              strict_removed_graph.linearize(strict_linearization_point);
          if (!strict_gaussian_subgraph || strict_gaussian_subgraph->empty()) {
            strict_failed_reason = "local_crossing_linearized_removed_graph_empty";
          } else {
            const auto strict_elimination_result =
                strict_gaussian_subgraph->eliminatePartialMultifrontal(
                    strict_eliminate_keys, gtsam::EliminatePreferCholesky);
            const auto& strict_summary_gaussian_graph =
                strict_elimination_result.second;
            if (!strict_summary_gaussian_graph ||
                strict_summary_gaussian_graph->empty()) {
              strict_failed_reason = "local_crossing_summary_gaussian_graph_empty";
            } else {
              const gtsam::NonlinearFactorGraph strict_summary_linear_container =
                  gtsam::LinearContainerFactor::ConvertLinearGraph(
                      *strict_summary_gaussian_graph, strict_linearization_point);
              size_t strict_emitted = 0u;
              for (const auto& factor : strict_summary_linear_container) {
                if (!factor) {
                  continue;
                }
                out_factors->push_back(factor);
                ++strict_emitted;
              }
              if (strict_emitted == 0u) {
                strict_failed_reason = "local_crossing_no_summary_factors_emitted";
              } else {
                target_coherence_ok = true;
                if (stats) {
                  stats->summary_requested_target_count = requested_targets.size();
                  stats->summary_realizable_target_count =
                      strict_realizable_targets.size();
                  stats->summary_dropped_target_count =
                      requested_targets.size() - strict_realizable_targets.size();
                  stats->summary_target_key_count =
                      stats->summary_realizable_target_count;
                  stats->summary_input_factor_count = strict_selected_slots.size();
                  stats->summary_eliminated_key_count =
                      strict_eliminate_keys.size();
                  stats->summary_factor_count_emitted = strict_emitted;
                  stats->expanded_summary_mode = "strict_local_crossing_primary";
                }
                std::sort(strict_selected_slots.begin(), strict_selected_slots.end());
                summary_selected_slots_for_coverage.clear();
                summary_selected_slots_for_coverage.insert(
                    strict_selected_slots.begin(), strict_selected_slots.end());
                if (!strict_selected_slots.empty()) {
                  const auto selected_slot = strict_selected_slots.front();
                  strict_first_selected_crossing_slot =
                      static_cast<long long>(selected_slot);
                  strict_first_covered_crossing_slot =
                      static_cast<long long>(selected_slot);
                  if (factorExists(graph, selected_slot)) {
                    const auto selected_factor = graph.at(selected_slot);
                    strict_first_selected_crossing_slot_class =
                        classifyFactorClassForDiag(selected_factor);
                    strict_first_selected_crossing_slot_keys =
                        formatFactorKeysForDiag(selected_factor);
                    strict_first_covered_crossing_slot_class =
                        strict_first_selected_crossing_slot_class;
                    strict_first_covered_crossing_slot_keys =
                        strict_first_selected_crossing_slot_keys;
                  }
                }
                refreshDirectLocalInternalClassification();
                return finishWith(true, "exact_local_crossing_schur", "none");
              }
            }
          }
        } catch (const std::exception& e) {
          strict_failed_reason =
              std::string("local_crossing_partial_elimination_failed:") + e.what();
        }
      }
    }
  }
  if (stats && stats->first_summary_failure_reason.empty() &&
      strict_failed_reason != "none") {
    stats->first_summary_failure_reason = strict_failed_reason;
  }

  // Expanded-graph first step: include crossing stale-side factors that can
  // be safely linearized now (including belief/mixed-key factors when all keys
  // have estimates). Double counting is prevented by summarizing only factors
  // in stale_local_factor_slots that are simultaneously removed this epoch.
  const auto expanded_start = std::chrono::steady_clock::now();
  bool expanded_success = false;
  std::vector<gtsam::FactorIndex> expanded_selected_slots;
  expanded_selected_slots.reserve(plan->stale_eviction.stale_local_factor_slots.size());
  std::unordered_set<Key> expanded_selected_keys_set;
  gtsam::NonlinearFactorGraph expanded_removed_graph;

  for (const gtsam::FactorIndex slot : plan->stale_eviction.stale_local_factor_slots) {
    if (!factorExists(graph, slot)) {
      continue;
    }
    const auto factor = graph.at(slot);
    if (!factor) {
      continue;
    }
    const FactorClass klass = classifyFactor(factor);

    bool touches_stale_local = false;
    bool touches_kept_side = false;
    bool all_keys_have_estimate = true;
    for (const Key key : factor->keys()) {
      if (plan->stale_eviction.stale_local_keys.count(key) > 0u) {
        touches_stale_local = true;
      } else {
        touches_kept_side = true;
      }
      if (!estimate.exists(key)) {
        all_keys_have_estimate = false;
      }
    }
    if (!(touches_stale_local && touches_kept_side)) {
      markExpandedExcludedClass(klass);
      continue;
    }
    if (!all_keys_have_estimate) {
      markExpandedExcludedClass(klass);
      noteExpandedFailure("expanded_crossing_factor_missing_estimate_key");
      continue;
    }

    markExpandedIncludedClass(klass);
    expanded_selected_slots.push_back(slot);
    expanded_removed_graph.push_back(factor);
    for (const Key key : factor->keys()) {
      expanded_selected_keys_set.insert(key);
    }
  }
  std::unordered_set<Key> stale_bootstrap_keys;
  stale_bootstrap_keys.reserve(3u);
  for (const Key key : plan->stale_eviction.stale_local_keys) {
    if (isFrame0BootstrapStateKey(key)) {
      stale_bootstrap_keys.insert(key);
    }
  }
  bool expanded_removed_crossing_uses_bootstrap_support = false;
  for (const Key key : expanded_selected_keys_set) {
    if (stale_bootstrap_keys.count(key) > 0u) {
      expanded_removed_crossing_uses_bootstrap_support = true;
      break;
    }
  }
  gtsam::FactorIndices expanded_support_slots;
  std::unordered_set<Key> expanded_support_keys;
  std::unordered_set<gtsam::FactorIndex> expanded_selected_slot_set(
      expanded_selected_slots.begin(), expanded_selected_slots.end());
  if (expanded_removed_crossing_uses_bootstrap_support) {
    for (const gtsam::FactorIndex slot :
         plan->stale_eviction.stale_local_factor_slots) {
      if (expanded_selected_slot_set.count(slot) > 0u || !factorExists(graph, slot)) {
        continue;
      }
      const auto factor = graph.at(slot);
      Key support_key = invalid_key;
      if (!isBootstrapSupportPriorFactor(factor, &support_key) ||
          stale_bootstrap_keys.count(support_key) == 0u ||
          !estimate.exists(support_key)) {
        continue;
      }
      expanded_support_slots.push_back(slot);
      expanded_support_keys.insert(support_key);
      expanded_removed_graph.push_back(factor);
      expanded_selected_keys_set.insert(support_key);
    }
    std::sort(expanded_support_slots.begin(), expanded_support_slots.end());
    if (expanded_support_slots.empty()) {
      noteExpandedFailure("missing_bootstrap_support_priors");
      if (stats) {
        stats->expanded_summary_support_factor_count = 0u;
        stats->expanded_summary_support_slots_first_few = "none";
        stats->expanded_summary_used_bootstrap_support_priors = false;
        stats->expanded_summary_bootstrap_support_keys = "none";
        stats->expanded_summary_removed_crossing_uses_bootstrap_support = true;
        stats->expanded_summary_mode = "failed";
      }
      refreshDirectLocalInternalClassification();
      return finishWith(false, "failed", "missing_bootstrap_support_priors");
    }
  }
  if (stats) {
    stats->expanded_selected_factor_slots_count = expanded_selected_slots.size();
    stats->expanded_selected_factor_slots.clear();
    stats->expanded_selected_factor_slots.insert(expanded_selected_slots.begin(),
                                                 expanded_selected_slots.end());
    stats->expanded_selected_key_count = expanded_selected_keys_set.size();
    stats->expanded_summary_support_factor_count = expanded_support_slots.size();
    stats->expanded_summary_support_slots_first_few =
        expanded_support_slots.empty()
            ? "none"
            : formatSlotsPreview(expanded_support_slots, 8u);
    stats->expanded_summary_used_bootstrap_support_priors =
        !expanded_support_slots.empty();
    stats->expanded_summary_bootstrap_support_keys =
        formatKeySet(expanded_support_keys, 8u);
    stats->expanded_summary_removed_crossing_uses_bootstrap_support =
        expanded_removed_crossing_uses_bootstrap_support;
  }

  std::vector<Key> expanded_realizable_targets;
  expanded_realizable_targets.reserve(expanded_requested_targets.size());
  for (const Key key : expanded_requested_targets) {
    if (expanded_selected_keys_set.count(key) == 0u) {
      continue;
    }
    if (!estimate.exists(key)) {
      continue;
    }
    expanded_realizable_targets.push_back(key);
  }
  if (stats) {
    stats->expanded_summary_realizable_target_count =
        expanded_realizable_targets.size();
  }

  if (expanded_selected_slots.empty()) {
    noteExpandedFailure("expanded_no_crossing_factors_selected");
  } else if (expanded_realizable_targets.empty()) {
    noteExpandedFailure("expanded_no_realizable_targets");
  } else {
    std::unordered_set<Key> expanded_target_set(expanded_realizable_targets.begin(),
                                                expanded_realizable_targets.end());
    gtsam::KeyVector expanded_eliminate_keys;
    expanded_eliminate_keys.reserve(expanded_selected_keys_set.size());
    for (const Key key : expanded_selected_keys_set) {
      if (expanded_target_set.count(key) == 0u) {
        expanded_eliminate_keys.push_back(key);
      }
    }
    std::sort(expanded_eliminate_keys.begin(), expanded_eliminate_keys.end());
    if (stats) {
      stats->expanded_eliminated_key_count = expanded_eliminate_keys.size();
    }

    if (expanded_eliminate_keys.empty()) {
      noteExpandedFailure("expanded_no_keys_to_eliminate");
    } else {
      gtsam::Values expanded_linearization_point;
      for (const Key key : expanded_selected_keys_set) {
        expanded_linearization_point.insert(key, estimate.at(key));
      }

      try {
        auto expanded_gaussian_subgraph =
            expanded_removed_graph.linearize(expanded_linearization_point);
        if (!expanded_gaussian_subgraph || expanded_gaussian_subgraph->empty()) {
          noteExpandedFailure("expanded_linearized_removed_graph_empty");
        } else {
          const auto elimination_result =
              expanded_gaussian_subgraph->eliminatePartialMultifrontal(
                  expanded_eliminate_keys, gtsam::EliminatePreferCholesky);
          const auto& expanded_summary_gaussian_graph = elimination_result.second;
          if (!expanded_summary_gaussian_graph ||
              expanded_summary_gaussian_graph->empty()) {
            noteExpandedFailure("expanded_summary_gaussian_graph_empty");
          } else {
            const gtsam::NonlinearFactorGraph expanded_summary_linear_container =
                gtsam::LinearContainerFactor::ConvertLinearGraph(
                    *expanded_summary_gaussian_graph, expanded_linearization_point);
            size_t emitted = 0u;
            size_t missing_estimate_keys = 0u;
            bool contains_nonlocal = false;
            bool contains_belief = false;
            bool references_stale_removed = false;
            bool references_orphan_pruned = false;
            std::unordered_set<Key> emitted_keys;
            gtsam::NonlinearFactorGraph expanded_validated_factors;
            size_t emitted_factor_index = 0u;
            for (const auto& factor : expanded_summary_linear_container) {
              if (!factor) {
                ++emitted_factor_index;
                continue;
              }
              ++emitted;
              for (const Key key : factor->keys()) {
                emitted_keys.insert(key);
                if (!isLocalStateKey(key)) {
                  contains_nonlocal = true;
                }
                if (belief_owned_keys.count(key) > 0u) {
                  contains_belief = true;
                }
                if (plan->stale_eviction.stale_local_keys.count(key) > 0u) {
                  references_stale_removed = true;
                  noteExpandedVeto("references_stale_removed_key",
                                   key,
                                   emitted_factor_index,
                                   "expanded_emitted_factor_references_stale_key");
                }
                if (orphan_pruned_keys.count(key) > 0u) {
                  references_orphan_pruned = true;
                  noteExpandedVeto("references_orphan_pruned_key",
                                   key,
                                   emitted_factor_index,
                                   "expanded_emitted_factor_references_orphan_key");
                }
                if (!estimate.exists(key)) {
                  ++missing_estimate_keys;
                  noteExpandedVeto("references_missing_estimate_key",
                                   key,
                                   emitted_factor_index,
                                   "expanded_emitted_factor_missing_estimate_key");
                }
                if (expanded_target_set.count(key) == 0u) {
                  noteExpandedVeto("kept_side_ownership_violation",
                                   key,
                                   emitted_factor_index,
                                   "expanded_emitted_factor_key_outside_kept_target_set");
                }
              }
              expanded_validated_factors.push_back(factor);
              ++emitted_factor_index;
            }
            if (stats) {
              stats->expanded_emitted_factor_count = emitted;
              stats->expanded_emitted_factor_key_count = emitted_keys.size();
              stats->expanded_emitted_factor_contains_nonlocal_keys =
                  contains_nonlocal;
              stats->expanded_emitted_factor_contains_belief_keys =
                  contains_belief;
              stats->expanded_emitted_factor_references_stale_removed_keys =
                  references_stale_removed;
              stats->expanded_emitted_factor_references_orphan_pruned_keys =
                  references_orphan_pruned;
              stats->expanded_emitted_factor_missing_estimate_keys =
                  missing_estimate_keys;
            }
            if (emitted > 0u && !expanded_vetoed) {
              for (const auto& factor : expanded_validated_factors) {
                if (factor) {
                  out_factors->push_back(factor);
                }
              }
              expanded_success = true;
              target_coherence_ok = true;
              if (stats) {
                stats->expanded_summary_factor_count_emitted = emitted;
                stats->summary_requested_target_count =
                    expanded_requested_targets.size();
                stats->summary_realizable_target_count =
                    expanded_realizable_targets.size();
                stats->summary_dropped_target_count =
                    expanded_requested_targets.size() -
                    expanded_realizable_targets.size();
                stats->summary_target_key_count =
                    stats->summary_realizable_target_count;
                stats->summary_input_factor_count =
                    expanded_selected_slots.size();
                stats->summary_eliminated_key_count =
                    expanded_eliminate_keys.size();
                stats->summary_factor_count_emitted = emitted;
              }
            } else {
              if (emitted == 0u) {
                noteExpandedFailure("expanded_no_summary_factors_emitted");
              } else if (expanded_vetoed) {
                noteExpandedFailure("expanded_vetoed");
              }
            }
          }
        }
      } catch (const std::exception& e) {
        noteExpandedExceptionContext(e.what());
        noteExpandedFailure(std::string("expanded_elimination_failed:") + e.what());
      }
    }
  }

  if (stats) {
    stats->expanded_summary_mode = expanded_success
                                       ? "exact_expanded_partial_scope"
                                       : (expanded_vetoed ? "vetoed" : "failed");
    stats->expanded_summary_build_ms =
        elapsedMs(expanded_start, std::chrono::steady_clock::now());
  }
  if (expanded_success) {
    summary_selected_slots_for_coverage.clear();
    summary_selected_slots_for_coverage.insert(expanded_selected_slots.begin(),
                                               expanded_selected_slots.end());
    refreshDirectLocalInternalClassification();
    return finishWith(true, "exact_expanded_partial_scope", "none");
  }

  if (requested_targets.empty()) {
    if (stats) {
      stats->expanded_summary_mode = "failed_local_fallback_failed";
    }
    refreshDirectLocalInternalClassification();
    return finishWith(false, "failed", "no_summary_targets");
  }

  // LOCAL-only fallback keeps the validated baseline path available.
  const std::string fallback_failed_mode =
      expanded_vetoed ? "vetoed_local_fallback_failed"
                      : "failed_local_fallback_failed";
  const std::string fallback_success_mode =
      expanded_vetoed ? "vetoed_local_fallback_exact_schur"
                      : "failed_local_fallback_exact_schur";
  std::vector<gtsam::FactorIndex> selected_slots;
  selected_slots.reserve(plan->stale_eviction.stale_local_factor_slots.size());
  std::unordered_set<Key> selected_keys_set;
  gtsam::NonlinearFactorGraph removed_local_graph;

  for (const gtsam::FactorIndex slot : plan->stale_eviction.stale_local_factor_slots) {
    if (!factorExists(graph, slot)) {
      continue;
    }
    const auto factor = graph.at(slot);
    if (!factor || isBeliefFactor(factor)) {
      continue;
    }
    bool all_local = true;
    bool touches_stale_local = false;
    for (const Key key : factor->keys()) {
      if (!isLocalStateKey(key)) {
        all_local = false;
        break;
      }
      if (plan->stale_eviction.stale_local_keys.count(key) > 0u) {
        touches_stale_local = true;
      }
    }
    if (!all_local || !touches_stale_local) {
      continue;
    }
    selected_slots.push_back(slot);
    removed_local_graph.push_back(factor);
    for (const Key key : factor->keys()) {
      selected_keys_set.insert(key);
    }
  }

  if (selected_slots.empty()) {
    if (stats) {
      stats->expanded_summary_mode = fallback_failed_mode;
    }
    refreshDirectLocalInternalClassification();
    return finishWith(false, "failed", "no_local_stale_factors_selected");
  }
  if (stats) {
    stats->summary_input_factor_count = selected_slots.size();
  }

  std::vector<Key> realizable_targets;
  realizable_targets.reserve(requested_targets.size());
  auto noteDroppedTarget = [&](const std::string& reason) {
    if (!stats) {
      return;
    }
    ++stats->summary_dropped_target_count;
    if (stats->first_dropped_target_reason.empty()) {
      stats->first_dropped_target_reason = reason;
    }
  };
  for (const Key key : requested_targets) {
    if (!selected_keys_set.count(key)) {
      noteDroppedTarget("requested_target_absent_from_selected_local_subgraph");
      continue;
    }
    if (!estimate.exists(key)) {
      noteDroppedTarget("requested_target_missing_estimate");
      continue;
    }
    realizable_targets.push_back(key);
  }
  if (stats) {
    stats->summary_realizable_target_count = realizable_targets.size();
    // Backward-compatible alias for the effective summarized target count.
    stats->summary_target_key_count = stats->summary_realizable_target_count;
  }
  if (realizable_targets.empty()) {
    if (stats) {
      stats->expanded_summary_mode = fallback_failed_mode;
    }
    refreshDirectLocalInternalClassification();
    return finishWith(false, "failed", "no_realizable_summary_targets");
  }
  target_coherence_ok = true;

  std::unordered_set<Key> target_set(realizable_targets.begin(),
                                     realizable_targets.end());

  gtsam::KeyVector eliminate_keys;
  eliminate_keys.reserve(selected_keys_set.size());
  for (const Key key : selected_keys_set) {
    if (target_set.count(key) == 0u) {
      if (!estimate.exists(key)) {
        target_coherence_ok = false;
        if (stats) {
          stats->expanded_summary_mode = fallback_failed_mode;
        }
        refreshDirectLocalInternalClassification();
        return finishWith(false, "failed", "elimination_key_missing_estimate");
      }
      eliminate_keys.push_back(key);
    }
  }
  std::sort(eliminate_keys.begin(), eliminate_keys.end());
  if (stats) {
    stats->summary_eliminated_key_count = eliminate_keys.size();
  }

  if (eliminate_keys.empty()) {
    if (stats) {
      stats->expanded_summary_mode = fallback_failed_mode;
    }
    refreshDirectLocalInternalClassification();
    return finishWith(false, "failed", "no_keys_to_eliminate_for_summary");
  }

  gtsam::Values linearization_point;
  for (const Key key : selected_keys_set) {
    if (!estimate.exists(key)) {
      target_coherence_ok = false;
      if (stats) {
        stats->expanded_summary_mode = fallback_failed_mode;
      }
    refreshDirectLocalInternalClassification();
    return finishWith(false, "failed", "selected_subgraph_key_missing_estimate");
    }
    linearization_point.insert(key, estimate.at(key));
  }

  gtsam::GaussianFactorGraph::shared_ptr gaussian_subgraph;
  try {
    gaussian_subgraph = removed_local_graph.linearize(linearization_point);
  } catch (const std::exception& e) {
    if (stats) {
      stats->expanded_summary_mode = fallback_failed_mode;
    }
    refreshDirectLocalInternalClassification();
    return finishWith(false,
                      "failed",
                      std::string("linearize_removed_graph_failed:") + e.what());
  }

  if (!gaussian_subgraph || gaussian_subgraph->empty()) {
    if (stats) {
      stats->expanded_summary_mode = fallback_failed_mode;
    }
    refreshDirectLocalInternalClassification();
    return finishWith(false, "failed", "linearized_removed_graph_empty");
  }

  decltype(gaussian_subgraph->eliminatePartialMultifrontal(
      eliminate_keys, gtsam::EliminatePreferCholesky))
      elimination_result;
  try {
    elimination_result = gaussian_subgraph->eliminatePartialMultifrontal(
        eliminate_keys, gtsam::EliminatePreferCholesky);
  } catch (const std::exception& e) {
    if (stats) {
      stats->expanded_summary_mode = fallback_failed_mode;
    }
      refreshDirectLocalInternalClassification();
      return finishWith(false,
                        "failed",
                        std::string("partial_elimination_failed:") + e.what());
  }

  const auto& summary_gaussian_graph = elimination_result.second;
  if (!summary_gaussian_graph || summary_gaussian_graph->empty()) {
    if (stats) {
      stats->expanded_summary_mode = fallback_failed_mode;
    }
    refreshDirectLocalInternalClassification();
    return finishWith(false, "failed", "summary_gaussian_graph_empty");
  }

  const gtsam::NonlinearFactorGraph summary_linear_container =
      gtsam::LinearContainerFactor::ConvertLinearGraph(*summary_gaussian_graph,
                                                       linearization_point);
  size_t emitted = 0u;
  for (const auto& factor : summary_linear_container) {
    if (!factor) {
      continue;
    }
    out_factors->push_back(factor);
    ++emitted;
  }
  if (stats) {
    stats->summary_factor_count_emitted = emitted;
  }

  if (emitted == 0u) {
    if (stats) {
      stats->expanded_summary_mode = fallback_failed_mode;
    }
    refreshDirectLocalInternalClassification();
    return finishWith(false, "failed", "no_summary_factors_emitted");
  }

  if (stats) {
    stats->expanded_summary_mode = fallback_success_mode;
  }
  summary_selected_slots_for_coverage.clear();
  summary_selected_slots_for_coverage.insert(selected_slots.begin(),
                                             selected_slots.end());
  refreshDirectLocalInternalClassification();
  return finishWith(true, "exact_schur", "none");
}

bool CbsFixedLagBpsamHeart::appendRootBootstrapDirectLocalSupportFactors(
    const LagWindowPlan& plan,
    const std::unordered_set<gtsam::FactorIndex>& blocked_component_slots,
    gtsam::NonlinearFactorGraph* out_factors,
    std::unordered_set<gtsam::FactorIndex>* out_supported_remove_slots,
    std::string* failure_reason,
    RootBootstrapSupportBuildStats* out_stats) const {
  if (out_stats) {
    *out_stats = RootBootstrapSupportBuildStats{};
  }
  auto format_slots_set_preview =
      [](const std::unordered_set<gtsam::FactorIndex>& slots,
         const size_t max_items) {
        if (slots.empty()) {
          return std::string("none");
        }
        std::vector<gtsam::FactorIndex> ordered(slots.begin(), slots.end());
        std::sort(ordered.begin(), ordered.end());
        std::ostringstream oss;
        const size_t limit = std::min(max_items, ordered.size());
        for (size_t i = 0; i < limit; ++i) {
          if (i > 0u) {
            oss << ",";
          }
          oss << ordered[i];
        }
        if (ordered.size() > limit) {
          oss << ",...";
        }
        return oss.str();
      };
  auto format_keys_set_preview = [](const std::unordered_set<Key>& keys,
                                    const size_t max_items) {
    if (keys.empty()) {
      return std::string("none");
    }
    std::vector<Key> ordered(keys.begin(), keys.end());
    std::sort(ordered.begin(), ordered.end());
    std::ostringstream oss;
    const size_t limit = std::min(max_items, ordered.size());
    for (size_t i = 0; i < limit; ++i) {
      if (i > 0u) {
        oss << "|";
      }
      oss << gtsam::DefaultKeyFormatter(ordered[i]);
    }
    if (ordered.size() > limit) {
      oss << "|...";
    }
    return oss.str();
  };
  auto set_failure = [&](const std::string& reason) {
    if (failure_reason) {
      *failure_reason = reason;
    }
    if (out_stats) {
      out_stats->first_failure_reason = reason;
    }
    return false;
  };
  if (out_factors) {
    out_factors->resize(0u);
  }
  if (out_supported_remove_slots) {
    out_supported_remove_slots->clear();
  }
  if (!out_factors || !out_supported_remove_slots) {
    return set_failure("null_output_container");
  }
  if (!bpsam_) {
    return set_failure("bpsam_not_set");
  }
  if (blocked_component_slots.empty()) {
    return set_failure("no_blocked_component_slots");
  }

  const auto& graph = bpsam_->getFactorsUnsafe();
  gtsam::Values estimate;
  try {
    estimate = bpsam_->calculateEstimate();
  } catch (const std::exception& e) {
    return set_failure(std::string("calculateEstimate_failed:") + e.what());
  }

  const Key invalid_key = std::numeric_limits<Key>::max();
  auto is_frame0_bootstrap_state_key = [](const Key key) {
    const gtsam::Symbol sym(key);
    return sym.index() == 0u &&
           (sym.chr() == kPoseSymbolChar || sym.chr() == kVelocitySymbolChar ||
            sym.chr() == kImuBiasSymbolChar);
  };
  auto is_bootstrap_support_prior_factor =
      [&](const gtsam::NonlinearFactor::shared_ptr& factor,
          Key* support_key_out) {
        if (support_key_out) {
          *support_key_out = invalid_key;
        }
        if (!factor || factor->keys().size() != 1u) {
          return false;
        }
        const Key key = factor->keys().front();
        if (!is_frame0_bootstrap_state_key(key)) {
          return false;
        }
        const bool is_pose_prior =
            dynamic_cast<const gtsam::PriorFactor<gtsam::Pose3>*>(factor.get()) !=
            nullptr;
        const bool is_vel_prior =
            dynamic_cast<const gtsam::PriorFactor<gtsam::Vector3>*>(
                factor.get()) != nullptr;
        const bool is_bias_prior =
            dynamic_cast<const gtsam::PriorFactor<gtsam::imuBias::ConstantBias>*>(
                factor.get()) != nullptr;
        if (!(is_pose_prior || is_vel_prior || is_bias_prior)) {
          return false;
        }
        if (support_key_out) {
          *support_key_out = key;
        }
        return true;
      };

  std::unordered_set<Key> stale_local_key_set(
      plan.stale_eviction.stale_local_keys.begin(),
      plan.stale_eviction.stale_local_keys.end());
  std::vector<gtsam::FactorIndex> ordered_component_slots(
      blocked_component_slots.begin(), blocked_component_slots.end());
  std::sort(ordered_component_slots.begin(), ordered_component_slots.end());

  gtsam::NonlinearFactorGraph selected_graph;
  std::unordered_set<Key> selected_keys_set;
  std::unordered_set<Key> blocked_component_local_keys;
  std::unordered_set<gtsam::FactorIndex> supported_remove_slots_set;
  selected_graph.reserve(ordered_component_slots.size() + 16u);
  for (const gtsam::FactorIndex slot : ordered_component_slots) {
    if (!factorExists(graph, slot)) {
      continue;
    }
    const auto factor = graph.at(slot);
    if (!factor || isBeliefFactor(factor)) {
      continue;
    }
    bool all_local = true;
    for (const Key key : factor->keys()) {
      if (!isLocalStateKey(key)) {
        all_local = false;
        break;
      }
    }
    if (!all_local) {
      continue;
    }
    selected_graph.push_back(factor);
    supported_remove_slots_set.insert(slot);
    for (const Key key : factor->keys()) {
      selected_keys_set.insert(key);
      blocked_component_local_keys.insert(key);
    }
  }
  if (supported_remove_slots_set.empty()) {
    return set_failure("no_supportable_blocked_slots_in_live_graph");
  }

  std::unordered_set<Key> stale_bootstrap_keys;
  for (const Key key : blocked_component_local_keys) {
    if (stale_local_key_set.count(key) > 0u &&
        is_frame0_bootstrap_state_key(key)) {
      stale_bootstrap_keys.insert(key);
    }
  }

  std::unordered_set<gtsam::FactorIndex> added_support_slots;
  for (const gtsam::FactorIndex slot :
       plan.stale_eviction.stale_local_factor_slots) {
    if (supported_remove_slots_set.count(slot) > 0u ||
        added_support_slots.count(slot) > 0u || !factorExists(graph, slot)) {
      continue;
    }
    const auto factor = graph.at(slot);
    Key support_key = invalid_key;
    if (!is_bootstrap_support_prior_factor(factor, &support_key)) {
      continue;
    }
    if (stale_bootstrap_keys.count(support_key) == 0u) {
      continue;
    }
    selected_graph.push_back(factor);
    added_support_slots.insert(slot);
    selected_keys_set.insert(support_key);
  }
  if (!stale_bootstrap_keys.empty() && added_support_slots.empty()) {
    return set_failure("missing_frame0_bootstrap_support_prior");
  }

  // Augment support construction with stale->kept local crossing factors that
  // are incident on the blocked component. These are support-only factors:
  // they must not enter the remove set.
  std::unordered_set<gtsam::FactorIndex> incident_crossing_support_slots;
  for (const gtsam::FactorIndex slot :
       plan.stale_eviction.stale_local_factor_slots) {
    if (supported_remove_slots_set.count(slot) > 0u ||
        added_support_slots.count(slot) > 0u ||
        incident_crossing_support_slots.count(slot) > 0u ||
        !factorExists(graph, slot)) {
      continue;
    }
    const auto factor = graph.at(slot);
    if (!factor || isBeliefFactor(factor)) {
      continue;
    }
    bool all_local = true;
    bool touches_component = false;
    bool touches_kept_local = false;
    for (const Key key : factor->keys()) {
      if (!isLocalStateKey(key)) {
        all_local = false;
        break;
      }
      if (blocked_component_local_keys.count(key) > 0u) {
        touches_component = true;
      }
      if (stale_local_key_set.count(key) == 0u) {
        touches_kept_local = true;
      }
    }
    if (!(all_local && touches_component && touches_kept_local)) {
      continue;
    }
    selected_graph.push_back(factor);
    incident_crossing_support_slots.insert(slot);
    for (const Key key : factor->keys()) {
      selected_keys_set.insert(key);
    }
  }

  std::unordered_set<Key> target_keys_set;
  for (const Key key : selected_keys_set) {
    if (!isLocalStateKey(key)) {
      continue;
    }
    if (stale_local_key_set.count(key) == 0u) {
      target_keys_set.insert(key);
    }
  }

  std::string augment_mode = "component_only_no_incident_crossing";
  if (!incident_crossing_support_slots.empty() && !added_support_slots.empty()) {
    augment_mode = "component_plus_incident_crossing_plus_bootstrap_prior";
  } else if (!incident_crossing_support_slots.empty()) {
    augment_mode = "component_plus_incident_crossing";
  } else if (!added_support_slots.empty()) {
    augment_mode = "component_plus_bootstrap_prior_only";
  }
  if (out_stats) {
    out_stats->incident_crossing_support_slot_count =
        incident_crossing_support_slots.size();
    out_stats->incident_crossing_support_slots_first_few =
        format_slots_set_preview(incident_crossing_support_slots, 16u);
    out_stats->kept_target_key_count = target_keys_set.size();
    out_stats->kept_target_keys_first_few =
        format_keys_set_preview(target_keys_set, 16u);
    out_stats->selected_graph_factor_count = selected_graph.size();
    out_stats->augment_mode = augment_mode;
  }

  if (target_keys_set.empty()) {
    return set_failure("no_kept_side_targets_for_root_bootstrap_component");
  }

  gtsam::KeyVector eliminate_keys;
  eliminate_keys.reserve(selected_keys_set.size());
  gtsam::Values linearization_point;
  for (const Key key : selected_keys_set) {
    if (!estimate.exists(key)) {
      return set_failure(std::string("selected_key_missing_estimate:") +
                         gtsam::DefaultKeyFormatter(key));
    }
    linearization_point.insert(key, estimate.at(key));
    if (target_keys_set.count(key) == 0u) {
      eliminate_keys.push_back(key);
    }
  }
  std::sort(eliminate_keys.begin(), eliminate_keys.end());
  if (out_stats) {
    out_stats->eliminate_key_count = eliminate_keys.size();
  }
  if (eliminate_keys.empty()) {
    return set_failure("no_keys_to_eliminate_for_root_bootstrap_support");
  }

  gtsam::GaussianFactorGraph::shared_ptr gaussian_subgraph;
  try {
    gaussian_subgraph = selected_graph.linearize(linearization_point);
  } catch (const std::exception& e) {
    return set_failure(std::string("linearize_selected_graph_failed:") + e.what());
  }
  if (!gaussian_subgraph || gaussian_subgraph->empty()) {
    return set_failure("linearized_selected_graph_empty");
  }

  decltype(gaussian_subgraph->eliminatePartialMultifrontal(
      eliminate_keys, gtsam::EliminatePreferCholesky))
      elimination_result;
  try {
    elimination_result = gaussian_subgraph->eliminatePartialMultifrontal(
        eliminate_keys, gtsam::EliminatePreferCholesky);
  } catch (const std::exception& e) {
    return set_failure(std::string("partial_elimination_failed:") + e.what());
  }

  const auto& summary_gaussian_graph = elimination_result.second;
  if (!summary_gaussian_graph || summary_gaussian_graph->empty()) {
    return set_failure("summary_gaussian_graph_empty");
  }

  const gtsam::NonlinearFactorGraph summary_linear_container =
      gtsam::LinearContainerFactor::ConvertLinearGraph(*summary_gaussian_graph,
                                                       linearization_point);
  size_t emitted = 0u;
  for (const auto& factor : summary_linear_container) {
    if (!factor) {
      continue;
    }
    out_factors->push_back(factor);
    ++emitted;
  }
  if (emitted == 0u) {
    return set_failure("no_root_bootstrap_support_factors_emitted");
  }

  *out_supported_remove_slots = std::move(supported_remove_slots_set);
  if (failure_reason) {
    *failure_reason = "none";
  }
  if (out_stats) {
    out_stats->first_failure_reason = "none";
  }
  return true;
}

gtsam::FactorIndices CbsFixedLagBpsamHeart::buildRemoveFactorIndices(
    const StaleFrameEvictionResult& stale_eviction,
    const OrphanPruneResult& orphan_prune) const {
  std::unordered_set<gtsam::FactorIndex> slots(
      stale_eviction.stale_local_factor_slots.begin(),
      stale_eviction.stale_local_factor_slots.end());
  slots.insert(stale_eviction.stale_belief_factor_slots.begin(),
               stale_eviction.stale_belief_factor_slots.end());
  slots.insert(orphan_prune.orphan_belief_factor_slots.begin(),
               orphan_prune.orphan_belief_factor_slots.end());

  gtsam::FactorIndices out(slots.begin(), slots.end());
  std::sort(out.begin(), out.end());
  return out;
}

void CbsFixedLagBpsamHeart::refreshBeliefOwnershipFromActiveGraphIfNeeded(
    const bool force) {
  if (!bpsam_) {
    incremental_belief_ownership_ = BeliefOwnedState{};
    belief_ownership_dirty_ = false;
    return;
  }
  if (!force && !belief_ownership_dirty_) {
    return;
  }
  incremental_belief_ownership_ = buildBeliefOwnershipFromActiveGraph();
  belief_ownership_dirty_ = false;
}

void CbsFixedLagBpsamHeart::pruneIncrementalStateWithPlan(
    const LagWindowPlan& plan) {
  const FrameId eviction_end = plan.stale_eviction.eviction_end_frame_id;
  if (eviction_end > 0u) {
    auto prune_frame_map = [&](auto* frame_map) {
      std::vector<FrameId> stale_frames;
      stale_frames.reserve(frame_map->size());
      for (const auto& kv : *frame_map) {
        if (kv.first < eviction_end) {
          stale_frames.push_back(kv.first);
        }
      }
      for (const FrameId frame : stale_frames) {
        frame_map->erase(frame);
      }
    };
    prune_frame_map(&incremental_local_ownership_.pose_keys_by_frame);
    prune_frame_map(&incremental_local_ownership_.vel_keys_by_frame);
    prune_frame_map(&incremental_local_ownership_.bias_keys_by_frame);
  }

  for (const Key key : plan.stale_eviction.stale_local_keys) {
    incremental_local_ownership_.all_local_state_keys.erase(key);
    incremental_key_timestamps_.erase(key);
  }

  for (const gtsam::FactorIndex slot : plan.stale_eviction.stale_belief_factor_slots) {
    incremental_belief_ownership_.belief_factor_by_slot.erase(slot);
  }
  for (const gtsam::FactorIndex slot : plan.orphan_prune.orphan_belief_factor_slots) {
    incremental_belief_ownership_.belief_factor_by_slot.erase(slot);
  }
  for (const Key key : plan.orphan_prune.orphan_robot_keys) {
    incremental_belief_ownership_.robot_keys.erase(key);
  }
  for (const Key key : plan.orphan_prune.orphan_gbp_keys) {
    incremental_belief_ownership_.gbp_keys.erase(key);
  }
  for (const Key key : plan.orphan_prune.orphan_consensus_keys) {
    incremental_belief_ownership_.consensus_keys.erase(key);
  }
}

void CbsFixedLagBpsamHeart::updateIncrementalDiagnosticsFromPlan(
    const LagWindowPlan& plan,
    const bool used_full_recompute_fallback,
    const double incremental_lag_update_ms) {
  last_incremental_diagnostics_ = makeDiagnostics(plan);
  last_incremental_diagnostics_.incremental_lag_update_ms =
      incremental_lag_update_ms;
  last_incremental_diagnostics_.used_full_recompute_fallback =
      used_full_recompute_fallback;
}

bool CbsFixedLagBpsamHeart::keyHasRemainingIncidentFactor(
    const Key key,
    const std::unordered_set<gtsam::FactorIndex>& removed_slots) const {
  if (!bpsam_) {
    return false;
  }

  const gtsam::NonlinearFactorGraph& graph = bpsam_->getFactorsUnsafe();
  const auto& variable_index = bpsam_->getVariableIndex();
  const auto it = variable_index.find(key);
  if (it == variable_index.end()) {
    return false;
  }

  for (const gtsam::FactorIndex slot : it->second) {
    if (removed_slots.count(slot) > 0u || !factorExists(graph, slot)) {
      continue;
    }
    return true;
  }
  return false;
}

bool CbsFixedLagBpsamHeart::isLocalStateKey(const Key key) const {
  const gtsam::Symbol symbol(key);
  const unsigned char c = static_cast<unsigned char>(symbol.chr());
  return c == kPoseSymbolChar || c == kVelocitySymbolChar ||
         c == kImuBiasSymbolChar;
}

bool CbsFixedLagBpsamHeart::isLocalPoseKey(const Key key) const {
  const gtsam::Symbol symbol(key);
  return static_cast<unsigned char>(symbol.chr()) == kPoseSymbolChar;
}

bool CbsFixedLagBpsamHeart::isRobotKey(const Key key) const {
  return cbs::isRobotKey(gtsam::LabeledSymbol(key));
}

bool CbsFixedLagBpsamHeart::isConsensusKey(const Key key) const {
  return cbs::isPoseKey(gtsam::LabeledSymbol(key)) && !isLocalStateKey(key);
}

bool CbsFixedLagBpsamHeart::isBeliefFactor(
    const gtsam::NonlinearFactor::shared_ptr& factor) const {
  if (!bpsam_ || !factor || factor->keys().size() != 2u) {
    return false;
  }
  const cbs::AgentId self_id = bpsam_->getParams().robot_id;
  return cbs::isPoseBeliefFactor(self_id, factor) ||
         cbs::isAnchorBeliefFactor(factor);
}

#endif  // KIMERA_USE_CBS

}  // namespace VIO
