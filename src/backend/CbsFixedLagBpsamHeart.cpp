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
    SummaryBuildStats* stats) const {
  const auto start = std::chrono::steady_clock::now();
  bool target_coherence_ok = false;
  if (stats) {
    *stats = SummaryBuildStats{};
  }
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
      if (!ok && stats->first_summary_failure_reason.empty()) {
        stats->first_summary_failure_reason = reason;
      }
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
  if (stats) {
    stats->expanded_selected_factor_slots_count = expanded_selected_slots.size();
    stats->expanded_selected_key_count = expanded_selected_keys_set.size();
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
    return finishWith(true, "exact_expanded_partial_scope", "none");
  }

  if (requested_targets.empty()) {
    if (stats) {
      stats->expanded_summary_mode = "failed_local_fallback_failed";
    }
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
    return finishWith(false, "failed", "no_keys_to_eliminate_for_summary");
  }

  gtsam::Values linearization_point;
  for (const Key key : selected_keys_set) {
    if (!estimate.exists(key)) {
      target_coherence_ok = false;
      if (stats) {
        stats->expanded_summary_mode = fallback_failed_mode;
      }
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
    return finishWith(false,
                      "failed",
                      std::string("linearize_removed_graph_failed:") + e.what());
  }

  if (!gaussian_subgraph || gaussian_subgraph->empty()) {
    if (stats) {
      stats->expanded_summary_mode = fallback_failed_mode;
    }
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
    return finishWith(false,
                      "failed",
                      std::string("partial_elimination_failed:") + e.what());
  }

  const auto& summary_gaussian_graph = elimination_result.second;
  if (!summary_gaussian_graph || summary_gaussian_graph->empty()) {
    if (stats) {
      stats->expanded_summary_mode = fallback_failed_mode;
    }
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
    return finishWith(false, "failed", "no_summary_factors_emitted");
  }

  if (stats) {
    stats->expanded_summary_mode = fallback_success_mode;
  }
  return finishWith(true, "exact_schur", "none");
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
