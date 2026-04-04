#pragma once

#include <limits>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <gtsam/base/Matrix.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>

#include "kimera-vio/common/vio_types.h"

#ifdef KIMERA_USE_CBS
#include <cbs/bpsam/bpsam.h>
#include <cbs/gbp/gaussian.h>
#endif

namespace VIO {

#ifdef KIMERA_USE_CBS

/**
 * @brief Incremental BPSAM-based fixed-lag heart helper.
 *
 * This class computes lag ownership/eviction over the expanded BPSAM graph and
 * builds LOCAL kept-side lag-edge summary factors from stale-side elimination.
 * Runtime adoption remains gated in VioBackend by experimental flags.
 */
class CbsFixedLagBpsamHeart {
 public:
  using Key = gtsam::Key;
  using KeyTimestampMap = std::map<Key, double>;

  struct LocalIngestPacket {
    gtsam::NonlinearFactorGraph local_factors;
    gtsam::Values local_values;
    KeyTimestampMap local_key_timestamps;
    FrameId newest_local_frame_id = 0u;
  };

  // Explicit local ownership model: keys owned by the local Kimera stream.
  struct LocalFrameOwnedKeys {
    std::unordered_map<FrameId, std::unordered_set<Key>> pose_keys_by_frame;
    std::unordered_map<FrameId, std::unordered_set<Key>> vel_keys_by_frame;
    std::unordered_map<FrameId, std::unordered_set<Key>> bias_keys_by_frame;
    std::unordered_set<Key> all_local_state_keys;

    size_t totalPoseKeys() const;
    size_t totalVelKeys() const;
    size_t totalBiasKeys() const;
  };

  // Explicit belief ownership model: factors/keys introduced by CBS belief flow.
  struct BeliefFactorOwnership {
    gtsam::FactorIndex slot = std::numeric_limits<gtsam::FactorIndex>::max();
    std::vector<Key> keys;
    bool is_pose_belief_factor = false;
    bool is_anchor_belief_factor = false;
    bool touches_local_pose = false;
    bool touches_local_state = false;
  };

  struct BeliefOwnedState {
    std::unordered_map<gtsam::FactorIndex, BeliefFactorOwnership>
        belief_factor_by_slot;
    std::unordered_set<Key> robot_keys;
    std::unordered_set<Key> gbp_keys;
    std::unordered_set<Key> consensus_keys;
  };

  // Eviction plan over stale frames (local + attached belief structure).
  struct StaleFrameEvictionResult {
    FrameId newest_frame_id = 0u;
    FrameId oldest_active_frame_id = 0u;
    FrameId eviction_start_frame_id = 0u;
    FrameId eviction_end_frame_id = 0u;  // exclusive
    bool full_rescan = true;

    std::unordered_set<Key> stale_local_keys;
    gtsam::FactorIndices stale_local_factor_slots;
    gtsam::FactorIndices stale_belief_factor_slots;
  };

  struct OrphanPruneResult {
    gtsam::FactorIndices orphan_belief_factor_slots;
    std::unordered_set<Key> orphan_robot_keys;
    std::unordered_set<Key> orphan_gbp_keys;
    std::unordered_set<Key> orphan_consensus_keys;
  };

  // Candidate separator keys where future true lag-edge summary should be formed.
  struct BoundarySeparatorCandidateSet {
    std::unordered_set<Key> separator_keys;
    std::unordered_set<Key> local_separator_keys;
    std::unordered_set<Key> belief_separator_keys;
    std::set<gtsam::FactorIndex> crossing_factor_slots;
  };

  // Explicit future insertion point for true lag-edge summary marginalization.
  struct SummaryInsertionPlan {
    BoundarySeparatorCandidateSet boundary;
    std::unordered_set<Key> summary_target_keys;
    bool implemented = false;
    std::string mode;
    std::string pending_reason;
  };

  // Explicit future API payload for a true lag-edge summary insertion step.
  struct BoundarySummaryInput {
    FrameId newest_frame_id = 0u;
    FrameId oldest_active_frame_id = 0u;
    std::unordered_set<Key> separator_keys;
    std::unordered_set<Key> local_separator_keys;
    std::unordered_set<Key> belief_separator_keys;
    std::set<gtsam::FactorIndex> crossing_factor_slots;
    std::unordered_set<Key> summary_target_keys;
    bool has_candidates = false;
  };

  // Temporary PHASE-2 bridge stats: covariance-driven priors emitted on
  // separator-local keys. This is not the final Schur-complement summary.
  struct SummaryPriorBridgeStats {
    size_t candidate_keys_total = 0u;
    size_t candidate_local_keys = 0u;
    size_t emitted_pose_priors = 0u;
    size_t emitted_vel_priors = 0u;
    size_t emitted_bias_priors = 0u;
    size_t skipped_non_local_keys = 0u;
    size_t skipped_missing_estimate = 0u;
    size_t skipped_covariance_failure = 0u;
    size_t covariance_queries_performed = 0u;
    size_t emitted_total_priors = 0u;
    bool used_fallback_sigmas = false;
    std::string first_failure_reason;
  };

  struct SummaryBuildStats {
    bool summary_implemented_this_epoch = false;
    bool summary_target_coherence_ok = false;
    size_t summary_target_key_count = 0u;
    size_t summary_requested_target_count = 0u;
    size_t summary_realizable_target_count = 0u;
    size_t summary_dropped_target_count = 0u;
    size_t summary_crossing_factor_count = 0u;
    size_t summary_crossing_candidate_slots_count = 0u;
    size_t summary_crossing_selected_slots_count = 0u;
    size_t summary_crossing_dropped_slots_count = 0u;
    long long summary_first_dropped_crossing_slot = -1;
    std::string summary_first_dropped_crossing_slot_class = "none";
    std::string summary_first_dropped_crossing_slot_keys = "none";
    std::string summary_first_dropped_crossing_slot_reason = "none";
    long long summary_first_selected_crossing_slot = -1;
    std::string summary_first_selected_crossing_slot_class = "none";
    std::string summary_first_selected_crossing_slot_keys = "none";
    long long summary_first_covered_crossing_slot = -1;
    std::string summary_first_covered_crossing_slot_class = "none";
    std::string summary_first_covered_crossing_slot_keys = "none";
    size_t summary_factor_count_emitted = 0u;
    size_t summary_input_factor_count = 0u;
    size_t summary_eliminated_key_count = 0u;
    double summary_build_ms = 0.0;
    std::string summary_mode = "failed";
    std::string first_summary_failure_reason;
    std::string first_dropped_target_reason;
    size_t summary_required_crossing_slots_count = 0u;
    std::string summary_required_crossing_slots_first_few = "none";
    std::unordered_set<gtsam::FactorIndex> summary_required_crossing_slots_exact;
    size_t summary_covered_crossing_slots_count = 0u;
    std::string summary_covered_crossing_slots_first_few = "none";
    std::unordered_set<gtsam::FactorIndex> summary_covered_crossing_slots_exact;
    size_t augmented_supported_requested_crossing_slots_count = 0u;
    std::string augmented_supported_requested_crossing_slots_first_few = "none";
    std::unordered_set<gtsam::FactorIndex>
        augmented_supported_requested_crossing_slots_exact;
    std::string augmented_requested_crossing_slots_first_few = "none";
    bool augmented_requested_supported_coherence_ok = true;
    std::unordered_set<gtsam::FactorIndex>
        caller_requested_crossing_slots_exact;
    size_t caller_requested_crossing_slots_count = 0u;
    std::string caller_requested_crossing_slots_first_few = "none";
    std::unordered_set<gtsam::FactorIndex>
        caller_requested_supported_crossing_slots_exact;
    size_t caller_requested_supported_crossing_slots_count = 0u;
    std::string caller_requested_supported_crossing_slots_first_few = "none";
    bool caller_requested_domain_coherence_ok = true;
    long long first_caller_requested_slot_missing_from_internal_crossing_domain =
        -1;
    std::string
        first_caller_requested_slot_missing_from_internal_crossing_domain_class =
            "none";
    std::string
        first_caller_requested_slot_missing_from_internal_crossing_domain_keys =
            "none";
    size_t caller_requested_missing_from_internal_domain_count = 0u;
    std::string caller_requested_missing_from_internal_domain_first_few = "none";
    std::string caller_requested_slot_classification_first_few = "none";
    std::string first_caller_requested_slot_exclusion_reason = "none";
    size_t directly_removable_local_internal_candidate_slots_count = 0u;
    std::string
        directly_removable_local_internal_candidate_slots_first_few = "none";
    std::unordered_set<gtsam::FactorIndex>
        directly_removable_local_internal_candidate_slots_exact;
    size_t directly_removable_local_internal_slots_count = 0u;
    std::string directly_removable_local_internal_slots_first_few = "none";
    std::unordered_set<gtsam::FactorIndex>
        directly_removable_local_internal_slots_exact;
    size_t retained_protected_local_support_slots_count = 0u;
    std::string retained_protected_local_support_slots_first_few = "none";
    std::unordered_set<gtsam::FactorIndex>
        retained_protected_local_support_slots_exact;
    std::string first_retained_protected_local_support_slot_class = "none";
    std::string first_retained_protected_local_support_slot_keys = "none";
    size_t
        direct_local_internal_rejected_due_to_incomplete_component_count = 0u;
    long long first_direct_local_internal_rejected_slot = -1;
    std::string first_direct_local_internal_rejected_slot_class = "none";
    std::string first_direct_local_internal_rejected_slot_keys = "none";
    std::string first_direct_local_internal_rejected_reason = "none";
    std::string first_direct_local_internal_rejected_blocking_key = "none";
    long long first_direct_local_internal_rejected_blocking_slot = -1;
    std::string first_direct_local_internal_rejected_blocking_slot_class =
        "none";
    std::string first_direct_local_internal_rejected_blocking_slot_keys =
        "none";
    long long
        first_direct_local_internal_rejected_due_to_retained_support_slot = -1;
    std::string
        first_direct_local_internal_rejected_due_to_retained_support_reason =
            "none";
    size_t
        direct_local_internal_rejected_due_to_retained_support_slots_count = 0u;
    std::string
        direct_local_internal_rejected_due_to_retained_support_slots_first_few =
            "none";
    std::unordered_set<gtsam::FactorIndex>
        direct_local_internal_rejected_due_to_retained_support_slots_exact;
    size_t root_bootstrap_blocked_component_count = 0u;
    size_t root_bootstrap_blocked_component_member_count = 0u;
    std::string root_bootstrap_blocked_component_member_slots_first_few = "none";
    std::unordered_set<gtsam::FactorIndex>
        root_bootstrap_blocked_component_member_slots_exact;
    size_t root_bootstrap_component_seed_count = 0u;
    std::string root_bootstrap_seed_slots_first_few = "none";
    std::string root_bootstrap_component_closure_mode = "none";
    std::string root_bootstrap_blocking_key = "none";
    long long root_bootstrap_blocking_support_slot = -1;
    std::string root_bootstrap_blocking_support_slot_class = "none";
    std::string root_bootstrap_blocking_support_slot_keys = "none";
    std::string root_bootstrap_blocking_incident_id = "none";
    size_t unsupported_deferred_lag_slots_count = 0u;
    std::string unsupported_deferred_lag_slots_first_few = "none";
    std::unordered_set<gtsam::FactorIndex> unsupported_deferred_lag_slots_exact;
    size_t lag_requested_slot_count = 0u;
    bool lag_requested_partition_coherence_ok = true;
    size_t unclassified_requested_lag_slots_count = 0u;
    std::string unclassified_requested_lag_slots_first_few = "none";
    long long first_unclassified_requested_lag_slot = -1;
    std::string first_unclassified_requested_lag_slot_class = "none";
    std::string first_unclassified_requested_lag_slot_keys = "none";
    std::string first_unclassified_requested_lag_slot_reason = "none";

    // Backward-compatible aggregate coverage fields (kept while we migrate
    // all diagnostics/consumers to split semantics).
    size_t summary_covered_remove_slots_count = 0u;
    std::string summary_covered_remove_slots_first_few = "none";
    std::unordered_set<gtsam::FactorIndex> summary_covered_remove_slots_exact;
    size_t summary_uncovered_requested_remove_slots_count = 0u;
    std::string summary_uncovered_requested_remove_slots_first_few = "none";
    std::string summary_remove_coverage_mode = "none";
    bool summary_remove_coverage_coherence_ok = true;
    size_t summary_remove_coverage_requested_count = 0u;
    size_t summary_remove_coverage_covered_count = 0u;
    size_t summary_remove_coverage_uncovered_count = 0u;
    std::string summary_remove_coverage_first_inconsistency_reason = "none";

    // Expanded-summary diagnostics (PHASE-3 first step).
    std::string expanded_summary_mode = "disabled";
    size_t expanded_summary_requested_target_count = 0u;
    size_t expanded_summary_realizable_target_count = 0u;
    size_t expanded_summary_crossing_factor_count = 0u;
    size_t expanded_summary_factor_count_emitted = 0u;
    size_t expanded_summary_support_factor_count = 0u;
    std::string expanded_summary_support_slots_first_few = "none";
    bool expanded_summary_used_bootstrap_support_priors = false;
    std::string expanded_summary_bootstrap_support_keys = "none";
    bool expanded_summary_removed_crossing_uses_bootstrap_support = false;
    double expanded_summary_build_ms = 0.0;
    std::string first_expanded_summary_failure_reason;
    size_t expanded_included_local_nonbelief_factor_count = 0u;
    size_t expanded_included_mixed_nonbelief_factor_count = 0u;
    size_t expanded_included_belief_factor_count = 0u;
    size_t expanded_excluded_local_nonbelief_factor_count = 0u;
    size_t expanded_excluded_mixed_nonbelief_factor_count = 0u;
    size_t expanded_excluded_belief_factor_count = 0u;

    // Expanded-path selection/injection consistency diagnostics.
    size_t expanded_selected_factor_slots_count = 0u;
    std::unordered_set<gtsam::FactorIndex> expanded_selected_factor_slots;
    size_t expanded_selected_key_count = 0u;
    size_t expanded_eliminated_key_count = 0u;
    size_t expanded_emitted_factor_count = 0u;
    size_t expanded_emitted_factor_key_count = 0u;
    bool expanded_emitted_factor_contains_nonlocal_keys = false;
    bool expanded_emitted_factor_contains_belief_keys = false;
    bool expanded_emitted_factor_references_stale_removed_keys = false;
    bool expanded_emitted_factor_references_orphan_pruned_keys = false;
    size_t expanded_emitted_factor_missing_estimate_keys = 0u;
    Key first_expanded_bad_key = std::numeric_limits<Key>::max();
    gtsam::FactorIndex first_expanded_bad_factor_slot =
        std::numeric_limits<gtsam::FactorIndex>::max();
    std::string first_expanded_exception_context;
    std::string expanded_veto_reason;
    size_t expanded_veto_count = 0u;
  };

  struct RootBootstrapSupportBuildStats {
    size_t incident_crossing_support_slot_count = 0u;
    std::string incident_crossing_support_slots_first_few = "none";
    size_t kept_target_key_count = 0u;
    std::string kept_target_keys_first_few = "none";
    size_t selected_graph_factor_count = 0u;
    size_t eliminate_key_count = 0u;
    std::string augment_mode = "none";
    std::string first_failure_reason = "none";
  };

  struct LagWindowPlan {
    LocalFrameOwnedKeys local_ownership;
    BeliefOwnedState belief_ownership;
    StaleFrameEvictionResult stale_eviction;
    OrphanPruneResult orphan_prune;
    BoundarySeparatorCandidateSet boundary_candidates;
    SummaryInsertionPlan future_summary_insertion;

    gtsam::FactorIndices remove_factor_indices;
  };

  struct Diagnostics {
    double incremental_lag_update_ms = 0.0;
    size_t active_local_key_count = 0u;
    size_t active_belief_factor_count = 0u;
    size_t active_robot_gbp_key_count = 0u;
    size_t stale_local_key_count = 0u;
    size_t stale_belief_factor_count = 0u;
    size_t orphan_key_count = 0u;
    size_t boundary_candidate_count = 0u;
    size_t covariance_queries_performed = 0u;
    size_t temporary_priors_emitted = 0u;
    bool used_full_recompute_fallback = false;
  };

 public:
  explicit CbsFixedLagBpsamHeart(cbs::BPSAM::Ptr bpsam, size_t lag_states);

  void setBpsam(cbs::BPSAM::Ptr bpsam);
  void setLagStates(size_t lag_states);
  void resetIncrementalLagState();

  // Local Kimera ingestion entrypoint for future runtime wiring.
  void ingestLocalKimeraPacket(const LocalIngestPacket& packet);

  // Incoming CBS beliefs are ingested with BPSAM Between semantics.
  size_t ingestIncomingBeliefs(
      const std::map<Key, std::map<cbs::AgentId, gbp::Gaussian>>& beliefs);

  // Incremental lag update path (intended online mode).
  const LagWindowPlan& advanceLagWindowIncremental(
      FrameId fallback_newest_frame_id);
  const LagWindowPlan& getIncrementalLagPlan() const;
  const gtsam::FactorIndices& getIncrementalRemoveFactorIndices() const;
  const Diagnostics& getLastIncrementalDiagnostics() const;
  bool hasPendingPrunePlan() const;
  bool commitPendingPrunePlan();
  void clearPendingPrunePlan();

  // Recompute lag window ownership/eviction/boundary over expanded graph.
  // NOTE: debug/validation/fallback path; not the intended online mode.
  LagWindowPlan recomputeLagWindow(const KeyTimestampMap& timestamps,
                                   FrameId fallback_newest_frame_id) const;

  Diagnostics makeDiagnostics(const LagWindowPlan& plan) const;

  // Future true summary insertion hook payload (no Schur summary yet).
  BoundarySummaryInput buildBoundarySummaryInput(
      const LagWindowPlan& plan) const;
  BoundarySummaryInput buildBoundarySummaryInputFromIncrementalState() const;

  // Build temporary separator priors into a factor graph. This is the PHASE-2
  // implementation nucleus and remains intentionally separate from runtime
  // wiring. Returns true if at least one prior was emitted.
  bool appendTemporarySummaryPriors(const LagWindowPlan& plan,
                                    gtsam::NonlinearFactorGraph* out_factors,
                                    SummaryPriorBridgeStats* stats) const;

  // Build a kept-side lag-edge summary factorization and inject it as
  // LinearContainerFactors. Current behavior tries an expanded crossing-factor
  // summary first and falls back to validated LOCAL-only summary if needed.
  bool appendLagEdgeSummaryFactors(LagWindowPlan* plan,
                                   gtsam::NonlinearFactorGraph* out_factors,
                                   SummaryBuildStats* stats,
                                   const std::unordered_set<gtsam::FactorIndex>*
                                       augmented_required_crossing_slots =
                                           nullptr) const;

  // Build a root/bootstrap-specific replacement support packet for direct-local
  // components blocked by retained frame-0 support, and report which blocked
  // slots are support-realizable for same-epoch remove.
  bool appendRootBootstrapDirectLocalSupportFactors(
      const LagWindowPlan& plan,
      const std::unordered_set<gtsam::FactorIndex>& blocked_component_slots,
      gtsam::NonlinearFactorGraph* out_factors,
      std::unordered_set<gtsam::FactorIndex>* out_supported_remove_slots,
      std::string* failure_reason,
      RootBootstrapSupportBuildStats* out_stats = nullptr) const;

  // Local-only covariance query stays explicitly separated from fused semantics.
  bool queryLocalOnlyCovariance(const Key& key,
                                gtsam::Matrix* covariance,
                                std::string* reason) const;

  static FrameId computeOldestActiveFrame(FrameId newest_frame_id,
                                          size_t lag_states);

 private:
  static constexpr unsigned char kPoseSymbolChar = 'x';
  static constexpr unsigned char kVelocitySymbolChar = 'v';
  static constexpr unsigned char kImuBiasSymbolChar = 'b';

 private:
  LocalFrameOwnedKeys buildLocalOwnershipFromVariableIndex() const;
  BeliefOwnedState buildBeliefOwnershipFromActiveGraph() const;

  StaleFrameEvictionResult computeStaleFrameEviction(
      const KeyTimestampMap& timestamps,
      FrameId fallback_newest_frame_id,
      const LocalFrameOwnedKeys& local_ownership) const;

  OrphanPruneResult computeOrphanPrune(
      const StaleFrameEvictionResult& stale_eviction,
      const BeliefOwnedState& belief_ownership,
      const LocalFrameOwnedKeys& local_ownership) const;

  BoundarySeparatorCandidateSet computeBoundaryCandidates(
      const StaleFrameEvictionResult& stale_eviction,
      const OrphanPruneResult& orphan_prune) const;

  SummaryInsertionPlan buildFutureSummaryInsertionPlan(
      const BoundarySeparatorCandidateSet& boundary_candidates,
      const StaleFrameEvictionResult& stale_eviction) const;

  gtsam::FactorIndices buildRemoveFactorIndices(
      const StaleFrameEvictionResult& stale_eviction,
      const OrphanPruneResult& orphan_prune) const;

  bool keyHasRemainingIncidentFactor(
      Key key,
      const std::unordered_set<gtsam::FactorIndex>& removed_slots) const;

  bool isLocalStateKey(Key key) const;
  bool isLocalPoseKey(Key key) const;
  bool isRobotKey(Key key) const;
  bool isConsensusKey(Key key) const;
  bool isBeliefFactor(const gtsam::NonlinearFactor::shared_ptr& factor) const;
  void refreshBeliefOwnershipFromActiveGraphIfNeeded(bool force);
  void pruneIncrementalStateWithPlan(const LagWindowPlan& plan);
  void updateIncrementalDiagnosticsFromPlan(
      const LagWindowPlan& plan,
      bool used_full_recompute_fallback,
      double incremental_lag_update_ms);

 private:
  cbs::BPSAM::Ptr bpsam_;
  size_t lag_states_ = 0u;

  // Cached ownership snapshots (scaffold state; no runtime side effects yet).
  LocalFrameOwnedKeys last_local_ownership_;
  BeliefOwnedState last_belief_ownership_;

  // Incremental lag state (intended online path).
  KeyTimestampMap incremental_key_timestamps_;
  LocalFrameOwnedKeys incremental_local_ownership_;
  BeliefOwnedState incremental_belief_ownership_;
  FrameId incremental_newest_local_frame_id_ = 0u;
  FrameId incremental_oldest_active_frame_id_ = 0u;
  BoundarySeparatorCandidateSet incremental_boundary_candidates_;
  SummaryInsertionPlan incremental_summary_insertion_;
  LagWindowPlan incremental_plan_cache_;
  LagWindowPlan pending_prune_plan_;
  mutable Diagnostics last_incremental_diagnostics_;
  bool incremental_plan_valid_ = false;
  bool pending_prune_plan_valid_ = false;
  bool incremental_state_seeded_from_full_recompute_ = false;
  bool belief_ownership_dirty_ = true;
};

#else

// Non-CBS builds keep a harmless placeholder to avoid include churn.
class CbsFixedLagBpsamHeart {};

#endif  // KIMERA_USE_CBS

}  // namespace VIO
