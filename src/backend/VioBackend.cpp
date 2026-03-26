/* ----------------------------------------------------------------------------
 * Copyright 2017, Massachusetts Institute of Technology,
 * Cambridge, MA 02139
 * All Rights Reserved
 * Authors: Luca Carlone, et al. (see THANKS for the full author list)
 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */

/**
 * @file   VioBackend.cpp
 * @brief  Visual-Inertial Odometry pipeline, as described in these papers:
 *
 * A. Rosinol, M. Abate, Y. Chang, L. Carlone.
 * Kimera: an Open-Source Library for Real-Time Metric-Semantic Localization
 * and Mapping. In IEEE Intl. Conf. on Robotics and Automation (ICRA), 2019.
 *
 * C. Forster, L. Carlone, F. Dellaert, and D. Scaramuzza.
 * On-Manifold Preintegration Theory for Fast and Accurate Visual-Inertial
 * Navigation. IEEE Trans. Robotics, 33(1):1-21, 2016.
 *
 * L. Carlone, Z. Kira, C. Beall, V. Indelman, and F. Dellaert.
 * Eliminating Conditionally Independent Sets in Factor Graphs: A Unifying
 * Perspective based on Smart Factors. In IEEE Intl. Conf. on Robotics and
 * Automation (ICRA), 2014.
 *
 * @author Antoni Rosinol
 * @author Luca Carlone
 */

#include "kimera-vio/backend/VioBackend.h"

#include <gflags/gflags.h>
#include <glog/logging.h>
// zy Step 11
#ifdef KIMERA_USE_CBS
#pragma push_macro("CHECK")
#pragma push_macro("CHECK_EQ")
#pragma push_macro("CHECK_NE")
#pragma push_macro("CHECK_LT")
#pragma push_macro("CHECK_LE")
#pragma push_macro("CHECK_GT")
#pragma push_macro("CHECK_GE")
#include <cbs/bpsam/bpsam.h>
#pragma pop_macro("CHECK_GE")
#pragma pop_macro("CHECK_GT")
#pragma pop_macro("CHECK_LE")
#pragma pop_macro("CHECK_LT")
#pragma pop_macro("CHECK_NE")
#pragma pop_macro("CHECK_EQ")
#pragma pop_macro("CHECK")
#endif



#include <limits>  // for numeric_limits<>
#include <algorithm>
#include <chrono>
#include <iterator>
#include <map>
#include <sstream>
#include <string>
#include <utility>  // for make_pair
#include <unordered_set>
#include <vector>
#include <cmath> // zy step 5_c
#include <iomanip>
#include <Eigen/Cholesky>
#include <Eigen/Eigenvalues>



#include "kimera-vio/common/VioNavState.h"
#include "kimera-vio/imu-frontend/ImuFrontend-definitions.h"
#include "kimera-vio/logging/Logger.h"
#include "kimera-vio/utils/GtsamPrinting.h"
#include "kimera-vio/utils/Statistics.h"
#include "kimera-vio/utils/Timer.h"
#include "kimera-vio/utils/UtilsNumerical.h"

DEFINE_bool(debug_graph_before_opt,
            false,
            "Store factor graph before optimization for later printing if the "
            "optimization fails.");
DEFINE_bool(process_cheirality,
            false,
            "Handle cheirality exception by removing problematic landmarks and "
            "re-running optimization.");
DEFINE_int32(max_number_of_cheirality_exceptions,
             5,
             "Sets the maximum number of times we process a cheirality "
             "exception for a given optimization problem. This is to avoid too "
             "many recursive calls to update the smoother");
DEFINE_bool(compute_state_covariance,
            false,
            "Flag to compute state covariance from optimization Backend");
DEFINE_bool(no_incremental_pose,
            false,
            "Flag to disable incremental pose usage in backend");
// zy Step 36
// Relax external-prior timestamp matching to a practical default for cross-stack exchange
// (Kimera/LIORF timestamps are close but not typically within 2ms under real runtime load).
DEFINE_int64(external_prior_timestamp_tolerance_ns,
             200000000,  // 200ms
             "Maximum absolute timestamp delta (ns) used to match incoming "
             "external pose priors to Kimera keyframes.");
// zy Step 37
// LIORF and Kimera can have multi-second processing lag differences during bag replay.
// Make age/future gates configurable so valid cross-estimator priors are not discarded.
DEFINE_int64(external_prior_max_age_ns,
             20000000000LL,  // 20s
             "Maximum age (ns) of an incoming external prior relative to the "
             "current Kimera optimize timestamp before dropping as too old.");
DEFINE_int64(external_prior_max_future_lead_ns,
             200000000,  // 200ms
             "Maximum future lead (ns) allowed for incoming external priors; "
             "larger lead is deferred for later optimize cycles.");
DEFINE_int32(external_prior_max_queue_size,
             1000,
             "Maximum number of staged external pose priors kept in memory.");
DEFINE_int64(external_prior_queue_time_horizon_ns,
             60000000000LL,  // 60s
             "Keep at most this trailing time window (ns) of staged external "
             "pose priors, measured against newest received prior timestamp.");
DEFINE_int32(external_prior_max_per_optimize,
             400,
             "Maximum number of external priors processed in one optimize() "
             "cycle before deferring the remainder.");
DEFINE_bool(external_pose_belief_safe_covariance_fallback,
            false,
            "If true, publish external pose beliefs with a safe covariance "
            "fallback when state covariance is unavailable. If false, keep "
            "legacy covariance fallback behavior.");

// zy Step 11a & 26
#ifdef KIMERA_USE_CBS
// zy
// Enable CBS belief exchange whenever requested at runtime.
DEFINE_bool(use_cbs_optimizer,
            false,
            "If true (and compiled with KIMERA_USE_CBS), enable CBS belief "
            "exchange (belief validation/merging).");
DEFINE_bool(cbs_replace_fixed_lag_optimizer,
            false,
            "If true together with --use_cbs_optimizer, make BPSAM the active "
            "optimizer heart (CBS-heart mode). If false, keep legacy fixed-lag "
            "smoother as optimization heart.");
//zy Step 40a
// Runtime toggle for CBS GkCM/PCM consistency filtering on incoming belief factors.
DEFINE_bool(cbs_enable_gkcm,
            false,
            "If true, enable CBS GkCM filtering of incoming belief factors.");
// Align CBS belief contraction defaults with the CBS offline examples.
DEFINE_double(cbs_belief_contract_alpha,
              0.5,
              "CBS belief contraction alpha (Hellinger target ratio).");
DEFINE_double(cbs_belief_d_reset,
              0.6,
              "CBS belief reset threshold in Hellinger distance.");
DEFINE_double(cbs_belief_gamma,
              0.1,
              "CBS contraction gamma (used when alpha is adaptive).");
//zy Step 40b
// Number of CBS inner update rounds per backend epoch (first round uses new factors, later rounds are belief-only).
DEFINE_int32(cbs_pose_rounds_per_epoch,
             3,
             "Maximum number of CBS pose-stage update rounds executed per backend optimize epoch.");
//zy Step 40c
// CBS pose-stage convergence criteria on residual change between consecutive inner rounds.
DEFINE_double(cbs_pose_convergence_abs_residual,
              1e-3,
              "Absolute residual-change threshold for early stopping of CBS pose rounds.");
DEFINE_double(cbs_pose_convergence_rel_residual,
              1e-3,
              "Relative residual-change threshold for early stopping of CBS pose rounds.");
DEFINE_bool(cbs_diag_align_incoming_mean,
            false,
            "Diagnostic mode: align incoming external belief mean with a "
            "per-source receiver_world<-sender_world transform before CBS "
            "addBeliefs(). Covariance path is unchanged.");
DEFINE_bool(cbs_skip_remove_indices_when_no_external_effect,
            true,
            "If true, skip CBS removeFactorIndices churn in epochs where no "
            "external beliefs were accepted/injected.");
DEFINE_bool(cbs_outgoing_cov_fastpath_when_no_external_effect,
            true,
            "If true, in CBS-heart mode publish outgoing covariance from the "
            "already-computed backend state covariance when the latest epoch "
            "had zero accepted/injected external effect.");
DEFINE_bool(cbs_outgoing_query_full_cov_for_diag,
            false,
            "If true, additionally query FULL covariance for outgoing "
            "diagnostics. If false, skip FULL query to reduce overhead.");
DEFINE_bool(cbs_query_receiver_local_cov_for_diag,
            false,
            "If true, query receiver LOCAL marginal covariance for incoming "
            "belief diagnostics before addBeliefs(). If false, skip these "
            "diagnostic covariance queries to reduce no-fusion runtime.");
DEFINE_bool(cbs_diag_disable_outgoing_external_pose_belief_callback,
            false,
            "If true, skip outgoing external pose belief production callback "
            "in spinOnce() for A/B runtime isolation.");
DEFINE_bool(cbs_outgoing_publish_decimation_enabled,
            true,
            "If true, decimate/event-gate outgoing external pose belief "
            "publication before triggering expensive covariance queries.");
DEFINE_double(cbs_outgoing_publish_min_period_sec,
              1.5,
              "Minimum elapsed publish period (sec) used by outgoing "
              "decimation gate.");
DEFINE_int32(cbs_outgoing_publish_min_kf_stride,
             8,
             "Minimum keyframe stride used by outgoing decimation gate.");
DEFINE_double(cbs_outgoing_publish_force_max_silence_sec,
              3.0,
              "Force outgoing publish when elapsed silence (sec) exceeds "
              "this bound.");
DEFINE_bool(cbs_outgoing_publish_on_external_effect,
            true,
            "If true, force outgoing publish when latest epoch accepted or "
            "injected any external belief/prior.");
DEFINE_double(cbs_outgoing_publish_pose_delta_trans_thresh_m,
              0.75,
              "Outgoing publish gate translation delta threshold (meters).");
DEFINE_double(cbs_outgoing_publish_pose_delta_rot_thresh_deg,
              8.0,
              "Outgoing publish gate rotation delta threshold (degrees).");
// Temporary diagnostic isolation toggles (no algorithm redesign).
DEFINE_bool(cbs_diag_disable_lag_boundary_anchor_priors,
            false,
            "If true, disable CBS lag-boundary anchor-prior insertion in "
            "CBS-heart mode for A/B isolation.");
DEFINE_bool(cbs_diag_disable_remove_factor_indices,
            false,
            "If true, disable removeFactorIndices application in CBS-heart "
            "mode for A/B isolation.");
DEFINE_bool(cbs_diag_force_no_external_fast_path_when_no_external_effect,
            false,
            "If true, force CBS no-external fast path whenever "
            "accepted=0/injected=0, even in epochs that add new states.");
DEFINE_bool(cbs_diag_disable_smart_factor_replacements,
            false,
            "If true, disable smart-factor replacement churn in CBS-heart "
            "mode for A/B isolation.");
DEFINE_bool(cbs_diag_disable_lag_eviction_remove_candidates,
            false,
            "If true, ignore lag-window eviction remove candidates before "
            "building removeFactorIndices in CBS-heart mode.");
DEFINE_bool(cbs_diag_disable_merge_delete_slots_into_remove_factor_indices,
            false,
            "If true, do not merge delete_slots into removeFactorIndices; "
            "use lag-eviction candidates only in CBS-heart mode.");
DEFINE_bool(cbs_diag_disable_repeated_remove_from_prev_epoch,
            false,
            "If true, drop removeFactorIndices that were already requested "
            "in the previous epoch (diagnostic isolation only).");
DEFINE_bool(cbs_diag_disable_first_attempt_remove_factor_indices,
            false,
            "If true, disable removeFactorIndices on the first CBS update "
            "attempt while preserving existing recovery behavior.");
DEFINE_bool(cbs_use_marginalization_prior_bridge,
            false,
            "If true, use covariance-driven lag-boundary priors and bypass "
            "delete_slots->removeFactorIndices bridging (B1 approximation).");
DEFINE_bool(cbs_b1_refresh_cov_only_on_boundary_change,
            false,
            "If true, refresh B1 LOCAL covariance queries only when lag "
            "boundary changes; otherwise use the existing cheap fallback.");
DEFINE_bool(cbs_b1_pose_only_cov_refresh,
            false,
            "If true in B1 mode, refresh only pose LOCAL covariance at the "
            "lag boundary; velocity and bias use existing fixed fallbacks.");
DEFINE_bool(cbs_h2_local_cov_sidecar,
            false,
            "If true (and CBS exchange is enabled while fixed-lag remains "
            "heart), build a LOCAL-only covariance sidecar and source outgoing "
            "covariance from it.");
DEFINE_bool(cbs_h2_replay_sidecar_sync,
            false,
            "If true, keep the legacy replay-sidecar synchronization path for "
            "H2 local covariance. If false, LOCAL covariance is queried "
            "directly from the active fixed-lag graph.");
DEFINE_bool(cbs_h2_publish_anchored_local_cov,
            true,
            "If true, publish H2 local covariance after adding a temporary "
            "pose anchor prior at the queried key (LIORF-aligned behavior).");
DEFINE_double(cbs_h2_local_cov_anchor_rot_var,
              1e-2,
              "Rotation variance for temporary pose-anchor prior used by H2 "
              "local covariance extraction.");
DEFINE_double(cbs_h2_local_cov_anchor_trans_var,
              1e-1,
              "Translation variance for temporary pose-anchor prior used by "
              "H2 local covariance extraction.");
DEFINE_int32(cbs_smart_replace_material_support_delta,
             3,
             "Minimum absolute support-size delta required to treat a smart "
             "factor replacement as materially changed.");
DEFINE_int32(cbs_smart_replace_material_pose_key_delta,
             3,
             "Minimum symmetric-difference cardinality between old/new smart "
             "pose-key supports required to treat pose-key-set change as "
             "material.");
DEFINE_int64(cbs_deferred_prior_initial_backoff_ns,
             200000000,  // 200ms
             "Initial defer backoff (ns) for unmatched/no-local/budgeted "
             "external priors in CBS-heart mode.");
DEFINE_int64(cbs_deferred_prior_max_backoff_ns,
             5000000000LL,  // 5s
             "Maximum defer backoff (ns) for deferred external priors in "
             "CBS-heart mode.");
DEFINE_int64(cbs_rejected_source_initial_backoff_ns,
             500000000,  // 500ms
             "Initial per-source cooldown (ns) after addBeliefs() rejection "
             "in CBS-heart mode.");
DEFINE_int64(cbs_rejected_source_max_backoff_ns,
             10000000000LL,  // 10s
             "Maximum per-source cooldown (ns) after repeated addBeliefs() "
             "rejections in CBS-heart mode.");
#endif

namespace {
inline double elapsedMs(const std::chrono::steady_clock::time_point& start,
                        const std::chrono::steady_clock::time_point& end) {
  return std::chrono::duration_cast<
             std::chrono::duration<double, std::milli>>(end - start)
      .count();
}

inline bool useCbsBeliefExchange() {
#ifdef KIMERA_USE_CBS
  return FLAGS_use_cbs_optimizer;
#else
  return false;
#endif
}

inline bool useCbsOptimizerHeart() {
#ifdef KIMERA_USE_CBS
  // CBS-heart is an explicit runtime mode: keep legacy fixed-lag smoother as
  // heart unless both flags are enabled.
  return FLAGS_use_cbs_optimizer && FLAGS_cbs_replace_fixed_lag_optimizer;
#else
  return false;
#endif
}

inline bool useCbsH2LocalCovSidecar() {
#ifdef KIMERA_USE_CBS
  return FLAGS_use_cbs_optimizer && !FLAGS_cbs_replace_fixed_lag_optimizer &&
         FLAGS_cbs_h2_local_cov_sidecar;
#else
  return false;
#endif
}

inline bool useCbsH2ReplaySidecarSync() {
#ifdef KIMERA_USE_CBS
  return useCbsH2LocalCovSidecar() && FLAGS_cbs_h2_replay_sidecar_sync;
#else
  return false;
#endif
}

inline const char* cbsH2SidecarModeName() {
  return "passive_local_snapshot";
}

#ifdef KIMERA_USE_CBS
class ExternalPosePriorFactor final : public gtsam::PriorFactor<gtsam::Pose3> {
 public:
  using Base = gtsam::PriorFactor<gtsam::Pose3>;
  using This = ExternalPosePriorFactor;
  ExternalPosePriorFactor(const gtsam::Key& key,
                          const gtsam::Pose3& prior,
                          const gtsam::SharedNoiseModel& model)
      : Base(key, prior, model) {}
  gtsam::NonlinearFactor::shared_ptr clone() const override {
    return boost::static_pointer_cast<gtsam::NonlinearFactor>(
        boost::make_shared<This>(*this));
  }
};

using Vec6 = Eigen::Matrix<double, 6, 1>;
using Mat6 = Eigen::Matrix<double, 6, 6>;

double logDetSym6(const Mat6& sigma) {
  if (!sigma.allFinite()) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  Eigen::SelfAdjointEigenSolver<Mat6> eig(sigma);
  if (eig.info() != Eigen::Success) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  double sum = 0.0;
  for (int i = 0; i < eig.eigenvalues().size(); ++i) {
    const double ev = eig.eigenvalues()(i);
    if (!(ev > 0.0) || !std::isfinite(ev)) {
      return std::numeric_limits<double>::quiet_NaN();
    }
    sum += std::log(ev);
  }
  return sum;
}

double hellingerDistance6(const Vec6& mu_a,
                          const Mat6& sigma_a,
                          const Vec6& mu_b,
                          const Mat6& sigma_b) {
  if (!mu_a.allFinite() || !mu_b.allFinite() || !sigma_a.allFinite() ||
      !sigma_b.allFinite()) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  const Mat6 sigma_bar = 0.5 * (sigma_a + sigma_b);
  const double log_det_a = logDetSym6(sigma_a);
  const double log_det_b = logDetSym6(sigma_b);
  const double log_det_bar = logDetSym6(sigma_bar);
  if (!std::isfinite(log_det_a) || !std::isfinite(log_det_b) ||
      !std::isfinite(log_det_bar)) {
    return std::numeric_limits<double>::quiet_NaN();
  }

  const Vec6 delta = mu_a - mu_b;
  Eigen::LDLT<Mat6> ldlt(sigma_bar);
  if (ldlt.info() != Eigen::Success) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  const auto d = ldlt.vectorD();
  for (int i = 0; i < d.size(); ++i) {
    if (!(d(i) > 0.0) || !std::isfinite(d(i))) {
      return std::numeric_limits<double>::quiet_NaN();
    }
  }
  const Vec6 x = ldlt.solve(delta);
  if (!x.allFinite()) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  const double quad = delta.dot(x);
  if (!std::isfinite(quad)) {
    return std::numeric_limits<double>::quiet_NaN();
  }

  const double log_coeff = 0.25 * (log_det_a + log_det_b) - 0.5 * log_det_bar;
  const double coeff = std::exp(log_coeff - 0.125 * quad);
  if (!std::isfinite(coeff)) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  const double h2 = std::clamp(1.0 - coeff, 0.0, 1.0);
  return std::sqrt(h2);
}
#endif

enum class PoseCovarianceStatus {
  kAccepted = 0,
  kRegularized = 1,
  kRejected = 2,
};

PoseCovarianceStatus sanitizePoseCovariance(gtsam::Matrix6* covariance,
                                            std::string* reason) {
  if (reason) {
    reason->clear();
  }
  if (covariance == nullptr) {
    if (reason) {
      *reason = "null_covariance";
    }
    return PoseCovarianceStatus::kRejected;
  }
  if (!covariance->allFinite()) {
    if (reason) {
      *reason = "non_finite_entries";
    }
    return PoseCovarianceStatus::kRejected;
  }

  gtsam::Matrix6 cov = 0.5 * ((*covariance) + covariance->transpose());
  bool regularized = false;
  constexpr double kMinRotVar = 1e-8;    // rad^2
  constexpr double kMinTransVar = 1e-8;  // m^2
  for (int i = 0; i < 6; ++i) {
    const double min_var = (i < 3) ? kMinRotVar : kMinTransVar;
    if (!std::isfinite(cov(i, i)) || cov(i, i) < min_var) {
      cov(i, i) = min_var;
      regularized = true;
    }
  }

  Eigen::SelfAdjointEigenSolver<gtsam::Matrix6> eig_solver(cov);
  if (eig_solver.info() != Eigen::Success ||
      !eig_solver.eigenvalues().allFinite()) {
    if (reason) {
      *reason = "eigendecomposition_failed";
    }
    return PoseCovarianceStatus::kRejected;
  }

  constexpr double kMinEigenvalue = 1e-10;
  const double min_eig = eig_solver.eigenvalues().minCoeff();
  if (min_eig <= kMinEigenvalue) {
    const double shift = (kMinEigenvalue - min_eig) + 1e-12;
    cov.diagonal().array() += shift;
    regularized = true;

    eig_solver.compute(cov);
    if (eig_solver.info() != Eigen::Success ||
        !eig_solver.eigenvalues().allFinite() ||
        eig_solver.eigenvalues().minCoeff() <= 0.0) {
      if (reason) {
        *reason = "regularization_failed";
      }
      return PoseCovarianceStatus::kRejected;
    }
  }

  *covariance = 0.5 * (cov + cov.transpose());
  if (regularized) {
    if (reason) {
      *reason = "regularized_to_spd";
    }
    return PoseCovarianceStatus::kRegularized;
  }
  if (reason) {
    *reason = "accepted";
  }
  return PoseCovarianceStatus::kAccepted;
}
}  // namespace



namespace VIO {

/* -------------------------------------------------------------------------- */
VioBackend::VioBackend(const gtsam::Pose3& B_Pose_leftCamRect,
                       const StereoCalibPtr& stereo_calibration,
                       const BackendParams& backend_params,
                       const ImuParams& imu_params,
                       const BackendOutputParams& backend_output_params,
                       bool log_output,
                       std::optional<OdometryParams> odom_params)
    : backend_state_(BackendState::Bootstrap),
      backend_params_(backend_params),
      imu_params_(imu_params),
      backend_output_params_(backend_output_params),
      odom_params_(odom_params),
      timestamp_lkf_(-1),
      imu_bias_lkf_(ImuBias()),
      W_Vel_B_lkf_(gtsam::Vector3::Zero()),
      W_Pose_B_lkf_from_increments_(gtsam::Pose3()),
      W_Pose_B_lkf_from_state_(gtsam::Pose3()),
      imu_bias_prev_kf_(ImuBias()),
      B_Pose_leftCamRect_(B_Pose_leftCamRect),
      stereo_cal_(stereo_calibration),
      last_kf_id_(-1),
      curr_kf_id_(0),
      landmark_count_(0),
      log_output_(log_output),
      logger_(log_output ? std::make_unique<BackendLogger>() : nullptr) {
// TODO the parsing of the params should be done inside here out from the
// path to the params file, otherwise other derived VIO Backends will be
// stuck with the parameters used by vanilla VIO, as there is no polymorphic
// container in C++...
// This way VioBackend can parse the params it cares about, while others can
// have the opportunity to parse their own parameters as well.
// Unfortunately, doing that would not work because many other modules use
// VioBackendParams as weird as this may sound...
// For now we have polymorphic params, with dynamic_cast to derived class,
// aka suboptimal...

//////////////////////////////////////////////////////////////////////////////
// Initialize smoother.
#ifdef INCREMENTAL_SMOOTHER
  gtsam::ISAM2Params isam_param;
  BackendParams::setIsam2Params(backend_params, &isam_param);

  smoother_ = std::make_unique<Smoother>(backend_params.nr_states_, isam_param);
#else  // BATCH SMOOTHER
  gtsam::LevenbergMarquardtParams lmParams;
  lmParams.setlambdaInitial(0.0);     // same as GN
  lmParams.setlambdaLowerBound(0.0);  // same as GN
  lmParams.setlambdaUpperBound(0.0);  // same as GN)
  smoother_ = std::make_unique<Smoother>(backend_params.nr_states_, lmParams);
#endif

// zy-------
// zy Step 10f
#ifdef KIMERA_USE_CBS
  // Build CBS optimizer with Kimera's current ISAM2 parameterization.
  gtsam::ISAM2Params cbs_isam_params;
  BackendParams::setIsam2Params(backend_params, &cbs_isam_params);

  cbs::BPSAM::Params cbs_params;
  // zy Step 33: use CBS-native printable agent IDs so robot/pose labeled keys behave consistently across modules.
  constexpr cbs::AgentId kKimeraAgentId = static_cast<cbs::AgentId>('a');
  cbs_params.robot_id = kKimeraAgentId;

  cbs_params.sam_params_ = cbs_isam_params;
  //zy Step 40d
  // Keep GkCM optional at runtime so we can compare plain contraction vs. CBS+PCM filtering.
  cbs_params.enable_gkcm = FLAGS_cbs_enable_gkcm;
  cbs_params.gbp_update_params.type = gbp::GaussianMergeType::Contract;
  cbs_params.gbp_update_params.metric_type = gbp::MetricType::Hellinger;
  cbs_params.gbp_update_params.contract_alpha =
      static_cast<float>(FLAGS_cbs_belief_contract_alpha);
  cbs_params.gbp_update_params.d_reset =
      static_cast<float>(FLAGS_cbs_belief_d_reset);
  cbs_params.gbp_update_params.gamma =
      static_cast<float>(FLAGS_cbs_belief_gamma);

  cbs_optimizer_ = std::make_shared<cbs::BPSAM>(cbs_params);
  // LOG(INFO) << "CBS BPSAM scaffold initialized (inactive)."; (zy cancelled it)
  // zy Step 27: startup log must state runtime mode so we can verify CBS-heart activation from logs.
  LOG(INFO) << "CBS BPSAM initialized. use_cbs_optimizer_flag="
            << (FLAGS_use_cbs_optimizer ? "true" : "false")
            << ", cbs_replace_fixed_lag_optimizer="
            << (FLAGS_cbs_replace_fixed_lag_optimizer ? "true" : "false")
            << ", cbs_heart_active="
            << (useCbsOptimizerHeart() ? "true" : "false")
            << ", cbs_h2_local_cov_sidecar="
            << (useCbsH2LocalCovSidecar() ? "true" : "false");
  LOG(INFO) << "CBS belief contraction params: type=Contract, metric=Hellinger"
            << ", alpha=" << FLAGS_cbs_belief_contract_alpha
            << ", d_reset=" << FLAGS_cbs_belief_d_reset
            << ", gamma=" << FLAGS_cbs_belief_gamma;
  //zy Step 40e
  // Surface CBS pose-round controls at startup for reproducible experiments.
  LOG(INFO) << "CBS config: gkcm=" << (FLAGS_cbs_enable_gkcm ? "true" : "false")
            << ", pose_rounds_per_epoch=" << FLAGS_cbs_pose_rounds_per_epoch
            << ", conv_abs=" << FLAGS_cbs_pose_convergence_abs_residual
            << ", conv_rel=" << FLAGS_cbs_pose_convergence_rel_residual;
  LOG(INFO) << "CBS H2 local covariance extraction: anchored_publish="
            << (FLAGS_cbs_h2_publish_anchored_local_cov ? "true" : "false")
            << ", anchor_rot_var=" << FLAGS_cbs_h2_local_cov_anchor_rot_var
            << ", anchor_trans_var="
            << FLAGS_cbs_h2_local_cov_anchor_trans_var;
  if (useCbsH2LocalCovSidecar()) {
    LOG(INFO) << "H2 local covariance mode = " << cbsH2SidecarModeName();
  }

#endif
// zy -------

  // Set parameters for all factors.
  setFactorsParams(backend_params,
                   &smart_noise_,
                   &smart_factors_params_,
                   &no_motion_prior_noise_,
                   &zero_velocity_prior_noise_,
                   &constant_velocity_prior_noise_);

  // zy Step 36_b
  // Allow runtime tuning via gflag and guard against invalid values.
  if (FLAGS_external_prior_timestamp_tolerance_ns > 0) {
    external_prior_timestamp_tolerance_ns_ =
        static_cast<Timestamp>(FLAGS_external_prior_timestamp_tolerance_ns);
  } else {
    LOG(WARNING) << "Invalid --external_prior_timestamp_tolerance_ns="
                 << FLAGS_external_prior_timestamp_tolerance_ns
                 << ", falling back to 200000000 ns.";
    external_prior_timestamp_tolerance_ns_ = 200000000;
  }
  LOG(INFO) << "External prior timestamp tolerance [ns]: "
            << external_prior_timestamp_tolerance_ns_;
  LOG(INFO) << "External prior max age [ns]: "
            << FLAGS_external_prior_max_age_ns;
  LOG(INFO) << "External prior max future lead [ns]: "
            << FLAGS_external_prior_max_future_lead_ns;
  if (FLAGS_external_prior_max_queue_size > 0) {
    max_external_pose_priors_queue_size_ =
        static_cast<size_t>(FLAGS_external_prior_max_queue_size);
  } else {
    LOG(WARNING) << "Invalid --external_prior_max_queue_size="
                 << FLAGS_external_prior_max_queue_size
                 << ", falling back to 1000.";
    max_external_pose_priors_queue_size_ = 1000;
  }
  LOG(INFO) << "External prior max queue size [count]: "
            << max_external_pose_priors_queue_size_;
  LOG(INFO) << "External prior queue time horizon [ns]: "
            << FLAGS_external_prior_queue_time_horizon_ns;
  LOG(INFO) << "External prior max per optimize [count]: "
            << FLAGS_external_prior_max_per_optimize;
  LOG(INFO) << "External belief safe covariance fallback: "
            << (FLAGS_external_pose_belief_safe_covariance_fallback ? "ON"
                                                                    : "OFF");
  LOG(INFO) << "CBS diagnostic: outgoing external pose belief callback is "
            << (FLAGS_cbs_diag_disable_outgoing_external_pose_belief_callback
                    ? "DISABLED"
                    : "ENABLED");
  LOG(INFO) << "CBS outgoing publish decimation: enabled="
            << (FLAGS_cbs_outgoing_publish_decimation_enabled ? "true"
                                                              : "false")
            << " min_period_sec=" << FLAGS_cbs_outgoing_publish_min_period_sec
            << " min_kf_stride=" << FLAGS_cbs_outgoing_publish_min_kf_stride
            << " force_max_silence_sec="
            << FLAGS_cbs_outgoing_publish_force_max_silence_sec
            << " publish_on_external_effect="
            << (FLAGS_cbs_outgoing_publish_on_external_effect ? "true"
                                                              : "false")
            << " pose_delta_trans_thresh_m="
            << FLAGS_cbs_outgoing_publish_pose_delta_trans_thresh_m
            << " pose_delta_rot_thresh_deg="
            << FLAGS_cbs_outgoing_publish_pose_delta_rot_thresh_deg;

  // zy
  // In fixed-lag mode, keep timestamp->key lookup bounded to the active lag
  // window so very old external beliefs can be identified and dropped.
  if (!useCbsOptimizerHeart()) {
    const size_t lag_states =
        backend_params_.nr_states_ > 0
            ? static_cast<size_t>(backend_params_.nr_states_)
            : static_cast<size_t>(25);
    max_timestamp_to_kf_id_map_size_ = std::max<size_t>(lag_states + 10, 50);
    LOG(INFO) << "Fixed-lag mode: timestamp->key map max entries set to "
              << max_timestamp_to_kf_id_map_size_;
  } else {
    LOG(INFO) << "CBS mode: timestamp->key map max entries set to "
              << max_timestamp_to_kf_id_map_size_;
  }

  // Reset debug info.
  resetDebugInfo(&debug_info_);

  // Print parameters if verbose
  if (VLOG_IS_ON(1)) print();
}

/* -------------------------------------------------------------------------- */
BackendOutput::UniquePtr VioBackend::spinOnce(const BackendInput& input) {
  if (VLOG_IS_ON(10)) {
    input.print();
  }

  if (logger_) {
    logger_->logBackendExtOdom(input);
  }

  bool backend_status = false;
  const BackendState backend_state = backend_state_;
  switch (backend_state) {
    case BackendState::Bootstrap: {
      initializeBackend(input);
      backend_status = true;
      break;
    }
    case BackendState::Nominal: {
      // Process data with VIO.
      backend_status = addVisualInertialStateAndOptimize(input);
      break;
    }
    default: {
      LOG(FATAL) << "Unrecognized Backend state.";
      break;
    }
  }

  // Fill ouput_payload (it will remain nullptr if the backend_status is not ok)
  BackendOutput::UniquePtr output_payload = nullptr;
  if (backend_status) {
    // If Backend is doing ok, fill and return ouput_payload;
    if (VLOG_IS_ON(10)) {
      LOG(INFO) << "Latest Backend IMU bias is: ";
      getLatestImuBias().print();
      LOG(INFO) << "Prev kf Backend IMU bias is: ";
      getImuBiasPrevKf().print();
    }

    // zy Step 23
    // Intuition: always read/output factors from the optimizer that is currently active (CBS or legacy smoother).
    const gtsam::NonlinearFactorGraph* output_factor_graph = nullptr;
#ifdef KIMERA_USE_CBS
    if (useCbsOptimizerHeart()) {
      CHECK(cbs_optimizer_) << "CBS optimizer flag is ON but cbs_optimizer_ is null.";
      output_factor_graph = &cbs_optimizer_->getFactorsUnsafe();
    } else
#endif
    {
      CHECK(smoother_);
      output_factor_graph = &smoother_->getFactors();
    }
    CHECK_NOTNULL(output_factor_graph);

    // TODO(Toni): remove all of this.... It should be done in 3DVisualizer
    // or in the Mesher depending on who needs what...
    // Generate extra optional backend ouputs.
    static const bool kOutputLmkMap =
        backend_output_params_.output_map_lmk_ids_to_3d_points_in_time_horizon_;
    static const bool kMinLmkObs =
        backend_output_params_.min_num_obs_for_lmks_in_time_horizon_;
    static const bool kOutputLmkTypeMap =
        backend_output_params_.output_lmk_id_to_lmk_type_map_;
    LmkIdToLmkTypeMap lmk_id_to_lmk_type_map;
    PointsWithIdMap lmk_ids_to_3d_points_in_time_horizon;
    if (kOutputLmkMap) {
      // Generate this map only if requested, since costly.
      // Also, if lmk type requested, fill lmk id to lmk type object.
      // WARNING this also cleans the lmks inside the old_smart_factors map!
      lmk_ids_to_3d_points_in_time_horizon =
          // getMapLmkIdsTo3dPointsInTimeHorizon(
          //     smoother_->getFactors(),
          //     kOutputLmkTypeMap ? &lmk_id_to_lmk_type_map : nullptr,
          //     kMinLmkObs); (zy cancelled it)
          // zy Step 23b: map extraction must use the same active factor graph used by the optimizer.
          getMapLmkIdsTo3dPointsInTimeHorizon(
              *output_factor_graph,
              kOutputLmkTypeMap ? &lmk_id_to_lmk_type_map : nullptr,
              kMinLmkObs);

    }
    // zy Step 8_d, edited existing code 
    if (map_update_callback_) {
      map_update_callback_(lmk_ids_to_3d_points_in_time_horizon);
    } else {
      LOG(FATAL) << "Did you forget to register the Map "
                    "Update callback for at least the "
                    "Frontend? Do so by using "
                    "registerMapUpdateCallback function.";
    }

    int outgoing_publish_gate_passed = 0;
    int outgoing_publish_callback_invoked = 0;
    std::string outgoing_publish_skip_reason = "none";
    if (external_pose_belief_callback_) {
      if (FLAGS_cbs_diag_disable_outgoing_external_pose_belief_callback) {
        outgoing_publish_skip_reason = "disabled";
      } else {
        const bool should_publish =
            shouldPublishOutgoingExternalPoseBelief(
                &outgoing_publish_skip_reason);
        outgoing_publish_gate_passed = should_publish ? 1 : 0;
        if (should_publish) {
          ExternalPoseBelief belief;
          if (getLatestExternalPoseBelief(&belief)) {
            external_pose_belief_callback_(belief);
            noteOutgoingExternalPoseBeliefPublished(belief);
            outgoing_publish_callback_invoked = 1;
          } else {
            VLOG(2) << "External pose belief callback gate passed, but no "
                       "valid belief was available this cycle.";
          }
        }
      }
    }
    std::cerr << std::setprecision(12)
              << "[CBS][OutgoingPublishGateDiag] timestamp_ns=" << timestamp_lkf_
              << " frame_id=" << curr_kf_id_
              << " outgoing_callback_registered="
              << (external_pose_belief_callback_ ? 1 : 0)
              << " outgoing_publish_gate_passed=" << outgoing_publish_gate_passed
              << " outgoing_publish_callback_invoked="
              << outgoing_publish_callback_invoked
              << " outgoing_publish_skip_reason="
              << outgoing_publish_skip_reason << std::endl;

    // Create Backend Output Payload.
    output_payload = std::make_unique<BackendOutput>(
        VioNavStateTimestamped(
            input.timestamp_,
            (FLAGS_no_incremental_pose ? W_Pose_B_lkf_from_state_
                                       : W_Pose_B_lkf_from_increments_),
            W_Vel_B_lkf_,
            imu_bias_lkf_),
        // TODO(Toni): Make all below optional!!
        state_,
        // smoother_->getFactors(), (zy cancelled it)
        // zy Step 23c: downstream modules should receive the active graph (CBS in CBS mode).
        *output_factor_graph,
        getCurrentStateCovariance(),
        curr_kf_id_,
        landmark_count_,
        debug_info_,
        lmk_ids_to_3d_points_in_time_horizon,
        lmk_id_to_lmk_type_map);

    if (logger_) {
      logger_->logBackendOutput(*output_payload);
    }
  }

  return output_payload;
}

/* -------------------------------------------------------------------------- */
void VioBackend::registerImuBiasUpdateCallback(
    const ImuBiasCallback& imu_bias_update_callback) {
  // Register callback.
  imu_bias_update_callback_ = imu_bias_update_callback;
  // Update imu bias just in case. This is useful specially because the
  // Backend initializes the imu bias to some value. So whoever is asking
  // to register this callback should have the newest imu bias.
  // But the imu bias is new iff the Backend is already initialized.
  if (backend_state_ != BackendState::Bootstrap) {
    CHECK(imu_bias_update_callback_);
    imu_bias_update_callback_(imu_bias_lkf_);
  }
}

void VioBackend::registerMapUpdateCallback(
    const MapCallback& map_update_callback) {
  map_update_callback_ = map_update_callback;
}

void VioBackend::saveGraph(const std::string& filepath) const {
#ifdef KIMERA_USE_CBS
  if (useCbsOptimizerHeart()) {
    // Intuition: when CBS is active, graph export must come from BPSAM (true optimization heart), not legacy smoother.
    CHECK(cbs_optimizer_) << "CBS optimizer flag is ON but cbs_optimizer_ is null.";
    cbs_optimizer_->getFactorsUnsafe().saveGraph(filepath);
    return;
  }
#endif
  // Intuition: keep original behavior for non-CBS mode.
  CHECK(smoother_);
  smoother_->getFactors().saveGraph(filepath);
}

// zy Step 8_c
// simple setter, same pattern as existing map/IMU callback registration.
void VioBackend::registerExternalPoseBeliefCallback(
    const std::function<void(const ExternalPoseBelief&)>&
        external_pose_belief_callback) {
  external_pose_belief_callback_ = external_pose_belief_callback;
}

bool VioBackend::shouldPublishOutgoingExternalPoseBelief(
    std::string* gate_reason) const {
  CHECK_NOTNULL(gate_reason);
  *gate_reason = "none";

  if (!FLAGS_cbs_outgoing_publish_decimation_enabled) {
    *gate_reason = "disabled";
    return true;
  }

  if (!cbs_outgoing_external_pose_belief_published_once_) {
    *gate_reason = "never";
    return true;
  }

  const double min_period_sec =
      std::max(0.0, FLAGS_cbs_outgoing_publish_min_period_sec);
  const int min_kf_stride = std::max(1, FLAGS_cbs_outgoing_publish_min_kf_stride);
  const double force_max_silence_sec =
      std::max(0.0, FLAGS_cbs_outgoing_publish_force_max_silence_sec);
  const double pose_delta_trans_thresh_m =
      std::max(0.0, FLAGS_cbs_outgoing_publish_pose_delta_trans_thresh_m);
  const double pose_delta_rot_thresh_rad =
      std::max(0.0, FLAGS_cbs_outgoing_publish_pose_delta_rot_thresh_deg) *
      (3.14159265358979323846 / 180.0);

  double elapsed_sec_since_last_publish = 0.0;
  if (timestamp_lkf_ >=
      cbs_outgoing_external_pose_belief_last_published_timestamp_ns_) {
    elapsed_sec_since_last_publish =
        1e-9 * static_cast<double>(
                   timestamp_lkf_ -
                   cbs_outgoing_external_pose_belief_last_published_timestamp_ns_);
  }
  int kf_stride_since_last_publish = 0;
  const int64_t curr_kf_id_i64 = static_cast<int64_t>(curr_kf_id_);
  const int64_t last_published_kf_id_i64 = static_cast<int64_t>(
      cbs_outgoing_external_pose_belief_last_published_frame_id_);
  if (curr_kf_id_i64 >= last_published_kf_id_i64) {
    kf_stride_since_last_publish =
        static_cast<int>(curr_kf_id_i64 - last_published_kf_id_i64);
  }
#ifdef KIMERA_USE_CBS
  // External effect is the only immediate override that can bypass min period.
  if (FLAGS_cbs_outgoing_publish_on_external_effect) {
    size_t last_epoch_beliefs_accepted = 0u;
    size_t last_epoch_priors_injected = 0u;
    {
      std::lock_guard<std::mutex> effect_lock(cbs_external_effect_state_mutex_);
      last_epoch_beliefs_accepted = cbs_last_epoch_beliefs_accepted_;
      last_epoch_priors_injected = cbs_last_epoch_priors_injected_;
    }
    if (last_epoch_beliefs_accepted > 0u || last_epoch_priors_injected > 0u) {
      *gate_reason = "external_effect";
      return true;
    }
  }
#endif

  // Heartbeat (slow cadence) keeps exchange alive when nothing changes.
  if (elapsed_sec_since_last_publish >= force_max_silence_sec) {
    *gate_reason = "force_silence";
    return true;
  }

  // Regular publish policy requires min period + min stride + pose delta.
  if (elapsed_sec_since_last_publish < min_period_sec) {
    *gate_reason = "below_min_period";
    return false;
  }

  if (kf_stride_since_last_publish < min_kf_stride) {
    *gate_reason = "below_min_stride";
    return false;
  }

  try {
    const gtsam::Pose3 delta_pose =
        cbs_outgoing_external_pose_belief_last_published_pose_.between(
            W_Pose_B_lkf_from_state_);
    const double trans_delta_m = delta_pose.translation().norm();
    const double rot_delta_rad = delta_pose.rotation().axisAngle().second;
    if (trans_delta_m > pose_delta_trans_thresh_m ||
        rot_delta_rad > pose_delta_rot_thresh_rad) {
      *gate_reason = "gated_pose_delta";
      return true;
    }
  } catch (const std::exception& e) {
    VLOG(2) << "Outgoing publish pose-delta gate failed: " << e.what();
    *gate_reason = "none";
    return false;
  } catch (...) {
    VLOG(2) << "Outgoing publish pose-delta gate failed with unknown "
               "exception.";
    *gate_reason = "none";
    return false;
  }

  *gate_reason = "below_pose_delta";
  return false;
}

void VioBackend::noteOutgoingExternalPoseBeliefPublished(
    const ExternalPoseBelief& belief) {
  cbs_outgoing_external_pose_belief_published_once_ = true;
  cbs_outgoing_external_pose_belief_last_published_timestamp_ns_ =
      belief.timestamp_kf_nsec_;
  cbs_outgoing_external_pose_belief_last_published_frame_id_ = belief.frame_id_;
  cbs_outgoing_external_pose_belief_last_published_pose_ = belief.W_Pose_B_;
}



/* -------------------------------------------------------------------------- */
bool VioBackend::initStateAndSetPriors(
    const VioNavStateTimestamped& vio_nav_state_initial_seed) {
  // Clean state
  new_values_.clear();
  new_imu_prior_and_other_factors_.resize(0);
  new_external_prior_factors_.resize(0);

  // Update member variables.
  timestamp_lkf_ = vio_nav_state_initial_seed.timestamp_;

  // zy Step 4_a
  // On backend (re)initialization, reset external-prior staging state, 
  // so old messages from previous runs don't leak into the new graph.
  // if Kimera restarts while ROS continues, old queued beliefs can corrupt a new optimization session. This guarantees a clean slate.
  {
    std::lock_guard<std::mutex> queue_lock(external_pose_priors_queue_mutex_);
    external_pose_priors_queue_.clear();
  }
  {
    std::lock_guard<std::mutex> map_lock(timestamp_to_kf_id_map_mutex_);
    timestamp_to_kf_id_map_.clear();
    timestamp_to_kf_id_map_[timestamp_lkf_] = curr_kf_id_;
  }
#ifdef KIMERA_USE_CBS
  cbs_local_cov_sidecar_.reset();
  cbs_local_cov_smoother_sidecar_.reset();
  cbs_h2_local_smoother_heart_to_sidecar_slot_map_.clear();
  cbs_h2_sidecar_heart_to_sidecar_slot_map_.clear();
  h2_local_graph_snapshot_.resize(0);
  h2_local_values_snapshot_.clear();
  h2_local_snapshot_timestamp_ns_ = -1;
  h2_local_snapshot_frame_id_ = 0;
  h2_local_snapshot_valid_ = false;
  cbs_h2_sidecar_sync_ok_ = false;
  cbs_h2_sidecar_desync_streak_ = 0u;
  cbs_h2_sidecar_fallback_cov_epochs_ = 0u;
  cbs_h2_sidecar_hard_reset_count_ = 0u;
  cbs_h2_sidecar_update_ms_last_epoch_ = 0.0;
  cbs_h2_local_snapshot_refresh_ms_last_epoch_ = 0.0;
  cbs_h2_sidecar_local_factor_count_last_epoch_ = 0u;
  cbs_h2_sidecar_filtered_external_count_last_epoch_ = 0u;
  cbs_h2_sidecar_remove_count_last_epoch_ = 0u;
  cbs_h2_sidecar_values_add_count_last_epoch_ = 0u;
  cbs_h2_sidecar_smart_factor_replacements_last_epoch_ = 0u;
  cbs_h2_sidecar_value_type_mismatch_count_last_epoch_ = 0u;
  cbs_h2_sidecar_missing_value_count_last_epoch_ = 0u;
  cbs_h2_sidecar_unsupported_key_type_count_last_epoch_ = 0u;
  cbs_h2_sidecar_unmapped_remove_slot_count_last_epoch_ = 0u;
  cbs_h2_sidecar_post_prune_remove_count_last_epoch_ = 0u;
  cbs_h2_sidecar_packet_pose_values_count_last_epoch_ = 0u;
  cbs_h2_sidecar_packet_vel_values_count_last_epoch_ = 0u;
  cbs_h2_sidecar_packet_bias_values_count_last_epoch_ = 0u;
  cbs_h2_sidecar_first_failure_key_last_epoch_.clear();
  cbs_h2_sidecar_first_failure_reason_last_epoch_.clear();
  cbs_h2_sidecar_last_failure_reason_.clear();
  cbs_h2_first_bad_epoch_logged_ = false;
  cbs_external_prior_factor_ptrs_.clear();
#endif


  // These two are identical in the beginning, but _from_state_ is used in
  // the optimizer and _from_increments_ is used as a smooth output
  W_Pose_B_lkf_from_state_ = vio_nav_state_initial_seed.pose_;
  W_Pose_B_lkf_from_increments_ = vio_nav_state_initial_seed.pose_;

  W_Vel_B_lkf_ = vio_nav_state_initial_seed.velocity_;
  imu_bias_lkf_ = vio_nav_state_initial_seed.imu_bias_;
  imu_bias_prev_kf_ = vio_nav_state_initial_seed.imu_bias_;

  VLOG(2) << "Initial state seed: \n"
          << " - Initial timestamp: " << timestamp_lkf_ << '\n'
          << " - Initial pose: " << W_Pose_B_lkf_from_state_ << '\n'
          << " - Initial vel: " << W_Vel_B_lkf_.transpose() << '\n'
          << " - Initial IMU bias: " << imu_bias_lkf_;

  // Can't add inertial prior factor until we have a state measurement.
  addInitialPriorFactors(curr_kf_id_);

  // Add initial state seed
  addStateValues(
      curr_kf_id_, W_Pose_B_lkf_from_state_, W_Vel_B_lkf_, imu_bias_lkf_);

  VLOG(2) << "Start optimize with initial state and priors!";
  return optimize(vio_nav_state_initial_seed.timestamp_,
                  curr_kf_id_,
                  backend_params_.numOptimize_);
}

/* -------------------------------------------------------------------------- */
// Workhorse that stores data and optimizes at each keyframe.
// [in] timestamp_kf_nsec, keyframe timestamp.
// [in] status_smart_stereo_measurements_kf, vision data.
bool VioBackend::addVisualInertialStateAndOptimize(
    const Timestamp& timestamp_kf_nsec,
    const StatusStereoMeasurements& status_smart_stereo_measurements_kf,
    const gtsam::PreintegrationType& pim,
    std::optional<gtsam::Pose3> odometry_body_pose,
    std::optional<gtsam::Velocity3> odometry_vel) {
  debug_info_.resetAddedFactorsStatistics();

  // Features and IMU line up --> do iSAM update
  last_kf_id_ = curr_kf_id_;
  ++curr_kf_id_;

  VLOG(1) << "VIO: adding keyframe " << curr_kf_id_
          << " at timestamp:" << UtilsNumerical::NsecToSec(timestamp_kf_nsec)
          << " (nsec).";
  
  // zy Step 3_c
  // every keyframe gets a stable timestamp->key entry so beliefs can target exact past poses. 
  {
  std::lock_guard<std::mutex> lock(timestamp_to_kf_id_map_mutex_);
  timestamp_to_kf_id_map_[timestamp_kf_nsec] = curr_kf_id_;

  while (timestamp_to_kf_id_map_.size() > max_timestamp_to_kf_id_map_size_) {
    timestamp_to_kf_id_map_.erase(timestamp_to_kf_id_map_.begin());
  }
  }

  // Add initial guess.
  addStateValues(curr_kf_id_,
                 status_smart_stereo_measurements_kf.first,
                 pim,
                 odometry_body_pose,
                 odometry_vel);

  /////////////////// MANAGE IMU MEASUREMENTS ///////////////////////////
  // Add imu factors between consecutive keyframe states
  addImuFactor(last_kf_id_, curr_kf_id_, pim);

  // Add between factor from RANSAC: first PnP, then Stereo, then Mono
  if (backend_params_.addBetweenStereoFactors_ &&
      status_smart_stereo_measurements_kf.first.kfTrackingStatus_stereo_ ==
          TrackingStatus::VALID) {
    addBetweenFactor(
        last_kf_id_,
        curr_kf_id_,
        // I think this should be B_Pose_leftCamRect_...
        B_Pose_leftCamRect_ *
            status_smart_stereo_measurements_kf.first.lkf_T_k_stereo_ *
            B_Pose_leftCamRect_.inverse(),
        backend_params_.betweenRotationPrecision_,
        backend_params_.betweenTranslationPrecision_);
  }

  /////////////////// MANAGE VISION MEASUREMENTS ///////////////////////////
  const StereoMeasurements& smart_stereo_measurements_kf =
      status_smart_stereo_measurements_kf.second;

  // if stereo ransac failed, remove all right pixels:
  // TrackingStatus kfTrackingStatus_stereo =
  //     status_smart_stereo_measurements_kf.first.kfTrackingStatus_stereo_;
  // if(kfTrackingStatus_stereo == TrackingStatus::INVALID){
  //   for(size_t i = 0; i < smartStereoMeasurements_kf.size(); i++)
  //     smartStereoMeasurements_kf[i].uR =
  //     std::numeric_limits<double>::quiet_NaN();;
  //}

  // extract relevant information from stereo frame
  LandmarkIds landmarks_kf;
  addStereoMeasurementsToFeatureTracks(
      curr_kf_id_, smart_stereo_measurements_kf, &landmarks_kf);

  if (VLOG_IS_ON(10)) {
    printFeatureTracks();
  }

  // decide which factors to add
  const TrackingStatus& kfTrackingStatus_mono =
      status_smart_stereo_measurements_kf.first.kfTrackingStatus_mono_;
  switch (kfTrackingStatus_mono) {
    // vehicle is not moving
    case TrackingStatus::LOW_DISPARITY: {
      if (backend_params_.low_disparity_use_imu_motion_gate_) {
        const gtsam::NavState navstate_lkf(W_Pose_B_lkf_from_state_,
                                           W_Vel_B_lkf_);
        const gtsam::NavState predicted_state =
            pim.predict(navstate_lkf, imu_bias_lkf_);
        const gtsam::Pose3 B_lkf_Pose_kf_imu =
            W_Pose_B_lkf_from_state_.between(predicted_state.pose());
        const double imu_dt_s = pim.deltaTij();
        const double imu_translation_norm_m =
            B_lkf_Pose_kf_imu.translation().norm();
        const double imu_rotation_norm_rad =
            gtsam::Rot3::Logmap(B_lkf_Pose_kf_imu.rotation()).norm();
        const double imu_delta_speed_norm_mps =
            (predicted_state.velocity() - W_Vel_B_lkf_).norm();

        const bool imu_indicates_motion =
            std::isfinite(imu_dt_s) && imu_dt_s > 0.0 &&
            (imu_translation_norm_m >
                 backend_params_
                     .low_disparity_motion_translation_threshold_m_ ||
             imu_rotation_norm_rad >
                 backend_params_.low_disparity_motion_rotation_threshold_rad_ ||
             imu_delta_speed_norm_mps >
                 backend_params_
                     .low_disparity_motion_delta_speed_threshold_mps_);

        if (imu_indicates_motion) {
          LOG(WARNING)
              << "Low disparity with IMU-indicated motion: using constant "
                 "velocity factor instead of no-motion priors."
              << " imu_dt_s=" << imu_dt_s
              << " imu_translation_m=" << imu_translation_norm_m
              << " imu_rotation_rad=" << imu_rotation_norm_rad
              << " imu_delta_speed_mps=" << imu_delta_speed_norm_mps;
          if (backend_params_.constant_vel_precision_ > 0.0) {
            addConstantVelocityFactor(last_kf_id_, curr_kf_id_);
          } else {
            LOG(WARNING)
                << "Low disparity with IMU motion but constant_vel_precision "
                   "is zero: skipping no-motion constraints for this keyframe.";
          }
          break;
        }
      }

      LOG(WARNING)
          << "Low disparity: adding zero velocity and no motion factors.";
      if (backend_params_.zero_velocity_precision_ > 0.0) {
        addZeroVelocityPrior(curr_kf_id_);
      } else {
        LOG(ERROR) << "Low disparity: not adding addZeroVelocityPrior because "
                      "precision is zero.";
      }
      if (backend_params_.no_motion_position_precision_ > 0.0 ||
          backend_params_.no_motion_rotation_precision_ > 0.0) {
        addNoMotionFactor(last_kf_id_, curr_kf_id_);
      } else {
        LOG(ERROR) << "Low disparity: not adding addNoMotionFactor because "
                      "precision is zero.";
      }
      break;
    }

    // This did not improve in any case
    //  case TrackingStatus::INVALID :// ransac failed hence we cannot
    //  trust features
    //    if (verbosity_ >= 7) {printf("Add constant velocity factor
    //    (monoRansac is INVALID)\n");}
    //    if (backend_params_.constant_vel_precision_ > 0.0) {
    //      addConstantVelocityFactor(last_id_, cur_id_); break;
    //    }

    // TrackingStatus::VALID, FEW_MATCHES, INVALID, DISABLED : //
    // we add features in VIO
    default: {
      addLandmarksToGraph(landmarks_kf);
      break;
    }
  }

  // Add odometry factors if they're available and have non-zero precision
  if (odometry_body_pose && odom_params_ &&
      (odom_params_->betweenRotationPrecision_ > 0.0 ||
       odom_params_->betweenTranslationPrecision_ > 0.0)) {
    VLOG(1) << "Added external factor between " << last_kf_id_ << " and "
            << curr_kf_id_;
    addBetweenFactor(last_kf_id_,
                     curr_kf_id_,
                     *odometry_body_pose,
                     odom_params_->betweenRotationPrecision_,
                     odom_params_->betweenTranslationPrecision_);
  }
  if (odometry_vel && odom_params_ && odom_params_->velocityPrecision_ > 0.0) {
    LOG_FIRST_N(ERROR, 1)
        << "Using velocity priors from external odometry: "
        << "This only works if you have velocity estimates in the world frame! "
        << "(not provided by typical odometry sensors)";
    addVelocityPrior(
        curr_kf_id_, *odometry_vel, odom_params_->velocityPrecision_);
  }

  // Why do we do this??
  // This lags 1 step behind to mimic hw.
  // imu_bias_lkf_ gets updated in the optimize call.
  imu_bias_prev_kf_ = imu_bias_lkf_;

  return optimize(timestamp_kf_nsec, curr_kf_id_, backend_params_.numOptimize_);
}

bool VioBackend::addVisualInertialStateAndOptimize(const BackendInput& input) {
  VLOG(10) << "Add visual inertial state and optimize.";
  CHECK(input.status_stereo_measurements_kf_);
  CHECK(input.pim_);
  bool is_smoother_ok = addVisualInertialStateAndOptimize(
      input.timestamp_,  // Current time for fixed lag smoother.
      *input.status_stereo_measurements_kf_,  // Vision data.
      *input.pim_,                            // Imu preintegrated data.
      input.body_lkf_OdomPose_body_kf_,
      input.body_kf_world_OdomVel_body_kf_);
  // Bookkeeping
  timestamp_lkf_ = input.timestamp_;
  return is_smoother_ok;
}

// TODO(Toni): no need to pass landmarks_kf, can iterate directly over feature
// tracks..
// Uses landmark table to add factors in graph.
void VioBackend::addLandmarksToGraph(const LandmarkIds& landmarks_kf) {
  // Add selected landmarks to graph:
  int n_new_landmarks = 0;
  int n_updated_landmarks = 0;
  debug_info_.numAddedSmartF_ += landmarks_kf.size();

  for (const LandmarkId& lmk_id : landmarks_kf) {
    FeatureTrack& ft = feature_tracks_.at(lmk_id);
    // TODO(TONI): parametrize this min_num_of_obs... should be in Frontend
    // rather than Backend though...
    if (ft.obs_.size() < 2) {  // we only insert feature tracks of length at
                               // least 2 (otherwise uninformative)
      continue;
    }

    if (!ft.in_ba_graph_) {
      const bool added = addLandmarkToGraph(lmk_id, ft);
      ft.in_ba_graph_ = added;
      if (added) {
        ++n_new_landmarks;
      }
    } else {
      const std::pair<FrameId, StereoPoint2> obs_kf = ft.obs_.back();

      LOG_IF(FATAL, obs_kf.first != static_cast<FrameId>(curr_kf_id_))
          << "addLandmarksToGraph: last obs is not from the current "
             "keyframe!\n";

      if (old_smart_factors_.find(lmk_id) == old_smart_factors_.end()) {
        // After backend recovery/cleanup a track may still be marked
        // in_ba_graph_ while its smart-factor bookkeeping entry is gone.
        // Recreate the factor from the full track instead of hard-failing.
        LOG(WARNING) << "Landmark " << lmk_id
                     << " missing from old_smart_factors_. Re-adding it from "
                        "feature track history.";
        const bool added = addLandmarkToGraph(lmk_id, ft);
        ft.in_ba_graph_ = added;
        if (added) {
          ++n_new_landmarks;
        }
        continue;
      }

      updateLandmarkInGraph(lmk_id, obs_kf);
      ++n_updated_landmarks;
    }
  }

  VLOG(10) << "Added " << n_new_landmarks << " new landmarks\n"
           << "Updated " << n_updated_landmarks << " landmarks in graph";
}

/* -------------------------------------------------------------------------- */
// Adds a landmark to the graph for the first time.
bool VioBackend::addLandmarkToGraph(const LandmarkId& lmk_id,
                                    const FeatureTrack& ft) {
  // We use a unit pinhole projection camera for the smart factors to be
  // more efficient.
  SmartStereoFactor::shared_ptr new_factor(new SmartStereoFactor(
      smart_noise_, smart_factors_params_, B_Pose_leftCamRect_));

  VLOG(10) << "Adding landmark with: " << ft.obs_.size()
           << " landmarks to graph, with keys: ";

  // Add observations to smart factor
  size_t num_active_observations = 0u;
  if (VLOG_IS_ON(10)) new_factor->print();
  std::stringstream ss;
  for (const std::pair<FrameId, StereoPoint2>& obs : ft.obs_) {
    const FrameId& frame_id = obs.first;
    const gtsam::Symbol& pose_symbol = gtsam::Symbol(kPoseSymbolChar, frame_id);
    bool pose_key_is_active = false;
#ifdef KIMERA_USE_CBS
    if (useCbsOptimizerHeart()) {
      CHECK(cbs_optimizer_)
          << "CBS optimizer flag is ON but cbs_optimizer_ is null.";
      pose_key_is_active =
          cbs_optimizer_->valueExists(pose_symbol) ||
          new_values_.exists(pose_symbol);
    } else
#endif
    {
      pose_key_is_active =
          state_.exists(pose_symbol) || new_values_.exists(pose_symbol);
    }
    if (!pose_key_is_active) {
      continue;
    }
    const StereoPoint2& measurement = obs.second;
    new_factor->add(measurement, pose_symbol, stereo_cal_);
    ++num_active_observations;

    if (VLOG_IS_ON(10)) ss << " " << obs.first;
  }
  VLOG(10) << ss.str() << std::endl;

  if (num_active_observations < 2u) {
    VLOG(2) << "Skipping smart-factor add for landmark " << lmk_id
            << ": only " << num_active_observations
            << " active observation(s) in current optimizer window.";
    return false;
  }

  // add new factor to suitable structures:
  new_smart_factors_.insert(std::make_pair(lmk_id, new_factor));
  old_smart_factors_.insert(
      std::make_pair(lmk_id, std::make_pair(new_factor, -1)));
  return true;
}

/* -------------------------------------------------------------------------- */
// Updates a landmark already in the graph.
void VioBackend::updateLandmarkInGraph(
    const LandmarkId& lmk_id,
    const std::pair<FrameId, StereoPoint2>& new_measurement) {
  // Update existing smart-factor
  auto old_smart_factors_it = old_smart_factors_.find(lmk_id);
  if (old_smart_factors_it == old_smart_factors_.end()) {
    LOG(WARNING) << "updateLandmarkInGraph: landmark " << lmk_id
                 << " is missing from old_smart_factors_. Skipping update.";
    return;
  }

  Slot slot = old_smart_factors_it->second.second;
  if (slot == -1) {
    // Factor not yet inserted in the graph: keep the queued version updated
    // incrementally.
    const auto& old_factor = old_smart_factors_it->second.first;
    SmartStereoFactor::shared_ptr new_factor(new SmartStereoFactor(*old_factor));
    const gtsam::Symbol pose_symbol(kPoseSymbolChar, new_measurement.first);
    const StereoPoint2& measurement = new_measurement.second;
    new_factor->add(measurement, pose_symbol, stereo_cal_);
    new_smart_factors_[lmk_id] = new_factor;
    old_smart_factors_it->second.first = new_factor;
    VLOG(10) << "updateLandmarkInGraph: updated queued factor for point: "
             << lmk_id;
    return;
  }

  // Rebuild candidate support from the full feature-track window. This keeps
  // smart-factor replacement decisions based on true support change, not only
  // the most recent observation.
  const auto ft_it = feature_tracks_.find(lmk_id);
  if (ft_it == feature_tracks_.end()) {
    LOG(WARNING) << "updateLandmarkInGraph: landmark " << lmk_id
                 << " missing in feature_tracks_. Skipping update.";
    return;
  }

  SmartStereoFactor::shared_ptr rebuilt_factor(new SmartStereoFactor(
      smart_noise_, smart_factors_params_, B_Pose_leftCamRect_));

  size_t num_active_observations = 0u;
  for (const std::pair<FrameId, StereoPoint2>& obs : ft_it->second.obs_) {
    const gtsam::Symbol pose_symbol(kPoseSymbolChar, obs.first);
    if (!isPoseKeyActiveInOptimizer(pose_symbol)) {
      continue;
    }
    rebuilt_factor->add(obs.second, pose_symbol, stereo_cal_);
    ++num_active_observations;
  }

  if (num_active_observations < 2u) {
    VLOG(2) << "updateLandmarkInGraph: skipping point " << lmk_id
            << " because rebuilt support in active window has only "
            << num_active_observations << " observation(s).";
    return;
  }

  new_smart_factors_[lmk_id] = rebuilt_factor;
  VLOG(10) << "updateLandmarkInGraph: rebuilt candidate support for point "
           << lmk_id << " with " << num_active_observations
           << " active observations.";
}

/* -------------------------------------------------------------------------- */
// Get valid 3D points and corresponding lmk id.
// Warning! it modifies old_smart_factors_!!
PointsWithIdMap VioBackend::getMapLmkIdsTo3dPointsInTimeHorizon(
    const gtsam::NonlinearFactorGraph& graph,
    LmkIdToLmkTypeMap* lmk_id_to_lmk_type_map,
    const size_t& min_age) {
  PointsWithIdMap points_with_id;

  if (lmk_id_to_lmk_type_map) {
    lmk_id_to_lmk_type_map->clear();
  }

  // Step 1:
  /////////////// Add landmarks encoded in the smart factors. //////////////////

  // old_smart_factors_ has all smart factors included so far.
  // Retrieve lmk ids from smart factors in state.
  size_t nr_valid_smart_lmks = 0, nr_smart_lmks = 0;
  for (SmartFactorMap::iterator old_smart_factor_it =
           old_smart_factors_.begin();
       old_smart_factor_it !=
       old_smart_factors_
           .end();) {  //!< landmarkId -> {SmartFactorPtr, SlotIndex}
    // Store number of smart lmks (one smart factor per landmark).
    nr_smart_lmks++;

    // Retrieve lmk_id of the smart factor.
    const LandmarkId& lmk_id = old_smart_factor_it->first;

    // Retrieve smart factor.
    const SmartStereoFactor::shared_ptr& smart_factor_ptr =
        old_smart_factor_it->second.first;
    // Check that pointer is well definied.
    CHECK(smart_factor_ptr) << "Smart factor is not well defined.";

    // Retrieve smart factor slot in the graph.
    const Slot& slot_id = old_smart_factor_it->second.second;

    // Check that slot is admissible.
    // Slot should be positive.
    DCHECK(slot_id >= 0) << "Slot of smart factor is not admissible.";
    // Ensure the graph size is small enough to cast to int.
    DCHECK_LT(graph.size(), std::numeric_limits<Slot>::max())
        << "Invalid cast, that would cause an overflow!";
    // Slot should be inferior to the size of the graph.
    DCHECK_LT(slot_id, static_cast<Slot>(graph.size()));

    // Check that this slot_id exists in the graph, aka check that it is
    // in bounds and that the pointer is live (aka at(slot_id) works).
    if (!graph.exists(slot_id)) {
      // This slot does not exist in the current graph...
      VLOG(5) << "The slot with id: " << slot_id
              << " does not exist in the graph.\n"
              << "Deleting old_smart_factor of lmk id: " << lmk_id;
      old_smart_factor_it = old_smart_factors_.erase(old_smart_factor_it);
      // Update as well the feature track....
      // TODO(TONI): please remove this and centralize how feature tracks
      // and new/old_smart_factors are added and removed!
      CHECK(deleteLmkFromFeatureTracks(lmk_id));
      continue;
    } else {
      VLOG(20) << "Slot id: " << slot_id
               << " for smart factor of lmk id: " << lmk_id;
    }

    // Check that the pointer smart_factor_ptr points to the right element
    // in the graph.
    if (smart_factor_ptr != graph.at(slot_id)) {
      // Pointer in the graph does not match
      // the one we stored in old_smart_factors_
      // ERROR: if the pointers don't match, then the code that follows does
      // not make any sense, since we are using lmk_id which comes from
      // smart_factor and result which comes from graph[slot_id], we should
      // use smart_factor_ptr instead then...
      LOG(ERROR) << "The factor with slot id: " << slot_id
                 << " in the graph does not match the old_smart_factor of "
                 << "lmk with id: " << lmk_id << "\n."
                 << "Deleting old_smart_factor of lmk id: " << lmk_id;
      old_smart_factor_it = old_smart_factors_.erase(old_smart_factor_it);
      CHECK(deleteLmkFromFeatureTracks(lmk_id));
      continue;
    }

    // Why do we do this? all info is in smart_factor_ptr
    // such as the triangulated point, whether it is valid or not
    // and the number of observations...
    // Is graph more up to date?
    const auto graph_factor = graph.at(slot_id);
    const auto gsf = dynamic_cast<const SmartStereoFactor*>(graph_factor.get());
    CHECK(gsf) << "Cannot cast factor in graph to a smart stereo factor.";

    // Get triangulation result from smart factor.
    const gtsam::TriangulationResult& result = gsf->point();
    if (result.valid()) {
      CHECK(result);
      if (gsf->measured().size() >= min_age) {
        // Triangulation result from smart factor is valid and
        // we have observed the lmk at least min_age times.
        VLOG(20) << "Adding lmk with id: " << lmk_id
                 << " to list of lmks in time horizon";
        // Check that we have not added this lmk already...
        CHECK(points_with_id.find(lmk_id) == points_with_id.end());
        points_with_id[lmk_id] = *result;
        if (lmk_id_to_lmk_type_map) {
          (*lmk_id_to_lmk_type_map)[lmk_id] = LandmarkType::SMART;
        }
        nr_valid_smart_lmks++;
      } else {
        VLOG(20) << "Rejecting lmk with id: " << lmk_id
                 << " from list of lmks in time horizon: "
                 << "not enough measurements, " << gsf->measured().size()
                 << ", vs min_age of " << min_age << ".";
      }  // gsf->measured().size() >= min_age ?
    } else {
      VLOG(20) << "Triangulation result for smart factor of lmk with id "
               << lmk_id << " is not initialized...";
    }

    // Next iteration.
    old_smart_factor_it++;
  }

  // Step 2:
  ////////////// Add landmarks that now are in projection factors. /////////////
  size_t nr_proj_lmks = 0;
  for (const auto& key_value : state_) {
    const gtsam::Symbol key(key_value.key);
    if (key.chr() != 'l') {
      continue;
    }

    const auto lmk_id = key.index();
    DCHECK(points_with_id.find(lmk_id) == points_with_id.end());
    points_with_id[lmk_id] = key_value.value.cast<gtsam::Point3>();
    if (lmk_id_to_lmk_type_map) {
      (*lmk_id_to_lmk_type_map)[lmk_id] = LandmarkType::PROJECTION;
    }
    nr_proj_lmks++;
  }

  // TODO aren't these points post-optimization? Shouldn't we instead add
  // the points before optimization? Then the regularities we enforce will
  // have the most impact, otherwise the points in the optimization horizon
  // do not move that much after optimizing... they are almost frozen and
  // are not visually changing much...
  // They might actually not be changing that much because we are not
  // enforcing the regularities on the points that are out of current frame
  // in the Backend currently...

  VLOG(10) << "Landmark typology to be used for the mesh:\n"
           << "Number of valid smart factors " << nr_valid_smart_lmks
           << " out of " << nr_smart_lmks << "\n"
           << "Number of landmarks (not involved in a smart factor) "
           << nr_proj_lmks << ".\n Total number of landmarks: "
           << (nr_valid_smart_lmks + nr_proj_lmks);
  return points_with_id;
}

// zy Step 17, "i replaced the whole original function"
/* -------------------------------------------------------------------------- */
// NOT TESTED (--> There is a UnitTest function in UtilsOpenCV)
void VioBackend::computeStateCovariance() {
#ifdef KIMERA_USE_CBS
  if (useCbsOptimizerHeart()) {
    // Intuition: in CBS-heart mode, covariance must come from BPSAM marginals, not from the legacy fixed-lag smoother.
    CHECK(cbs_optimizer_) << "CBS optimizer flag is ON but cbs_optimizer_ is null.";
    
    // Intuition: BPSAM exposes per-key marginals; we build a conservative block-diagonal [x(6), v(3), b(6)] covariance.
    state_covariance_lkf_ = gtsam::Matrix::Identity(15, 15) * 1e-6;

    const gtsam::Symbol pose_key(kPoseSymbolChar, curr_kf_id_);
    const gtsam::Symbol vel_key(kVelocitySymbolChar, curr_kf_id_);
    const gtsam::Symbol bias_key(kImuBiasSymbolChar, curr_kf_id_);

    if (cbs_optimizer_->valueExists(pose_key)) {
      try {
        const gtsam::Matrix pose_cov = cbs_optimizer_->marginalCovariance(
            pose_key, cbs::BPSAM::MarginalizationType::FULL);
        if (pose_cov.rows() >= 6 && pose_cov.cols() >= 6 &&
            pose_cov.allFinite()) {
          state_covariance_lkf_.block(0, 0, 6, 6) =
              pose_cov.topLeftCorner(6, 6);
        } else {
          VLOG(2) << "Invalid CBS pose covariance for key: " << pose_key;
        }
      } catch (const std::exception& e) {
        VLOG(2) << "CBS pose covariance query failed for key " << pose_key
                << ": " << e.what();
      }
    } else {
      VLOG(2) << "CBS pose key not found for covariance: " << pose_key;
    }

    if (cbs_optimizer_->valueExists(vel_key)) {
      try {
        const gtsam::Matrix vel_cov = cbs_optimizer_->marginalCovariance(
            vel_key, cbs::BPSAM::MarginalizationType::FULL);
        if (vel_cov.rows() >= 3 && vel_cov.cols() >= 3 &&
            vel_cov.allFinite()) {
          state_covariance_lkf_.block(6, 6, 3, 3) =
              vel_cov.topLeftCorner(3, 3);
        } else {
          VLOG(2) << "Invalid CBS velocity covariance for key: " << vel_key;
        }
      } catch (const std::exception& e) {
        VLOG(2) << "CBS velocity covariance query failed for key " << vel_key
                << ": " << e.what();
      }
    } else {
      VLOG(2) << "CBS velocity key not found for covariance: " << vel_key;
    }

    if (cbs_optimizer_->valueExists(bias_key)) {
      try {
        const gtsam::Matrix bias_cov = cbs_optimizer_->marginalCovariance(
            bias_key, cbs::BPSAM::MarginalizationType::FULL);
        if (bias_cov.rows() >= 6 && bias_cov.cols() >= 6 &&
            bias_cov.allFinite()) {
          state_covariance_lkf_.block(9, 9, 6, 6) =
              bias_cov.topLeftCorner(6, 6);
        } else {
          VLOG(2) << "Invalid CBS bias covariance for key: " << bias_key;
        }
      } catch (const std::exception& e) {
        VLOG(2) << "CBS bias covariance query failed for key " << bias_key
                << ": " << e.what();
      }
    } else {
      VLOG(2) << "CBS bias key not found for covariance: " << bias_key;
    }

    // Intuition: enforce numeric symmetry before publishing/consuming covariance downstream.
    state_covariance_lkf_ =
        0.5 * (state_covariance_lkf_ + state_covariance_lkf_.transpose());
    state_covariance_lkf_valid_ = true;
    return;
  }
#endif

  gtsam::Marginals marginals(smoother_->getFactors(),
                             state_,
                             gtsam::Marginals::Factorization::CHOLESKY);

  // Current state includes pose, velocity and imu biases.
  gtsam::KeyVector keys;
  keys.push_back(gtsam::Symbol(kPoseSymbolChar, curr_kf_id_));
  keys.push_back(gtsam::Symbol(kVelocitySymbolChar, curr_kf_id_));
  keys.push_back(gtsam::Symbol(kImuBiasSymbolChar, curr_kf_id_));

  // Return the marginal covariance matrix.
  state_covariance_lkf_ = UtilsOpenCV::Covariance_bvx2xvb(
      marginals.jointMarginalCovariance(keys)
          .fullMatrix());  // 6 + 3 + 6 = 15x15matrix
  state_covariance_lkf_valid_ = true;
}

#ifdef KIMERA_USE_CBS
bool VioBackend::queryH2LocalPoseCovFromActiveSmoother(
    const gtsam::Symbol& pose_symbol,
    gtsam::Pose3* pose_out,
    gtsam::Matrix66* cov_out,
    std::string* reason_out,
    std::string* source_path_out) const {
  auto set_reason = [&](const std::string& reason) {
    if (reason_out) {
      *reason_out = reason;
    }
  };
  auto set_source_path = [&](const std::string& source_path) {
    if (source_path_out) {
      *source_path_out = source_path;
    }
  };
  auto key_to_string = [](const gtsam::Symbol& symbol) -> std::string {
    std::ostringstream oss;
    oss << symbol.chr() << symbol.index();
    return oss.str();
  };
  auto sanitize_anchor_var = [](double requested, double fallback) -> double {
    if (!std::isfinite(requested) || requested <= 0.0) {
      return fallback;
    }
    return requested;
  };

  CHECK_NOTNULL(pose_out);
  CHECK_NOTNULL(cov_out);
  set_reason("none");
  set_source_path("unset");

  if (!useCbsH2LocalCovSidecar() || useCbsOptimizerHeart()) {
    set_reason("h2_mode_inactive");
    return false;
  }
  if (!h2_local_snapshot_valid_) {
    set_reason("h2_local_snapshot_invalid");
    return false;
  }

  const double anchor_rot_var =
      sanitize_anchor_var(FLAGS_cbs_h2_local_cov_anchor_rot_var, 1e-2);
  const double anchor_trans_var =
      sanitize_anchor_var(FLAGS_cbs_h2_local_cov_anchor_trans_var, 1e-1);
  const bool use_anchored_covariance = FLAGS_cbs_h2_publish_anchored_local_cov;

  const auto finalize_covariance = [&](const gtsam::Matrix& cov,
                                       std::string* finalize_reason) -> bool {
    CHECK_NOTNULL(finalize_reason);
    if (cov.rows() < 6 || cov.cols() < 6) {
      *finalize_reason = "covariance_not_6x6";
      return false;
    }
    const gtsam::Matrix66 pose_cov = cov.topLeftCorner<6, 6>();
    if (!pose_cov.allFinite()) {
      *finalize_reason = "covariance_non_finite";
      return false;
    }
    *cov_out = 0.5 * (pose_cov + pose_cov.transpose());
    constexpr double kMinVar = 1e-8;
    for (int i = 0; i < 6; ++i) {
      if (!std::isfinite((*cov_out)(i, i)) || (*cov_out)(i, i) < kMinVar) {
        (*cov_out)(i, i) = kMinVar;
      }
    }
    *finalize_reason = "none";
    return true;
  };
  const auto query_covariance = [&](const gtsam::NonlinearFactorGraph& graph,
                                    const gtsam::Values& values,
                                    const gtsam::Pose3& anchor_pose,
                                    const std::string& raw_source_path,
                                    const std::string& anchored_source_path,
                                    std::string* query_reason) -> bool {
    CHECK_NOTNULL(query_reason);
    *query_reason = "none";

    if (use_anchored_covariance) {
      try {
        gtsam::NonlinearFactorGraph anchored_graph = graph;
        gtsam::Vector6 anchor_var;
        anchor_var << anchor_rot_var, anchor_rot_var, anchor_rot_var,
            anchor_trans_var, anchor_trans_var, anchor_trans_var;
        const auto anchor_noise =
            gtsam::noiseModel::Diagonal::Variances(anchor_var);
        anchored_graph.emplace_shared<gtsam::PriorFactor<gtsam::Pose3>>(
            pose_symbol, anchor_pose, anchor_noise);
        const gtsam::Marginals anchored_marginals(anchored_graph, values);
        const gtsam::Matrix anchored_cov =
            anchored_marginals.marginalCovariance(pose_symbol);
        std::string finalize_reason;
        if (finalize_covariance(anchored_cov, &finalize_reason)) {
          set_source_path(anchored_source_path);
          return true;
        }
        *query_reason = "anchored_covariance_invalid:" + finalize_reason;
      } catch (const std::exception& e) {
        *query_reason = std::string("anchored_covariance_exception:") + e.what();
      } catch (...) {
        *query_reason = "anchored_covariance_unknown_exception";
      }
    }

    try {
      const gtsam::Marginals raw_marginals(graph, values);
      const gtsam::Matrix raw_cov = raw_marginals.marginalCovariance(pose_symbol);
      std::string finalize_reason;
      if (finalize_covariance(raw_cov, &finalize_reason)) {
        set_source_path(raw_source_path);
        return true;
      }
      if (use_anchored_covariance && *query_reason != "none") {
        *query_reason += "|raw_covariance_invalid:" + finalize_reason;
      } else {
        *query_reason = "raw_covariance_invalid:" + finalize_reason;
      }
      return false;
    } catch (const std::exception& e) {
      if (use_anchored_covariance && *query_reason != "none") {
        *query_reason += "|raw_covariance_exception:" + std::string(e.what());
      } else {
        *query_reason = std::string("raw_covariance_exception:") + e.what();
      }
      return false;
    } catch (...) {
      if (use_anchored_covariance && *query_reason != "none") {
        *query_reason += "|raw_covariance_unknown_exception";
      } else {
        *query_reason = "raw_covariance_unknown_exception";
      }
      return false;
    }
  };

  if (h2_local_graph_snapshot_.empty()) {
    set_reason("h2_local_snapshot_graph_empty");
    return false;
  }
  if (!h2_local_values_snapshot_.exists(pose_symbol)) {
    set_reason("missing_h2_local_snapshot_value_for_key:" +
               key_to_string(pose_symbol));
    return false;
  }

  try {
    *pose_out = h2_local_values_snapshot_.at<gtsam::Pose3>(pose_symbol);
  } catch (const std::exception& e) {
    set_reason(std::string("h2_local_snapshot_pose_cast_exception:") + e.what());
    return false;
  } catch (...) {
    set_reason("h2_local_snapshot_pose_cast_unknown_exception");
    return false;
  }

  std::string query_reason;
  if (query_covariance(h2_local_graph_snapshot_,
                       h2_local_values_snapshot_,
                       *pose_out,
                       "h2_local_snapshot_marginal",
                       "h2_local_snapshot_marginal_with_pose_anchor_prior",
                       &query_reason)) {
    set_reason("none");
    return true;
  }

  set_reason("h2_local_snapshot_covariance_query_failed:" + query_reason);
  return false;
}
#endif


// zy Step 7_b
// When CBS is driving estimates, outgoing belief should carry CBS covariance, not legacy smoother covariance.
// Fallbacks keep behavior robust when CBS marginals are temporarily unavailable.
bool VioBackend::getLatestExternalPoseBelief(
    ExternalPoseBelief* belief) const {
  CHECK_NOTNULL(belief);
  const auto outgoing_start = std::chrono::steady_clock::now();
  double outgoing_cov_query_ms = 0.0;
  double outgoing_diag_emit_ms = 0.0;

  if (backend_state_ == BackendState::Bootstrap) {
    return false;
  }

  belief->timestamp_kf_nsec_ = timestamp_lkf_;
  belief->frame_id_ = curr_kf_id_;
  belief->W_Pose_B_ = W_Pose_B_lkf_from_state_;

  bool covariance_set = false;
  bool outgoing_cov_fastpath_no_external_effect = false;
  bool cbs_last_epoch_no_external_effect = false;
  size_t cbs_last_epoch_beliefs_accepted = 0u;
  size_t cbs_last_epoch_priors_injected = 0u;
  const gtsam::Symbol pose_symbol(kPoseSymbolChar, curr_kf_id_);
  std::string outgoing_mean_source = "heart_state";
  std::string outgoing_cov_source = "unset";
  std::string h2_outgoing_cov_source_path = "unset";
  bool h2_sidecar_key_exists_for_outgoing = false;
  bool h2_cov_fallback_used = false;
  std::string h2_cov_fallback_reason = "none";
  const auto covariance_logdet = [](const gtsam::Matrix66& cov) -> double {
    Eigen::LLT<gtsam::Matrix66> llt(cov);
    if (llt.info() != Eigen::Success) {
      return std::numeric_limits<double>::quiet_NaN();
    }
    const auto& L = llt.matrixL();
    double sum_log_diag = 0.0;
    for (int i = 0; i < L.rows(); ++i) {
      const double diag = L(i, i);
      if (!(diag > 0.0) || !std::isfinite(diag)) {
        return std::numeric_limits<double>::quiet_NaN();
      }
      sum_log_diag += std::log(diag);
    }
    return 2.0 * sum_log_diag;
  };
  const auto covariance_lambda_min = [](const gtsam::Matrix66& cov) -> double {
    Eigen::SelfAdjointEigenSolver<gtsam::Matrix66> eig(cov);
    if (eig.info() != Eigen::Success) {
      return std::numeric_limits<double>::quiet_NaN();
    }
    return eig.eigenvalues().minCoeff();
  };
  auto emit_outgoing_cov_diag =
      [&](const gtsam::Matrix66& local_cov,
          const std::optional<gtsam::Matrix66>& fused_cov_opt,
          const char* source_path) {
        const double local_trace = local_cov.trace();
        const double local_logdet = covariance_logdet(local_cov);
        const double local_lambda_min = covariance_lambda_min(local_cov);

        double fused_trace = std::numeric_limits<double>::quiet_NaN();
        double fused_logdet = std::numeric_limits<double>::quiet_NaN();
        double fused_lambda_min = std::numeric_limits<double>::quiet_NaN();
        double trace_ratio = std::numeric_limits<double>::quiet_NaN();
        double logdet_delta = std::numeric_limits<double>::quiet_NaN();
        double lambda_min_ratio = std::numeric_limits<double>::quiet_NaN();
        int fused_available = 0;

        if (fused_cov_opt) {
          fused_available = 1;
          const gtsam::Matrix66& fused_cov = *fused_cov_opt;
          fused_trace = fused_cov.trace();
          fused_logdet = covariance_logdet(fused_cov);
          fused_lambda_min = covariance_lambda_min(fused_cov);
          if (std::isfinite(local_trace) && local_trace > 0.0 &&
              std::isfinite(fused_trace)) {
            trace_ratio = fused_trace / local_trace;
          }
          if (std::isfinite(local_logdet) && std::isfinite(fused_logdet)) {
            logdet_delta = fused_logdet - local_logdet;
          }
          if (std::isfinite(local_lambda_min) && local_lambda_min > 0.0 &&
              std::isfinite(fused_lambda_min)) {
            lambda_min_ratio = fused_lambda_min / local_lambda_min;
          }
        }

        std::cerr << std::setprecision(12)
                  << "[CBS][OutgoingBeliefCov] key=" << pose_symbol.key()
                  << " frame_id=" << curr_kf_id_
                  << " timestamp_ns=" << timestamp_lkf_
                  << " local_only_trace=" << local_trace
                  << " local_only_logdet=" << local_logdet
                  << " local_only_lambda_min=" << local_lambda_min
                  << " fused_posterior_available=" << fused_available
                  << " fused_posterior_trace=" << fused_trace
                  << " fused_posterior_logdet=" << fused_logdet
                  << " fused_posterior_lambda_min=" << fused_lambda_min
                  << " local_to_fused_trace_ratio=" << trace_ratio
                  << " fused_minus_local_logdet_delta=" << logdet_delta
                  << " fused_to_local_lambda_min_ratio=" << lambda_min_ratio
                  // Backward-compatible aliases:
                  << " local_trace=" << local_trace
                  << " local_logdet=" << local_logdet
                  << " local_lambda_min=" << local_lambda_min
                  << " fused_available=" << fused_available
                  << " fused_trace=" << fused_trace
                  << " fused_logdet=" << fused_logdet
                  << " fused_lambda_min=" << fused_lambda_min
                  << " trace_ratio=" << trace_ratio
                  << " logdet_delta=" << logdet_delta
                  << " lambda_min_ratio=" << lambda_min_ratio
                  << " source_path=" << source_path
                  << std::endl;
      };

#ifdef KIMERA_USE_CBS
  {
    std::lock_guard<std::mutex> effect_lock(cbs_external_effect_state_mutex_);
    cbs_last_epoch_no_external_effect = cbs_last_epoch_no_external_effect_;
    cbs_last_epoch_beliefs_accepted = cbs_last_epoch_beliefs_accepted_;
    cbs_last_epoch_priors_injected = cbs_last_epoch_priors_injected_;
  }

  if (useCbsH2LocalCovSidecar() && !useCbsOptimizerHeart() && !covariance_set) {
    gtsam::Pose3 local_pose = belief->W_Pose_B_;
    gtsam::Matrix66 local_cov = gtsam::Matrix66::Identity();
    std::string local_reason;
    std::string local_source_path;
    const auto cov_query_start = std::chrono::steady_clock::now();
    const bool local_ok = queryH2LocalPoseCovFromActiveSmoother(
        pose_symbol, &local_pose, &local_cov, &local_reason, &local_source_path);
    outgoing_cov_query_ms +=
        elapsedMs(cov_query_start, std::chrono::steady_clock::now());
    if (local_ok) {
      h2_sidecar_key_exists_for_outgoing = true;
      belief->W_Pose_B_ = local_pose;
      belief->covariance_ = local_cov;
      covariance_set = true;
      outgoing_mean_source = "h2_local_snapshot_graph";
      outgoing_cov_source = "h2_sidecar_local_only";
      h2_outgoing_cov_source_path =
          local_source_path.empty() ? "h2_sidecar_local_only" : local_source_path;
      const auto emit_diag_start = std::chrono::steady_clock::now();
      emit_outgoing_cov_diag(
          belief->covariance_, std::nullopt, h2_outgoing_cov_source_path.c_str());
      outgoing_diag_emit_ms +=
          elapsedMs(emit_diag_start, std::chrono::steady_clock::now());
    } else {
      h2_cov_fallback_reason = local_reason.empty()
                                   ? "h2_local_snapshot_query_failed"
                                   : local_reason;
    }
  }

  // In CBS mode, export covariance from BPSAM marginals so shared beliefs reflect the CBS state.
  if (useCbsOptimizerHeart()) {
    CHECK(cbs_optimizer_) << "CBS optimizer flag is ON but cbs_optimizer_ is null.";
    if (FLAGS_cbs_outgoing_cov_fastpath_when_no_external_effect &&
        cbs_last_epoch_no_external_effect &&
        state_covariance_lkf_valid_ && state_covariance_lkf_.rows() >= 6 &&
        state_covariance_lkf_.cols() >= 6) {
      const gtsam::Matrix66 fallback_pose_cov =
          state_covariance_lkf_.topLeftCorner<6, 6>();
      if (fallback_pose_cov.allFinite()) {
        belief->covariance_ = fallback_pose_cov;
        covariance_set = true;
        outgoing_cov_source = "state_covariance_fastpath_no_external_effect";
        outgoing_cov_fastpath_no_external_effect = true;
        const auto emit_diag_start = std::chrono::steady_clock::now();
        emit_outgoing_cov_diag(
            belief->covariance_,
            std::nullopt,
            "state_covariance_fastpath_no_external_effect");
        outgoing_diag_emit_ms +=
            elapsedMs(emit_diag_start, std::chrono::steady_clock::now());
      }
    }

    if (!covariance_set) {
      try {
        if (cbs_optimizer_->valueExists(pose_symbol)) {
          const auto cov_query_start = std::chrono::steady_clock::now();
          // Match CBS pose-sharing stage: export pose covariance from LOCAL
          // marginalization (exclude belief factors from covariance computation).
          const gtsam::Matrix cov = cbs_optimizer_->marginalCovariance(
              pose_symbol, cbs::BPSAM::MarginalizationType::LOCAL);
          if (cov.rows() >= 6 && cov.cols() >= 6 && cov.allFinite()) {
            belief->covariance_ = cov.topLeftCorner(6, 6);
            covariance_set = true;
            outgoing_cov_source =
                FLAGS_cbs_outgoing_query_full_cov_for_diag
                    ? "cbs_local_marginal_with_full_diag"
                    : "cbs_local_marginal";
            std::optional<gtsam::Matrix66> fused_cov_opt = std::nullopt;
            if (FLAGS_cbs_outgoing_query_full_cov_for_diag) {
              try {
                const gtsam::Matrix full_cov = cbs_optimizer_->marginalCovariance(
                    pose_symbol, cbs::BPSAM::MarginalizationType::FULL);
                if (full_cov.rows() >= 6 && full_cov.cols() >= 6 &&
                    full_cov.allFinite()) {
                  fused_cov_opt = full_cov.topLeftCorner(6, 6);
                }
              } catch (const std::exception& e) {
                VLOG(2) << "CBS fused covariance query failed: " << e.what();
              }
            }
            outgoing_cov_query_ms +=
                elapsedMs(cov_query_start, std::chrono::steady_clock::now());
            const auto emit_diag_start = std::chrono::steady_clock::now();
            emit_outgoing_cov_diag(
                belief->covariance_,
                fused_cov_opt,
                FLAGS_cbs_outgoing_query_full_cov_for_diag
                    ? "cbs_local_and_full_marginal"
                    : "cbs_local_marginal");
            outgoing_diag_emit_ms +=
                elapsedMs(emit_diag_start, std::chrono::steady_clock::now());
          } else {
            VLOG(2) << "CBS covariance unavailable/invalid for pose key: "
                    << pose_symbol;
            outgoing_cov_query_ms +=
                elapsedMs(cov_query_start, std::chrono::steady_clock::now());
          }
        } else {
          VLOG(2) << "CBS value not found for pose key: " << pose_symbol;
        }
      } catch (const std::exception& e) {
        VLOG(2) << "CBS marginal covariance query failed: " << e.what();
      }
    }
  }
#endif

  // Optional safe path: only use backend covariance when it has been explicitly
  // computed and validated.
  if (!covariance_set) {
    h2_cov_fallback_used = useCbsH2LocalCovSidecar() && !useCbsOptimizerHeart();
    if (h2_cov_fallback_used && h2_cov_fallback_reason == "none") {
      h2_cov_fallback_reason = "legacy_covariance_fallback_path";
    }
    if (FLAGS_external_pose_belief_safe_covariance_fallback) {
      if (state_covariance_lkf_valid_ && state_covariance_lkf_.rows() >= 6 &&
          state_covariance_lkf_.cols() >= 6) {
        const gtsam::Matrix66 pose_cov =
            state_covariance_lkf_.topLeftCorner<6, 6>();
        if (pose_cov.allFinite()) {
          belief->covariance_ = pose_cov;
          covariance_set = true;
          outgoing_cov_source = "state_covariance_safe_fallback";
        }
      }
    } else {
      // Legacy path (preserve existing behavior outside explicitly tuned runs).
      if (state_covariance_lkf_.rows() >= 6 && state_covariance_lkf_.cols() >= 6) {
        belief->covariance_ = state_covariance_lkf_.topLeftCorner<6, 6>();
        covariance_set = true;
        outgoing_cov_source = "state_covariance_legacy";
      }
    }
  }

  // Fallback when covariance is unavailable.
  if (!covariance_set) {
    h2_cov_fallback_used = useCbsH2LocalCovSidecar() && !useCbsOptimizerHeart();
    if (h2_cov_fallback_used && h2_cov_fallback_reason == "none") {
      h2_cov_fallback_reason = "identity_or_conservative_covariance_fallback";
    }
    if (FLAGS_external_pose_belief_safe_covariance_fallback) {
      // Conservative fallback for explicitly enabled datasets/profiles.
      belief->covariance_.setZero();
      constexpr double kFallbackRotVar = 5e-2;    // rad^2
      constexpr double kFallbackTransVar = 5e-1;  // m^2
      belief->covariance_.topLeftCorner<3, 3>().diagonal().setConstant(
          kFallbackRotVar);
      belief->covariance_.bottomRightCorner<3, 3>().diagonal().setConstant(
          kFallbackTransVar);
      VLOG(1) << "State covariance unavailable, using conservative fallback "
                 "pose covariance (rot_var="
              << kFallbackRotVar << ", trans_var=" << kFallbackTransVar
              << ").";
    } else {
      // Legacy fallback.
      belief->covariance_.setIdentity();
      belief->covariance_.topLeftCorner<3, 3>() *= 1e-2;      // rot
      belief->covariance_.bottomRightCorner<3, 3>() *= 1e-2;  // trans
      VLOG(2) << "State covariance unavailable, using fallback identity covariance.";
    }
    if (outgoing_cov_source == "unset") {
      outgoing_cov_source = FLAGS_external_pose_belief_safe_covariance_fallback
                                ? "conservative_fallback_covariance"
                                : "identity_fallback_covariance";
    }
  }

  if (!belief->covariance_.allFinite()) {
    LOG(WARNING) << "Latest external pose belief covariance is non-finite.";
    return false;
  }

  // Symmetrize + clamp tiny/negative variances to keep the transmitted Gaussian numerically stable.
  belief->covariance_ =
      0.5 * (belief->covariance_ + belief->covariance_.transpose());

  constexpr double kMinVar = 1e-8;
  for (int i = 0; i < 6; ++i) {
    if (!std::isfinite(belief->covariance_(i, i)) ||
        belief->covariance_(i, i) < kMinVar) {
      belief->covariance_(i, i) = kMinVar;
    }
  }

#ifdef KIMERA_USE_CBS
  if (useCbsH2LocalCovSidecar() && !useCbsOptimizerHeart() &&
      h2_cov_fallback_used) {
    ++cbs_h2_sidecar_fallback_cov_epochs_;
  }
#endif

  std::cerr << std::setprecision(12)
            << "[CBS][OutgoingBeliefTiming] timestamp_ns=" << timestamp_lkf_
            << " frame_id=" << curr_kf_id_
            << " outgoing_total_ms="
            << elapsedMs(outgoing_start, std::chrono::steady_clock::now())
            << " outgoing_cov_query_ms=" << outgoing_cov_query_ms
            << " outgoing_diag_emit_ms=" << outgoing_diag_emit_ms
            << " outgoing_cov_fastpath_no_external_effect="
            << (outgoing_cov_fastpath_no_external_effect ? 1 : 0)
            << " last_epoch_beliefs_accepted="
            << cbs_last_epoch_beliefs_accepted
            << " last_epoch_priors_injected="
            << cbs_last_epoch_priors_injected
            << " cbs_heart_active=" << (useCbsOptimizerHeart() ? 1 : 0)
            << " outgoing_mean_source=" << outgoing_mean_source
            << " outgoing_cov_source=" << outgoing_cov_source
#ifdef KIMERA_USE_CBS
            << " h2_mode=" << (useCbsH2LocalCovSidecar() ? 1 : 0)
            << " h2_sidecar_sync_ok=" << (cbs_h2_sidecar_sync_ok_ ? 1 : 0)
            << " h2_sidecar_key_exists_for_outgoing="
            << (h2_sidecar_key_exists_for_outgoing ? 1 : 0)
            << " h2_cov_source_path=" << h2_outgoing_cov_source_path
            << " h2_cov_fallback_used=" << (h2_cov_fallback_used ? 1 : 0)
            << " h2_cov_fallback_reason=" << h2_cov_fallback_reason
            << " h2_sidecar_desync_streak=" << cbs_h2_sidecar_desync_streak_
            << " h2_sidecar_fallback_cov_epochs="
            << cbs_h2_sidecar_fallback_cov_epochs_
#endif
            << std::endl;

  return true;
}

// zy Step 14b
// Intuition: query a timestamp-aligned pose belief (mean + covariance), using CBS marginals when enabled and safe fallback behavior otherwise.
bool VioBackend::getExternalPoseBeliefAtTimestamp(
    const Timestamp& query_timestamp_kf_nsec,
    ExternalPoseBelief* belief,
    const Timestamp& tolerance_ns) const {
  CHECK_NOTNULL(belief);

  if (backend_state_ == BackendState::Bootstrap) {
    return false;
  }

  const Timestamp effective_tolerance_ns =
      (tolerance_ns >= 0) ? tolerance_ns : external_prior_timestamp_tolerance_ns_;

  auto absDiffNs = [](Timestamp a, Timestamp b) -> Timestamp {
    return (a >= b) ? (a - b) : (b - a);
  };

  bool matched = false;
  Timestamp matched_timestamp = -1;
  FrameId matched_frame_id = -1;

  {
    std::lock_guard<std::mutex> map_lock(timestamp_to_kf_id_map_mutex_);
    if (timestamp_to_kf_id_map_.empty()) {
      return false;
    }

    auto it = timestamp_to_kf_id_map_.lower_bound(query_timestamp_kf_nsec);
    std::map<Timestamp, FrameId>::const_iterator best_it =
        timestamp_to_kf_id_map_.end();
    Timestamp best_dt = std::numeric_limits<Timestamp>::max();

    if (it != timestamp_to_kf_id_map_.end()) {
      best_it = it;
      best_dt = absDiffNs(it->first, query_timestamp_kf_nsec);
    }
    if (it != timestamp_to_kf_id_map_.begin()) {
      auto prev_it = std::prev(it);
      const Timestamp dt = absDiffNs(prev_it->first, query_timestamp_kf_nsec);
      if (dt < best_dt) {
        best_it = prev_it;
        best_dt = dt;
      }
    }

    if (best_it != timestamp_to_kf_id_map_.end() &&
        best_dt <= effective_tolerance_ns) {
      matched = true;
      matched_timestamp = best_it->first;
      matched_frame_id = best_it->second;
    }
  }

  if (!matched) {
    VLOG(2) << "No timestamp match for external belief query. query_ts="
            << query_timestamp_kf_nsec << ", tol_ns=" << effective_tolerance_ns;
    return false;
  }

  const gtsam::Symbol pose_symbol(kPoseSymbolChar, matched_frame_id);

#ifdef KIMERA_USE_CBS
  if (useCbsH2LocalCovSidecar() && !useCbsOptimizerHeart()) {
    gtsam::Pose3 local_pose;
    gtsam::Matrix66 local_cov = gtsam::Matrix66::Identity();
    std::string local_reason;
    std::string local_source_path;
    if (!queryH2LocalPoseCovFromActiveSmoother(
            pose_symbol, &local_pose, &local_cov, &local_reason,
            &local_source_path)) {
      VLOG(2) << "H2 local snapshot covariance query failed for key "
              << pose_symbol << ": "
              << (local_reason.empty() ? "unknown_reason" : local_reason);
      return false;
    }
    VLOG(3) << "H2 timestamp belief covariance source for key " << pose_symbol
            << ": "
            << (local_source_path.empty() ? "h2_sidecar_local_only"
                                          : local_source_path);
    belief->timestamp_kf_nsec_ = matched_timestamp;
    belief->frame_id_ = matched_frame_id;
    belief->W_Pose_B_ = local_pose;
    belief->covariance_ = local_cov;
    belief->covariance_ =
        0.5 * (belief->covariance_ + belief->covariance_.transpose());
    constexpr double kMinVar = 1e-8;
    for (int i = 0; i < 6; ++i) {
      if (!std::isfinite(belief->covariance_(i, i)) ||
          belief->covariance_(i, i) < kMinVar) {
        belief->covariance_(i, i) = kMinVar;
      }
    }
    return true;
  }

  // Intuition: in CBS mode, query arbitrary historical pose beliefs directly from BPSAM marginals.
  if (useCbsOptimizerHeart()) {
    CHECK(cbs_optimizer_) << "CBS optimizer flag is ON but cbs_optimizer_ is null.";

    if (!cbs_optimizer_->valueExists(pose_symbol)) {
      VLOG(2) << "CBS does not contain requested key: " << pose_symbol;
      return false;
    }

    try {
      belief->timestamp_kf_nsec_ = matched_timestamp;
      belief->frame_id_ = matched_frame_id;
      belief->W_Pose_B_ =
          cbs_optimizer_->calculateEstimate<gtsam::Pose3>(pose_symbol);

      const gtsam::Matrix cov = cbs_optimizer_->marginalCovariance(
          pose_symbol, cbs::BPSAM::MarginalizationType::LOCAL);
      if (cov.rows() < 6 || cov.cols() < 6 || !cov.allFinite()) {
        VLOG(2) << "Invalid CBS covariance for key: " << pose_symbol;
        return false;
      }

      belief->covariance_ = cov.topLeftCorner(6, 6);
      belief->covariance_ =
          0.5 * (belief->covariance_ + belief->covariance_.transpose());

      constexpr double kMinVar = 1e-8;
      for (int i = 0; i < 6; ++i) {
        if (!std::isfinite(belief->covariance_(i, i)) ||
            belief->covariance_(i, i) < kMinVar) {
            belief->covariance_(i, i) = kMinVar;
        }
      }
      return true;
    } catch (const std::exception& e) {
      VLOG(2) << "CBS timestamp belief query failed: " << e.what();
      return false;
    }
  }
#endif

  // zy 19b keep key-type comparison explicit and warning-free across platforms.
  if (matched_frame_id == static_cast<FrameId>(curr_kf_id_)) {
    return getLatestExternalPoseBelief(belief);
  }

  VLOG(2) << "Timestamp belief query for historical frame requires CBS mode. "
          << "matched_frame_id=" << matched_frame_id
          << ", curr_kf_id=" << curr_kf_id_;
  return false;
}



/* -------------------------------------------------------------------------- */
// TODO this function doesn't do just one thing... Should be refactored!
// It returns the landmark ids of the stereo measurements
// It also updates the feature tracks. Why is this in the Backend???
// TODO(Toni): the FeatureTracks can be fully replaced by the StereoMeasurements
// class...
void VioBackend::addStereoMeasurementsToFeatureTracks(
    const int& frame_num,
    const StereoMeasurements& stereo_meas_kf,
    LandmarkIds* landmarks_kf) {
  CHECK_NOTNULL(landmarks_kf);

  // TODO: feature tracks will grow unbounded.

  // Make sure the landmarks_kf vector is empty and has a suitable size.
  const size_t& n_stereo_measurements = stereo_meas_kf.size();
  landmarks_kf->resize(n_stereo_measurements);

  // Store landmark ids.
  // TODO(Toni): the concept of feature tracks should not be in the Backend...
  for (size_t i = 0u; i < n_stereo_measurements; ++i) {
    const LandmarkId& lmk_id_in_kf_i = stereo_meas_kf[i].first;
    const StereoPoint2& stereo_px_i = stereo_meas_kf[i].second;

    // We filtered invalid lmks in the StereoTracker, so this should not happen.
    CHECK_NE(lmk_id_in_kf_i, -1) << "landmarkId_kf_i == -1?";

    // Thinner structure that only keeps landmarkIds.
    // These landmark ids are only the ones visible in current keyframe,
    // with a valid track...
    // CHECK that we do not have repeated lmk ids!
    DCHECK(std::find(landmarks_kf->begin(),
                     landmarks_kf->end(),
                     lmk_id_in_kf_i) == landmarks_kf->end());
    (*landmarks_kf)[i] = lmk_id_in_kf_i;

    // Add features to vio->featureTracks_ if they are new.
    const FeatureTracks::iterator& feature_track_it =
        feature_tracks_.find(lmk_id_in_kf_i);
    if (feature_track_it == feature_tracks_.end()) {
      // New feature.
      VLOG(20) << "Creating new feature track for lmk: " << lmk_id_in_kf_i
               << '.';
      feature_tracks_.insert(
          std::make_pair(lmk_id_in_kf_i, FeatureTrack(frame_num, stereo_px_i)));
      ++landmark_count_;
    } else {
      // @TODO: It seems that this else condition does not help --
      // conjecture that it creates long feature tracks with low information
      // (i.e. we're not moving)
      // This is problematic in conjunction with our landmark selection
      // mechanism which prioritizes long feature tracks

      // TODO: to avoid making the feature tracks grow unbounded we could
      // use a tmp feature tracks container to which we would add the old
      // feature track plus the new observation on it. (for new tracks, it
      // would be the same as above, using the tmp structure of course).

      // Add observation to existing landmark.
      VLOG(20) << "Updating feature track for lmk: " << lmk_id_in_kf_i << ".";
      feature_track_it->second.obs_.push_back(
          std::make_pair(frame_num, stereo_px_i));

      // TODO(Toni):
      // Mark feature tracks that have been re-observed, so that we can delete
      // the broken feature tracks efficiently.
    }
  }
}

/// Value adders.
/* -------------------------------------------------------------------------- */
void VioBackend::addStateValues(const FrameId& frame_id,
                                const TrackerStatusSummary& tracker_status,
                                const gtsam::PreintegrationType& pim,
                                std::optional<gtsam::Pose3> odom_pose,
                                std::optional<gtsam::Vector3> odom_vel) {
  // NOTE: we use the latest state instead of W_Pose_B_lkf_from_increments_
  // because that one is generated by chaining relative poses from the
  // optimization, and might be far from the state estimate of the VIO.
  // Initializing the smoother_ optimization with W_Pose_B_lkf_from_increments_
  // would cause crashes because it's different from the latest state in
  // smoother_.
  gtsam::NavState navstate_lkf(W_Pose_B_lkf_from_state_, W_Vel_B_lkf_);
  const gtsam::NavState& navstate_k = pim.predict(navstate_lkf, imu_bias_lkf_);
  debug_info_.navstate_k_ = navstate_k;

  switch (backend_params_.pose_guess_source_) {
    case PoseGuessSource::IMU: {
      addStateValuesFromNavState(frame_id, navstate_k);
      break;
    }
    case PoseGuessSource::MONO: {
      if (tracker_status.kfTrackingStatus_mono_ == TrackingStatus::VALID) {
        gtsam::Pose3 W_Pose_B_k_mono =
            W_Pose_B_lkf_from_state_ * B_Pose_leftCamRect_ *
            tracker_status.lkf_T_k_mono_ * B_Pose_leftCamRect_.inverse();
        gtsam::Point3 W_ScaledTranslation_B_k_mono =
            W_Pose_B_k_mono.translation() *
            backend_params_.mono_translation_scale_factor_;
        addStateValues(frame_id,
                       gtsam::Pose3(W_Pose_B_k_mono.rotation(),
                                    W_ScaledTranslation_B_k_mono),
                       navstate_k.velocity(),
                       imu_bias_lkf_);
      } else {
        LOG(WARNING) << "Mono tracking failure... Using IMU for pose guess.";
        addStateValuesFromNavState(frame_id, navstate_k);
      }
      break;
    }
    case PoseGuessSource::STEREO: {
      if (tracker_status.kfTrackingStatus_stereo_ == TrackingStatus::VALID) {
        addStateValues(frame_id,
                       W_Pose_B_lkf_from_state_ * B_Pose_leftCamRect_ *
                           tracker_status.lkf_T_k_stereo_ *
                           B_Pose_leftCamRect_.inverse(),
                       navstate_k.velocity(),
                       imu_bias_lkf_);
      } else {
        LOG(WARNING) << "Stereo tracking failure... Using IMU for pose guess.";
        addStateValuesFromNavState(frame_id, navstate_k);
      }
      break;
    }
    case PoseGuessSource::PNP: {
      if (tracker_status.kfTracking_status_pnp_ == TrackingStatus::VALID) {
        addStateValues(
            frame_id,
            tracker_status.W_T_k_pnp_ * B_Pose_leftCamRect_.inverse(),
            navstate_k.velocity(),
            imu_bias_lkf_);
      } else {
        LOG(WARNING) << "PnP tracking failure... Using IMU for pose guess.";
        addStateValuesFromNavState(frame_id, navstate_k);
      }
      break;
    }
    case PoseGuessSource::EXTERNAL_ODOM: {
      if (odom_pose) {
        // odom_pose is relative (body_lkf_odomPose_body_kf)
        gtsam::Pose3 W_Pose_B_odom =
            W_Pose_B_lkf_from_state_ * odom_pose.value();
        if (odom_vel && odom_params_->velocityPrecision_ > 0.0) {
          LOG(ERROR) << "Using external odometry velocity is not "
                        "recommended! Set odomVelPrecision = 0. Ignore this "
                        "only after serious consideration.";
          addStateValues(
              frame_id, W_Pose_B_odom, odom_vel.value(), imu_bias_lkf_);
        } else {
          addStateValues(
              frame_id, W_Pose_B_odom, navstate_k.velocity(), imu_bias_lkf_);
        }
      } else {
        LOG(WARNING) << "External odometry tracking failure (no odom pose "
                        "provided)... Using IMU for pose guess.";
        addStateValuesFromNavState(frame_id, navstate_k);
      }
      break;
    }
    default: {
      LOG(FATAL) << "Unrecognized Initial Pose Guess source: "
                 << VIO::to_underlying(backend_params_.pose_guess_source_);
      break;
    }
  }
}

void VioBackend::addStateValuesFromNavState(const FrameId& frame_id,
                                            const gtsam::NavState& nav_state) {
  addStateValues(
      frame_id, nav_state.pose(), nav_state.velocity(), imu_bias_lkf_);
}

void VioBackend::addStateValues(const FrameId& cur_id,
                                const gtsam::Pose3& pose,
                                const gtsam::Velocity3& velocity,
                                const ImuBias& imu_bias) {
  new_values_.insert(gtsam::Symbol(kPoseSymbolChar, cur_id), pose);
  new_values_.insert(gtsam::Symbol(kVelocitySymbolChar, cur_id), velocity);
  new_values_.insert(gtsam::Symbol(kImuBiasSymbolChar, cur_id), imu_bias);
}

/// Factor adders.
/* -------------------------------------------------------------------------- */
void VioBackend::addImuFactor(const FrameId& from_id,
                              const FrameId& to_id,
                              const gtsam::PreintegrationType& pim) {
  switch (imu_params_.imu_preintegration_type_) {
    case ImuPreintegrationType::kPreintegratedCombinedMeasurements: {
      new_imu_prior_and_other_factors_.emplace_shared<gtsam::CombinedImuFactor>(
          gtsam::Symbol(kPoseSymbolChar, from_id),
          gtsam::Symbol(kVelocitySymbolChar, from_id),
          gtsam::Symbol(kPoseSymbolChar, to_id),
          gtsam::Symbol(kVelocitySymbolChar, to_id),
          gtsam::Symbol(kImuBiasSymbolChar, from_id),
          gtsam::Symbol(kImuBiasSymbolChar, to_id),
          safeCastToPreintegratedCombinedImuMeasurements(pim));
      break;
    }
    case ImuPreintegrationType::kPreintegratedImuMeasurements: {
      new_imu_prior_and_other_factors_.emplace_shared<gtsam::ImuFactor>(
          gtsam::Symbol(kPoseSymbolChar, from_id),
          gtsam::Symbol(kVelocitySymbolChar, from_id),
          gtsam::Symbol(kPoseSymbolChar, to_id),
          gtsam::Symbol(kVelocitySymbolChar, to_id),
          gtsam::Symbol(kImuBiasSymbolChar, from_id),
          safeCastToPreintegratedImuMeasurements(pim));

      static const gtsam::imuBias::ConstantBias zero_bias(
          gtsam::Vector3(0.0, 0.0, 0.0), gtsam::Vector3(0.0, 0.0, 0.0));

      // Factor to discretize and move normalize by the interval between
      // measurements:
      CHECK_NE(imu_params_.nominal_sampling_time_s_, 0.0)
          << "Nominal IMU sampling time cannot be 0 s.";
      // See Trawny05 http://mars.cs.umn.edu/tr/reports/Trawny05b.pdf
      // Eq. 130
      const double& sqrt_delta_t_ij = std::sqrt(pim.deltaTij());
      gtsam::Vector6 bias_sigmas;
      bias_sigmas.head<3>().setConstant(sqrt_delta_t_ij *
                                        imu_params_.acc_random_walk_);
      bias_sigmas.tail<3>().setConstant(sqrt_delta_t_ij *
                                        imu_params_.gyro_random_walk_);
      const gtsam::SharedNoiseModel& bias_noise_model =
          gtsam::noiseModel::Diagonal::Sigmas(bias_sigmas);

      new_imu_prior_and_other_factors_
          .emplace_shared<gtsam::BetweenFactor<gtsam::imuBias::ConstantBias>>(
              gtsam::Symbol(kImuBiasSymbolChar, from_id),
              gtsam::Symbol(kImuBiasSymbolChar, to_id),
              zero_bias,
              bias_noise_model);
      break;
    }
    default: {
      LOG(FATAL) << "Unknown IMU Preintegration Type.";
      break;
    }
  }

  debug_info_.imuR_lkf_kf = pim.deltaRij();
  debug_info_.numAddedImuF_++;
}

/* -------------------------------------------------------------------------- */
void VioBackend::addBetweenFactor(const FrameId& from_id,
                                  const FrameId& to_id,
                                  const gtsam::Pose3& from_id_POSE_to_id,
                                  const double& between_rotation_precision,
                                  const double& between_translation_precision) {
  // TODO(Toni): make noise models const members of Backend...
  Vector6 precisions;
  precisions.head<3>().setConstant(between_rotation_precision);
  precisions.tail<3>().setConstant(between_translation_precision);
  const gtsam::SharedNoiseModel& betweenNoise_ =
      gtsam::noiseModel::Diagonal::Precisions(precisions);

  new_imu_prior_and_other_factors_
      .emplace_shared<gtsam::BetweenFactor<gtsam::Pose3>>(
          gtsam::Symbol(kPoseSymbolChar, from_id),
          gtsam::Symbol(kPoseSymbolChar, to_id),
          from_id_POSE_to_id,
          betweenNoise_);

  debug_info_.numAddedBetweenStereoF_++;
}

/* -------------------------------------------------------------------------- */
// zy Step 1
void VioBackend::addExternalPosePrior(const FrameId& frame_id,
                                      const gtsam::Pose3& W_Pose_B,
                                      const gtsam::SharedNoiseModel& noise_model) {
  // Add a prior on the pose node x(frame_id).
#ifdef KIMERA_USE_CBS
  const auto external_prior_factor =
      boost::make_shared<ExternalPosePriorFactor>(
          gtsam::Symbol(kPoseSymbolChar, frame_id), W_Pose_B, noise_model);
#else
  const auto external_prior_factor =
      boost::make_shared<gtsam::PriorFactor<gtsam::Pose3>>(
          gtsam::Symbol(kPoseSymbolChar, frame_id), W_Pose_B, noise_model);
#endif
  // Keep external priors isolated from LOCAL factors so H2 sidecar packets can
  // be assembled by construction without pointer-based filtering.
  new_external_prior_factors_.push_back(external_prior_factor);
#ifdef KIMERA_USE_CBS
  cbs_external_prior_factor_ptrs_.insert(external_prior_factor.get());
#endif

  VLOG(1) << "Enqueued external pose prior on key: "
          << gtsam::Symbol(kPoseSymbolChar, frame_id)
          << " (frame_id=" << frame_id << ").";
}
/* -------------------------------------------------------------------------- */

/* -------------------------------------------------------------------------- */
// zy Step 2_d and 6_c
// this is only staging. No factor injection yet
void VioBackend::enqueueExternalPosePrior(
    const Timestamp& timestamp_kf_nsec,
    const gtsam::Pose3& W_Pose_B,
    const gtsam::SharedNoiseModel& noise_model,
    const std::string& source,
    uint64_t source_seq) {
  CHECK(noise_model) << "enqueueExternalPosePrior received null noise model.";

  std::lock_guard<std::mutex> lock(external_pose_priors_queue_mutex_);

  ExternalPosePrior prior;
  prior.timestamp_kf_nsec_ = timestamp_kf_nsec;
  prior.W_Pose_B_ = W_Pose_B;
  prior.noise_model_ = noise_model;
    
  prior.source_ = source;
  prior.source_seq_ = source_seq;

  external_pose_priors_queue_.push_back(std::move(prior));

  // Keep only a trailing timestamp window so queue growth is bounded even if
  // backend lags behind incoming external messages.
  if (FLAGS_external_prior_queue_time_horizon_ns > 0) {
    const Timestamp horizon_ns =
        static_cast<Timestamp>(FLAGS_external_prior_queue_time_horizon_ns);
    const Timestamp newest_ts = external_pose_priors_queue_.back().timestamp_kf_nsec_;
    size_t dropped_horizon = 0;
    while (!external_pose_priors_queue_.empty() &&
           external_pose_priors_queue_.front().timestamp_kf_nsec_ + horizon_ns <
               newest_ts) {
      external_pose_priors_queue_.pop_front();
      ++dropped_horizon;
    }
    if (dropped_horizon > 0) {
      LOG_EVERY_N(WARNING, 100)
          << "External pose prior queue horizon trim dropped "
          << dropped_horizon << " stale priors.";
    }
  }

  if (external_pose_priors_queue_.size() > max_external_pose_priors_queue_size_) {
    external_pose_priors_queue_.pop_front();
    LOG_EVERY_N(WARNING, 100)
        << "External pose prior queue overflow. Dropping oldest prior.";
  }

  VLOG(1) << "Queued external pose prior ts[nsec]=" << timestamp_kf_nsec
          << ", source=" << source
          << ", seq=" << source_seq
          << ", queue size=" << external_pose_priors_queue_.size();
}
/* -------------------------------------------------------------------------- */

/* -------------------------------------------------------------------------- */
// zy Step 5_b and 6_c
// bridge code can pass raw covariance matrix directly; backend owns conversion and validation.
bool VioBackend::enqueueExternalPosePriorFromCovariance(
    const Timestamp& timestamp_kf_nsec,
    const gtsam::Pose3& W_Pose_B,
    const gtsam::Matrix6& covariance,
    const std::string& source,
    uint64_t source_seq) {
  if (!useCbsBeliefExchange()) {
    LOG_EVERY_N(WARNING, 200)
        << "Dropping external pose prior because CBS belief exchange is OFF.";
    return false;
  }

  if (!covariance.allFinite()) {
    LOG(WARNING) << "enqueueExternalPosePriorFromCovariance: covariance has "
                    "non-finite entries, dropping prior.";
    return false;
  }

  gtsam::Matrix6 cov = covariance;
  std::string cov_reason = "unknown";
  const PoseCovarianceStatus cov_status =
      sanitizePoseCovariance(&cov, &cov_reason);
  if (cov_status == PoseCovarianceStatus::kRejected) {
    LOG(WARNING)
        << "enqueueExternalPosePriorFromCovariance: covariance rejected ("
        << cov_reason << "), dropping prior.";
    return false;
  }
  if (cov_status == PoseCovarianceStatus::kRegularized) {
    LOG_EVERY_N(WARNING, 50)
        << "enqueueExternalPosePriorFromCovariance: covariance regularized "
           "before noise-model creation. regularization_count="
        << google::COUNTER;
  }

  gtsam::SharedNoiseModel noise_model;
  try {
    noise_model = gtsam::noiseModel::Gaussian::Covariance(cov);
  } catch (const std::exception& e) {
    LOG(WARNING) << "enqueueExternalPosePriorFromCovariance: failed to build "
                    "Gaussian noise from covariance: "
                 << e.what();
    return false;
  }

enqueueExternalPosePrior(
  timestamp_kf_nsec, W_Pose_B, noise_model, source, source_seq);  
return true;}

/* -------------------------------------------------------------------------- */



void VioBackend::addNoMotionFactor(const FrameId& from_id,
                                   const FrameId& to_id) {
  new_imu_prior_and_other_factors_
      .emplace_shared<gtsam::BetweenFactor<gtsam::Pose3>>(
          gtsam::Symbol(kPoseSymbolChar, from_id),
          gtsam::Symbol(kPoseSymbolChar, to_id),
          gtsam::Pose3(),
          no_motion_prior_noise_);

  debug_info_.numAddedNoMotionF_++;

  VLOG(10) << "No motion detected, adding no relative motion prior";
}

/* -------------------------------------------------------------------------- */
void VioBackend::addZeroVelocityPrior(const FrameId& frame_id) {
  VLOG(10) << "No motion detected, adding zero velocity prior.";
  new_imu_prior_and_other_factors_
      .emplace_shared<gtsam::PriorFactor<gtsam::Vector3>>(
          gtsam::Symbol(kVelocitySymbolChar, frame_id),
          gtsam::Vector3::Zero(),
          zero_velocity_prior_noise_);
}

void VioBackend::addVelocityPrior(const FrameId& frame_id,
                                  const gtsam::Velocity3& vel,
                                  const double& precision) {
  VLOG(10) << "Adding odometry pose velocity prior factor.";
  gtsam::Vector3 precisions;
  precisions.head<3>().setConstant(precision);
  const gtsam::SharedNoiseModel& noise_model =
      gtsam::noiseModel::Diagonal::Precisions(precisions);
  new_imu_prior_and_other_factors_
      .emplace_shared<gtsam::PriorFactor<gtsam::Vector3>>(
          gtsam::Symbol(kVelocitySymbolChar, frame_id), vel, noise_model);
}

/* -------------------------------------------------------------------------- */
bool VioBackend::isPoseKeyActiveInOptimizer(
    const gtsam::Symbol& pose_symbol) const {
#ifdef KIMERA_USE_CBS
  if (useCbsOptimizerHeart()) {
    CHECK(cbs_optimizer_) << "CBS optimizer flag is ON but cbs_optimizer_ is null.";
    return cbs_optimizer_->valueExists(pose_symbol) ||
           new_values_.exists(pose_symbol);
  }
#endif
  return state_.exists(pose_symbol) || new_values_.exists(pose_symbol);
}

/* -------------------------------------------------------------------------- */
bool VioBackend::isReceiverLocalBeliefReadyForCbs(
    const gtsam::Symbol& pose_symbol) const {
#ifdef KIMERA_USE_CBS
  if (!useCbsOptimizerHeart()) {
    return true;
  }
  CHECK(cbs_optimizer_) << "CBS optimizer flag is ON but cbs_optimizer_ is null.";

  if (!cbs_optimizer_->valueExists(pose_symbol)) {
    return false;
  }

  try {
    const gtsam::Matrix cov = cbs_optimizer_->marginalCovariance(
        pose_symbol, cbs::BPSAM::MarginalizationType::LOCAL);
    return cov.rows() >= 6 && cov.cols() >= 6 &&
           cov.topLeftCorner(6, 6).allFinite();
  } catch (const std::exception&) {
    return false;
  } catch (...) {
    return false;
  }
#else
  (void)pose_symbol;
  return false;
#endif
}

/* -------------------------------------------------------------------------- */
void VioBackend::pruneTimestampToKeyframeMap(
    const bool cbs_heart_active,
    const FrameId& oldest_active_frame_id_by_lag,
    size_t* num_timestamp_map_pruned,
    Timestamp* oldest_active_pose_timestamp,
    Timestamp* newest_active_pose_timestamp) {
  CHECK_NOTNULL(num_timestamp_map_pruned);
  CHECK_NOTNULL(oldest_active_pose_timestamp);
  CHECK_NOTNULL(newest_active_pose_timestamp);
  *num_timestamp_map_pruned = 0u;
  *oldest_active_pose_timestamp = -1;
  *newest_active_pose_timestamp = -1;

  std::lock_guard<std::mutex> map_lock(timestamp_to_kf_id_map_mutex_);
  for (auto it = timestamp_to_kf_id_map_.begin();
       it != timestamp_to_kf_id_map_.end();) {
    if (cbs_heart_active &&
        static_cast<FrameId>(it->second) < oldest_active_frame_id_by_lag) {
      it = timestamp_to_kf_id_map_.erase(it);
      ++(*num_timestamp_map_pruned);
      continue;
    }

    const gtsam::Symbol pose_symbol(kPoseSymbolChar, it->second);
    if (isPoseKeyActiveInOptimizer(pose_symbol)) {
      ++it;
    } else {
      it = timestamp_to_kf_id_map_.erase(it);
      ++(*num_timestamp_map_pruned);
    }
  }

  if (!timestamp_to_kf_id_map_.empty()) {
    *oldest_active_pose_timestamp = timestamp_to_kf_id_map_.begin()->first;
    *newest_active_pose_timestamp = timestamp_to_kf_id_map_.rbegin()->first;
  }
}

/* -------------------------------------------------------------------------- */
FrameId VioBackend::computeCbsOldestActiveFrame(
    const FrameId& newest_frame_id) const {
  const FrameId lag_states = static_cast<FrameId>(backend_params_.nr_states_);
  if (lag_states == 0u || newest_frame_id + 1 <= lag_states) {
    return 0u;
  }
  return newest_frame_id - lag_states + 1;
}

#ifdef KIMERA_USE_CBS
/* -------------------------------------------------------------------------- */
VioBackend::CbsFixedLagWindowState VioBackend::buildCbsFixedLagWindowState(
    const std::map<Key, double>& timestamps) {
  CHECK(cbs_optimizer_) << "CBS optimizer flag is ON but cbs_optimizer_ is null.";

  CbsFixedLagWindowState window_state;
  window_state.newest_frame_id = curr_kf_id_;
  for (const auto& key_ts : timestamps) {
    window_state.newest_frame_id = std::max(
        window_state.newest_frame_id,
        static_cast<FrameId>(std::llround(key_ts.second)));
  }
  window_state.oldest_active_frame_id =
      computeCbsOldestActiveFrame(window_state.newest_frame_id);
  window_state.prev_oldest_active_frame_id =
      cbs_has_prev_oldest_active_frame_id_
          ? cbs_prev_oldest_active_frame_id_
          : window_state.oldest_active_frame_id;
  window_state.eviction_start_frame_id = 0;
  window_state.eviction_end_frame_id = window_state.oldest_active_frame_id;
  window_state.eviction_incremental = false;
  window_state.eviction_full_rescan = true;
  window_state.eviction_frames = 0u;

  if (backend_params_.nr_states_ <= 0) {
    cbs_prev_oldest_active_frame_id_ = window_state.oldest_active_frame_id;
    cbs_has_prev_oldest_active_frame_id_ = true;
    return window_state;
  }

  if (cbs_has_prev_oldest_active_frame_id_) {
    if (window_state.oldest_active_frame_id > cbs_prev_oldest_active_frame_id_) {
      window_state.eviction_start_frame_id = cbs_prev_oldest_active_frame_id_;
      window_state.eviction_end_frame_id = window_state.oldest_active_frame_id;
      window_state.eviction_incremental = true;
      window_state.eviction_full_rescan = false;
      window_state.eviction_frames =
          static_cast<size_t>(window_state.eviction_end_frame_id -
                              window_state.eviction_start_frame_id);
    } else if (window_state.oldest_active_frame_id ==
               cbs_prev_oldest_active_frame_id_) {
      window_state.eviction_start_frame_id = window_state.oldest_active_frame_id;
      window_state.eviction_end_frame_id = window_state.oldest_active_frame_id;
      window_state.eviction_incremental = true;
      window_state.eviction_full_rescan = false;
      window_state.eviction_frames = 0u;
    } else {
      // Window moved backward (restart/recovery): conservatively rescan stale tail.
      window_state.eviction_start_frame_id = 0;
      window_state.eviction_end_frame_id = window_state.oldest_active_frame_id;
      window_state.eviction_incremental = false;
      window_state.eviction_full_rescan = true;
      window_state.eviction_frames =
          static_cast<size_t>(window_state.eviction_end_frame_id);
    }
  } else {
    window_state.eviction_frames =
        static_cast<size_t>(window_state.eviction_end_frame_id);
  }

  std::unordered_set<gtsam::FactorIndex> lag_remove_slots_set;
  if (window_state.eviction_end_frame_id <= window_state.eviction_start_frame_id) {
    cbs_prev_oldest_active_frame_id_ = window_state.oldest_active_frame_id;
    cbs_has_prev_oldest_active_frame_id_ = true;
    return window_state;
  }

  const auto& variable_index = cbs_optimizer_->getVariableIndex();
  for (const auto& key_and_slots : variable_index) {
    const gtsam::Symbol key_symbol(key_and_slots.first);
    const char key_char = key_symbol.chr();
    if (key_char != kPoseSymbolChar && key_char != kVelocitySymbolChar &&
        key_char != kImuBiasSymbolChar) {
      continue;
    }
    const FrameId key_frame = static_cast<FrameId>(key_symbol.index());
    if (key_frame < window_state.eviction_start_frame_id ||
        key_frame >= window_state.eviction_end_frame_id) {
      continue;
    }

    ++window_state.stale_state_keys;
    if (key_char == kPoseSymbolChar) {
      ++window_state.stale_pose_keys;
    }
    lag_remove_slots_set.insert(
        key_and_slots.second.begin(), key_and_slots.second.end());
  }

  window_state.remove_factor_indices.assign(
      lag_remove_slots_set.begin(), lag_remove_slots_set.end());
  std::sort(window_state.remove_factor_indices.begin(),
            window_state.remove_factor_indices.end());
  cbs_prev_oldest_active_frame_id_ = window_state.oldest_active_frame_id;
  cbs_has_prev_oldest_active_frame_id_ = true;
  return window_state;
}

/* -------------------------------------------------------------------------- */
bool VioBackend::refreshH2LocalCovariancePassiveSnapshot(
    const H2LocalCovSidecarPacket& packet,
    const FrameId& curr_kf_id,
    H2LocalCovSidecarSyncStats* stats) {
  CHECK_NOTNULL(stats);
  stats->update_ms = 0.0;
  stats->snapshot_refresh_ms = 0.0;
  stats->local_factor_count = packet.local_factors.size();
  stats->filtered_external_factor_count = packet.filtered_external_factor_count;
  stats->remove_count = 0u;
  stats->values_add_count = packet.local_values.size();
  stats->hard_reset_count = cbs_h2_sidecar_hard_reset_count_;
  stats->smart_factor_replacements = packet.smart_factor_replacements;
  stats->unmapped_remove_slot_count = 0u;
  stats->packet_pose_values_count = packet.pose_values_count;
  stats->packet_vel_values_count = packet.vel_values_count;
  stats->packet_bias_values_count = packet.bias_values_count;
  stats->packet_remove_count = 0u;
  stats->packet_mapped_heart_remove_slots_count = 0u;
  stats->packet_heart_new_factor_positions_count =
      packet.heart_new_factor_positions.size();
  stats->first_failure_key.clear();
  stats->first_failure_reason.clear();
  stats->failure_reason = "none";

  const auto sync_start = std::chrono::steady_clock::now();
  auto key_to_string = [](const gtsam::Key key) -> std::string {
    const gtsam::Symbol symbol(key);
    std::ostringstream oss;
    oss << symbol.chr() << symbol.index();
    return oss.str();
  };
  auto fail_and_reset = [&](const std::string& reason,
                            const std::string& failure_key = std::string()) {
    h2_local_graph_snapshot_.resize(0);
    h2_local_values_snapshot_.clear();
    h2_local_snapshot_timestamp_ns_ = -1;
    h2_local_snapshot_frame_id_ = 0;
    h2_local_snapshot_valid_ = false;
    ++cbs_h2_sidecar_hard_reset_count_;
    stats->hard_reset_count = cbs_h2_sidecar_hard_reset_count_;
    stats->first_failure_key = failure_key;
    stats->first_failure_reason = reason;
    stats->failure_reason = reason;
    stats->snapshot_refresh_ms =
        elapsedMs(sync_start, std::chrono::steady_clock::now());
    return false;
  };

  if (!useCbsH2LocalCovSidecar()) {
    stats->failure_reason = "h2_mode_disabled";
    return false;
  }
  if (useCbsOptimizerHeart()) {
    stats->failure_reason = "heart_is_cbs";
    return false;
  }
  if (packet.local_factors.empty()) {
    return fail_and_reset("h2_local_snapshot_empty_factors");
  }
  if (packet.local_values.empty()) {
    return fail_and_reset("h2_local_snapshot_empty_values");
  }

  for (const auto& factor : packet.local_factors) {
    if (!factor) {
      continue;
    }
    for (const gtsam::Key key : factor->keys()) {
      if (!packet.local_values.exists(key)) {
        return fail_and_reset("h2_local_snapshot_missing_factor_value",
                              key_to_string(key));
      }
    }
  }

  gtsam::NonlinearFactorGraph cloned_local_graph;
  cloned_local_graph.reserve(packet.local_factors.size());
  for (const auto& factor : packet.local_factors) {
    if (!factor) {
      continue;
    }
    try {
      cloned_local_graph.push_back(factor->clone());
    } catch (const std::exception&) {
      return fail_and_reset("h2_local_snapshot_factor_clone_exception");
    } catch (...) {
      return fail_and_reset("h2_local_snapshot_factor_clone_unknown_exception");
    }
  }

  h2_local_graph_snapshot_ = std::move(cloned_local_graph);
  h2_local_values_snapshot_ = packet.local_values;
  h2_local_snapshot_timestamp_ns_ = timestamp_lkf_;
  h2_local_snapshot_frame_id_ = curr_kf_id;
  h2_local_snapshot_valid_ = true;
  stats->failure_reason = "none";
  stats->snapshot_refresh_ms =
      elapsedMs(sync_start, std::chrono::steady_clock::now());
  return true;
}

/* -------------------------------------------------------------------------- */
bool VioBackend::refreshH2LocalCovarianceIncrementalSmootherSidecar(
    const H2LocalCovSidecarPacket& packet,
    const gtsam::FactorIndices& heart_new_factor_indices,
    const gtsam::FactorIndices& heart_delete_slots,
    const FrameId& curr_kf_id,
    H2LocalCovSidecarSyncStats* stats) {
  CHECK_NOTNULL(stats);
  stats->update_ms = 0.0;
  stats->snapshot_refresh_ms = 0.0;
  stats->local_factor_count = packet.local_factors.size();
  stats->filtered_external_factor_count = packet.filtered_external_factor_count;
  stats->remove_count = 0u;
  stats->values_add_count = packet.local_values.size();
  stats->hard_reset_count = cbs_h2_sidecar_hard_reset_count_;
  stats->smart_factor_replacements = packet.smart_factor_replacements;
  stats->unmapped_remove_slot_count = packet.unmapped_remove_slot_count;
  stats->packet_pose_values_count = packet.pose_values_count;
  stats->packet_vel_values_count = packet.vel_values_count;
  stats->packet_bias_values_count = packet.bias_values_count;
  stats->packet_remove_count = 0u;
  stats->packet_mapped_heart_remove_slots_count = 0u;
  stats->packet_heart_new_factor_positions_count =
      packet.heart_new_factor_positions.size();
  stats->failure_reason = "none";

  const auto sync_start = std::chrono::steady_clock::now();
  auto fail_and_hard_reset = [&](const std::string& reason) {
    cbs_local_cov_smoother_sidecar_.reset();
    cbs_h2_local_smoother_heart_to_sidecar_slot_map_.clear();
    ++cbs_h2_sidecar_hard_reset_count_;
    stats->hard_reset_count = cbs_h2_sidecar_hard_reset_count_;
    stats->failure_reason = reason;
    stats->update_ms = elapsedMs(sync_start, std::chrono::steady_clock::now());
    return false;
  };

  if (!useCbsH2LocalCovSidecar()) {
    stats->failure_reason = "h2_mode_disabled";
    return false;
  }
  if (useCbsOptimizerHeart()) {
    stats->failure_reason = "heart_is_cbs";
    return false;
  }
  if (!smoother_) {
    return fail_and_hard_reset("missing_fixed_lag_smoother");
  }

  if (!cbs_local_cov_smoother_sidecar_) {
#ifdef INCREMENTAL_SMOOTHER
    gtsam::ISAM2Params isam_param;
    BackendParams::setIsam2Params(backend_params_, &isam_param);
    cbs_local_cov_smoother_sidecar_ =
        std::make_unique<Smoother>(backend_params_.nr_states_, isam_param);
#else
    gtsam::LevenbergMarquardtParams lm_params;
    lm_params.setlambdaInitial(0.0);
    lm_params.setlambdaLowerBound(0.0);
    lm_params.setlambdaUpperBound(0.0);
    cbs_local_cov_smoother_sidecar_ =
        std::make_unique<Smoother>(backend_params_.nr_states_, lm_params);
#endif
    cbs_h2_local_smoother_heart_to_sidecar_slot_map_.clear();
  }

  CHECK(cbs_local_cov_smoother_sidecar_);
  stats->sidecar_slot_map_size_before_update =
      cbs_h2_local_smoother_heart_to_sidecar_slot_map_.size();
  const auto& local_graph_before = cbs_local_cov_smoother_sidecar_->getFactors();
  stats->sidecar_graph_size_before_update = local_graph_before.size();

  gtsam::FactorIndices local_remove_slots;
  local_remove_slots.reserve(heart_delete_slots.size());
  for (const auto heart_slot : heart_delete_slots) {
    const auto map_it =
        cbs_h2_local_smoother_heart_to_sidecar_slot_map_.find(heart_slot);
    if (map_it == cbs_h2_local_smoother_heart_to_sidecar_slot_map_.end()) {
      continue;
    }
    const auto local_slot = map_it->second;
    if (local_graph_before.exists(local_slot)) {
      local_remove_slots.push_back(local_slot);
      ++stats->translated_remove_count;
      if (stats->first_translated_sidecar_remove_slot ==
          std::numeric_limits<gtsam::FactorIndex>::max()) {
        stats->first_translated_sidecar_remove_slot = local_slot;
      }
    } else {
      ++stats->filtered_stale_remove_count;
      if (stats->first_removed_stale_heart_slot ==
          std::numeric_limits<gtsam::FactorIndex>::max()) {
        stats->first_removed_stale_heart_slot = heart_slot;
      }
      if (stats->first_removed_stale_sidecar_slot ==
          std::numeric_limits<gtsam::FactorIndex>::max()) {
        stats->first_removed_stale_sidecar_slot = local_slot;
      }
    }
    cbs_h2_local_smoother_heart_to_sidecar_slot_map_.erase(map_it);
  }
  std::sort(local_remove_slots.begin(), local_remove_slots.end());
  local_remove_slots.erase(
      std::unique(local_remove_slots.begin(), local_remove_slots.end()),
      local_remove_slots.end());
  stats->remove_count = local_remove_slots.size();
  stats->packet_remove_count = local_remove_slots.size();
  stats->packet_mapped_heart_remove_slots_count = local_remove_slots.size();

  std::map<Key, double> local_key_frame_count;
  for (const auto& key_value : packet.local_values) {
    local_key_frame_count[key_value.key] = static_cast<double>(curr_kf_id);
  }

  Smoother::Result sidecar_result;
  try {
    sidecar_result = cbs_local_cov_smoother_sidecar_->update(
        packet.local_factors,
        packet.local_values,
        local_key_frame_count,
        local_remove_slots);
  } catch (const std::exception& e) {
    return fail_and_hard_reset(std::string("local_side_smoother_update:") +
                               e.what());
  } catch (...) {
    return fail_and_hard_reset("local_side_smoother_update:unknown_exception");
  }

  const gtsam::ISAM2Result& sidecar_isam2_result =
      cbs_local_cov_smoother_sidecar_->getISAM2Result();
  const auto& sidecar_new_indices = sidecar_isam2_result.newFactorsIndices;
  stats->sidecar_new_factor_indices_count = sidecar_new_indices.size();
  const size_t mapped_count = std::min(packet.heart_new_factor_positions.size(),
                                       sidecar_new_indices.size());
  for (size_t i = 0u; i < mapped_count; ++i) {
    const size_t heart_position = packet.heart_new_factor_positions[i];
    if (heart_position >= heart_new_factor_indices.size()) {
      continue;
    }
    const gtsam::FactorIndex heart_slot = heart_new_factor_indices[heart_position];
    const gtsam::FactorIndex local_slot = sidecar_new_indices[i];
    cbs_h2_local_smoother_heart_to_sidecar_slot_map_[heart_slot] = local_slot;
  }

  stats->update_ms = elapsedMs(sync_start, std::chrono::steady_clock::now());
  stats->failure_reason = "none";
  return true;
}

bool VioBackend::refreshH2LocalCovarianceSidecar(
    const H2LocalCovSidecarPacket& packet,
    const gtsam::FactorIndices& heart_new_factor_indices,
    const FrameId& curr_kf_id,
    H2LocalCovSidecarSyncStats* stats) {
  CHECK_NOTNULL(stats);
  stats->update_ms = 0.0;
  stats->snapshot_refresh_ms = 0.0;
  stats->local_factor_count = packet.local_factors.size();
  stats->filtered_external_factor_count = packet.filtered_external_factor_count;
  stats->remove_count = 0u;
  stats->values_add_count = packet.local_values.size();
  stats->hard_reset_count = cbs_h2_sidecar_hard_reset_count_;
  stats->smart_factor_replacements = packet.smart_factor_replacements;
  stats->unmapped_remove_slot_count = packet.unmapped_remove_slot_count;
  stats->packet_pose_values_count = packet.pose_values_count;
  stats->packet_vel_values_count = packet.vel_values_count;
  stats->packet_bias_values_count = packet.bias_values_count;
  stats->packet_remove_count = packet.sidecar_remove_factor_indices.size();
  stats->packet_mapped_heart_remove_slots_count =
      packet.mapped_heart_remove_factor_slots.size();
  stats->packet_heart_new_factor_positions_count =
      packet.heart_new_factor_positions.size();
  stats->sidecar_slot_map_size_before_update = 0u;
  stats->sidecar_graph_size_before_update = 0u;
  stats->translated_remove_count = 0u;
  stats->filtered_stale_remove_count = 0u;
  stats->sidecar_new_factor_indices_count = 0u;
  stats->h2_repair_epoch_detected = false;
  stats->h2_repair_epoch_original_new_index_count = 0u;
  stats->h2_replay_add_only_attempted = false;
  stats->h2_replay_add_only_succeeded = false;
  stats->h2_replay_add_only_new_index_count = 0u;
  stats->h2_replay_add_only_failure_reason = "none";
  stats->h2_replay_subset_probe_attempted = false;
  stats->h2_replay_subset_probe_clone_available = false;
  stats->h2_replay_subset_probe_clone_setup_reason = "none";
  stats->h2_replay_subset_smart_count = 0u;
  stats->h2_replay_subset_non_smart_count = 0u;
  stats->h2_replay_subset_imu_count = 0u;
  stats->h2_replay_subset_between_count = 0u;
  stats->h2_replay_subset_empty_succeeded = false;
  stats->h2_replay_subset_non_smart_succeeded = false;
  stats->h2_replay_subset_smart_succeeded = false;
  stats->h2_replay_subset_imu_only_succeeded = false;
  stats->h2_replay_subset_between_only_succeeded = false;
  stats->h2_replay_subset_full_probe_succeeded = false;
  stats->h2_replay_subset_empty_reason = "not_run";
  stats->h2_replay_subset_non_smart_reason = "not_run";
  stats->h2_replay_subset_smart_reason = "not_run";
  stats->h2_replay_subset_imu_only_reason = "not_run";
  stats->h2_replay_subset_between_only_reason = "not_run";
  stats->h2_replay_subset_full_probe_reason = "not_run";
  stats->h2_replay_probe_bootstrap_empty_succeeded = false;
  stats->h2_replay_probe_bootstrap_non_smart_succeeded = false;
  stats->h2_replay_probe_bootstrap_smart_succeeded = false;
  stats->h2_replay_probe_bootstrap_imu_only_succeeded = false;
  stats->h2_replay_probe_bootstrap_between_only_succeeded = false;
  stats->h2_replay_probe_bootstrap_full_succeeded = false;
  stats->h2_replay_probe_update_empty_succeeded = false;
  stats->h2_replay_probe_update_non_smart_succeeded = false;
  stats->h2_replay_probe_update_smart_succeeded = false;
  stats->h2_replay_probe_update_imu_only_succeeded = false;
  stats->h2_replay_probe_update_between_only_succeeded = false;
  stats->h2_replay_probe_update_full_succeeded = false;
  stats->h2_replay_probe_bootstrap_empty_reason = "not_run";
  stats->h2_replay_probe_bootstrap_non_smart_reason = "not_run";
  stats->h2_replay_probe_bootstrap_smart_reason = "not_run";
  stats->h2_replay_probe_bootstrap_imu_only_reason = "not_run";
  stats->h2_replay_probe_bootstrap_between_only_reason = "not_run";
  stats->h2_replay_probe_bootstrap_full_reason = "not_run";
  stats->h2_replay_probe_update_empty_reason = "not_run";
  stats->h2_replay_probe_update_non_smart_reason = "not_run";
  stats->h2_replay_probe_update_smart_reason = "not_run";
  stats->h2_replay_probe_update_imu_only_reason = "not_run";
  stats->h2_replay_probe_update_between_only_reason = "not_run";
  stats->h2_replay_probe_update_full_reason = "not_run";
  stats->h2_partial_replay_non_smart_only_applied = false;
  stats->h2_partial_replay_reason = "none";
  stats->remap_existing_heart_slot_conflict_count = 0u;
  stats->post_prune_candidate_count = 0u;
  stats->post_prune_filtered_stale_count = 0u;
  stats->stale_slot_map_entries_removed_pre_update = 0u;
  stats->stale_slot_map_entries_removed_during_translation = 0u;
  stats->stale_slot_map_entries_removed_during_remap = 0u;
  stats->primary_remove_attempted = false;
  stats->primary_remove_deferred_due_to_stale_map = false;
  stats->primary_retry_attempted = false;
  stats->primary_retry_succeeded = false;
  stats->retry_without_remove_factor_indices = false;
  stats->post_prune_retry_with_filtered_slots = false;
  stats->post_prune_threw_map_at = false;
  stats->post_prune_skipped_deferred = false;
  stats->first_unmapped_heart_slot = packet.first_unmapped_heart_slot;
  stats->first_unmapped_sidecar_slot = packet.first_unmapped_sidecar_slot;
  stats->first_translated_sidecar_remove_slot =
      std::numeric_limits<gtsam::FactorIndex>::max();
  stats->first_post_prune_sidecar_slot =
      std::numeric_limits<gtsam::FactorIndex>::max();
  stats->first_conflicting_heart_slot =
      std::numeric_limits<gtsam::FactorIndex>::max();
  stats->first_conflicting_old_sidecar_slot =
      std::numeric_limits<gtsam::FactorIndex>::max();
  stats->first_conflicting_new_sidecar_slot =
      std::numeric_limits<gtsam::FactorIndex>::max();
  stats->first_removed_stale_heart_slot =
      std::numeric_limits<gtsam::FactorIndex>::max();
  stats->first_removed_stale_sidecar_slot =
      std::numeric_limits<gtsam::FactorIndex>::max();
  stats->sample_remove_indices.clear();
  stats->sample_heart_new_factor_positions.clear();
  stats->sample_heart_new_factor_indices.clear();
  stats->sample_sidecar_new_factor_indices.clear();
  stats->required_local_key_count = 0u;
  stats->required_packet_key_count = 0u;
  stats->packet_local_values_exact_required = false;
  stats->packet_missing_required_local_values_count = 0u;
  stats->packet_extra_local_values_count = 0u;
  stats->packet_local_value_keys.clear();
  stats->packet_missing_required_local_value_keys.clear();
  stats->packet_extra_local_value_keys.clear();
  stats->value_type_mismatch_count = 0u;
  stats->missing_value_count = 0u;
  stats->unsupported_key_type_count = 0u;
  stats->post_prune_remove_count = 0u;
  stats->first_failure_key.clear();
  stats->first_failure_reason.clear();
  stats->failure_reason.clear();

  if (!useCbsH2LocalCovSidecar()) {
    stats->failure_reason = "h2_mode_disabled";
    return false;
  }
  if (useCbsOptimizerHeart()) {
    stats->failure_reason = "heart_is_cbs";
    return false;
  }
  if (!smoother_) {
    stats->failure_reason = "missing_fixed_lag_smoother";
    return false;
  }

  const auto sync_start = std::chrono::steady_clock::now();
  auto fail_and_hard_reset = [&](const std::string& reason) {
    cbs_local_cov_sidecar_.reset();
    cbs_local_cov_smoother_sidecar_.reset();
    cbs_h2_local_smoother_heart_to_sidecar_slot_map_.clear();
    cbs_h2_sidecar_heart_to_sidecar_slot_map_.clear();
    ++cbs_h2_sidecar_hard_reset_count_;
    stats->hard_reset_count = cbs_h2_sidecar_hard_reset_count_;
    stats->failure_reason = reason;
    stats->update_ms = elapsedMs(sync_start, std::chrono::steady_clock::now());
    return false;
  };
  auto key_to_string = [](const gtsam::Key key) -> std::string {
    const gtsam::Symbol symbol(key);
    std::ostringstream oss;
    oss << symbol.chr() << symbol.index();
    return oss.str();
  };
  auto indices_to_sample_csv = [](const gtsam::FactorIndices& indices,
                                  const size_t max_items = 8u) -> std::string {
    std::ostringstream oss;
    for (size_t i = 0u; i < indices.size() && i < max_items; ++i) {
      if (i > 0u) {
        oss << ",";
      }
      oss << indices[i];
    }
    if (indices.size() > max_items) {
      oss << ",...(" << (indices.size() - max_items) << "_more)";
    }
    return oss.str();
  };
  auto size_t_to_sample_csv = [](const std::vector<size_t>& values,
                                 const size_t max_items = 8u) -> std::string {
    std::ostringstream oss;
    for (size_t i = 0u; i < values.size() && i < max_items; ++i) {
      if (i > 0u) {
        oss << ",";
      }
      oss << values[i];
    }
    if (values.size() > max_items) {
      oss << ",...(" << (values.size() - max_items) << "_more)";
    }
    return oss.str();
  };
  stats->sample_heart_new_factor_positions =
      size_t_to_sample_csv(packet.heart_new_factor_positions);
  auto keys_to_compact_csv =
      [&](const std::unordered_set<gtsam::Key>& keys,
          const size_t max_items = 24u) -> std::string {
    std::vector<gtsam::Key> sorted_keys(keys.begin(), keys.end());
    std::sort(sorted_keys.begin(), sorted_keys.end());
    std::ostringstream oss;
    for (size_t i = 0u; i < sorted_keys.size() && i < max_items; ++i) {
      if (i > 0u) {
        oss << ",";
      }
      oss << key_to_string(sorted_keys[i]);
    }
    if (sorted_keys.size() > max_items) {
      oss << ",...(" << (sorted_keys.size() - max_items) << "_more)";
    }
    return oss.str();
  };
  auto fail_with_key = [&](const std::string& reason_prefix,
                           const gtsam::Key key) {
    const std::string key_str = key_to_string(key);
    stats->first_failure_key = key_str;
    stats->first_failure_reason = reason_prefix;
    return fail_and_hard_reset(reason_prefix + ":" + key_str);
  };
  auto record_removed_stale_slot_map_entry =
      [&](const gtsam::FactorIndex heart_slot,
          const gtsam::FactorIndex sidecar_slot,
          size_t* removed_counter) {
        if (removed_counter) {
          ++(*removed_counter);
        }
        if (stats->first_removed_stale_heart_slot ==
            std::numeric_limits<gtsam::FactorIndex>::max()) {
          stats->first_removed_stale_heart_slot = heart_slot;
        }
        if (stats->first_removed_stale_sidecar_slot ==
            std::numeric_limits<gtsam::FactorIndex>::max()) {
          stats->first_removed_stale_sidecar_slot = sidecar_slot;
        }
      };
  auto packet_has_correct_type = [&](const gtsam::Symbol& symbol) -> bool {
    if (!packet.local_values.exists(symbol)) {
      return false;
    }
    try {
      if (symbol.chr() == kPoseSymbolChar) {
        (void)packet.local_values.at<gtsam::Pose3>(symbol);
        return true;
      }
      if (symbol.chr() == kVelocitySymbolChar) {
        (void)packet.local_values.at<gtsam::Vector3>(symbol);
        return true;
      }
      if (symbol.chr() == kImuBiasSymbolChar) {
        (void)packet.local_values.at<gtsam::imuBias::ConstantBias>(symbol);
        return true;
      }
      return false;
    } catch (...) {
      return false;
    }
  };
  auto ensure_sidecar_initialized = [&]() -> bool {
    if (cbs_local_cov_sidecar_) {
      return true;
    }
    gtsam::ISAM2Params sidecar_isam_params;
    BackendParams::setIsam2Params(backend_params_, &sidecar_isam_params);

    cbs::BPSAM::Params sidecar_params;
    constexpr cbs::AgentId kKimeraAgentId = static_cast<cbs::AgentId>('a');
    sidecar_params.robot_id = kKimeraAgentId;
    sidecar_params.sam_params_ = sidecar_isam_params;
    sidecar_params.enable_gkcm = FLAGS_cbs_enable_gkcm;
    sidecar_params.gbp_update_params.type = gbp::GaussianMergeType::Contract;
    sidecar_params.gbp_update_params.metric_type = gbp::MetricType::Hellinger;
    sidecar_params.gbp_update_params.contract_alpha =
        static_cast<float>(FLAGS_cbs_belief_contract_alpha);
    sidecar_params.gbp_update_params.d_reset =
        static_cast<float>(FLAGS_cbs_belief_d_reset);
    sidecar_params.gbp_update_params.gamma =
        static_cast<float>(FLAGS_cbs_belief_gamma);
    cbs_local_cov_sidecar_ = std::make_shared<cbs::BPSAM>(sidecar_params);
    cbs_h2_sidecar_heart_to_sidecar_slot_map_.clear();
    return true;
  };

  if (!ensure_sidecar_initialized()) {
    return fail_and_hard_reset("h2_sidecar_init_failed");
  }

  try {
    const bool trace_focus_epoch = (curr_kf_id == 51 || curr_kf_id == 52);
    const bool nonsmart_probe_epoch = (curr_kf_id == 52);
    auto classify_factor_type = [](const gtsam::NonlinearFactor* factor)
        -> std::string {
      if (!factor) {
        return "other";
      }
      if (dynamic_cast<const SmartStereoFactor*>(factor) != nullptr) {
        return "smart";
      }
      if (dynamic_cast<const gtsam::CombinedImuFactor*>(factor) != nullptr ||
          dynamic_cast<const gtsam::ImuFactor*>(factor) != nullptr) {
        return "imu";
      }
      if (dynamic_cast<const gtsam::BetweenFactor<gtsam::Pose3>*>(factor) !=
              nullptr ||
          dynamic_cast<const gtsam::BetweenFactor<gtsam::Vector3>*>(factor) !=
              nullptr ||
          dynamic_cast<
              const gtsam::BetweenFactor<gtsam::imuBias::ConstantBias>*>(
              factor) != nullptr) {
        return "between";
      }
      if (dynamic_cast<const gtsam::PriorFactor<gtsam::Pose3>*>(factor) !=
              nullptr ||
          dynamic_cast<const gtsam::PriorFactor<gtsam::Vector3>*>(factor) !=
              nullptr ||
          dynamic_cast<const gtsam::PriorFactor<gtsam::imuBias::ConstantBias>*>(
              factor) != nullptr) {
        return "prior";
      }
      return "other";
    };
    auto run_sidecar_self_audit = [&](const std::string& audit_point,
                                      const bool apply_targeted_purge) {
      const gtsam::NonlinearFactorGraph& sidecar_graph_snapshot =
          cbs_local_cov_sidecar_->getFactorsUnsafe();
      const size_t sidecar_graph_size = sidecar_graph_snapshot.size();
      const size_t slot_map_size = cbs_h2_sidecar_heart_to_sidecar_slot_map_.size();

      gtsam::Values sidecar_values;
      bool sidecar_values_available = false;
      std::string value_snapshot_error = "none";
      try {
        sidecar_values = cbs_local_cov_sidecar_->calculateEstimate();
        sidecar_values_available = true;
      } catch (const std::exception& e) {
        value_snapshot_error =
            e.what() ? std::string(e.what())
                     : std::string("calculate_estimate_std_exception");
      } catch (...) {
        value_snapshot_error = "calculate_estimate_unknown_exception";
      }
      const size_t sidecar_value_count =
          sidecar_values_available ? sidecar_values.size() : 0u;

      size_t existing_factor_slots = 0u;
      size_t factors_with_missing_keys = 0u;
      size_t smart_factors_with_missing_keys = 0u;
      size_t imu_factors_with_missing_keys = 0u;
      size_t between_factors_with_missing_keys = 0u;
      size_t prior_factors_with_missing_keys = 0u;
      size_t other_factors_with_missing_keys = 0u;
      size_t stale_slot_map_entries = 0u;
      size_t slot_map_entries_with_missing_keys = 0u;
      gtsam::FactorIndex first_bad_sidecar_slot =
          std::numeric_limits<gtsam::FactorIndex>::max();
      gtsam::FactorIndex first_bad_heart_slot =
          std::numeric_limits<gtsam::FactorIndex>::max();
      std::string first_missing_key = "none";
      std::string first_bad_factor_type = "none";

      std::unordered_set<gtsam::FactorIndex> missing_key_factor_slots;
      gtsam::FactorIndices missing_key_slots_for_purge;
      missing_key_slots_for_purge.reserve(sidecar_graph_snapshot.size());

      for (gtsam::FactorIndex sidecar_slot = 0;
           sidecar_slot < sidecar_graph_snapshot.size();
           ++sidecar_slot) {
        if (!sidecar_graph_snapshot.exists(sidecar_slot)) {
          continue;
        }
        ++existing_factor_slots;
        const auto& factor = sidecar_graph_snapshot.at(sidecar_slot);
        if (!factor) {
          continue;
        }

        gtsam::Key first_missing_factor_key =
            std::numeric_limits<gtsam::Key>::max();
        bool has_missing_key = false;
        for (const gtsam::Key key : factor->keys()) {
          bool sidecar_has_key = false;
          if (sidecar_values_available) {
            sidecar_has_key = sidecar_values.exists(key);
          } else {
            sidecar_has_key = cbs_local_cov_sidecar_->valueExists(gtsam::Symbol(key));
          }
          if (!sidecar_has_key) {
            has_missing_key = true;
            first_missing_factor_key = key;
            break;
          }
        }
        if (!has_missing_key) {
          continue;
        }

        ++factors_with_missing_keys;
        missing_key_factor_slots.insert(sidecar_slot);
        missing_key_slots_for_purge.push_back(sidecar_slot);

        const std::string factor_type = classify_factor_type(factor.get());
        if (factor_type == "smart") {
          ++smart_factors_with_missing_keys;
        } else if (factor_type == "imu") {
          ++imu_factors_with_missing_keys;
        } else if (factor_type == "between") {
          ++between_factors_with_missing_keys;
        } else if (factor_type == "prior") {
          ++prior_factors_with_missing_keys;
        } else {
          ++other_factors_with_missing_keys;
        }

        if (first_bad_sidecar_slot ==
            std::numeric_limits<gtsam::FactorIndex>::max()) {
          first_bad_sidecar_slot = sidecar_slot;
          first_bad_factor_type = factor_type;
          if (first_missing_factor_key != std::numeric_limits<gtsam::Key>::max()) {
            first_missing_key = key_to_string(first_missing_factor_key);
          }
        }
      }

      for (const auto& heart_to_sidecar :
           cbs_h2_sidecar_heart_to_sidecar_slot_map_) {
        const gtsam::FactorIndex heart_slot = heart_to_sidecar.first;
        const gtsam::FactorIndex sidecar_slot = heart_to_sidecar.second;
        if (!sidecar_graph_snapshot.exists(sidecar_slot)) {
          ++stale_slot_map_entries;
          if (first_bad_heart_slot ==
              std::numeric_limits<gtsam::FactorIndex>::max()) {
            first_bad_heart_slot = heart_slot;
            if (first_bad_sidecar_slot ==
                std::numeric_limits<gtsam::FactorIndex>::max()) {
              first_bad_sidecar_slot = sidecar_slot;
            }
          }
          continue;
        }
        if (missing_key_factor_slots.count(sidecar_slot) > 0u) {
          ++slot_map_entries_with_missing_keys;
          if (first_bad_heart_slot ==
              std::numeric_limits<gtsam::FactorIndex>::max()) {
            first_bad_heart_slot = heart_slot;
            if (first_bad_sidecar_slot ==
                std::numeric_limits<gtsam::FactorIndex>::max()) {
              first_bad_sidecar_slot = sidecar_slot;
            }
          }
        }
      }

      bool targeted_purge_applied = false;
      bool targeted_purge_success = false;
      size_t targeted_purge_slots = 0u;
      size_t targeted_purge_map_entries_removed = 0u;
      std::string targeted_purge_error = "none";
      if (apply_targeted_purge && !missing_key_slots_for_purge.empty()) {
        targeted_purge_applied = true;
        std::sort(missing_key_slots_for_purge.begin(),
                  missing_key_slots_for_purge.end());
        missing_key_slots_for_purge.erase(
            std::unique(missing_key_slots_for_purge.begin(),
                        missing_key_slots_for_purge.end()),
            missing_key_slots_for_purge.end());

        std::unordered_set<gtsam::FactorIndex> purge_slot_set(
            missing_key_slots_for_purge.begin(), missing_key_slots_for_purge.end());
        for (auto it = cbs_h2_sidecar_heart_to_sidecar_slot_map_.begin();
             it != cbs_h2_sidecar_heart_to_sidecar_slot_map_.end();) {
          if (purge_slot_set.count(it->second) > 0u) {
            it = cbs_h2_sidecar_heart_to_sidecar_slot_map_.erase(it);
            ++targeted_purge_map_entries_removed;
          } else {
            ++it;
          }
        }

        auto filter_existing_purge_slots = [&](const gtsam::FactorIndices& slots)
            -> gtsam::FactorIndices {
          gtsam::FactorIndices filtered_slots;
          filtered_slots.reserve(slots.size());
          const gtsam::NonlinearFactorGraph& sidecar_graph_now =
              cbs_local_cov_sidecar_->getFactorsUnsafe();
          for (const gtsam::FactorIndex slot : slots) {
            if (sidecar_graph_now.exists(slot)) {
              filtered_slots.push_back(slot);
            }
          }
          if (!filtered_slots.empty()) {
            std::sort(filtered_slots.begin(), filtered_slots.end());
            filtered_slots.erase(
                std::unique(filtered_slots.begin(), filtered_slots.end()),
                filtered_slots.end());
          }
          return filtered_slots;
        };

        auto try_targeted_purge =
            [&](const gtsam::FactorIndices& purge_slots, std::string* error_reason)
            -> bool {
          CHECK_NOTNULL(error_reason);
          *error_reason = "none";
          if (purge_slots.empty()) {
            return true;
          }
          cbs::BPSAM::UpdateParams purge_params;
          purge_params.removeFactorIndices = purge_slots;
          try {
            cbs_local_cov_sidecar_->update(gtsam::NonlinearFactorGraph(),
                                           gtsam::Values(),
                                           purge_params);
            return true;
          } catch (const std::exception& e) {
            *error_reason =
                e.what() ? std::string(e.what())
                         : std::string("h2_sidecar_targeted_purge_std_exception");
            return false;
          } catch (...) {
            *error_reason = "h2_sidecar_targeted_purge_unknown_exception";
            return false;
          }
        };

        gtsam::FactorIndices purge_slots =
            filter_existing_purge_slots(missing_key_slots_for_purge);
        targeted_purge_slots = purge_slots.size();
        if (try_targeted_purge(purge_slots, &targeted_purge_error)) {
          targeted_purge_success = true;
        } else if (targeted_purge_error.find("map::at") != std::string::npos) {
          gtsam::FactorIndices retry_slots = filter_existing_purge_slots(purge_slots);
          targeted_purge_slots = retry_slots.size();
          if (retry_slots.empty()) {
            targeted_purge_success = true;
            targeted_purge_error = "none";
          } else {
            targeted_purge_success =
                try_targeted_purge(retry_slots, &targeted_purge_error);
          }
        }
      }

      auto factor_index_to_log = [](const gtsam::FactorIndex idx) -> long long {
        if (idx == std::numeric_limits<gtsam::FactorIndex>::max()) {
          return -1;
        }
        return static_cast<long long>(idx);
      };

      std::cerr << std::setprecision(12)
                << "[CBS][H2SidecarSelfAudit]"
                << " curr_kf_id=" << curr_kf_id
                << " audit_point=" << audit_point
                << " sidecar_graph_size=" << sidecar_graph_size
                << " sidecar_value_count=" << sidecar_value_count
                << " slot_map_size=" << slot_map_size
                << " existing_factor_slots=" << existing_factor_slots
                << " factors_with_missing_keys=" << factors_with_missing_keys
                << " smart_factors_with_missing_keys="
                << smart_factors_with_missing_keys
                << " imu_factors_with_missing_keys="
                << imu_factors_with_missing_keys
                << " between_factors_with_missing_keys="
                << between_factors_with_missing_keys
                << " prior_factors_with_missing_keys="
                << prior_factors_with_missing_keys
                << " other_factors_with_missing_keys="
                << other_factors_with_missing_keys
                << " stale_slot_map_entries=" << stale_slot_map_entries
                << " slot_map_entries_with_missing_keys="
                << slot_map_entries_with_missing_keys
                << " stale_slot_map_entries_removed_pre_update="
                << stats->stale_slot_map_entries_removed_pre_update
                << " stale_slot_map_entries_removed_during_translation="
                << stats->stale_slot_map_entries_removed_during_translation
                << " stale_slot_map_entries_removed_during_remap="
                << stats->stale_slot_map_entries_removed_during_remap
                << " first_bad_sidecar_slot="
                << factor_index_to_log(first_bad_sidecar_slot)
                << " first_bad_heart_slot="
                << factor_index_to_log(first_bad_heart_slot)
                << " first_removed_stale_heart_slot="
                << factor_index_to_log(stats->first_removed_stale_heart_slot)
                << " first_removed_stale_sidecar_slot="
                << factor_index_to_log(stats->first_removed_stale_sidecar_slot)
                << " first_missing_key=" << first_missing_key
                << " first_bad_factor_type=" << first_bad_factor_type
                << " targeted_purge_applied="
                << (targeted_purge_applied ? 1 : 0)
                << " targeted_purge_slots=" << targeted_purge_slots
                << " targeted_purge_map_entries_removed="
                << targeted_purge_map_entries_removed
                << " targeted_purge_success="
                << (targeted_purge_success ? 1 : 0)
                << " value_snapshot_error="
                << (value_snapshot_error.empty() ? "none" : value_snapshot_error)
                << " targeted_purge_error="
                << (targeted_purge_error.empty() ? "none" : targeted_purge_error)
                << std::endl;
    };
    if (curr_kf_id == 52) {
      run_sidecar_self_audit("start_of_epoch_kf52", true);
    }
    size_t local_factor_smart_count = 0u;
    size_t local_factor_imu_count = 0u;
    size_t local_factor_prior_count = 0u;
    size_t local_factor_between_count = 0u;
    size_t local_factor_other_count = 0u;
    size_t local_factor_all_keys_in_sidecar_count = 0u;
    size_t local_factor_packet_only_key_count = 0u;
    size_t smart_replacement_candidate_count = 0u;
    gtsam::NonlinearFactorGraph smart_only_factors;
    gtsam::NonlinearFactorGraph imu_only_factors;
    gtsam::NonlinearFactorGraph between_only_factors;
    gtsam::NonlinearFactorGraph no_smart_replacement_factors;
    gtsam::NonlinearFactorGraph no_smart_any_factors;
    std::vector<size_t> non_smart_local_factor_positions;
    std::vector<std::string> first_factor_key_samples;
    first_factor_key_samples.reserve(6u);

    smart_only_factors.reserve(packet.local_factors.size());
    imu_only_factors.reserve(packet.local_factors.size());
    between_only_factors.reserve(packet.local_factors.size());
    no_smart_replacement_factors.reserve(packet.local_factors.size());
    no_smart_any_factors.reserve(packet.local_factors.size());
    non_smart_local_factor_positions.reserve(packet.local_factors.size());

    for (size_t factor_idx = 0u; factor_idx < packet.local_factors.size();
         ++factor_idx) {
      const auto& factor = packet.local_factors.at(factor_idx);
      if (!factor) {
        ++local_factor_other_count;
        continue;
      }

      const std::string factor_type = classify_factor_type(factor.get());
      const bool is_smart_factor = factor_type == "smart";
      const bool is_imu_factor = factor_type == "imu";
      const bool is_prior_factor = factor_type == "prior";
      const bool is_between_factor = factor_type == "between";

      if (is_smart_factor) {
        ++local_factor_smart_count;
      } else if (is_imu_factor) {
        ++local_factor_imu_count;
      } else if (is_prior_factor) {
        ++local_factor_prior_count;
      } else if (is_between_factor) {
        ++local_factor_between_count;
      } else {
        ++local_factor_other_count;
      }

      std::ostringstream keys_oss;
      bool all_keys_in_sidecar = true;
      bool has_packet_only_key = false;
      const gtsam::KeyVector factor_keys = factor->keys();
      for (size_t key_i = 0u; key_i < factor_keys.size(); ++key_i) {
        const gtsam::Key key = factor_keys[key_i];
        if (key_i > 0u) {
          keys_oss << ",";
        }
        keys_oss << key_to_string(key);

        const gtsam::Symbol key_symbol(key);
        const bool sidecar_has_key = cbs_local_cov_sidecar_->valueExists(key_symbol);
        const bool packet_has_key = packet.local_values.exists(key);
        all_keys_in_sidecar = all_keys_in_sidecar && sidecar_has_key;
        if (!sidecar_has_key && packet_has_key) {
          has_packet_only_key = true;
        }
      }

      if (first_factor_key_samples.size() < 6u) {
        std::ostringstream sample_oss;
        sample_oss << "f" << factor_idx << ":" << keys_oss.str();
        first_factor_key_samples.push_back(sample_oss.str());
      }
      if (all_keys_in_sidecar) {
        ++local_factor_all_keys_in_sidecar_count;
      }
      if (has_packet_only_key) {
        ++local_factor_packet_only_key_count;
      }

      // Smart replacement proxy for A/B localization:
      // smart factor whose keys are already present in sidecar state.
      const bool smart_replacement_candidate =
          is_smart_factor && all_keys_in_sidecar;
      if (smart_replacement_candidate) {
        ++smart_replacement_candidate_count;
      } else {
        no_smart_replacement_factors.push_back(factor);
      }
      if (is_smart_factor) {
        smart_only_factors.push_back(factor);
      }
      if (is_imu_factor) {
        imu_only_factors.push_back(factor);
      }
      if (is_between_factor) {
        between_only_factors.push_back(factor);
      }
      if (!is_smart_factor) {
        no_smart_any_factors.push_back(factor);
        non_smart_local_factor_positions.push_back(factor_idx);
      }
    }

    if (trace_focus_epoch) {
      std::ostringstream first_keys_oss;
      for (size_t i = 0u; i < first_factor_key_samples.size(); ++i) {
        if (i > 0u) {
          first_keys_oss << "|";
        }
        first_keys_oss << first_factor_key_samples[i];
      }
      std::cerr << std::setprecision(12)
                << "[CBS][H2PacketFactorDiag]"
                << " curr_kf_id=" << curr_kf_id
                << " packet_local_factor_count=" << packet.local_factors.size()
                << " smart_factor_count=" << local_factor_smart_count
                << " imu_factor_count=" << local_factor_imu_count
                << " prior_factor_count=" << local_factor_prior_count
                << " between_factor_count=" << local_factor_between_count
                << " other_factor_count=" << local_factor_other_count
                << " smart_replacement_candidate_count="
                << smart_replacement_candidate_count
                << " factors_all_keys_in_sidecar_count="
                << local_factor_all_keys_in_sidecar_count
                << " factors_with_packet_only_key_count="
                << local_factor_packet_only_key_count
                << " first_factor_key_samples="
                << (first_keys_oss.str().empty() ? "none" : first_keys_oss.str())
                << std::endl;
    }

    // H2.3 pre-update consistency check: each key used by local factors must
    // either already exist in sidecar, or be provided in this packet with the
    // correct semantic type.
    std::unordered_set<gtsam::Key> required_keys;
    for (const auto& factor : packet.local_factors) {
      if (!factor) {
        continue;
      }
      for (const gtsam::Key key : factor->keys()) {
        required_keys.insert(key);
      }
    }
    stats->required_local_key_count = required_keys.size();

    std::unordered_set<gtsam::Key> packet_value_keys;
    for (const gtsam::Key key : packet.local_values.keys()) {
      packet_value_keys.insert(key);
    }
    stats->packet_local_value_keys = keys_to_compact_csv(packet_value_keys);

    std::unordered_set<gtsam::Key> required_packet_keys;
    gtsam::Key first_wrong_type_key = std::numeric_limits<gtsam::Key>::max();
    bool has_wrong_type_key = false;
    for (const gtsam::Key key : required_keys) {
      const gtsam::Symbol symbol(key);
      const auto key_char = symbol.chr();
      if (key_char != kPoseSymbolChar && key_char != kVelocitySymbolChar &&
          key_char != kImuBiasSymbolChar) {
        ++stats->unsupported_key_type_count;
        return fail_with_key("unsupported_key_type", key);
      }

      const bool sidecar_has_value = cbs_local_cov_sidecar_->valueExists(symbol);
      const bool packet_has_value = packet.local_values.exists(symbol);
      if (!sidecar_has_value) {
        required_packet_keys.insert(key);
      }
      if (packet_has_value && !packet_has_correct_type(symbol) &&
          !has_wrong_type_key) {
        has_wrong_type_key = true;
        first_wrong_type_key = key;
      }
    }
    stats->required_packet_key_count = required_packet_keys.size();

    std::unordered_set<gtsam::Key> missing_required_packet_keys;
    for (const gtsam::Key key : required_packet_keys) {
      if (packet_value_keys.find(key) == packet_value_keys.end()) {
        missing_required_packet_keys.insert(key);
      }
    }
    std::unordered_set<gtsam::Key> extra_packet_keys;
    for (const gtsam::Key key : packet_value_keys) {
      if (required_packet_keys.find(key) == required_packet_keys.end()) {
        extra_packet_keys.insert(key);
      }
    }
    stats->packet_missing_required_local_values_count =
        missing_required_packet_keys.size();
    stats->packet_extra_local_values_count = extra_packet_keys.size();
    stats->packet_missing_required_local_value_keys =
        keys_to_compact_csv(missing_required_packet_keys);
    stats->packet_extra_local_value_keys = keys_to_compact_csv(extra_packet_keys);
    stats->packet_local_values_exact_required =
        missing_required_packet_keys.empty() && extra_packet_keys.empty();

    if (has_wrong_type_key) {
      ++stats->value_type_mismatch_count;
      return fail_with_key("wrong_value_type_for_key", first_wrong_type_key);
    }
    if (!missing_required_packet_keys.empty()) {
      ++stats->missing_value_count;
      return fail_with_key("missing_value_for_key",
                           *missing_required_packet_keys.begin());
    }

    gtsam::FactorIndices remove_indices =
        packet.sidecar_remove_factor_indices;
    std::sort(remove_indices.begin(), remove_indices.end());
    remove_indices.erase(
        std::unique(remove_indices.begin(), remove_indices.end()),
        remove_indices.end());

    const gtsam::NonlinearFactorGraph& sidecar_graph_before_update =
        cbs_local_cov_sidecar_->getFactorsUnsafe();
    stats->sidecar_slot_map_size_before_update =
        cbs_h2_sidecar_heart_to_sidecar_slot_map_.size();
    stats->sidecar_graph_size_before_update = sidecar_graph_before_update.size();

    // Harden slot map before update: drop stale sidecar slots so we never
    // forward invalid remove indices into BPSAM.
    for (auto it = cbs_h2_sidecar_heart_to_sidecar_slot_map_.begin();
         it != cbs_h2_sidecar_heart_to_sidecar_slot_map_.end();) {
      const gtsam::FactorIndex heart_slot = it->first;
      const gtsam::FactorIndex sidecar_slot = it->second;
      if (!sidecar_graph_before_update.exists(sidecar_slot)) {
        ++stats->unmapped_remove_slot_count;
        ++stats->filtered_stale_remove_count;
        record_removed_stale_slot_map_entry(
            heart_slot,
            sidecar_slot,
            &stats->stale_slot_map_entries_removed_pre_update);
        if (stats->first_unmapped_heart_slot ==
            std::numeric_limits<gtsam::FactorIndex>::max()) {
          stats->first_unmapped_heart_slot = heart_slot;
        }
        if (stats->first_unmapped_sidecar_slot ==
            std::numeric_limits<gtsam::FactorIndex>::max()) {
          stats->first_unmapped_sidecar_slot = sidecar_slot;
        }
        it = cbs_h2_sidecar_heart_to_sidecar_slot_map_.erase(it);
      } else {
        ++it;
      }
    }

    // Re-translate mapped heart remove slots against the current map/graph.
    std::unordered_set<gtsam::FactorIndex> translated_remove_slot_set;
    gtsam::FactorIndices translated_remove_indices;
    translated_remove_indices.reserve(packet.mapped_heart_remove_factor_slots.size() +
                                      remove_indices.size());
    for (const gtsam::FactorIndex heart_slot :
         packet.mapped_heart_remove_factor_slots) {
      auto map_it = cbs_h2_sidecar_heart_to_sidecar_slot_map_.find(heart_slot);
      if (map_it == cbs_h2_sidecar_heart_to_sidecar_slot_map_.end()) {
        ++stats->unmapped_remove_slot_count;
        if (stats->first_unmapped_heart_slot ==
            std::numeric_limits<gtsam::FactorIndex>::max()) {
          stats->first_unmapped_heart_slot = heart_slot;
        }
        continue;
      }
      const gtsam::FactorIndex sidecar_slot = map_it->second;
      if (!sidecar_graph_before_update.exists(sidecar_slot)) {
        ++stats->unmapped_remove_slot_count;
        ++stats->filtered_stale_remove_count;
        record_removed_stale_slot_map_entry(
            heart_slot,
            sidecar_slot,
            &stats->stale_slot_map_entries_removed_during_translation);
        if (stats->first_unmapped_heart_slot ==
            std::numeric_limits<gtsam::FactorIndex>::max()) {
          stats->first_unmapped_heart_slot = heart_slot;
        }
        if (stats->first_unmapped_sidecar_slot ==
            std::numeric_limits<gtsam::FactorIndex>::max()) {
          stats->first_unmapped_sidecar_slot = sidecar_slot;
        }
        cbs_h2_sidecar_heart_to_sidecar_slot_map_.erase(map_it);
        continue;
      }
      if (translated_remove_slot_set.insert(sidecar_slot).second) {
        translated_remove_indices.push_back(sidecar_slot);
      }
    }
    // Keep packet-translated slots only if they still exist.
    for (const gtsam::FactorIndex sidecar_slot : remove_indices) {
      if (sidecar_graph_before_update.exists(sidecar_slot)) {
        if (translated_remove_slot_set.insert(sidecar_slot).second) {
          translated_remove_indices.push_back(sidecar_slot);
        }
      } else {
        ++stats->unmapped_remove_slot_count;
        ++stats->filtered_stale_remove_count;
        if (stats->first_unmapped_sidecar_slot ==
            std::numeric_limits<gtsam::FactorIndex>::max()) {
          stats->first_unmapped_sidecar_slot = sidecar_slot;
        }
      }
    }
    std::sort(translated_remove_indices.begin(), translated_remove_indices.end());
    translated_remove_indices.erase(
        std::unique(translated_remove_indices.begin(),
                    translated_remove_indices.end()),
        translated_remove_indices.end());
    remove_indices.swap(translated_remove_indices);
    stats->translated_remove_count = remove_indices.size();
    if (!remove_indices.empty()) {
      stats->first_translated_sidecar_remove_slot = remove_indices.front();
    }
    stats->sample_remove_indices = indices_to_sample_csv(remove_indices);

    gtsam::ISAM2Result sidecar_result;
    bool remove_applied_in_primary_update = false;
    std::string update_error_reason;
    bool full_update_failure_after_insert = false;
    bool retry_update_failure_after_insert = false;
    bool no_smart_ab_attempted = false;
    bool no_smart_ab_succeeded = false;
    bool no_smart_ab_failure_after_insert = false;
    std::string no_smart_ab_failure_reason = "none";
    std::string no_smart_ab_mode = "none";
    size_t no_smart_ab_new_factor_indices_count = 0u;
    const size_t no_smart_replacement_ab_excluded_count =
        packet.local_factors.size() - no_smart_replacement_factors.size();
    const size_t no_smart_any_ab_excluded_count =
        packet.local_factors.size() - no_smart_any_factors.size();
    size_t no_smart_ab_excluded_count = 0u;
    size_t no_smart_ab_factor_count = packet.local_factors.size();
    auto run_sidecar_update =
        [&](const gtsam::NonlinearFactorGraph& update_factors,
            const bool enable_remove,
            bool* should_retry_without_remove,
            bool* failure_after_insert) -> bool {
      if (failure_after_insert) {
        *failure_after_insert = false;
      }
      cbs::BPSAM::UpdateParams update_params;
      if (enable_remove) {
        update_params.removeFactorIndices = remove_indices;
      }
      const size_t graph_size_before_attempt =
          cbs_local_cov_sidecar_->getFactorsUnsafe().size();
      try {
        sidecar_result = cbs_local_cov_sidecar_->update(update_factors,
                                                        packet.local_values,
                                                        update_params);
        if (enable_remove) {
          remove_applied_in_primary_update = true;
          stats->remove_count += remove_indices.size();
        }
        return true;
      } catch (const std::exception& e) {
        const std::string err_msg =
            e.what() ? std::string(e.what())
                     : std::string("h2_sidecar_update_std_exception");
        if (failure_after_insert) {
          *failure_after_insert =
              cbs_local_cov_sidecar_->getFactorsUnsafe().size() >
              graph_size_before_attempt;
        }
        if (err_msg.find("map::at") != std::string::npos) {
          if (enable_remove && should_retry_without_remove) {
            *should_retry_without_remove = true;
            return false;
          }
          update_error_reason = "map::at_primary_update";
          return false;
        }
        update_error_reason = err_msg;
        return false;
      } catch (...) {
        if (failure_after_insert) {
          *failure_after_insert =
              cbs_local_cov_sidecar_->getFactorsUnsafe().size() >
              graph_size_before_attempt;
        }
        update_error_reason = "h2_sidecar_update_unknown_exception";
        return false;
      }
    };

    bool retry_without_remove = false;
    const bool enable_remove_first_attempt = !remove_indices.empty();
    stats->primary_remove_attempted = enable_remove_first_attempt;
    if (stats->filtered_stale_remove_count > 0u && enable_remove_first_attempt) {
      stats->primary_remove_deferred_due_to_stale_map = true;
    }
    if (!run_sidecar_update(packet.local_factors,
                            enable_remove_first_attempt,
                            &retry_without_remove,
                            &full_update_failure_after_insert)) {
      if (retry_without_remove) {
        stats->primary_retry_attempted = true;
      }
      if (!(retry_without_remove &&
            run_sidecar_update(packet.local_factors,
                               false,
                               nullptr,
                               &retry_update_failure_after_insert))) {
        if (trace_focus_epoch) {
          const gtsam::NonlinearFactorGraph* no_smart_ab_factors = nullptr;
          if (no_smart_replacement_ab_excluded_count > 0u) {
            no_smart_ab_mode = "exclude_replacement_candidates";
            no_smart_ab_excluded_count = no_smart_replacement_ab_excluded_count;
            no_smart_ab_factors = &no_smart_replacement_factors;
          } else if (no_smart_any_ab_excluded_count > 0u) {
            no_smart_ab_mode = "exclude_all_smart";
            no_smart_ab_excluded_count = no_smart_any_ab_excluded_count;
            no_smart_ab_factors = &no_smart_any_factors;
          }
          if (no_smart_ab_factors) {
            no_smart_ab_factor_count = no_smart_ab_factors->size();
          }
          no_smart_ab_attempted = no_smart_ab_factors != nullptr;
          if (no_smart_ab_attempted) {
            no_smart_ab_failure_reason = "skipped_for_h2_nonsmart_probe";
          }
          std::cerr << std::setprecision(12)
                    << "[CBS][H2PrimaryABDiag]"
                    << " curr_kf_id=" << curr_kf_id
                    << " full_packet_failed=1"
                    << " full_packet_failure_reason="
                    << (update_error_reason.empty() ? "none"
                                                   : update_error_reason)
                    << " full_packet_failure_after_insert="
                    << ((full_update_failure_after_insert ||
                         retry_update_failure_after_insert)
                            ? 1
                            : 0)
                    << " no_smart_ab_attempted="
                    << (no_smart_ab_attempted ? 1 : 0)
                    << " no_smart_ab_excluded_count="
                    << no_smart_ab_excluded_count
                    << " no_smart_ab_mode=" << no_smart_ab_mode
                    << " no_smart_ab_factor_count="
                    << no_smart_ab_factor_count
                    << " no_smart_ab_succeeded="
                    << (no_smart_ab_succeeded ? 1 : 0)
                    << " no_smart_ab_new_factor_indices_count="
                    << no_smart_ab_new_factor_indices_count
                    << " no_smart_ab_failure_reason="
                    << (no_smart_ab_failure_reason.empty()
                            ? "none"
                            : no_smart_ab_failure_reason)
                    << " no_smart_ab_failure_after_insert="
                    << (no_smart_ab_failure_after_insert ? 1 : 0)
                    << " smart_replacement_candidate_count="
                    << smart_replacement_candidate_count
                    << " factors_all_keys_in_sidecar_count="
                    << local_factor_all_keys_in_sidecar_count
                    << " factors_with_packet_only_key_count="
                    << local_factor_packet_only_key_count
                    << " stale_slot_map_entries_removed_pre_update="
                    << stats->stale_slot_map_entries_removed_pre_update
                    << " stale_slot_map_entries_removed_during_translation="
                    << stats->stale_slot_map_entries_removed_during_translation
                    << " stale_slot_map_entries_removed_during_remap="
                    << stats->stale_slot_map_entries_removed_during_remap
                    << " first_removed_stale_heart_slot="
                    << (stats->first_removed_stale_heart_slot ==
                                std::numeric_limits<gtsam::FactorIndex>::max()
                            ? -1
                            : static_cast<long long>(
                                  stats->first_removed_stale_heart_slot))
                    << " first_removed_stale_sidecar_slot="
                    << (stats->first_removed_stale_sidecar_slot ==
                                std::numeric_limits<gtsam::FactorIndex>::max()
                            ? -1
                            : static_cast<long long>(
                                  stats->first_removed_stale_sidecar_slot))
                    << std::endl;
          if (nonsmart_probe_epoch) {
            bool probe_clone_available = true;
            std::string probe_clone_setup_reason = "none";
            gtsam::Values probe_base_values;
            gtsam::NonlinearFactorGraph probe_base_graph;
            try {
              probe_base_values = cbs_local_cov_sidecar_->calculateEstimate();
              probe_base_graph = cbs_local_cov_sidecar_->getFactorsUnsafe();
            } catch (const std::exception& e) {
              probe_clone_available = false;
              probe_clone_setup_reason =
                  e.what() ? std::string(e.what())
                           : std::string("probe_clone_setup_std_exception");
            } catch (...) {
              probe_clone_available = false;
              probe_clone_setup_reason = "probe_clone_setup_unknown_exception";
            }

            auto make_probe_sidecar = [&]() -> std::shared_ptr<cbs::BPSAM> {
              gtsam::ISAM2Params sidecar_isam_params;
              BackendParams::setIsam2Params(backend_params_, &sidecar_isam_params);

              cbs::BPSAM::Params sidecar_params;
              constexpr cbs::AgentId kKimeraAgentId =
                  static_cast<cbs::AgentId>('a');
              sidecar_params.robot_id = kKimeraAgentId;
              sidecar_params.sam_params_ = sidecar_isam_params;
              sidecar_params.enable_gkcm = FLAGS_cbs_enable_gkcm;
              sidecar_params.gbp_update_params.type =
                  gbp::GaussianMergeType::Contract;
              sidecar_params.gbp_update_params.metric_type =
                  gbp::MetricType::Hellinger;
              sidecar_params.gbp_update_params.contract_alpha =
                  static_cast<float>(FLAGS_cbs_belief_contract_alpha);
              sidecar_params.gbp_update_params.d_reset =
                  static_cast<float>(FLAGS_cbs_belief_d_reset);
              sidecar_params.gbp_update_params.gamma =
                  static_cast<float>(FLAGS_cbs_belief_gamma);
              return std::make_shared<cbs::BPSAM>(sidecar_params);
            };

            auto run_post_failure_probe =
                [&](const gtsam::NonlinearFactorGraph& probe_factors,
                    bool* attempted,
                    bool* succeeded,
                    std::string* reason,
                    bool* failure_after_insert) {
                  *attempted = true;
                  *succeeded = false;
                  *reason = "none";
                  *failure_after_insert = false;

                  if (probe_clone_available) {
                    std::shared_ptr<cbs::BPSAM> probe_sidecar;
                    try {
                      probe_sidecar = make_probe_sidecar();
                    } catch (const std::exception& e) {
                      *reason = e.what() ? std::string(e.what())
                                         : std::string(
                                               "probe_make_sidecar_std_exception");
                      return;
                    } catch (...) {
                      *reason = "probe_make_sidecar_unknown_exception";
                      return;
                    }

                    try {
                      cbs::BPSAM::UpdateParams bootstrap_params;
                      probe_sidecar->update(probe_base_graph,
                                            probe_base_values,
                                            bootstrap_params);
                    } catch (const std::exception& e) {
                      *reason =
                          std::string("probe_clone_bootstrap_failed:") +
                          (e.what() ? std::string(e.what())
                                    : std::string("std_exception"));
                      return;
                    } catch (...) {
                      *reason = "probe_clone_bootstrap_failed:unknown_exception";
                      return;
                    }

                    const size_t graph_size_before_probe =
                        probe_sidecar->getFactorsUnsafe().size();
                    try {
                      cbs::BPSAM::UpdateParams probe_params;
                      probe_sidecar->update(probe_factors,
                                            packet.local_values,
                                            probe_params);
                      *succeeded = true;
                    } catch (const std::exception& e) {
                      *reason = e.what() ? std::string(e.what())
                                         : std::string("probe_std_exception");
                      *failure_after_insert =
                          probe_sidecar->getFactorsUnsafe().size() >
                          graph_size_before_probe;
                    } catch (...) {
                      *reason = "probe_unknown_exception";
                      *failure_after_insert =
                          probe_sidecar->getFactorsUnsafe().size() >
                          graph_size_before_probe;
                    }
                    return;
                  }

                  // Clone setup not feasible: run the minimum safe probes
                  // in-place after failure, just for localization, then reset.
                  const size_t graph_size_before_probe =
                      cbs_local_cov_sidecar_->getFactorsUnsafe().size();
                  try {
                    cbs::BPSAM::UpdateParams probe_params;
                    cbs_local_cov_sidecar_->update(probe_factors,
                                                   packet.local_values,
                                                   probe_params);
                    *succeeded = true;
                    *reason = "inplace_post_failure_probe";
                  } catch (const std::exception& e) {
                    *reason = e.what() ? std::string(e.what())
                                       : std::string("probe_std_exception");
                    *failure_after_insert =
                        cbs_local_cov_sidecar_->getFactorsUnsafe().size() >
                        graph_size_before_probe;
                  } catch (...) {
                    *reason = "probe_unknown_exception";
                    *failure_after_insert =
                        cbs_local_cov_sidecar_->getFactorsUnsafe().size() >
                        graph_size_before_probe;
                  }
                };

            bool empty_attempted = false;
            bool empty_succeeded = false;
            bool empty_failure_after_insert = false;
            std::string empty_reason = "none";

            bool imu_attempted = false;
            bool imu_succeeded = false;
            bool imu_failure_after_insert = false;
            std::string imu_reason = "none";

            bool between_attempted = false;
            bool between_succeeded = false;
            bool between_failure_after_insert = false;
            std::string between_reason = "none";

            bool non_smart_attempted = false;
            bool non_smart_succeeded = false;
            bool non_smart_failure_after_insert = false;
            std::string non_smart_reason = "none";

            const gtsam::NonlinearFactorGraph empty_factors;
            run_post_failure_probe(empty_factors,
                                   &empty_attempted,
                                   &empty_succeeded,
                                   &empty_reason,
                                   &empty_failure_after_insert);
            run_post_failure_probe(imu_only_factors,
                                   &imu_attempted,
                                   &imu_succeeded,
                                   &imu_reason,
                                   &imu_failure_after_insert);
            run_post_failure_probe(between_only_factors,
                                   &between_attempted,
                                   &between_succeeded,
                                   &between_reason,
                                   &between_failure_after_insert);
            run_post_failure_probe(no_smart_any_factors,
                                   &non_smart_attempted,
                                   &non_smart_succeeded,
                                   &non_smart_reason,
                                   &non_smart_failure_after_insert);

            std::cerr << std::setprecision(12)
                      << "[CBS][H2NonSmartABDiag]"
                      << " curr_kf_id=" << curr_kf_id
                      << " full_failed=1"
                      << " full_failure_reason="
                      << (update_error_reason.empty() ? "none"
                                                     : update_error_reason)
                      << " empty_attempted=" << (empty_attempted ? 1 : 0)
                      << " empty_succeeded=" << (empty_succeeded ? 1 : 0)
                      << " empty_reason="
                      << (empty_reason.empty() ? "none" : empty_reason)
                      << " imu_only_count=" << imu_only_factors.size()
                      << " imu_only_succeeded=" << (imu_succeeded ? 1 : 0)
                      << " imu_only_reason="
                      << (imu_reason.empty() ? "none" : imu_reason)
                      << " between_only_count=" << between_only_factors.size()
                      << " between_only_succeeded="
                      << (between_succeeded ? 1 : 0)
                      << " between_only_reason="
                      << (between_reason.empty() ? "none" : between_reason)
                      << " non_smart_only_count="
                      << no_smart_any_factors.size()
                      << " non_smart_only_succeeded="
                      << (non_smart_succeeded ? 1 : 0)
                      << " non_smart_only_reason="
                      << (non_smart_reason.empty() ? "none" : non_smart_reason)
                      << " full_failure_after_insert="
                      << ((full_update_failure_after_insert ||
                           retry_update_failure_after_insert)
                              ? 1
                              : 0)
                      << " empty_failure_after_insert="
                      << (empty_failure_after_insert ? 1 : 0)
                      << " imu_failure_after_insert="
                      << (imu_failure_after_insert ? 1 : 0)
                      << " between_failure_after_insert="
                      << (between_failure_after_insert ? 1 : 0)
                      << " non_smart_failure_after_insert="
                      << (non_smart_failure_after_insert ? 1 : 0)
                      << " smart_count=" << local_factor_smart_count
                      << " imu_count=" << local_factor_imu_count
                      << " between_count=" << local_factor_between_count
                      << " prior_count=" << local_factor_prior_count
                      << " other_count=" << local_factor_other_count
                      << " probe_clone_available="
                      << (probe_clone_available ? 1 : 0)
                      << " probe_clone_setup_reason="
                      << (probe_clone_setup_reason.empty()
                              ? "none"
                              : probe_clone_setup_reason)
                      << std::endl;
          }
        }
        return fail_and_hard_reset(update_error_reason.empty()
                                       ? "h2_sidecar_update_failed"
                                       : update_error_reason);
      }
      stats->retry_without_remove_factor_indices = true;
      stats->primary_retry_succeeded = true;
      stats->primary_remove_deferred_due_to_stale_map = true;
      stats->unmapped_remove_slot_count += remove_indices.size();
      LOG(WARNING) << "H2 sidecar update hit map::at with removeFactorIndices; "
                   << "retrying without removeFactorIndices for this epoch. "
                   << "Deferred remove slots=" << remove_indices.size();
    }

    if (remove_applied_in_primary_update) {
      for (const gtsam::FactorIndex heart_slot :
           packet.mapped_heart_remove_factor_slots) {
        cbs_h2_sidecar_heart_to_sidecar_slot_map_.erase(heart_slot);
      }
    }

    const auto& sidecar_new_indices = sidecar_result.newFactorsIndices;
    gtsam::FactorIndices resolved_sidecar_new_indices(sidecar_new_indices.begin(),
                                                      sidecar_new_indices.end());
    std::vector<size_t> resolved_heart_new_factor_positions =
        packet.heart_new_factor_positions;
    stats->sidecar_new_factor_indices_count = resolved_sidecar_new_indices.size();
    stats->sample_sidecar_new_factor_indices =
        indices_to_sample_csv(resolved_sidecar_new_indices);

    const size_t expected_sidecar_new_indices_count =
        packet.heart_new_factor_positions.size();
    if (expected_sidecar_new_indices_count == 0u &&
        !resolved_sidecar_new_indices.empty()) {
      std::ostringstream oss;
      oss << "h2_sidecar_new_index_mismatch sidecar="
          << resolved_sidecar_new_indices.size()
          << " local_packet=0";
      return fail_and_hard_reset(oss.str());
    }
    if (expected_sidecar_new_indices_count > 0u &&
        resolved_sidecar_new_indices.size() !=
            expected_sidecar_new_indices_count) {
      const bool repair_epoch_detected =
          resolved_sidecar_new_indices.size() <
          expected_sidecar_new_indices_count;
      if (!repair_epoch_detected) {
        std::ostringstream oss;
        oss << "h2_sidecar_new_index_mismatch sidecar="
            << resolved_sidecar_new_indices.size()
            << " local_packet=" << expected_sidecar_new_indices_count;
        return fail_and_hard_reset(oss.str());
      }

      stats->h2_repair_epoch_detected = true;
      stats->h2_repair_epoch_original_new_index_count =
          resolved_sidecar_new_indices.size();
      stats->h2_replay_add_only_attempted = true;

      stats->h2_replay_subset_smart_count = local_factor_smart_count;
      stats->h2_replay_subset_non_smart_count = no_smart_any_factors.size();
      stats->h2_replay_subset_imu_count = imu_only_factors.size();
      stats->h2_replay_subset_between_count = between_only_factors.size();

      gtsam::Values replay_probe_base_values;
      gtsam::NonlinearFactorGraph replay_probe_base_graph;
      bool replay_probe_base_available = false;
      std::string replay_probe_base_reason = "none";
      try {
        replay_probe_base_values = cbs_local_cov_sidecar_->calculateEstimate();
        replay_probe_base_graph = cbs_local_cov_sidecar_->getFactorsUnsafe();
        replay_probe_base_available = true;
      } catch (const std::exception& e) {
        replay_probe_base_reason =
            e.what() ? std::string(e.what())
                     : std::string("probe_clone_setup_std_exception");
      } catch (...) {
        replay_probe_base_reason = "probe_clone_setup_unknown_exception";
      }

      auto make_probe_sidecar = [&]() -> std::shared_ptr<cbs::BPSAM> {
        gtsam::ISAM2Params sidecar_isam_params;
        BackendParams::setIsam2Params(backend_params_, &sidecar_isam_params);

        cbs::BPSAM::Params sidecar_params;
        constexpr cbs::AgentId kKimeraAgentId = static_cast<cbs::AgentId>('a');
        sidecar_params.robot_id = kKimeraAgentId;
        sidecar_params.sam_params_ = sidecar_isam_params;
        sidecar_params.enable_gkcm = FLAGS_cbs_enable_gkcm;
        sidecar_params.gbp_update_params.type = gbp::GaussianMergeType::Contract;
        sidecar_params.gbp_update_params.metric_type = gbp::MetricType::Hellinger;
        sidecar_params.gbp_update_params.contract_alpha =
            static_cast<float>(FLAGS_cbs_belief_contract_alpha);
        sidecar_params.gbp_update_params.d_reset =
            static_cast<float>(FLAGS_cbs_belief_d_reset);
        sidecar_params.gbp_update_params.gamma =
            static_cast<float>(FLAGS_cbs_belief_gamma);
        return std::make_shared<cbs::BPSAM>(sidecar_params);
      };

      gtsam::ISAM2Result replay_add_only_result;
      bool replay_add_only_threw = false;
      std::string replay_reason = "none";
      try {
        cbs::BPSAM::UpdateParams replay_add_only_params;
        replay_add_only_result = cbs_local_cov_sidecar_->update(
            packet.local_factors, packet.local_values, replay_add_only_params);
      } catch (const std::exception& e) {
        replay_add_only_threw = true;
        replay_reason =
            e.what() ? std::string(e.what())
                     : std::string("h2_sidecar_replay_add_only_std_exception");
      } catch (...) {
        replay_add_only_threw = true;
        replay_reason = "h2_sidecar_replay_add_only_unknown_exception";
      }

      if (replay_add_only_threw) {
        stats->h2_replay_add_only_failure_reason = replay_reason;
        const bool replay_map_at =
            replay_reason.find("map::at") != std::string::npos;
        if (replay_map_at) {
          stats->h2_replay_subset_probe_attempted = true;
          stats->h2_replay_subset_probe_clone_available = replay_probe_base_available;
          stats->h2_replay_subset_probe_clone_setup_reason =
              replay_probe_base_reason.empty() ? "none" : replay_probe_base_reason;

          struct ReplayBootstrapAuditResult {
            size_t base_graph_size = 0u;
            size_t base_nonnull_factor_count = 0u;
            size_t base_null_factor_count = 0u;
            size_t base_values_count = 0u;
            size_t base_factor_with_missing_key_count = 0u;
            size_t base_smart_with_missing_key_count = 0u;
            size_t base_imu_with_missing_key_count = 0u;
            size_t base_between_with_missing_key_count = 0u;
            size_t base_prior_with_missing_key_count = 0u;
            size_t base_other_with_missing_key_count = 0u;
            gtsam::FactorIndex first_bad_base_slot =
                std::numeric_limits<gtsam::FactorIndex>::max();
            std::string first_bad_base_factor_type = "none";
            std::string first_missing_base_key = "none";
            gtsam::NonlinearFactorGraph sanitized_bootstrap_graph;
          };

          auto audit_bootstrap_snapshot = [&]() -> ReplayBootstrapAuditResult {
            ReplayBootstrapAuditResult audit;
            if (!replay_probe_base_available) {
              return audit;
            }

            audit.base_graph_size = replay_probe_base_graph.size();
            audit.base_values_count = replay_probe_base_values.size();
            audit.sanitized_bootstrap_graph.reserve(audit.base_graph_size);

            for (gtsam::FactorIndex slot = 0; slot < replay_probe_base_graph.size();
                 ++slot) {
              if (!replay_probe_base_graph.exists(slot)) {
                continue;
              }
              const auto& factor = replay_probe_base_graph.at(slot);
              if (!factor) {
                ++audit.base_null_factor_count;
                if (audit.first_bad_base_slot ==
                    std::numeric_limits<gtsam::FactorIndex>::max()) {
                  audit.first_bad_base_slot = slot;
                  audit.first_bad_base_factor_type = "null";
                }
                continue;
              }

              ++audit.base_nonnull_factor_count;
              bool has_missing_key = false;
              gtsam::Key first_missing_key =
                  std::numeric_limits<gtsam::Key>::max();
              for (const gtsam::Key key : factor->keys()) {
                if (!replay_probe_base_values.exists(key)) {
                  has_missing_key = true;
                  first_missing_key = key;
                  break;
                }
              }

              if (!has_missing_key) {
                audit.sanitized_bootstrap_graph.push_back(factor);
                continue;
              }

              ++audit.base_factor_with_missing_key_count;
              const std::string factor_type = classify_factor_type(factor.get());
              if (factor_type == "smart") {
                ++audit.base_smart_with_missing_key_count;
              } else if (factor_type == "imu") {
                ++audit.base_imu_with_missing_key_count;
              } else if (factor_type == "between") {
                ++audit.base_between_with_missing_key_count;
              } else if (factor_type == "prior") {
                ++audit.base_prior_with_missing_key_count;
              } else {
                ++audit.base_other_with_missing_key_count;
              }

              if (audit.first_bad_base_slot ==
                  std::numeric_limits<gtsam::FactorIndex>::max()) {
                audit.first_bad_base_slot = slot;
                audit.first_bad_base_factor_type = factor_type;
                if (first_missing_key != std::numeric_limits<gtsam::Key>::max()) {
                  audit.first_missing_base_key = key_to_string(first_missing_key);
                }
              }
            }
            return audit;
          };

          const ReplayBootstrapAuditResult replay_bootstrap_audit =
              audit_bootstrap_snapshot();

          auto run_bootstrap_probe_only =
              [&](const gtsam::NonlinearFactorGraph& bootstrap_graph,
                  bool* succeeded,
                  std::string* reason) {
                CHECK_NOTNULL(succeeded);
                CHECK_NOTNULL(reason);
                *succeeded = false;
                *reason = "not_run";
                if (!replay_probe_base_available) {
                  *reason = "probe_clone_unavailable";
                  return;
                }

                try {
                  std::shared_ptr<cbs::BPSAM> probe_sidecar = make_probe_sidecar();
                  cbs::BPSAM::UpdateParams bootstrap_params;
                  probe_sidecar->update(
                      bootstrap_graph, replay_probe_base_values, bootstrap_params);
                  *succeeded = true;
                  *reason = "none";
                } catch (const std::exception& e) {
                  *reason = e.what()
                                ? std::string(e.what())
                                : std::string("probe_bootstrap_std_exception");
                } catch (...) {
                  *reason = "probe_bootstrap_unknown_exception";
                }
              };

          bool raw_bootstrap_succeeded = false;
          std::string raw_bootstrap_reason = "not_run";
          run_bootstrap_probe_only(
              replay_probe_base_graph, &raw_bootstrap_succeeded, &raw_bootstrap_reason);

          const size_t sanitized_bootstrap_graph_size =
              replay_bootstrap_audit.sanitized_bootstrap_graph.size();
          bool sanitized_bootstrap_succeeded = false;
          std::string sanitized_bootstrap_reason = "not_run";
          run_bootstrap_probe_only(replay_bootstrap_audit.sanitized_bootstrap_graph,
                                   &sanitized_bootstrap_succeeded,
                                   &sanitized_bootstrap_reason);

          struct PrefixProbeResult {
            bool succeeded = false;
            std::string reason = "not_run";
          };
          std::map<size_t, PrefixProbeResult> sanitized_prefix_probe_cache;
          auto run_sanitized_bootstrap_prefix_probe =
              [&](const size_t requested_prefix_len) -> PrefixProbeResult {
            const size_t prefix_len =
                std::min(requested_prefix_len, sanitized_bootstrap_graph_size);
            const auto cache_it = sanitized_prefix_probe_cache.find(prefix_len);
            if (cache_it != sanitized_prefix_probe_cache.end()) {
              return cache_it->second;
            }
            PrefixProbeResult result;
            gtsam::NonlinearFactorGraph prefix_graph;
            prefix_graph.reserve(prefix_len);
            for (size_t idx = 0u; idx < prefix_len; ++idx) {
              if (!replay_bootstrap_audit.sanitized_bootstrap_graph.exists(idx)) {
                continue;
              }
              const auto& factor =
                  replay_bootstrap_audit.sanitized_bootstrap_graph.at(idx);
              if (factor) {
                prefix_graph.push_back(factor);
              }
            }
            run_bootstrap_probe_only(prefix_graph, &result.succeeded, &result.reason);
            sanitized_prefix_probe_cache.emplace(prefix_len, result);
            return result;
          };
          auto factor_keys_to_sample =
              [&](const gtsam::NonlinearFactor* factor,
                  const size_t max_items = 8u) -> std::string {
            if (!factor) {
              return "none";
            }
            const gtsam::KeyVector& keys = factor->keys();
            if (keys.empty()) {
              return "none";
            }
            std::ostringstream oss;
            for (size_t i = 0u; i < keys.size() && i < max_items; ++i) {
              if (i > 0u) {
                oss << ",";
              }
              oss << key_to_string(keys[i]);
            }
            if (keys.size() > max_items) {
              oss << ",...(" << (keys.size() - max_items) << "_more)";
            }
            return oss.str();
          };

          long long replay_bootstrap_last_success_prefix_len = -1;
          long long replay_bootstrap_first_failing_prefix_len = -1;
          long long replay_bootstrap_first_failing_factor_local_index = -1;
          std::string replay_bootstrap_first_failing_factor_type = "none";
          std::string replay_bootstrap_first_failing_factor_keys = "none";
          std::string replay_bootstrap_failing_window_types = "none";
          std::string replay_bootstrap_failing_window_keys = "none";
          bool replay_bootstrap_single_factor_skip_succeeded = false;
          std::string replay_bootstrap_single_factor_skip_reason = "not_run";

          const PrefixProbeResult zero_prefix_result =
              run_sanitized_bootstrap_prefix_probe(0u);
          if (zero_prefix_result.succeeded) {
            replay_bootstrap_last_success_prefix_len = 0;
          }

          if (!sanitized_bootstrap_succeeded) {
            if (!zero_prefix_result.succeeded) {
              replay_bootstrap_first_failing_prefix_len = 0;
            } else if (sanitized_bootstrap_graph_size > 0u) {
              size_t low_success = 0u;
              size_t high_fail = sanitized_bootstrap_graph_size;
              while (low_success + 1u < high_fail) {
                const size_t mid = low_success + (high_fail - low_success) / 2u;
                const PrefixProbeResult mid_result =
                    run_sanitized_bootstrap_prefix_probe(mid);
                if (mid_result.succeeded) {
                  low_success = mid;
                } else {
                  high_fail = mid;
                }
              }
              replay_bootstrap_last_success_prefix_len =
                  static_cast<long long>(low_success);
              replay_bootstrap_first_failing_prefix_len =
                  static_cast<long long>(high_fail);

              const size_t local_window_start =
                  (high_fail > 2u) ? (high_fail - 2u) : 0u;
              const size_t local_window_end =
                  std::min(sanitized_bootstrap_graph_size, high_fail + 2u);
              for (size_t prefix_len = local_window_start;
                   prefix_len <= local_window_end;
                   ++prefix_len) {
                const PrefixProbeResult local_result =
                    run_sanitized_bootstrap_prefix_probe(prefix_len);
                if (!local_result.succeeded) {
                  replay_bootstrap_first_failing_prefix_len =
                      static_cast<long long>(prefix_len);
                  if (prefix_len == 0u) {
                    replay_bootstrap_last_success_prefix_len = -1;
                  } else {
                    const PrefixProbeResult previous_result =
                        run_sanitized_bootstrap_prefix_probe(prefix_len - 1u);
                    replay_bootstrap_last_success_prefix_len =
                        previous_result.succeeded
                            ? static_cast<long long>(prefix_len - 1u)
                            : replay_bootstrap_last_success_prefix_len;
                  }
                  break;
                }
              }
            }
          } else {
            replay_bootstrap_last_success_prefix_len =
                static_cast<long long>(sanitized_bootstrap_graph_size);
          }

          if (replay_bootstrap_first_failing_prefix_len > 0 &&
              replay_bootstrap_first_failing_prefix_len <=
                  static_cast<long long>(sanitized_bootstrap_graph_size)) {
            replay_bootstrap_first_failing_factor_local_index =
                replay_bootstrap_first_failing_prefix_len - 1;
          }

          if (replay_bootstrap_first_failing_factor_local_index >= 0 &&
              replay_bootstrap_first_failing_factor_local_index <
                  static_cast<long long>(sanitized_bootstrap_graph_size)) {
            const size_t failing_index = static_cast<size_t>(
                replay_bootstrap_first_failing_factor_local_index);
            const auto& failing_factor =
                replay_bootstrap_audit.sanitized_bootstrap_graph.at(failing_index);
            replay_bootstrap_first_failing_factor_type =
                classify_factor_type(failing_factor.get());
            replay_bootstrap_first_failing_factor_keys =
                factor_keys_to_sample(failing_factor.get());

            const size_t window_start =
                (failing_index > 2u) ? (failing_index - 2u) : 0u;
            const size_t window_end = std::min(
                sanitized_bootstrap_graph_size - 1u, failing_index + 2u);
            std::ostringstream type_oss;
            std::ostringstream key_oss;
            for (size_t idx = window_start; idx <= window_end; ++idx) {
              const auto& window_factor =
                  replay_bootstrap_audit.sanitized_bootstrap_graph.at(idx);
              if (idx > window_start) {
                type_oss << "|";
                key_oss << "|";
              }
              type_oss << idx << ":" << classify_factor_type(window_factor.get());
              key_oss << idx << ":" << factor_keys_to_sample(window_factor.get());
            }
            replay_bootstrap_failing_window_types = type_oss.str().empty()
                                                        ? "none"
                                                        : type_oss.str();
            replay_bootstrap_failing_window_keys = key_oss.str().empty()
                                                       ? "none"
                                                       : key_oss.str();

            gtsam::NonlinearFactorGraph skip_single_graph;
            skip_single_graph.reserve(
                replay_bootstrap_audit.sanitized_bootstrap_graph.size());
            for (size_t idx = 0u;
                 idx < replay_bootstrap_audit.sanitized_bootstrap_graph.size();
                 ++idx) {
              if (idx == failing_index ||
                  !replay_bootstrap_audit.sanitized_bootstrap_graph.exists(idx)) {
                continue;
              }
              const auto& factor =
                  replay_bootstrap_audit.sanitized_bootstrap_graph.at(idx);
              if (factor) {
                skip_single_graph.push_back(factor);
              }
            }
            run_bootstrap_probe_only(skip_single_graph,
                                     &replay_bootstrap_single_factor_skip_succeeded,
                                     &replay_bootstrap_single_factor_skip_reason);
          }

          const bool use_sanitized_bootstrap_for_stage_probes =
              !raw_bootstrap_succeeded && sanitized_bootstrap_succeeded;

          const auto factor_index_to_log = [](const gtsam::FactorIndex idx)
              -> long long {
            if (idx == std::numeric_limits<gtsam::FactorIndex>::max()) {
              return -1;
            }
            return static_cast<long long>(idx);
          };

          std::cerr << std::setprecision(12)
                    << "[CBS][H2ReplayBootstrapABDiag]"
                    << " curr_kf_id=" << curr_kf_id
                    << " base_graph_size=" << replay_bootstrap_audit.base_graph_size
                    << " base_nonnull_factor_count="
                    << replay_bootstrap_audit.base_nonnull_factor_count
                    << " base_null_factor_count="
                    << replay_bootstrap_audit.base_null_factor_count
                    << " base_values_count="
                    << replay_bootstrap_audit.base_values_count
                    << " base_factor_with_missing_key_count="
                    << replay_bootstrap_audit.base_factor_with_missing_key_count
                    << " base_smart_with_missing_key_count="
                    << replay_bootstrap_audit.base_smart_with_missing_key_count
                    << " base_imu_with_missing_key_count="
                    << replay_bootstrap_audit.base_imu_with_missing_key_count
                    << " base_between_with_missing_key_count="
                    << replay_bootstrap_audit.base_between_with_missing_key_count
                    << " base_prior_with_missing_key_count="
                    << replay_bootstrap_audit.base_prior_with_missing_key_count
                    << " base_other_with_missing_key_count="
                    << replay_bootstrap_audit.base_other_with_missing_key_count
                    << " first_bad_base_slot="
                    << factor_index_to_log(replay_bootstrap_audit.first_bad_base_slot)
                    << " first_bad_base_factor_type="
                    << (replay_bootstrap_audit.first_bad_base_factor_type.empty()
                            ? "none"
                            : replay_bootstrap_audit.first_bad_base_factor_type)
                    << " first_missing_base_key="
                    << (replay_bootstrap_audit.first_missing_base_key.empty()
                            ? "none"
                            : replay_bootstrap_audit.first_missing_base_key)
                    << " raw_bootstrap_succeeded="
                    << (raw_bootstrap_succeeded ? 1 : 0)
                    << " raw_bootstrap_reason="
                    << (raw_bootstrap_reason.empty() ? "none" : raw_bootstrap_reason)
                    << " sanitized_bootstrap_graph_size="
                    << sanitized_bootstrap_graph_size
                    << " sanitized_bootstrap_succeeded="
                    << (sanitized_bootstrap_succeeded ? 1 : 0)
                    << " sanitized_bootstrap_reason="
                    << (sanitized_bootstrap_reason.empty()
                            ? "none"
                            : sanitized_bootstrap_reason)
                    << std::endl;
          std::cerr << std::setprecision(12)
                    << "[CBS][H2ReplayBootstrapBisectDiag]"
                    << " curr_kf_id=" << curr_kf_id
                    << " sanitized_bootstrap_graph_size="
                    << sanitized_bootstrap_graph_size
                    << " full_sanitized_bootstrap_succeeded="
                    << (sanitized_bootstrap_succeeded ? 1 : 0)
                    << " full_sanitized_bootstrap_reason="
                    << (sanitized_bootstrap_reason.empty()
                            ? "none"
                            : sanitized_bootstrap_reason)
                    << " last_success_prefix_len="
                    << replay_bootstrap_last_success_prefix_len
                    << " first_failing_prefix_len="
                    << replay_bootstrap_first_failing_prefix_len
                    << " first_failing_factor_local_index="
                    << replay_bootstrap_first_failing_factor_local_index
                    << " first_failing_factor_type="
                    << (replay_bootstrap_first_failing_factor_type.empty()
                            ? "none"
                            : replay_bootstrap_first_failing_factor_type)
                    << " first_failing_factor_keys="
                    << (replay_bootstrap_first_failing_factor_keys.empty()
                            ? "none"
                            : replay_bootstrap_first_failing_factor_keys)
                    << " failing_window_types="
                    << (replay_bootstrap_failing_window_types.empty()
                            ? "none"
                            : replay_bootstrap_failing_window_types)
                    << " failing_window_keys="
                    << (replay_bootstrap_failing_window_keys.empty()
                            ? "none"
                            : replay_bootstrap_failing_window_keys)
                    << " single_factor_skip_succeeded="
                    << (replay_bootstrap_single_factor_skip_succeeded ? 1 : 0)
                    << " single_factor_skip_reason="
                    << (replay_bootstrap_single_factor_skip_reason.empty()
                            ? "none"
                            : replay_bootstrap_single_factor_skip_reason)
                    << std::endl;

          struct BootstrapWindowProbeResult {
            bool succeeded = false;
            std::string reason = "not_run";
          };
          long long replay_bootstrap_window_fail_idx = -1;
          long long replay_bootstrap_window_good_prefix_len = -1;
          long long replay_bootstrap_window_bad_prefix_len = -1;
          long long replay_bootstrap_window_start = -1;
          long long replay_bootstrap_window_end = -1;
          bool replay_bootstrap_window_baseline_good_prefix_succeeded = false;
          std::string replay_bootstrap_window_baseline_good_prefix_reason =
              "not_run";
          bool replay_bootstrap_window_baseline_bad_prefix_succeeded = false;
          std::string replay_bootstrap_window_baseline_bad_prefix_reason =
              "not_run";
          std::string replay_bootstrap_window_single_remove_success_mask = "none";
          std::string replay_bootstrap_window_adjacent_pair_success_mask = "none";
          bool replay_bootstrap_window_full_window_removed_succeeded = false;
          std::string replay_bootstrap_window_full_window_removed_reason =
              "not_run";
          bool replay_bootstrap_window_only_window_succeeded = false;
          std::string replay_bootstrap_window_only_window_reason = "not_run";
          bool replay_bootstrap_window_window_appended_succeeded = false;
          std::string replay_bootstrap_window_window_appended_reason = "not_run";
          bool replay_bootstrap_window_window_reversed_succeeded = false;
          std::string replay_bootstrap_window_window_reversed_reason = "not_run";
          std::string replay_bootstrap_window_first_success_variant = "none";
          std::string replay_bootstrap_window_first_success_reason = "none";
          std::string replay_bootstrap_window_local_window_types = "none";
          std::string replay_bootstrap_window_local_window_keys = "none";
          std::string replay_bootstrap_append_single_success_mask = "none";
          std::string replay_bootstrap_append_single_reason_mask = "none";
          std::string replay_bootstrap_append_adjacent_pair_success_mask =
              "none";
          std::string replay_bootstrap_append_adjacent_pair_reason_mask = "none";
          std::string replay_bootstrap_append_triple_success_mask = "none";
          std::string replay_bootstrap_append_triple_reason_mask = "none";
          bool replay_bootstrap_append_whole_window_succeeded = false;
          std::string replay_bootstrap_append_whole_window_reason = "not_run";
          bool replay_bootstrap_append_whole_minus_fail_succeeded = false;
          std::string replay_bootstrap_append_whole_minus_fail_reason = "not_run";
          bool replay_bootstrap_append_only_fail_succeeded = false;
          std::string replay_bootstrap_append_only_fail_reason = "not_run";
          bool replay_bootstrap_append_smart_only_succeeded = false;
          std::string replay_bootstrap_append_smart_only_reason = "not_run";
          bool replay_bootstrap_append_non_smart_only_succeeded = false;
          std::string replay_bootstrap_append_non_smart_only_reason = "not_run";
          bool replay_bootstrap_append_fail_front_succeeded = false;
          std::string replay_bootstrap_append_fail_front_reason = "not_run";
          bool replay_bootstrap_append_fail_end_succeeded = false;
          std::string replay_bootstrap_append_fail_end_reason = "not_run";
          std::string replay_factor_243_type = "none";
          std::string replay_factor_243_keys = "none";
          long long replay_factor_243_key_count = -1;
          long long replay_factor_243_measured_count = -1;
          int replay_factor_243_is_smart = 0;
          std::string replay_factor_243_pointer = "none";
          int replay_factor_243_point_valid = -1;
          std::string replay_factor_244_type = "none";
          std::string replay_factor_244_keys = "none";
          long long replay_factor_244_key_count = -1;
          long long replay_factor_244_measured_count = -1;
          int replay_factor_244_is_smart = 0;
          std::string replay_factor_244_pointer = "none";
          int replay_factor_244_point_valid = -1;
          std::string replay_factor_245_type = "none";
          std::string replay_factor_245_keys = "none";
          long long replay_factor_245_key_count = -1;
          long long replay_factor_245_measured_count = -1;
          int replay_factor_245_is_smart = 0;
          std::string replay_factor_245_pointer = "none";
          int replay_factor_245_point_valid = -1;
          std::string replay_factor_246_type = "none";
          std::string replay_factor_246_keys = "none";
          long long replay_factor_246_key_count = -1;
          long long replay_factor_246_measured_count = -1;
          int replay_factor_246_is_smart = 0;
          std::string replay_factor_246_pointer = "none";
          int replay_factor_246_point_valid = -1;
          int replay_factor_244_245_same_keys = 0;
          int replay_factor_243_244_same_keys = 0;
          int replay_factor_245_246_same_keys = 0;
          std::string replay_smart_cluster_factor_diag = "none";
          std::string replay_smart_cluster_window_type_counts = "none";
          std::string replay_smart_cluster_window_order_signature = "none";
          int replay_smart_cluster_window_repeated_key_signature = 0;
          int replay_smart_cluster_window_repeated_measure_signature = 0;
          std::string replay_smart_cluster_factor_244_key_signature = "none";
          std::string replay_smart_cluster_factor_245_key_signature = "none";
          std::string replay_smart_cluster_factor_244_measure_signature = "none";
          std::string replay_smart_cluster_factor_245_measure_signature = "none";
          std::string replay_smart_cluster_factor_244_noise_signature = "none";
          std::string replay_smart_cluster_factor_245_noise_signature = "none";
          long long replay_smart_cluster_factor_244_key_match_count = 0;
          long long replay_smart_cluster_factor_245_key_match_count = 0;
          long long replay_smart_cluster_factor_244_measure_match_count = 0;
          long long replay_smart_cluster_factor_245_measure_match_count = 0;
          std::string replay_smart_cluster_factor_244_key_match_indices = "none";
          std::string replay_smart_cluster_factor_245_key_match_indices = "none";
          std::string replay_smart_cluster_factor_244_measure_match_indices =
              "none";
          std::string replay_smart_cluster_factor_245_measure_match_indices =
              "none";
          std::vector<size_t>
              replay_smart_cluster_factor_244_key_match_index_list;
          std::vector<size_t>
              replay_smart_cluster_factor_245_key_match_index_list;
          int replay_smart_cluster_244_245_same_key_signature = 0;
          int replay_smart_cluster_244_245_same_measure_signature = 0;
          int replay_smart_cluster_244_245_same_noise_signature = 0;
          int replay_smart_cluster_order_any_success = 0;
          std::string replay_smart_cluster_order_success_mask = "none";
          std::string replay_smart_cluster_order_reason_mask = "none";
          std::string replay_smart_cluster_order_first_success_variant = "none";
          std::string replay_smart_cluster_order_first_success_reason = "none";
          std::string replay_family244_single_append_success_mask = "none";
          std::string replay_family244_single_append_reason_mask = "none";
          std::string replay_family245_single_append_success_mask = "none";
          std::string replay_family245_single_append_reason_mask = "none";
          std::string replay_family244_standalone_success_mask = "none";
          std::string replay_family244_standalone_reason_mask = "none";
          std::string replay_family245_standalone_success_mask = "none";
          std::string replay_family245_standalone_reason_mask = "none";
          int replay_family244_any_append_success = 0;
          int replay_family245_any_append_success = 0;
          int replay_family244_all_append_fail = 0;
          int replay_family245_all_append_fail = 0;
          long long replay_family244_first_append_success_idx = -1;
          long long replay_family245_first_append_success_idx = -1;
          long long replay_family244_first_append_fail_idx = -1;
          long long replay_family245_first_append_fail_idx = -1;
          long long replay_append_frontier_idx = -1;
          long long replay_append_frontier_idx0 = -1;
          long long replay_append_frontier_idx1 = -1;
          long long replay_append_frontier_idx2 = -1;
          std::string replay_append_frontier_idx0_type = "none";
          std::string replay_append_frontier_idx1_type = "none";
          std::string replay_append_frontier_idx2_type = "none";
          std::string replay_append_frontier_idx0_keysig = "none";
          std::string replay_append_frontier_idx1_keysig = "none";
          std::string replay_append_frontier_idx2_keysig = "none";
          std::string replay_append_frontier_idx0_factor_keys = "none";
          std::string replay_append_frontier_idx1_factor_keys = "none";
          std::string replay_append_frontier_idx2_factor_keys = "none";
          std::string replay_append_frontier_idx0_meassig = "none";
          std::string replay_append_frontier_idx1_meassig = "none";
          std::string replay_append_frontier_idx2_meassig = "none";
          std::string replay_append_frontier_idx0_pointer = "none";
          std::string replay_append_frontier_idx1_pointer = "none";
          std::string replay_append_frontier_idx2_pointer = "none";
          std::string replay_append_frontier_idx0_key_presence_mask = "none";
          std::string replay_append_frontier_idx1_key_presence_mask = "none";
          std::string replay_append_frontier_idx2_key_presence_mask = "none";
          int replay_append_frontier_idx0_introduces_new_key = 0;
          int replay_append_frontier_idx1_introduces_new_key = 0;
          int replay_append_frontier_idx2_introduces_new_key = 0;
          int replay_append_frontier_idx0_all_keys_in_good_prefix_values = 0;
          int replay_append_frontier_idx1_all_keys_in_good_prefix_values = 0;
          int replay_append_frontier_idx2_all_keys_in_good_prefix_values = 0;
          long long replay_append_frontier_idx0_measured_count = -1;
          long long replay_append_frontier_idx1_measured_count = -1;
          long long replay_append_frontier_idx2_measured_count = -1;
          int replay_append_frontier_idx0_point_valid = -1;
          int replay_append_frontier_idx1_point_valid = -1;
          int replay_append_frontier_idx2_point_valid = -1;
          std::string replay_append_frontier_pair_union_key_summary = "none";
          long long replay_append_frontier_prefix_touch_count_idx0 = 0;
          long long replay_append_frontier_prefix_touch_count_idx1 = 0;
          std::string replay_append_frontier_prefix_touch_sample_idx0 = "none";
          std::string replay_append_frontier_prefix_touch_sample_idx1 = "none";
          int replay_append_frontier_prefix_has_imu_like_idx0 = 0;
          int replay_append_frontier_prefix_has_between_like_idx1 = 0;
          long long replay_prefix_smart_touch_idx0_count = 0;
          long long replay_prefix_smart_touch_idx1_count = 0;
          long long replay_prefix_smart_touch_both_count = 0;
          std::string replay_prefix_smart_touch_idx0_samples = "none";
          std::string replay_prefix_smart_touch_idx1_samples = "none";
          std::string replay_prefix_smart_touch_both_samples = "none";
          long long replay_prefix_same_keysig_as_idx0_count = 0;
          long long replay_prefix_same_keysig_as_idx1_count = 0;
          long long replay_prefix_same_meassig_as_idx0_count = 0;
          long long replay_prefix_same_meassig_as_idx1_count = 0;
          std::string replay_prefix_same_keysig_as_idx0_indices = "none";
          std::string replay_prefix_same_keysig_as_idx1_indices = "none";
          std::string replay_prefix_same_meassig_as_idx0_indices = "none";
          std::string replay_prefix_same_meassig_as_idx1_indices = "none";
          std::string replay_prefix_prev_m3_context = "none";
          std::string replay_prefix_prev_m2_context = "none";
          std::string replay_prefix_prev_m1_context = "none";
          int replay_append_frontier_idx0_idx1_same_keys = 0;
          int replay_append_frontier_idx1_idx2_same_keys = 0;
          int replay_append_frontier_idx0_idx2_same_keys = 0;
          int replay_append_frontier_idx0_idx1_same_measure = 0;
          int replay_append_frontier_idx1_idx2_same_measure = 0;
          int replay_append_frontier_idx0_idx2_same_measure = 0;
          bool replay_append_frontier_append_idx0_succeeded = false;
          std::string replay_append_frontier_append_idx0_reason = "not_run";
          bool replay_append_frontier_append_idx1_succeeded = false;
          std::string replay_append_frontier_append_idx1_reason = "not_run";
          bool replay_append_frontier_append_idx2_succeeded = false;
          std::string replay_append_frontier_append_idx2_reason = "not_run";
          bool replay_append_frontier_append_idx0_idx1_succeeded = false;
          std::string replay_append_frontier_append_idx0_idx1_reason = "not_run";
          bool replay_append_frontier_append_idx1_idx0_succeeded = false;
          std::string replay_append_frontier_append_idx1_idx0_reason = "not_run";
          bool replay_append_frontier_append_idx0_idx1_idx2_succeeded = false;
          std::string replay_append_frontier_append_idx0_idx1_idx2_reason =
              "not_run";
          bool replay_append_frontier_append_idx1_idx2_succeeded = false;
          std::string replay_append_frontier_append_idx1_idx2_reason = "not_run";
          bool replay_append_frontier_append_idx0_idx2_succeeded = false;
          std::string replay_append_frontier_append_idx0_idx2_reason = "not_run";
          bool replay_append_frontier_standalone_idx0_succeeded = false;
          std::string replay_append_frontier_standalone_idx0_reason = "not_run";
          bool replay_append_frontier_standalone_idx1_succeeded = false;
          std::string replay_append_frontier_standalone_idx1_reason = "not_run";
          bool replay_append_frontier_standalone_idx2_succeeded = false;
          std::string replay_append_frontier_standalone_idx2_reason = "not_run";
          bool replay_factor_clone_orig_244_succeeded = false;
          std::string replay_factor_clone_orig_244_reason = "not_run";
          bool replay_factor_clone_orig_245_succeeded = false;
          std::string replay_factor_clone_orig_245_reason = "not_run";
          bool replay_factor_clone_clone_244_attempted = false;
          bool replay_factor_clone_clone_244_succeeded = false;
          std::string replay_factor_clone_clone_244_reason = "not_run";
          bool replay_factor_clone_clone_245_attempted = false;
          bool replay_factor_clone_clone_245_succeeded = false;
          std::string replay_factor_clone_clone_245_reason = "not_run";
          bool replay_factor_clone_orig_244_245_pair_succeeded = false;
          std::string replay_factor_clone_orig_244_245_pair_reason = "not_run";
          bool replay_factor_clone_clone_244_245_pair_attempted = false;
          bool replay_factor_clone_clone_244_245_pair_succeeded = false;
          std::string replay_factor_clone_clone_244_245_pair_reason = "not_run";

          auto run_bootstrap_window_probe =
              [&](const std::vector<size_t>& ordered_indices)
              -> BootstrapWindowProbeResult {
            BootstrapWindowProbeResult result;
            gtsam::NonlinearFactorGraph probe_graph;
            probe_graph.reserve(ordered_indices.size());
            for (const size_t idx : ordered_indices) {
              if (idx >= replay_bootstrap_audit.sanitized_bootstrap_graph.size() ||
                  !replay_bootstrap_audit.sanitized_bootstrap_graph.exists(idx)) {
                continue;
              }
              const auto& factor =
                  replay_bootstrap_audit.sanitized_bootstrap_graph.at(idx);
              if (factor) {
                probe_graph.push_back(factor);
              }
            }
            run_bootstrap_probe_only(probe_graph, &result.succeeded, &result.reason);
            return result;
          };
          auto collect_prefix_indices = [](const size_t prefix_len)
              -> std::vector<size_t> {
            std::vector<size_t> indices;
            indices.reserve(prefix_len);
            for (size_t idx = 0u; idx < prefix_len; ++idx) {
              indices.push_back(idx);
            }
            return indices;
          };
          auto maybe_record_window_first_success =
              [&](const std::string& variant_name,
                  const BootstrapWindowProbeResult& result) {
            if (replay_bootstrap_window_first_success_variant == "none" &&
                result.succeeded) {
              replay_bootstrap_window_first_success_variant = variant_name;
              replay_bootstrap_window_first_success_reason = result.reason.empty()
                                                                 ? "none"
                                                                 : result.reason;
            }
          };
          auto factor_ptr_to_string = [](const gtsam::NonlinearFactor* factor)
              -> std::string {
            if (!factor) {
              return "null";
            }
            std::ostringstream oss;
            oss << static_cast<const void*>(factor);
            return oss.str();
          };
          auto factor_keys_sorted = [&](const gtsam::NonlinearFactor* factor)
              -> std::vector<gtsam::Key> {
            std::vector<gtsam::Key> keys;
            if (!factor) {
              return keys;
            }
            const gtsam::KeyVector factor_keys = factor->keys();
            keys.assign(factor_keys.begin(), factor_keys.end());
            std::sort(keys.begin(), keys.end());
            return keys;
          };
          struct SmartMeasurementSummary {
            long long measured_count = -1;
            std::string measurement_signature = "na";
            int point_valid = -1;
          };
          const auto join_index_list =
              [](const std::vector<size_t>& indices) -> std::string {
            if (indices.empty()) {
              return "none";
            }
            std::ostringstream oss;
            for (size_t i = 0u; i < indices.size(); ++i) {
              if (i > 0u) {
                oss << ",";
              }
              oss << indices[i];
            }
            return oss.str();
          };
          auto key_signature_sorted = [&](const gtsam::NonlinearFactor* factor)
              -> std::string {
            if (!factor) {
              return "none";
            }
            std::vector<gtsam::Key> keys = factor_keys_sorted(factor);
            if (keys.empty()) {
              return "none";
            }
            std::ostringstream oss;
            for (size_t i = 0u; i < keys.size(); ++i) {
              if (i > 0u) {
                oss << ",";
              }
              oss << key_to_string(keys[i]);
            }
            return oss.str();
          };
          const auto fnv1a_mix_u64 = [](uint64_t* hash, const uint64_t value) {
            CHECK_NOTNULL(hash);
            constexpr uint64_t kFnvPrime = 1099511628211ull;
            for (size_t b = 0u; b < sizeof(value); ++b) {
              const uint8_t byte =
                  static_cast<uint8_t>((value >> (8u * b)) & 0xffu);
              *hash ^= static_cast<uint64_t>(byte);
              *hash *= kFnvPrime;
            }
          };
          const auto quantize_double_for_signature = [](const double value)
              -> long long {
            if (!std::isfinite(value)) {
              return (value >= 0.0) ? std::numeric_limits<long long>::max()
                                    : std::numeric_limits<long long>::min();
            }
            return static_cast<long long>(std::llround(value * 1e9));
          };
          auto summarize_smart_measurement =
              [&](const gtsam::NonlinearFactor* factor)
              -> SmartMeasurementSummary {
            SmartMeasurementSummary summary;
            const auto* smart_factor =
                dynamic_cast<const SmartStereoFactor*>(factor);
            if (!smart_factor) {
              return summary;
            }
            summary.measured_count =
                static_cast<long long>(smart_factor->measured().size());
            summary.point_valid = smart_factor->point().valid() ? 1 : 0;
            uint64_t hash = 1469598103934665603ull;
            for (const auto& measurement : smart_factor->measured()) {
              const long long q_u_l =
                  quantize_double_for_signature(measurement.uL());
              const long long q_u_r =
                  quantize_double_for_signature(measurement.uR());
              const long long q_v =
                  quantize_double_for_signature(measurement.v());
              fnv1a_mix_u64(&hash, static_cast<uint64_t>(q_u_l));
              fnv1a_mix_u64(&hash, static_cast<uint64_t>(q_u_r));
              fnv1a_mix_u64(&hash, static_cast<uint64_t>(q_v));
            }
            std::ostringstream oss;
            oss << "n" << summary.measured_count << "_h" << std::hex << hash;
            summary.measurement_signature = oss.str();
            return summary;
          };
          auto summarize_noise_signature =
              [&](const gtsam::NonlinearFactor* factor) -> std::string {
            const auto* noise_model_factor =
                dynamic_cast<const gtsam::NoiseModelFactor*>(factor);
            if (!noise_model_factor || !noise_model_factor->noiseModel()) {
              return "none";
            }
            const auto noise_model = noise_model_factor->noiseModel();
            std::string model_type = "unknown";
            gtsam::noiseModel::Gaussian::shared_ptr gaussian_model;
            if (auto robust_model =
                    boost::dynamic_pointer_cast<gtsam::noiseModel::Robust>(
                        noise_model)) {
              model_type = "robust";
              const auto wrapped_model = robust_model->noise();
              if (boost::dynamic_pointer_cast<gtsam::noiseModel::Isotropic>(
                      wrapped_model)) {
                model_type += "_isotropic";
              } else if (boost::dynamic_pointer_cast<
                             gtsam::noiseModel::Diagonal>(wrapped_model)) {
                model_type += "_diagonal";
              } else if (boost::dynamic_pointer_cast<
                             gtsam::noiseModel::Gaussian>(wrapped_model)) {
                model_type += "_gaussian";
              } else {
                model_type += "_other";
              }
              gaussian_model = boost::dynamic_pointer_cast<
                  gtsam::noiseModel::Gaussian>(wrapped_model);
            } else if (boost::dynamic_pointer_cast<
                           gtsam::noiseModel::Isotropic>(noise_model)) {
              model_type = "isotropic";
              gaussian_model = boost::dynamic_pointer_cast<
                  gtsam::noiseModel::Gaussian>(noise_model);
            } else if (boost::dynamic_pointer_cast<gtsam::noiseModel::Diagonal>(
                           noise_model)) {
              model_type = "diagonal";
              gaussian_model = boost::dynamic_pointer_cast<
                  gtsam::noiseModel::Gaussian>(noise_model);
            } else if (boost::dynamic_pointer_cast<gtsam::noiseModel::Gaussian>(
                           noise_model)) {
              model_type = "gaussian";
              gaussian_model = boost::dynamic_pointer_cast<
                  gtsam::noiseModel::Gaussian>(noise_model);
            }
            uint64_t sigma_hash = 1469598103934665603ull;
            long long sigma_count = -1;
            if (gaussian_model) {
              const gtsam::Vector sigmas = gaussian_model->sigmas();
              sigma_count = static_cast<long long>(sigmas.size());
              for (long long i = 0; i < sigma_count; ++i) {
                const long long q_sigma =
                    quantize_double_for_signature(sigmas(i));
                fnv1a_mix_u64(&sigma_hash, static_cast<uint64_t>(q_sigma));
              }
            }
            std::ostringstream oss;
            oss << model_type << "_d" << noise_model->dim();
            if (sigma_count >= 0) {
              oss << "_n" << sigma_count << "_h" << std::hex << sigma_hash;
            }
            return oss.str();
          };
          auto build_cluster_factor_token =
              [&](const size_t index,
                  const gtsam::NonlinearFactor::shared_ptr& factor) -> std::string {
            if (!factor) {
              std::ostringstream oss;
              oss << index << "{missing=1}";
              return oss.str();
            }
            const SmartMeasurementSummary smart_summary =
                summarize_smart_measurement(factor.get());
            std::ostringstream oss;
            oss << index << "{type=" << classify_factor_type(factor.get())
                << ";keysig=" << key_signature_sorted(factor.get())
                << ";measured_count=" << smart_summary.measured_count
                << ";measurement_sig=" << smart_summary.measurement_signature
                << ";noise_sig=" << summarize_noise_signature(factor.get())
                << ";pointer=" << factor_ptr_to_string(factor.get())
                << ";point_valid=" << smart_summary.point_valid << "}";
            return oss.str();
          };
          auto get_sanitized_factor_at = [&](const size_t index)
              -> gtsam::NonlinearFactor::shared_ptr {
            if (index >= replay_bootstrap_audit.sanitized_bootstrap_graph.size() ||
                !replay_bootstrap_audit.sanitized_bootstrap_graph.exists(index)) {
              return gtsam::NonlinearFactor::shared_ptr();
            }
            return replay_bootstrap_audit.sanitized_bootstrap_graph.at(index);
          };
          auto run_probe_from_factors =
              [&](const std::vector<gtsam::NonlinearFactor::shared_ptr>& factors)
              -> BootstrapWindowProbeResult {
            BootstrapWindowProbeResult result;
            gtsam::NonlinearFactorGraph probe_graph;
            probe_graph.reserve(factors.size());
            for (const auto& factor : factors) {
              if (factor) {
                probe_graph.push_back(factor);
              }
            }
            run_bootstrap_probe_only(probe_graph, &result.succeeded, &result.reason);
            if (result.reason.empty()) {
              result.reason = "none";
            }
            return result;
          };
          auto clone_smart_factor =
              [&](const gtsam::NonlinearFactor::shared_ptr& factor,
                  bool* attempted,
                  bool* succeeded,
                  std::string* reason) -> gtsam::NonlinearFactor::shared_ptr {
            CHECK_NOTNULL(attempted);
            CHECK_NOTNULL(succeeded);
            CHECK_NOTNULL(reason);
            *attempted = false;
            *succeeded = false;
            *reason = "not_run";
            if (!factor) {
              *reason = "missing_factor";
              return gtsam::NonlinearFactor::shared_ptr();
            }
            const auto* smart_factor =
                dynamic_cast<const SmartStereoFactor*>(factor.get());
            if (!smart_factor) {
              *reason = "not_smart_factor";
              return gtsam::NonlinearFactor::shared_ptr();
            }
            *attempted = true;
            try {
              SmartStereoFactor::shared_ptr smart_clone(
                  new SmartStereoFactor(*smart_factor));
              *succeeded = true;
              *reason = "none";
              return smart_clone;
            } catch (const std::exception& e) {
              *reason = e.what() ? std::string(e.what())
                                 : std::string("clone_std_exception");
            } catch (...) {
              *reason = "clone_unknown_exception";
            }
            return gtsam::NonlinearFactor::shared_ptr();
          };

          if (replay_bootstrap_first_failing_factor_local_index >= 0 &&
              replay_bootstrap_first_failing_factor_local_index <
                  static_cast<long long>(sanitized_bootstrap_graph_size)) {
            const size_t fail_idx = static_cast<size_t>(
                replay_bootstrap_first_failing_factor_local_index);
            const size_t good_prefix_len = fail_idx;
            const size_t bad_prefix_len = fail_idx + 1u;
            const size_t local_window_start =
                (fail_idx > 2u) ? (fail_idx - 2u) : 0u;
            const size_t local_window_end =
                std::min(sanitized_bootstrap_graph_size - 1u, fail_idx + 2u);

            replay_bootstrap_window_fail_idx = static_cast<long long>(fail_idx);
            replay_bootstrap_window_good_prefix_len =
                static_cast<long long>(good_prefix_len);
            replay_bootstrap_window_bad_prefix_len =
                static_cast<long long>(bad_prefix_len);
            replay_bootstrap_window_start =
                static_cast<long long>(local_window_start);
            replay_bootstrap_window_end =
                static_cast<long long>(local_window_end);

            std::ostringstream window_types_oss;
            std::ostringstream window_keys_oss;
            for (size_t idx = local_window_start; idx <= local_window_end; ++idx) {
              const auto& factor =
                  replay_bootstrap_audit.sanitized_bootstrap_graph.at(idx);
              if (idx > local_window_start) {
                window_types_oss << "|";
                window_keys_oss << "|";
              }
              window_types_oss << idx << ":" << classify_factor_type(factor.get());
              window_keys_oss << idx << ":" << factor_keys_to_sample(factor.get());
            }
            replay_bootstrap_window_local_window_types =
                window_types_oss.str().empty() ? "none" : window_types_oss.str();
            replay_bootstrap_window_local_window_keys =
                window_keys_oss.str().empty() ? "none" : window_keys_oss.str();

            const size_t inspect_idx_242 = 242u;
            const size_t inspect_idx_243 = 243u;
            const size_t inspect_idx_244 = 244u;
            const size_t inspect_idx_245 = 245u;
            const size_t inspect_idx_246 = 246u;
            const gtsam::NonlinearFactor::shared_ptr factor_242 =
                get_sanitized_factor_at(inspect_idx_242);
            const gtsam::NonlinearFactor::shared_ptr factor_243 =
                get_sanitized_factor_at(inspect_idx_243);
            const gtsam::NonlinearFactor::shared_ptr factor_244 =
                get_sanitized_factor_at(inspect_idx_244);
            const gtsam::NonlinearFactor::shared_ptr factor_245 =
                get_sanitized_factor_at(inspect_idx_245);
            const gtsam::NonlinearFactor::shared_ptr factor_246 =
                get_sanitized_factor_at(inspect_idx_246);

            auto fill_factor_object_diag =
                [&](const gtsam::NonlinearFactor::shared_ptr& factor,
                    std::string* factor_type,
                    std::string* factor_keys,
                    long long* factor_key_count,
                    long long* factor_measured_count,
                    int* factor_is_smart,
                    std::string* factor_pointer,
                    int* factor_point_valid) {
                  CHECK_NOTNULL(factor_type);
                  CHECK_NOTNULL(factor_keys);
                  CHECK_NOTNULL(factor_key_count);
                  CHECK_NOTNULL(factor_measured_count);
                  CHECK_NOTNULL(factor_is_smart);
                  CHECK_NOTNULL(factor_pointer);
                  CHECK_NOTNULL(factor_point_valid);
                  *factor_type = "none";
                  *factor_keys = "none";
                  *factor_key_count = -1;
                  *factor_measured_count = -1;
                  *factor_is_smart = 0;
                  *factor_pointer = "null";
                  *factor_point_valid = -1;
                  if (!factor) {
                    return;
                  }
                  *factor_type = classify_factor_type(factor.get());
                  *factor_keys = factor_keys_to_sample(factor.get());
                  const gtsam::KeyVector keys = factor->keys();
                  *factor_key_count = static_cast<long long>(keys.size());
                  *factor_pointer = factor_ptr_to_string(factor.get());
                  const auto* smart_factor =
                      dynamic_cast<const SmartStereoFactor*>(factor.get());
                  if (smart_factor) {
                    *factor_is_smart = 1;
                    *factor_measured_count =
                        static_cast<long long>(smart_factor->measured().size());
                    *factor_point_valid = smart_factor->point().valid() ? 1 : 0;
                  }
                };
            fill_factor_object_diag(factor_243,
                                    &replay_factor_243_type,
                                    &replay_factor_243_keys,
                                    &replay_factor_243_key_count,
                                    &replay_factor_243_measured_count,
                                    &replay_factor_243_is_smart,
                                    &replay_factor_243_pointer,
                                    &replay_factor_243_point_valid);
            fill_factor_object_diag(factor_244,
                                    &replay_factor_244_type,
                                    &replay_factor_244_keys,
                                    &replay_factor_244_key_count,
                                    &replay_factor_244_measured_count,
                                    &replay_factor_244_is_smart,
                                    &replay_factor_244_pointer,
                                    &replay_factor_244_point_valid);
            fill_factor_object_diag(factor_245,
                                    &replay_factor_245_type,
                                    &replay_factor_245_keys,
                                    &replay_factor_245_key_count,
                                    &replay_factor_245_measured_count,
                                    &replay_factor_245_is_smart,
                                    &replay_factor_245_pointer,
                                    &replay_factor_245_point_valid);
            fill_factor_object_diag(factor_246,
                                    &replay_factor_246_type,
                                    &replay_factor_246_keys,
                                    &replay_factor_246_key_count,
                                    &replay_factor_246_measured_count,
                                    &replay_factor_246_is_smart,
                                    &replay_factor_246_pointer,
                                    &replay_factor_246_point_valid);
            if (factor_244 && factor_245) {
              replay_factor_244_245_same_keys =
                  factor_keys_sorted(factor_244.get()) ==
                          factor_keys_sorted(factor_245.get())
                      ? 1
                      : 0;
            }
            if (factor_243 && factor_244) {
              replay_factor_243_244_same_keys =
                  factor_keys_sorted(factor_243.get()) ==
                          factor_keys_sorted(factor_244.get())
                      ? 1
                      : 0;
            }
            if (factor_245 && factor_246) {
              replay_factor_245_246_same_keys =
                  factor_keys_sorted(factor_245.get()) ==
                          factor_keys_sorted(factor_246.get())
                      ? 1
                      : 0;
            }

            {
              const std::vector<size_t> cluster_indices = {
                  inspect_idx_242, inspect_idx_243, inspect_idx_244,
                  inspect_idx_245, inspect_idx_246};
              std::ostringstream cluster_diag_oss;
              for (size_t c = 0u; c < cluster_indices.size(); ++c) {
                if (c > 0u) {
                  cluster_diag_oss << "|";
                }
                const size_t idx = cluster_indices[c];
                cluster_diag_oss
                    << build_cluster_factor_token(idx, get_sanitized_factor_at(idx));
              }
              replay_smart_cluster_factor_diag =
                  cluster_diag_oss.str().empty() ? "none" : cluster_diag_oss.str();

              replay_smart_cluster_factor_244_key_signature =
                  key_signature_sorted(factor_244.get());
              replay_smart_cluster_factor_245_key_signature =
                  key_signature_sorted(factor_245.get());
              replay_smart_cluster_factor_244_measure_signature =
                  summarize_smart_measurement(factor_244.get())
                      .measurement_signature;
              replay_smart_cluster_factor_245_measure_signature =
                  summarize_smart_measurement(factor_245.get())
                      .measurement_signature;
              replay_smart_cluster_factor_244_noise_signature =
                  summarize_noise_signature(factor_244.get());
              replay_smart_cluster_factor_245_noise_signature =
                  summarize_noise_signature(factor_245.get());

              replay_smart_cluster_factor_244_key_match_index_list.clear();
              replay_smart_cluster_factor_245_key_match_index_list.clear();
              std::vector<size_t> measure_matches_244;
              std::vector<size_t> measure_matches_245;
              for (size_t idx = 0u; idx < sanitized_bootstrap_graph_size; ++idx) {
                if (!replay_bootstrap_audit.sanitized_bootstrap_graph.exists(idx)) {
                  continue;
                }
                const auto& factor =
                    replay_bootstrap_audit.sanitized_bootstrap_graph.at(idx);
                if (!factor) {
                  continue;
                }
                const std::string key_signature = key_signature_sorted(factor.get());
                const std::string measure_signature =
                    summarize_smart_measurement(factor.get()).measurement_signature;
                if (factor_244 &&
                    key_signature == replay_smart_cluster_factor_244_key_signature) {
                  replay_smart_cluster_factor_244_key_match_index_list.push_back(idx);
                }
                if (factor_245 &&
                    key_signature == replay_smart_cluster_factor_245_key_signature) {
                  replay_smart_cluster_factor_245_key_match_index_list.push_back(idx);
                }
                if (factor_244 &&
                    measure_signature ==
                        replay_smart_cluster_factor_244_measure_signature) {
                  measure_matches_244.push_back(idx);
                }
                if (factor_245 &&
                    measure_signature ==
                        replay_smart_cluster_factor_245_measure_signature) {
                  measure_matches_245.push_back(idx);
                }
              }
              replay_smart_cluster_factor_244_key_match_count =
                  static_cast<long long>(
                      replay_smart_cluster_factor_244_key_match_index_list.size());
              replay_smart_cluster_factor_245_key_match_count =
                  static_cast<long long>(
                      replay_smart_cluster_factor_245_key_match_index_list.size());
              replay_smart_cluster_factor_244_measure_match_count =
                  static_cast<long long>(measure_matches_244.size());
              replay_smart_cluster_factor_245_measure_match_count =
                  static_cast<long long>(measure_matches_245.size());
              replay_smart_cluster_factor_244_key_match_indices =
                  join_index_list(replay_smart_cluster_factor_244_key_match_index_list);
              replay_smart_cluster_factor_245_key_match_indices =
                  join_index_list(replay_smart_cluster_factor_245_key_match_index_list);
              replay_smart_cluster_factor_244_measure_match_indices =
                  join_index_list(measure_matches_244);
              replay_smart_cluster_factor_245_measure_match_indices =
                  join_index_list(measure_matches_245);

              replay_smart_cluster_244_245_same_key_signature =
                  (factor_244 && factor_245 &&
                   replay_smart_cluster_factor_244_key_signature ==
                       replay_smart_cluster_factor_245_key_signature)
                      ? 1
                      : 0;
              replay_smart_cluster_244_245_same_measure_signature =
                  (factor_244 && factor_245 &&
                   replay_smart_cluster_factor_244_measure_signature ==
                       replay_smart_cluster_factor_245_measure_signature)
                      ? 1
                      : 0;
              replay_smart_cluster_244_245_same_noise_signature =
                  (factor_244 && factor_245 &&
                   replay_smart_cluster_factor_244_noise_signature ==
                       replay_smart_cluster_factor_245_noise_signature)
                      ? 1
                      : 0;

              std::map<std::string, long long> window_type_counts;
              std::map<std::string, long long> window_key_signature_counts;
              std::map<std::string, long long> window_measure_signature_counts;
              std::ostringstream window_order_oss;
              for (size_t idx = inspect_idx_242; idx <= inspect_idx_246; ++idx) {
                if (idx > inspect_idx_242) {
                  window_order_oss << "|";
                }
                const auto factor = get_sanitized_factor_at(idx);
                if (!factor) {
                  window_order_oss << idx << ":missing";
                  continue;
                }
                const std::string factor_type = classify_factor_type(factor.get());
                const std::string key_signature = key_signature_sorted(factor.get());
                const SmartMeasurementSummary measurement_summary =
                    summarize_smart_measurement(factor.get());
                window_order_oss << idx << ":" << factor_type << "[k="
                                 << key_signature << ";m="
                                 << measurement_summary.measurement_signature
                                 << ";n=" << summarize_noise_signature(factor.get())
                                 << "]";
                ++window_type_counts[factor_type];
                ++window_key_signature_counts[key_signature];
                if (measurement_summary.measurement_signature != "na") {
                  ++window_measure_signature_counts
                      [measurement_summary.measurement_signature];
                }
              }
              std::ostringstream type_counts_oss;
              for (auto it = window_type_counts.begin();
                   it != window_type_counts.end();
                   ++it) {
                if (it != window_type_counts.begin()) {
                  type_counts_oss << ",";
                }
                type_counts_oss << it->first << ":" << it->second;
              }
              replay_smart_cluster_window_type_counts =
                  type_counts_oss.str().empty() ? "none" : type_counts_oss.str();
              replay_smart_cluster_window_order_signature =
                  window_order_oss.str().empty() ? "none" : window_order_oss.str();
              replay_smart_cluster_window_repeated_key_signature = 0;
              for (const auto& kv : window_key_signature_counts) {
                if (kv.second > 1) {
                  replay_smart_cluster_window_repeated_key_signature = 1;
                  break;
                }
              }
              replay_smart_cluster_window_repeated_measure_signature = 0;
              for (const auto& kv : window_measure_signature_counts) {
                if (kv.second > 1) {
                  replay_smart_cluster_window_repeated_measure_signature = 1;
                  break;
                }
              }
            }

            auto run_original_factor_probe =
                [&](const gtsam::NonlinearFactor::shared_ptr& factor,
                    bool* succeeded,
                    std::string* reason) {
                  CHECK_NOTNULL(succeeded);
                  CHECK_NOTNULL(reason);
                  *succeeded = false;
                  *reason = "not_run";
                  if (!factor) {
                    *reason = "missing_factor";
                    return;
                  }
                  const BootstrapWindowProbeResult result =
                      run_probe_from_factors({factor});
                  *succeeded = result.succeeded;
                  *reason = result.reason.empty() ? "none" : result.reason;
                };
            run_original_factor_probe(factor_244,
                                      &replay_factor_clone_orig_244_succeeded,
                                      &replay_factor_clone_orig_244_reason);
            run_original_factor_probe(factor_245,
                                      &replay_factor_clone_orig_245_succeeded,
                                      &replay_factor_clone_orig_245_reason);
            if (factor_244 && factor_245) {
              const BootstrapWindowProbeResult orig_pair_result =
                  run_probe_from_factors({factor_244, factor_245});
              replay_factor_clone_orig_244_245_pair_succeeded =
                  orig_pair_result.succeeded;
              replay_factor_clone_orig_244_245_pair_reason =
                  orig_pair_result.reason.empty() ? "none"
                                                  : orig_pair_result.reason;
            } else {
              replay_factor_clone_orig_244_245_pair_succeeded = false;
              replay_factor_clone_orig_244_245_pair_reason = "missing_factor";
            }

            bool clone_244_construct_attempted = false;
            bool clone_244_construct_succeeded = false;
            std::string clone_244_construct_reason = "not_run";
            const gtsam::NonlinearFactor::shared_ptr clone_244 =
                clone_smart_factor(factor_244,
                                   &clone_244_construct_attempted,
                                   &clone_244_construct_succeeded,
                                   &clone_244_construct_reason);
            replay_factor_clone_clone_244_attempted = clone_244_construct_attempted;
            if (clone_244_construct_succeeded && clone_244) {
              const BootstrapWindowProbeResult clone_244_result =
                  run_probe_from_factors({clone_244});
              replay_factor_clone_clone_244_succeeded = clone_244_result.succeeded;
              replay_factor_clone_clone_244_reason =
                  clone_244_result.reason.empty() ? "none"
                                                  : clone_244_result.reason;
            } else {
              replay_factor_clone_clone_244_succeeded = false;
              replay_factor_clone_clone_244_reason =
                  clone_244_construct_reason.empty() ? "none"
                                                     : clone_244_construct_reason;
            }

            bool clone_245_construct_attempted = false;
            bool clone_245_construct_succeeded = false;
            std::string clone_245_construct_reason = "not_run";
            const gtsam::NonlinearFactor::shared_ptr clone_245 =
                clone_smart_factor(factor_245,
                                   &clone_245_construct_attempted,
                                   &clone_245_construct_succeeded,
                                   &clone_245_construct_reason);
            replay_factor_clone_clone_245_attempted = clone_245_construct_attempted;
            if (clone_245_construct_succeeded && clone_245) {
              const BootstrapWindowProbeResult clone_245_result =
                  run_probe_from_factors({clone_245});
              replay_factor_clone_clone_245_succeeded = clone_245_result.succeeded;
              replay_factor_clone_clone_245_reason =
                  clone_245_result.reason.empty() ? "none"
                                                  : clone_245_result.reason;
            } else {
              replay_factor_clone_clone_245_succeeded = false;
              replay_factor_clone_clone_245_reason =
                  clone_245_construct_reason.empty() ? "none"
                                                     : clone_245_construct_reason;
            }

            replay_factor_clone_clone_244_245_pair_attempted =
                clone_244_construct_succeeded && clone_245_construct_succeeded &&
                static_cast<bool>(clone_244) && static_cast<bool>(clone_245);
            if (replay_factor_clone_clone_244_245_pair_attempted) {
              const BootstrapWindowProbeResult clone_pair_result =
                  run_probe_from_factors({clone_244, clone_245});
              replay_factor_clone_clone_244_245_pair_succeeded =
                  clone_pair_result.succeeded;
              replay_factor_clone_clone_244_245_pair_reason =
                  clone_pair_result.reason.empty() ? "none"
                                                   : clone_pair_result.reason;
            } else {
              replay_factor_clone_clone_244_245_pair_succeeded = false;
              if (!clone_244_construct_succeeded && !clone_245_construct_succeeded) {
                replay_factor_clone_clone_244_245_pair_reason =
                    "clone_244_and_245_unavailable";
              } else if (!clone_244_construct_succeeded) {
                replay_factor_clone_clone_244_245_pair_reason =
                    "clone_244_unavailable";
              } else if (!clone_245_construct_succeeded) {
                replay_factor_clone_clone_244_245_pair_reason =
                    "clone_245_unavailable";
              } else {
                replay_factor_clone_clone_244_245_pair_reason = "clone_missing";
              }
            }

            const std::vector<size_t> good_prefix_indices =
                collect_prefix_indices(good_prefix_len);
            const std::vector<size_t> bad_prefix_indices =
                collect_prefix_indices(bad_prefix_len);
            std::vector<size_t> local_window_indices;
            local_window_indices.reserve(local_window_end - local_window_start + 1u);
            for (size_t idx = local_window_start; idx <= local_window_end; ++idx) {
              local_window_indices.push_back(idx);
            }

            auto run_append_probe =
                [&](const std::vector<size_t>& append_indices)
                -> BootstrapWindowProbeResult {
              std::vector<size_t> variant_indices;
              variant_indices.reserve(good_prefix_indices.size() +
                                      append_indices.size());
              variant_indices.insert(variant_indices.end(),
                                     good_prefix_indices.begin(),
                                     good_prefix_indices.end());
              variant_indices.insert(variant_indices.end(),
                                     append_indices.begin(),
                                     append_indices.end());
              return run_bootstrap_window_probe(variant_indices);
            };

            const BootstrapWindowProbeResult baseline_good_result =
                run_bootstrap_window_probe(good_prefix_indices);
            replay_bootstrap_window_baseline_good_prefix_succeeded =
                baseline_good_result.succeeded;
            replay_bootstrap_window_baseline_good_prefix_reason =
                baseline_good_result.reason.empty() ? "none"
                                                    : baseline_good_result.reason;
            maybe_record_window_first_success("baseline_good_prefix",
                                              baseline_good_result);

            const BootstrapWindowProbeResult baseline_bad_result =
                run_bootstrap_window_probe(bad_prefix_indices);
            replay_bootstrap_window_baseline_bad_prefix_succeeded =
                baseline_bad_result.succeeded;
            replay_bootstrap_window_baseline_bad_prefix_reason =
                baseline_bad_result.reason.empty() ? "none"
                                                   : baseline_bad_result.reason;
            maybe_record_window_first_success("baseline_bad_prefix",
                                              baseline_bad_result);

            auto is_in_bad_prefix_window = [&](const size_t idx) {
              return idx < bad_prefix_len && idx >= local_window_start &&
                     idx <= local_window_end;
            };
            std::vector<size_t> window_indices_in_bad_prefix;
            for (size_t idx = local_window_start; idx <= local_window_end; ++idx) {
              if (is_in_bad_prefix_window(idx)) {
                window_indices_in_bad_prefix.push_back(idx);
              }
            }

            std::ostringstream single_mask_oss;
            for (size_t idx = local_window_start; idx <= local_window_end; ++idx) {
              std::vector<size_t> variant_indices;
              variant_indices.reserve(bad_prefix_indices.size());
              for (const size_t candidate : bad_prefix_indices) {
                if (candidate == idx) {
                  continue;
                }
                variant_indices.push_back(candidate);
              }
              const BootstrapWindowProbeResult variant_result =
                  run_bootstrap_window_probe(variant_indices);
              if (idx > local_window_start) {
                single_mask_oss << ",";
              }
              single_mask_oss << idx << ":" << (variant_result.succeeded ? 1 : 0);
              std::ostringstream variant_name;
              variant_name << "single_remove_" << idx;
              maybe_record_window_first_success(variant_name.str(), variant_result);
            }
            replay_bootstrap_window_single_remove_success_mask =
                single_mask_oss.str().empty() ? "none" : single_mask_oss.str();

            std::ostringstream pair_mask_oss;
            if (local_window_start < local_window_end) {
              for (size_t idx = local_window_start; idx < local_window_end; ++idx) {
                const size_t next_idx = idx + 1u;
                std::vector<size_t> variant_indices;
                variant_indices.reserve(bad_prefix_indices.size());
                for (const size_t candidate : bad_prefix_indices) {
                  if (candidate == idx || candidate == next_idx) {
                    continue;
                  }
                  variant_indices.push_back(candidate);
                }
                const BootstrapWindowProbeResult variant_result =
                    run_bootstrap_window_probe(variant_indices);
                if (idx > local_window_start) {
                  pair_mask_oss << ",";
                }
                pair_mask_oss << idx << "-" << next_idx << ":"
                              << (variant_result.succeeded ? 1 : 0);
                std::ostringstream variant_name;
                variant_name << "adjacent_pair_remove_" << idx << "_" << next_idx;
                maybe_record_window_first_success(variant_name.str(), variant_result);
              }
            }
            replay_bootstrap_window_adjacent_pair_success_mask =
                pair_mask_oss.str().empty() ? "none" : pair_mask_oss.str();

            std::vector<size_t> full_window_removed_indices;
            full_window_removed_indices.reserve(bad_prefix_indices.size());
            for (const size_t candidate : bad_prefix_indices) {
              if (is_in_bad_prefix_window(candidate)) {
                continue;
              }
              full_window_removed_indices.push_back(candidate);
            }
            const BootstrapWindowProbeResult full_window_removed_result =
                run_bootstrap_window_probe(full_window_removed_indices);
            replay_bootstrap_window_full_window_removed_succeeded =
                full_window_removed_result.succeeded;
            replay_bootstrap_window_full_window_removed_reason =
                full_window_removed_result.reason.empty()
                    ? "none"
                    : full_window_removed_result.reason;
            maybe_record_window_first_success("full_window_removed",
                                              full_window_removed_result);

            std::vector<size_t> only_window_indices;
            only_window_indices.reserve(window_indices_in_bad_prefix.size());
            for (const size_t candidate : bad_prefix_indices) {
              if (is_in_bad_prefix_window(candidate)) {
                only_window_indices.push_back(candidate);
              }
            }
            const BootstrapWindowProbeResult only_window_result =
                run_bootstrap_window_probe(only_window_indices);
            replay_bootstrap_window_only_window_succeeded =
                only_window_result.succeeded;
            replay_bootstrap_window_only_window_reason =
                only_window_result.reason.empty() ? "none" : only_window_result.reason;
            maybe_record_window_first_success("only_window", only_window_result);

            std::vector<size_t> window_appended_indices;
            window_appended_indices.reserve(bad_prefix_indices.size());
            for (const size_t candidate : bad_prefix_indices) {
              if (is_in_bad_prefix_window(candidate)) {
                continue;
              }
              window_appended_indices.push_back(candidate);
            }
            window_appended_indices.insert(window_appended_indices.end(),
                                           window_indices_in_bad_prefix.begin(),
                                           window_indices_in_bad_prefix.end());
            const BootstrapWindowProbeResult window_appended_result =
                run_bootstrap_window_probe(window_appended_indices);
            replay_bootstrap_window_window_appended_succeeded =
                window_appended_result.succeeded;
            replay_bootstrap_window_window_appended_reason =
                window_appended_result.reason.empty()
                    ? "none"
                    : window_appended_result.reason;
            maybe_record_window_first_success("window_appended",
                                              window_appended_result);

            std::vector<size_t> window_reversed_indices = bad_prefix_indices;
            if (!window_indices_in_bad_prefix.empty()) {
              const size_t reverse_start = window_indices_in_bad_prefix.front();
              const size_t reverse_end = window_indices_in_bad_prefix.back();
              if (reverse_start <= reverse_end &&
                  reverse_end < window_reversed_indices.size()) {
                std::reverse(window_reversed_indices.begin() + reverse_start,
                             window_reversed_indices.begin() + reverse_end + 1u);
              }
            }
            const BootstrapWindowProbeResult window_reversed_result =
                run_bootstrap_window_probe(window_reversed_indices);
            replay_bootstrap_window_window_reversed_succeeded =
                window_reversed_result.succeeded;
            replay_bootstrap_window_window_reversed_reason =
                window_reversed_result.reason.empty()
                    ? "none"
                    : window_reversed_result.reason;
            maybe_record_window_first_success("window_reversed",
                                              window_reversed_result);

            std::ostringstream append_single_mask_oss;
            std::ostringstream append_single_reason_oss;
            for (size_t w = 0u; w < local_window_indices.size(); ++w) {
              const size_t idx = local_window_indices[w];
              const BootstrapWindowProbeResult append_single_result =
                  run_append_probe({idx});
              if (w > 0u) {
                append_single_mask_oss << ",";
                append_single_reason_oss << "|";
              }
              append_single_mask_oss << idx << ":"
                                     << (append_single_result.succeeded ? 1 : 0);
              append_single_reason_oss
                  << idx << ":"
                  << (append_single_result.reason.empty()
                          ? "none"
                          : append_single_result.reason);
            }
            replay_bootstrap_append_single_success_mask =
                append_single_mask_oss.str().empty()
                    ? "none"
                    : append_single_mask_oss.str();
            replay_bootstrap_append_single_reason_mask =
                append_single_reason_oss.str().empty()
                    ? "none"
                    : append_single_reason_oss.str();

            std::ostringstream append_pair_mask_oss;
            std::ostringstream append_pair_reason_oss;
            for (size_t w = 0u; w + 1u < local_window_indices.size(); ++w) {
              const size_t first_idx = local_window_indices[w];
              const size_t second_idx = local_window_indices[w + 1u];
              const BootstrapWindowProbeResult append_pair_result =
                  run_append_probe({first_idx, second_idx});
              if (w > 0u) {
                append_pair_mask_oss << ",";
                append_pair_reason_oss << "|";
              }
              append_pair_mask_oss << first_idx << "-" << second_idx << ":"
                                   << (append_pair_result.succeeded ? 1 : 0);
              append_pair_reason_oss
                  << first_idx << "-" << second_idx << ":"
                  << (append_pair_result.reason.empty()
                          ? "none"
                          : append_pair_result.reason);
            }
            replay_bootstrap_append_adjacent_pair_success_mask =
                append_pair_mask_oss.str().empty()
                    ? "none"
                    : append_pair_mask_oss.str();
            replay_bootstrap_append_adjacent_pair_reason_mask =
                append_pair_reason_oss.str().empty()
                    ? "none"
                    : append_pair_reason_oss.str();

            std::vector<std::pair<std::string, std::vector<size_t>>>
                append_triple_variants;
            const long long fail_idx_ll = static_cast<long long>(fail_idx);
            auto maybe_add_append_triple = [&](const long long a,
                                               const long long b,
                                               const long long c) {
              if (a < 0 || b < 0 || c < 0) {
                return;
              }
              const size_t ia = static_cast<size_t>(a);
              const size_t ib = static_cast<size_t>(b);
              const size_t ic = static_cast<size_t>(c);
              if (ia >= sanitized_bootstrap_graph_size ||
                  ib >= sanitized_bootstrap_graph_size ||
                  ic >= sanitized_bootstrap_graph_size) {
                return;
              }
              if (ia < local_window_start || ic > local_window_end) {
                return;
              }
              std::ostringstream label_oss;
              label_oss << ia << "-" << ib << "-" << ic;
              append_triple_variants.emplace_back(label_oss.str(),
                                                  std::vector<size_t>{ia, ib, ic});
            };
            maybe_add_append_triple(fail_idx_ll - 1, fail_idx_ll, fail_idx_ll + 1);
            maybe_add_append_triple(fail_idx_ll - 2, fail_idx_ll - 1, fail_idx_ll);
            maybe_add_append_triple(fail_idx_ll, fail_idx_ll + 1, fail_idx_ll + 2);

            std::ostringstream append_triple_mask_oss;
            std::ostringstream append_triple_reason_oss;
            for (size_t t = 0u; t < append_triple_variants.size(); ++t) {
              const auto& triple_variant = append_triple_variants[t];
              const BootstrapWindowProbeResult append_triple_result =
                  run_append_probe(triple_variant.second);
              if (t > 0u) {
                append_triple_mask_oss << ",";
                append_triple_reason_oss << "|";
              }
              append_triple_mask_oss
                  << triple_variant.first << ":"
                  << (append_triple_result.succeeded ? 1 : 0);
              append_triple_reason_oss
                  << triple_variant.first << ":"
                  << (append_triple_result.reason.empty()
                          ? "none"
                          : append_triple_result.reason);
            }
            replay_bootstrap_append_triple_success_mask =
                append_triple_mask_oss.str().empty()
                    ? "none"
                    : append_triple_mask_oss.str();
            replay_bootstrap_append_triple_reason_mask =
                append_triple_reason_oss.str().empty()
                    ? "none"
                    : append_triple_reason_oss.str();

            const BootstrapWindowProbeResult append_whole_window_result =
                run_append_probe(local_window_indices);
            replay_bootstrap_append_whole_window_succeeded =
                append_whole_window_result.succeeded;
            replay_bootstrap_append_whole_window_reason =
                append_whole_window_result.reason.empty()
                    ? "none"
                    : append_whole_window_result.reason;

            std::vector<size_t> append_whole_minus_fail_indices;
            append_whole_minus_fail_indices.reserve(local_window_indices.size());
            for (const size_t idx : local_window_indices) {
              if (idx == fail_idx) {
                continue;
              }
              append_whole_minus_fail_indices.push_back(idx);
            }
            const BootstrapWindowProbeResult append_whole_minus_fail_result =
                run_append_probe(append_whole_minus_fail_indices);
            replay_bootstrap_append_whole_minus_fail_succeeded =
                append_whole_minus_fail_result.succeeded;
            replay_bootstrap_append_whole_minus_fail_reason =
                append_whole_minus_fail_result.reason.empty()
                    ? "none"
                    : append_whole_minus_fail_result.reason;

            const BootstrapWindowProbeResult append_only_fail_result =
                run_append_probe({fail_idx});
            replay_bootstrap_append_only_fail_succeeded =
                append_only_fail_result.succeeded;
            replay_bootstrap_append_only_fail_reason =
                append_only_fail_result.reason.empty()
                    ? "none"
                    : append_only_fail_result.reason;

            std::vector<size_t> append_smart_only_indices;
            std::vector<size_t> append_non_smart_only_indices;
            append_smart_only_indices.reserve(local_window_indices.size());
            append_non_smart_only_indices.reserve(local_window_indices.size());
            for (const size_t idx : local_window_indices) {
              const auto& factor =
                  replay_bootstrap_audit.sanitized_bootstrap_graph.at(idx);
              const std::string factor_type = classify_factor_type(factor.get());
              if (factor_type == "smart") {
                append_smart_only_indices.push_back(idx);
              } else {
                append_non_smart_only_indices.push_back(idx);
              }
            }
            if (!append_smart_only_indices.empty()) {
              const BootstrapWindowProbeResult append_smart_only_result =
                  run_append_probe(append_smart_only_indices);
              replay_bootstrap_append_smart_only_succeeded =
                  append_smart_only_result.succeeded;
              replay_bootstrap_append_smart_only_reason =
                  append_smart_only_result.reason.empty()
                      ? "none"
                      : append_smart_only_result.reason;
            } else {
              replay_bootstrap_append_smart_only_succeeded = false;
              replay_bootstrap_append_smart_only_reason = "not_present";
            }
            if (!append_non_smart_only_indices.empty()) {
              const BootstrapWindowProbeResult append_non_smart_only_result =
                  run_append_probe(append_non_smart_only_indices);
              replay_bootstrap_append_non_smart_only_succeeded =
                  append_non_smart_only_result.succeeded;
              replay_bootstrap_append_non_smart_only_reason =
                  append_non_smart_only_result.reason.empty()
                      ? "none"
                      : append_non_smart_only_result.reason;
            } else {
              replay_bootstrap_append_non_smart_only_succeeded = false;
              replay_bootstrap_append_non_smart_only_reason = "not_present";
            }

            std::vector<size_t> append_fail_front_indices;
            append_fail_front_indices.reserve(local_window_indices.size());
            append_fail_front_indices.push_back(fail_idx);
            for (const size_t idx : local_window_indices) {
              if (idx == fail_idx) {
                continue;
              }
              append_fail_front_indices.push_back(idx);
            }
            const BootstrapWindowProbeResult append_fail_front_result =
                run_append_probe(append_fail_front_indices);
            replay_bootstrap_append_fail_front_succeeded =
                append_fail_front_result.succeeded;
            replay_bootstrap_append_fail_front_reason =
                append_fail_front_result.reason.empty()
                    ? "none"
                    : append_fail_front_result.reason;

            std::vector<size_t> append_fail_end_indices;
            append_fail_end_indices.reserve(local_window_indices.size());
            for (const size_t idx : local_window_indices) {
              if (idx == fail_idx) {
                continue;
              }
              append_fail_end_indices.push_back(idx);
            }
            append_fail_end_indices.push_back(fail_idx);
            const BootstrapWindowProbeResult append_fail_end_result =
                run_append_probe(append_fail_end_indices);
            replay_bootstrap_append_fail_end_succeeded =
                append_fail_end_result.succeeded;
            replay_bootstrap_append_fail_end_reason =
                append_fail_end_result.reason.empty()
                    ? "none"
                    : append_fail_end_result.reason;

            auto run_checked_append_probe =
                [&](const std::vector<size_t>& append_indices)
                -> BootstrapWindowProbeResult {
              for (const size_t idx : append_indices) {
                if (idx >= sanitized_bootstrap_graph_size ||
                    !replay_bootstrap_audit.sanitized_bootstrap_graph.exists(idx)) {
                  BootstrapWindowProbeResult missing_result;
                  missing_result.succeeded = false;
                  missing_result.reason =
                      "missing_idx_" + std::to_string(idx);
                  return missing_result;
                }
                const auto& factor =
                    replay_bootstrap_audit.sanitized_bootstrap_graph.at(idx);
                if (!factor) {
                  BootstrapWindowProbeResult missing_result;
                  missing_result.succeeded = false;
                  missing_result.reason =
                      "null_factor_" + std::to_string(idx);
                  return missing_result;
                }
              }
              return run_append_probe(append_indices);
            };
            const std::vector<std::pair<std::string, std::vector<size_t>>>
                smart_cluster_order_variants = {
                    {"244", {244u}},
                    {"245", {245u}},
                    {"244-245", {244u, 245u}},
                    {"245-244", {245u, 244u}},
                    {"243-244", {243u, 244u}},
                    {"244-243", {244u, 243u}},
                    {"244-246", {244u, 246u}},
                    {"246-244", {246u, 244u}},
                    {"243-244-245", {243u, 244u, 245u}},
                    {"243-245-244", {243u, 245u, 244u}},
                    {"244-245-246", {244u, 245u, 246u}},
                    {"245-244-246", {245u, 244u, 246u}}};
            std::ostringstream smart_cluster_order_success_oss;
            std::ostringstream smart_cluster_order_reason_oss;
            for (size_t variant_idx = 0u;
                 variant_idx < smart_cluster_order_variants.size();
                 ++variant_idx) {
              const auto& variant = smart_cluster_order_variants[variant_idx];
              const BootstrapWindowProbeResult variant_result =
                  run_checked_append_probe(variant.second);
              if (variant_idx > 0u) {
                smart_cluster_order_success_oss << ",";
                smart_cluster_order_reason_oss << "|";
              }
              smart_cluster_order_success_oss
                  << variant.first << ":" << (variant_result.succeeded ? 1 : 0);
              smart_cluster_order_reason_oss
                  << variant.first << ":"
                  << (variant_result.reason.empty() ? "none"
                                                    : variant_result.reason);
              if (variant_result.succeeded) {
                replay_smart_cluster_order_any_success = 1;
                if (replay_smart_cluster_order_first_success_variant == "none") {
                  replay_smart_cluster_order_first_success_variant = variant.first;
                  replay_smart_cluster_order_first_success_reason =
                      variant_result.reason.empty() ? "none"
                                                    : variant_result.reason;
                }
              }
            }
            replay_smart_cluster_order_success_mask =
                smart_cluster_order_success_oss.str().empty()
                    ? "none"
                    : smart_cluster_order_success_oss.str();
            replay_smart_cluster_order_reason_mask =
                smart_cluster_order_reason_oss.str().empty()
                    ? "none"
                    : smart_cluster_order_reason_oss.str();

            auto run_checked_standalone_probe =
                [&](const size_t idx) -> BootstrapWindowProbeResult {
              if (idx >= sanitized_bootstrap_graph_size ||
                  !replay_bootstrap_audit.sanitized_bootstrap_graph.exists(idx)) {
                BootstrapWindowProbeResult missing_result;
                missing_result.succeeded = false;
                missing_result.reason = "missing_idx_" + std::to_string(idx);
                return missing_result;
              }
              const auto& factor =
                  replay_bootstrap_audit.sanitized_bootstrap_graph.at(idx);
              if (!factor) {
                BootstrapWindowProbeResult missing_result;
                missing_result.succeeded = false;
                missing_result.reason = "null_factor_" + std::to_string(idx);
                return missing_result;
              }
              return run_bootstrap_window_probe({idx});
            };
            auto run_family_single_probes =
                [&](const std::vector<size_t>& family_indices,
                    std::string* append_success_mask,
                    std::string* append_reason_mask,
                    std::string* standalone_success_mask,
                    std::string* standalone_reason_mask,
                    int* any_append_success,
                    int* all_append_fail,
                    long long* first_append_success_idx,
                    long long* first_append_fail_idx) {
              CHECK_NOTNULL(append_success_mask);
              CHECK_NOTNULL(append_reason_mask);
              CHECK_NOTNULL(standalone_success_mask);
              CHECK_NOTNULL(standalone_reason_mask);
              CHECK_NOTNULL(any_append_success);
              CHECK_NOTNULL(all_append_fail);
              CHECK_NOTNULL(first_append_success_idx);
              CHECK_NOTNULL(first_append_fail_idx);
              *append_success_mask = "none";
              *append_reason_mask = "none";
              *standalone_success_mask = "none";
              *standalone_reason_mask = "none";
              *any_append_success = 0;
              *all_append_fail = 0;
              *first_append_success_idx = -1;
              *first_append_fail_idx = -1;
              int append_success_count = 0;
              std::ostringstream append_success_oss;
              std::ostringstream append_reason_oss;
              std::ostringstream standalone_success_oss;
              std::ostringstream standalone_reason_oss;
              for (size_t i = 0u; i < family_indices.size(); ++i) {
                const size_t idx = family_indices[i];
                const BootstrapWindowProbeResult append_result =
                    run_checked_append_probe({idx});
                const BootstrapWindowProbeResult standalone_result =
                    run_checked_standalone_probe(idx);
                if (i > 0u) {
                  append_success_oss << ",";
                  append_reason_oss << "|";
                  standalone_success_oss << ",";
                  standalone_reason_oss << "|";
                }
                append_success_oss << idx << ":"
                                   << (append_result.succeeded ? 1 : 0);
                append_reason_oss
                    << idx << ":"
                    << (append_result.reason.empty() ? "none"
                                                     : append_result.reason);
                standalone_success_oss
                    << idx << ":" << (standalone_result.succeeded ? 1 : 0);
                standalone_reason_oss
                    << idx << ":"
                    << (standalone_result.reason.empty()
                            ? "none"
                            : standalone_result.reason);
                if (append_result.succeeded) {
                  ++append_success_count;
                  if (*first_append_success_idx < 0) {
                    *first_append_success_idx = static_cast<long long>(idx);
                  }
                } else if (*first_append_fail_idx < 0) {
                  *first_append_fail_idx = static_cast<long long>(idx);
                }
              }
              if (!family_indices.empty()) {
                *append_success_mask = append_success_oss.str();
                *append_reason_mask = append_reason_oss.str();
                *standalone_success_mask = standalone_success_oss.str();
                *standalone_reason_mask = standalone_reason_oss.str();
              }
              *any_append_success = append_success_count > 0 ? 1 : 0;
              *all_append_fail =
                  (!family_indices.empty() && append_success_count == 0) ? 1 : 0;
            };
            run_family_single_probes(
                replay_smart_cluster_factor_244_key_match_index_list,
                &replay_family244_single_append_success_mask,
                &replay_family244_single_append_reason_mask,
                &replay_family244_standalone_success_mask,
                &replay_family244_standalone_reason_mask,
                &replay_family244_any_append_success,
                &replay_family244_all_append_fail,
                &replay_family244_first_append_success_idx,
                &replay_family244_first_append_fail_idx);
            run_family_single_probes(
                replay_smart_cluster_factor_245_key_match_index_list,
                &replay_family245_single_append_success_mask,
                &replay_family245_single_append_reason_mask,
                &replay_family245_standalone_success_mask,
                &replay_family245_standalone_reason_mask,
                &replay_family245_any_append_success,
                &replay_family245_all_append_fail,
                &replay_family245_first_append_success_idx,
                &replay_family245_first_append_fail_idx);

            const size_t frontier_idx =
                replay_bootstrap_window_good_prefix_len;
            const size_t frontier_idx0 = frontier_idx;
            const size_t frontier_idx1 = frontier_idx + 1u;
            const size_t frontier_idx2 = frontier_idx + 2u;
            replay_append_frontier_idx = static_cast<long long>(frontier_idx);
            replay_append_frontier_idx0 = static_cast<long long>(frontier_idx0);
            replay_append_frontier_idx1 = static_cast<long long>(frontier_idx1);
            replay_append_frontier_idx2 = static_cast<long long>(frontier_idx2);
            auto normalize_probe_reason = [](const std::string& reason)
                -> std::string { return reason.empty() ? "none" : reason; };
            auto fill_frontier_meta = [&](const size_t idx,
                                          std::string* factor_type,
                                          std::string* key_sig,
                                          std::string* meas_sig,
                                          std::string* pointer,
                                          long long* measured_count,
                                          int* point_valid,
                                          bool* is_present) {
              CHECK_NOTNULL(factor_type);
              CHECK_NOTNULL(key_sig);
              CHECK_NOTNULL(meas_sig);
              CHECK_NOTNULL(pointer);
              CHECK_NOTNULL(measured_count);
              CHECK_NOTNULL(point_valid);
              CHECK_NOTNULL(is_present);
              *factor_type = "missing";
              *key_sig = "missing";
              *meas_sig = "missing";
              *pointer = "none";
              *measured_count = -1;
              *point_valid = -1;
              *is_present = false;
              if (idx >= sanitized_bootstrap_graph_size ||
                  !replay_bootstrap_audit.sanitized_bootstrap_graph.exists(idx)) {
                return;
              }
              const auto factor =
                  replay_bootstrap_audit.sanitized_bootstrap_graph.at(idx);
              if (!factor) {
                *factor_type = "null";
                *key_sig = "null";
                *meas_sig = "null";
                *pointer = "null";
                return;
              }
              *is_present = true;
              *factor_type = classify_factor_type(factor.get());
              *key_sig = key_signature_sorted(factor.get());
              const SmartMeasurementSummary smart_summary =
                  summarize_smart_measurement(factor.get());
              *meas_sig = smart_summary.measurement_signature;
              *pointer = factor_ptr_to_string(factor.get());
              *measured_count = smart_summary.measured_count;
              *point_valid = smart_summary.point_valid;
            };
            bool frontier_idx0_present = false;
            bool frontier_idx1_present = false;
            bool frontier_idx2_present = false;
            fill_frontier_meta(frontier_idx0,
                               &replay_append_frontier_idx0_type,
                               &replay_append_frontier_idx0_keysig,
                               &replay_append_frontier_idx0_meassig,
                               &replay_append_frontier_idx0_pointer,
                               &replay_append_frontier_idx0_measured_count,
                               &replay_append_frontier_idx0_point_valid,
                               &frontier_idx0_present);
            fill_frontier_meta(frontier_idx1,
                               &replay_append_frontier_idx1_type,
                               &replay_append_frontier_idx1_keysig,
                               &replay_append_frontier_idx1_meassig,
                               &replay_append_frontier_idx1_pointer,
                               &replay_append_frontier_idx1_measured_count,
                               &replay_append_frontier_idx1_point_valid,
                               &frontier_idx1_present);
            fill_frontier_meta(frontier_idx2,
                               &replay_append_frontier_idx2_type,
                               &replay_append_frontier_idx2_keysig,
                               &replay_append_frontier_idx2_meassig,
                               &replay_append_frontier_idx2_pointer,
                               &replay_append_frontier_idx2_measured_count,
                               &replay_append_frontier_idx2_point_valid,
                               &frontier_idx2_present);
            replay_append_frontier_idx0_idx1_same_keys =
                (frontier_idx0_present && frontier_idx1_present &&
                 replay_append_frontier_idx0_keysig ==
                     replay_append_frontier_idx1_keysig)
                    ? 1
                    : 0;
            replay_append_frontier_idx1_idx2_same_keys =
                (frontier_idx1_present && frontier_idx2_present &&
                 replay_append_frontier_idx1_keysig ==
                     replay_append_frontier_idx2_keysig)
                    ? 1
                    : 0;
            replay_append_frontier_idx0_idx2_same_keys =
                (frontier_idx0_present && frontier_idx2_present &&
                 replay_append_frontier_idx0_keysig ==
                     replay_append_frontier_idx2_keysig)
                    ? 1
                    : 0;
            replay_append_frontier_idx0_idx1_same_measure =
                (frontier_idx0_present && frontier_idx1_present &&
                 replay_append_frontier_idx0_meassig ==
                     replay_append_frontier_idx1_meassig)
                    ? 1
                    : 0;
            replay_append_frontier_idx1_idx2_same_measure =
                (frontier_idx1_present && frontier_idx2_present &&
                 replay_append_frontier_idx1_meassig ==
                     replay_append_frontier_idx2_meassig)
                    ? 1
                    : 0;
            replay_append_frontier_idx0_idx2_same_measure =
                (frontier_idx0_present && frontier_idx2_present &&
                 replay_append_frontier_idx0_meassig ==
                     replay_append_frontier_idx2_meassig)
                    ? 1
                    : 0;
            auto collect_frontier_factor_keys = [&](const size_t idx)
                -> std::vector<gtsam::Key> {
              std::vector<gtsam::Key> keys;
              if (idx >= sanitized_bootstrap_graph_size ||
                  !replay_bootstrap_audit.sanitized_bootstrap_graph.exists(idx)) {
                return keys;
              }
              const auto factor =
                  replay_bootstrap_audit.sanitized_bootstrap_graph.at(idx);
              if (!factor) {
                return keys;
              }
              const gtsam::KeyVector factor_keys = factor->keys();
              keys.assign(factor_keys.begin(), factor_keys.end());
              std::sort(keys.begin(), keys.end());
              return keys;
            };
            auto keys_to_csv = [&](const std::vector<gtsam::Key>& keys)
                -> std::string {
              if (keys.empty()) {
                return "none";
              }
              std::ostringstream oss;
              for (size_t i = 0u; i < keys.size(); ++i) {
                if (i > 0u) {
                  oss << ",";
                }
                oss << key_to_string(keys[i]);
              }
              return oss.str();
            };
            auto build_key_presence_mask = [&](const std::vector<gtsam::Key>& keys,
                                               std::string* mask,
                                               int* introduces_new_key,
                                               int* all_keys_in_prefix_values) {
              CHECK_NOTNULL(mask);
              CHECK_NOTNULL(introduces_new_key);
              CHECK_NOTNULL(all_keys_in_prefix_values);
              *mask = "none";
              *introduces_new_key = 0;
              *all_keys_in_prefix_values = 0;
              if (keys.empty()) {
                return;
              }
              *all_keys_in_prefix_values = 1;
              std::ostringstream oss;
              for (size_t i = 0u; i < keys.size(); ++i) {
                const gtsam::Key key = keys[i];
                const bool in_prefix_values = replay_probe_base_values.exists(key);
                const bool in_packet_local_values = packet.local_values.exists(key);
                if (!in_prefix_values) {
                  *introduces_new_key = 1;
                  *all_keys_in_prefix_values = 0;
                }
                if (i > 0u) {
                  oss << "|";
                }
                oss << key_to_string(key) << ":pfx="
                    << (in_prefix_values ? 1 : 0) << ",pkt="
                    << (in_packet_local_values ? 1 : 0);
              }
              *mask = oss.str().empty() ? "none" : oss.str();
            };
            const std::vector<gtsam::Key> frontier_keys0 =
                collect_frontier_factor_keys(frontier_idx0);
            const std::vector<gtsam::Key> frontier_keys1 =
                collect_frontier_factor_keys(frontier_idx1);
            const std::vector<gtsam::Key> frontier_keys2 =
                collect_frontier_factor_keys(frontier_idx2);
            replay_append_frontier_idx0_factor_keys = keys_to_csv(frontier_keys0);
            replay_append_frontier_idx1_factor_keys = keys_to_csv(frontier_keys1);
            replay_append_frontier_idx2_factor_keys = keys_to_csv(frontier_keys2);
            build_key_presence_mask(frontier_keys0,
                                    &replay_append_frontier_idx0_key_presence_mask,
                                    &replay_append_frontier_idx0_introduces_new_key,
                                    &replay_append_frontier_idx0_all_keys_in_good_prefix_values);
            build_key_presence_mask(frontier_keys1,
                                    &replay_append_frontier_idx1_key_presence_mask,
                                    &replay_append_frontier_idx1_introduces_new_key,
                                    &replay_append_frontier_idx1_all_keys_in_good_prefix_values);
            build_key_presence_mask(frontier_keys2,
                                    &replay_append_frontier_idx2_key_presence_mask,
                                    &replay_append_frontier_idx2_introduces_new_key,
                                    &replay_append_frontier_idx2_all_keys_in_good_prefix_values);

            const std::set<gtsam::Key> frontier_key_set0(frontier_keys0.begin(),
                                                         frontier_keys0.end());
            const std::set<gtsam::Key> frontier_key_set1(frontier_keys1.begin(),
                                                         frontier_keys1.end());
            std::set<gtsam::Key> frontier_union_key_set(frontier_key_set0.begin(),
                                                        frontier_key_set0.end());
            frontier_union_key_set.insert(frontier_key_set1.begin(),
                                          frontier_key_set1.end());
            std::map<gtsam::Key, long long> frontier_union_prefix_touch_counts;
            for (const gtsam::Key key : frontier_union_key_set) {
              frontier_union_prefix_touch_counts[key] = 0;
            }
            replay_append_frontier_prefix_touch_count_idx0 = 0;
            replay_append_frontier_prefix_touch_count_idx1 = 0;
            replay_append_frontier_prefix_has_imu_like_idx0 = 0;
            replay_append_frontier_prefix_has_between_like_idx1 = 0;
            replay_prefix_smart_touch_idx0_count = 0;
            replay_prefix_smart_touch_idx1_count = 0;
            replay_prefix_smart_touch_both_count = 0;
            replay_prefix_same_keysig_as_idx0_count = 0;
            replay_prefix_same_keysig_as_idx1_count = 0;
            replay_prefix_same_meassig_as_idx0_count = 0;
            replay_prefix_same_meassig_as_idx1_count = 0;
            std::vector<std::string> frontier_touch_sample_idx0;
            std::vector<std::string> frontier_touch_sample_idx1;
            std::vector<std::string> prefix_smart_touch_idx0_samples;
            std::vector<std::string> prefix_smart_touch_idx1_samples;
            std::vector<std::string> prefix_smart_touch_both_samples;
            std::vector<size_t> prefix_same_keysig_as_idx0_indices;
            std::vector<size_t> prefix_same_keysig_as_idx1_indices;
            std::vector<size_t> prefix_same_meassig_as_idx0_indices;
            std::vector<size_t> prefix_same_meassig_as_idx1_indices;
            const size_t prefix_factor_limit =
                std::min(good_prefix_len, sanitized_bootstrap_graph_size);
            for (size_t prefix_idx = 0u; prefix_idx < prefix_factor_limit;
                 ++prefix_idx) {
              if (!replay_bootstrap_audit.sanitized_bootstrap_graph.exists(
                      prefix_idx)) {
                continue;
              }
              const auto factor =
                  replay_bootstrap_audit.sanitized_bootstrap_graph.at(prefix_idx);
              if (!factor) {
                continue;
              }
              const std::vector<gtsam::Key> factor_keys =
                  factor_keys_sorted(factor.get());
              bool touches_idx0 = false;
              bool touches_idx1 = false;
              for (const gtsam::Key key : factor_keys) {
                if (frontier_key_set0.find(key) != frontier_key_set0.end()) {
                  touches_idx0 = true;
                }
                if (frontier_key_set1.find(key) != frontier_key_set1.end()) {
                  touches_idx1 = true;
                }
                const auto union_it = frontier_union_prefix_touch_counts.find(key);
                if (union_it != frontier_union_prefix_touch_counts.end()) {
                  ++union_it->second;
                }
              }
              const std::string factor_type = classify_factor_type(factor.get());
              if (touches_idx0) {
                ++replay_append_frontier_prefix_touch_count_idx0;
                if (factor_type == "imu") {
                  replay_append_frontier_prefix_has_imu_like_idx0 = 1;
                }
                if (frontier_touch_sample_idx0.size() < 8u) {
                  std::ostringstream token;
                  token << prefix_idx << ":" << factor_type << ":"
                        << factor_keys_to_sample(factor.get());
                  frontier_touch_sample_idx0.push_back(token.str());
                }
              }
              if (touches_idx1) {
                ++replay_append_frontier_prefix_touch_count_idx1;
                if (factor_type == "between") {
                  replay_append_frontier_prefix_has_between_like_idx1 = 1;
                }
                if (frontier_touch_sample_idx1.size() < 8u) {
                  std::ostringstream token;
                  token << prefix_idx << ":" << factor_type << ":"
                        << factor_keys_to_sample(factor.get());
                  frontier_touch_sample_idx1.push_back(token.str());
                }
              }
              if (factor_type == "smart") {
                const std::string factor_keysig = key_signature_sorted(factor.get());
                const SmartMeasurementSummary factor_smart_summary =
                    summarize_smart_measurement(factor.get());
                const std::string factor_meassig =
                    factor_smart_summary.measurement_signature;
                if (touches_idx0) {
                  ++replay_prefix_smart_touch_idx0_count;
                  if (prefix_smart_touch_idx0_samples.size() < 12u) {
                    std::ostringstream token;
                    token << prefix_idx << ":" << factor_type << ":"
                          << factor_keys_to_sample(factor.get());
                    prefix_smart_touch_idx0_samples.push_back(token.str());
                  }
                }
                if (touches_idx1) {
                  ++replay_prefix_smart_touch_idx1_count;
                  if (prefix_smart_touch_idx1_samples.size() < 12u) {
                    std::ostringstream token;
                    token << prefix_idx << ":" << factor_type << ":"
                          << factor_keys_to_sample(factor.get());
                    prefix_smart_touch_idx1_samples.push_back(token.str());
                  }
                }
                if (touches_idx0 && touches_idx1) {
                  ++replay_prefix_smart_touch_both_count;
                  if (prefix_smart_touch_both_samples.size() < 12u) {
                    std::ostringstream token;
                    token << prefix_idx << ":" << factor_type << ":"
                          << factor_keys_to_sample(factor.get());
                    prefix_smart_touch_both_samples.push_back(token.str());
                  }
                }
                if (factor_keysig == replay_append_frontier_idx0_keysig) {
                  ++replay_prefix_same_keysig_as_idx0_count;
                  prefix_same_keysig_as_idx0_indices.push_back(prefix_idx);
                }
                if (factor_keysig == replay_append_frontier_idx1_keysig) {
                  ++replay_prefix_same_keysig_as_idx1_count;
                  prefix_same_keysig_as_idx1_indices.push_back(prefix_idx);
                }
                if (factor_meassig == replay_append_frontier_idx0_meassig) {
                  ++replay_prefix_same_meassig_as_idx0_count;
                  prefix_same_meassig_as_idx0_indices.push_back(prefix_idx);
                }
                if (factor_meassig == replay_append_frontier_idx1_meassig) {
                  ++replay_prefix_same_meassig_as_idx1_count;
                  prefix_same_meassig_as_idx1_indices.push_back(prefix_idx);
                }
              }
            }
            auto join_tokens = [](const std::vector<std::string>& tokens)
                -> std::string {
              if (tokens.empty()) {
                return "none";
              }
              std::ostringstream oss;
              for (size_t i = 0u; i < tokens.size(); ++i) {
                if (i > 0u) {
                  oss << "|";
                }
                oss << tokens[i];
              }
              return oss.str();
            };
            replay_append_frontier_prefix_touch_sample_idx0 =
                join_tokens(frontier_touch_sample_idx0);
            replay_append_frontier_prefix_touch_sample_idx1 =
                join_tokens(frontier_touch_sample_idx1);
            replay_prefix_smart_touch_idx0_samples =
                join_tokens(prefix_smart_touch_idx0_samples);
            replay_prefix_smart_touch_idx1_samples =
                join_tokens(prefix_smart_touch_idx1_samples);
            replay_prefix_smart_touch_both_samples =
                join_tokens(prefix_smart_touch_both_samples);
            auto join_indices = [](const std::vector<size_t>& indices)
                -> std::string {
              if (indices.empty()) {
                return "none";
              }
              std::ostringstream oss;
              for (size_t i = 0u; i < indices.size(); ++i) {
                if (i > 0u) {
                  oss << ",";
                }
                oss << indices[i];
              }
              return oss.str();
            };
            replay_prefix_same_keysig_as_idx0_indices =
                join_indices(prefix_same_keysig_as_idx0_indices);
            replay_prefix_same_keysig_as_idx1_indices =
                join_indices(prefix_same_keysig_as_idx1_indices);
            replay_prefix_same_meassig_as_idx0_indices =
                join_indices(prefix_same_meassig_as_idx0_indices);
            replay_prefix_same_meassig_as_idx1_indices =
                join_indices(prefix_same_meassig_as_idx1_indices);
            auto build_prefix_context_token = [&](long long idx_value)
                -> std::string {
              if (idx_value < 0 ||
                  static_cast<size_t>(idx_value) >= sanitized_bootstrap_graph_size) {
                return "none";
              }
              const size_t idx = static_cast<size_t>(idx_value);
              if (!replay_bootstrap_audit.sanitized_bootstrap_graph.exists(idx)) {
                return "none";
              }
              const auto factor =
                  replay_bootstrap_audit.sanitized_bootstrap_graph.at(idx);
              if (!factor) {
                return "none";
              }
              const std::string factor_type = classify_factor_type(factor.get());
              const std::string factor_keys = factor_keys_to_sample(factor.get());
              const std::string factor_keysig = key_signature_sorted(factor.get());
              const SmartMeasurementSummary smart_summary =
                  summarize_smart_measurement(factor.get());
              const std::string factor_meassig =
                  smart_summary.measurement_signature;
              std::ostringstream token;
              token << idx << ":" << factor_type << ":keys=" << factor_keys
                    << ":keysig=" << factor_keysig
                    << ":meassig=" << factor_meassig;
              return token.str();
            };
            const long long prefix_prev_m3_index =
                (good_prefix_len >= 3u) ? static_cast<long long>(good_prefix_len) - 3
                                        : -1;
            const long long prefix_prev_m2_index =
                (good_prefix_len >= 2u) ? static_cast<long long>(good_prefix_len) - 2
                                        : -1;
            const long long prefix_prev_m1_index =
                (good_prefix_len >= 1u) ? static_cast<long long>(good_prefix_len) - 1
                                        : -1;
            replay_prefix_prev_m3_context =
                build_prefix_context_token(prefix_prev_m3_index);
            replay_prefix_prev_m2_context =
                build_prefix_context_token(prefix_prev_m2_index);
            replay_prefix_prev_m1_context =
                build_prefix_context_token(prefix_prev_m1_index);
            {
              std::ostringstream boundary_union_oss;
              for (const gtsam::Key key : frontier_union_key_set) {
                if (boundary_union_oss.tellp() > 0) {
                  boundary_union_oss << "|";
                }
                const bool in_prefix_values = replay_probe_base_values.exists(key);
                const long long prefix_touch_count =
                    frontier_union_prefix_touch_counts[key];
                boundary_union_oss << key_to_string(key) << ":pfx="
                                   << (in_prefix_values ? 1 : 0)
                                   << ",touch=" << prefix_touch_count;
              }
              replay_append_frontier_pair_union_key_summary =
                  boundary_union_oss.str().empty() ? "none"
                                                   : boundary_union_oss.str();
            }

            const BootstrapWindowProbeResult append_idx0_result =
                run_checked_append_probe({frontier_idx0});
            replay_append_frontier_append_idx0_succeeded =
                append_idx0_result.succeeded;
            replay_append_frontier_append_idx0_reason =
                normalize_probe_reason(append_idx0_result.reason);
            const BootstrapWindowProbeResult append_idx1_result =
                run_checked_append_probe({frontier_idx1});
            replay_append_frontier_append_idx1_succeeded =
                append_idx1_result.succeeded;
            replay_append_frontier_append_idx1_reason =
                normalize_probe_reason(append_idx1_result.reason);
            const BootstrapWindowProbeResult append_idx2_result =
                run_checked_append_probe({frontier_idx2});
            replay_append_frontier_append_idx2_succeeded =
                append_idx2_result.succeeded;
            replay_append_frontier_append_idx2_reason =
                normalize_probe_reason(append_idx2_result.reason);
            const BootstrapWindowProbeResult append_idx0_idx1_result =
                run_checked_append_probe({frontier_idx0, frontier_idx1});
            replay_append_frontier_append_idx0_idx1_succeeded =
                append_idx0_idx1_result.succeeded;
            replay_append_frontier_append_idx0_idx1_reason =
                normalize_probe_reason(append_idx0_idx1_result.reason);
            const BootstrapWindowProbeResult append_idx1_idx0_result =
                run_checked_append_probe({frontier_idx1, frontier_idx0});
            replay_append_frontier_append_idx1_idx0_succeeded =
                append_idx1_idx0_result.succeeded;
            replay_append_frontier_append_idx1_idx0_reason =
                normalize_probe_reason(append_idx1_idx0_result.reason);
            const BootstrapWindowProbeResult append_idx0_idx1_idx2_result =
                run_checked_append_probe(
                    {frontier_idx0, frontier_idx1, frontier_idx2});
            replay_append_frontier_append_idx0_idx1_idx2_succeeded =
                append_idx0_idx1_idx2_result.succeeded;
            replay_append_frontier_append_idx0_idx1_idx2_reason =
                normalize_probe_reason(append_idx0_idx1_idx2_result.reason);
            const BootstrapWindowProbeResult append_idx1_idx2_result =
                run_checked_append_probe({frontier_idx1, frontier_idx2});
            replay_append_frontier_append_idx1_idx2_succeeded =
                append_idx1_idx2_result.succeeded;
            replay_append_frontier_append_idx1_idx2_reason =
                normalize_probe_reason(append_idx1_idx2_result.reason);
            const BootstrapWindowProbeResult append_idx0_idx2_result =
                run_checked_append_probe({frontier_idx0, frontier_idx2});
            replay_append_frontier_append_idx0_idx2_succeeded =
                append_idx0_idx2_result.succeeded;
            replay_append_frontier_append_idx0_idx2_reason =
                normalize_probe_reason(append_idx0_idx2_result.reason);
            const BootstrapWindowProbeResult standalone_idx0_result =
                run_checked_standalone_probe(frontier_idx0);
            replay_append_frontier_standalone_idx0_succeeded =
                standalone_idx0_result.succeeded;
            replay_append_frontier_standalone_idx0_reason =
                normalize_probe_reason(standalone_idx0_result.reason);
            const BootstrapWindowProbeResult standalone_idx1_result =
                run_checked_standalone_probe(frontier_idx1);
            replay_append_frontier_standalone_idx1_succeeded =
                standalone_idx1_result.succeeded;
            replay_append_frontier_standalone_idx1_reason =
                normalize_probe_reason(standalone_idx1_result.reason);
            const BootstrapWindowProbeResult standalone_idx2_result =
                run_checked_standalone_probe(frontier_idx2);
            replay_append_frontier_standalone_idx2_succeeded =
                standalone_idx2_result.succeeded;
            replay_append_frontier_standalone_idx2_reason =
                normalize_probe_reason(standalone_idx2_result.reason);
          }

          std::cerr << std::setprecision(12)
                    << "[CBS][H2ReplayBootstrapWindowDiag]"
                    << " curr_kf_id=" << curr_kf_id
                    << " fail_idx=" << replay_bootstrap_window_fail_idx
                    << " good_prefix_len="
                    << replay_bootstrap_window_good_prefix_len
                    << " bad_prefix_len="
                    << replay_bootstrap_window_bad_prefix_len
                    << " local_window_start="
                    << replay_bootstrap_window_start
                    << " local_window_end="
                    << replay_bootstrap_window_end
                    << " baseline_good_prefix_succeeded="
                    << (replay_bootstrap_window_baseline_good_prefix_succeeded ? 1
                                                                                : 0)
                    << " baseline_good_prefix_reason="
                    << (replay_bootstrap_window_baseline_good_prefix_reason.empty()
                            ? "none"
                            : replay_bootstrap_window_baseline_good_prefix_reason)
                    << " baseline_bad_prefix_succeeded="
                    << (replay_bootstrap_window_baseline_bad_prefix_succeeded ? 1
                                                                               : 0)
                    << " baseline_bad_prefix_reason="
                    << (replay_bootstrap_window_baseline_bad_prefix_reason.empty()
                            ? "none"
                            : replay_bootstrap_window_baseline_bad_prefix_reason)
                    << " single_remove_success_mask="
                    << (replay_bootstrap_window_single_remove_success_mask.empty()
                            ? "none"
                            : replay_bootstrap_window_single_remove_success_mask)
                    << " adjacent_pair_success_mask="
                    << (replay_bootstrap_window_adjacent_pair_success_mask.empty()
                            ? "none"
                            : replay_bootstrap_window_adjacent_pair_success_mask)
                    << " full_window_removed_succeeded="
                    << (replay_bootstrap_window_full_window_removed_succeeded ? 1
                                                                              : 0)
                    << " full_window_removed_reason="
                    << (replay_bootstrap_window_full_window_removed_reason.empty()
                            ? "none"
                            : replay_bootstrap_window_full_window_removed_reason)
                    << " only_window_succeeded="
                    << (replay_bootstrap_window_only_window_succeeded ? 1 : 0)
                    << " only_window_reason="
                    << (replay_bootstrap_window_only_window_reason.empty()
                            ? "none"
                            : replay_bootstrap_window_only_window_reason)
                    << " window_appended_succeeded="
                    << (replay_bootstrap_window_window_appended_succeeded ? 1 : 0)
                    << " window_appended_reason="
                    << (replay_bootstrap_window_window_appended_reason.empty()
                            ? "none"
                            : replay_bootstrap_window_window_appended_reason)
                    << " window_reversed_succeeded="
                    << (replay_bootstrap_window_window_reversed_succeeded ? 1 : 0)
                    << " window_reversed_reason="
                    << (replay_bootstrap_window_window_reversed_reason.empty()
                            ? "none"
                            : replay_bootstrap_window_window_reversed_reason)
                    << " first_success_variant="
                    << (replay_bootstrap_window_first_success_variant.empty()
                            ? "none"
                            : replay_bootstrap_window_first_success_variant)
                    << " first_success_reason="
                    << (replay_bootstrap_window_first_success_reason.empty()
                            ? "none"
                            : replay_bootstrap_window_first_success_reason)
                    << " local_window_types="
                    << (replay_bootstrap_window_local_window_types.empty()
                            ? "none"
                            : replay_bootstrap_window_local_window_types)
                    << " local_window_keys="
                    << (replay_bootstrap_window_local_window_keys.empty()
                            ? "none"
                            : replay_bootstrap_window_local_window_keys)
                    << std::endl;
          std::cerr << std::setprecision(12)
                    << "[CBS][H2ReplayBootstrapAppendDiag]"
                    << " curr_kf_id=" << curr_kf_id
                    << " good_prefix_len="
                    << replay_bootstrap_window_good_prefix_len
                    << " fail_idx=" << replay_bootstrap_window_fail_idx
                    << " local_window_start="
                    << replay_bootstrap_window_start
                    << " local_window_end="
                    << replay_bootstrap_window_end
                    << " failing_factor_type="
                    << (replay_bootstrap_first_failing_factor_type.empty()
                            ? "none"
                            : replay_bootstrap_first_failing_factor_type)
                    << " failing_factor_keys="
                    << (replay_bootstrap_first_failing_factor_keys.empty()
                            ? "none"
                            : replay_bootstrap_first_failing_factor_keys)
                    << " local_window_types="
                    << (replay_bootstrap_window_local_window_types.empty()
                            ? "none"
                            : replay_bootstrap_window_local_window_types)
                    << " local_window_keys="
                    << (replay_bootstrap_window_local_window_keys.empty()
                            ? "none"
                            : replay_bootstrap_window_local_window_keys)
                    << " append_single_success_mask="
                    << (replay_bootstrap_append_single_success_mask.empty()
                            ? "none"
                            : replay_bootstrap_append_single_success_mask)
                    << " append_single_reason_mask="
                    << (replay_bootstrap_append_single_reason_mask.empty()
                            ? "none"
                            : replay_bootstrap_append_single_reason_mask)
                    << " append_adjacent_pair_success_mask="
                    << (replay_bootstrap_append_adjacent_pair_success_mask.empty()
                            ? "none"
                            : replay_bootstrap_append_adjacent_pair_success_mask)
                    << " append_adjacent_pair_reason_mask="
                    << (replay_bootstrap_append_adjacent_pair_reason_mask.empty()
                            ? "none"
                            : replay_bootstrap_append_adjacent_pair_reason_mask)
                    << " append_triple_success_mask="
                    << (replay_bootstrap_append_triple_success_mask.empty()
                            ? "none"
                            : replay_bootstrap_append_triple_success_mask)
                    << " append_triple_reason_mask="
                    << (replay_bootstrap_append_triple_reason_mask.empty()
                            ? "none"
                            : replay_bootstrap_append_triple_reason_mask)
                    << " append_whole_window_succeeded="
                    << (replay_bootstrap_append_whole_window_succeeded ? 1 : 0)
                    << " append_whole_window_reason="
                    << (replay_bootstrap_append_whole_window_reason.empty()
                            ? "none"
                            : replay_bootstrap_append_whole_window_reason)
                    << " append_whole_minus_fail_succeeded="
                    << (replay_bootstrap_append_whole_minus_fail_succeeded ? 1 : 0)
                    << " append_whole_minus_fail_reason="
                    << (replay_bootstrap_append_whole_minus_fail_reason.empty()
                            ? "none"
                            : replay_bootstrap_append_whole_minus_fail_reason)
                    << " append_only_fail_succeeded="
                    << (replay_bootstrap_append_only_fail_succeeded ? 1 : 0)
                    << " append_only_fail_reason="
                    << (replay_bootstrap_append_only_fail_reason.empty()
                            ? "none"
                            : replay_bootstrap_append_only_fail_reason)
                    << " append_smart_only_succeeded="
                    << (replay_bootstrap_append_smart_only_succeeded ? 1 : 0)
                    << " append_smart_only_reason="
                    << (replay_bootstrap_append_smart_only_reason.empty()
                            ? "none"
                            : replay_bootstrap_append_smart_only_reason)
                    << " append_non_smart_only_succeeded="
                    << (replay_bootstrap_append_non_smart_only_succeeded ? 1 : 0)
                    << " append_non_smart_only_reason="
                    << (replay_bootstrap_append_non_smart_only_reason.empty()
                            ? "none"
                            : replay_bootstrap_append_non_smart_only_reason)
                    << " append_fail_front_succeeded="
                    << (replay_bootstrap_append_fail_front_succeeded ? 1 : 0)
                    << " append_fail_front_reason="
                    << (replay_bootstrap_append_fail_front_reason.empty()
                            ? "none"
                            : replay_bootstrap_append_fail_front_reason)
                    << " append_fail_end_succeeded="
                    << (replay_bootstrap_append_fail_end_succeeded ? 1 : 0)
                    << " append_fail_end_reason="
                    << (replay_bootstrap_append_fail_end_reason.empty()
                            ? "none"
                            : replay_bootstrap_append_fail_end_reason)
                    << std::endl;
          std::cerr << std::setprecision(12)
                    << "[CBS][H2ReplayFactorObjectDiag]"
                    << " curr_kf_id=" << curr_kf_id
                    << " sanitized_bootstrap_graph_size="
                    << sanitized_bootstrap_graph_size
                    << " factor_243_type=" << replay_factor_243_type
                    << " factor_243_keys=" << replay_factor_243_keys
                    << " factor_243_key_count=" << replay_factor_243_key_count
                    << " factor_243_measured_count="
                    << replay_factor_243_measured_count
                    << " factor_243_is_smart=" << replay_factor_243_is_smart
                    << " factor_243_point_valid=" << replay_factor_243_point_valid
                    << " factor_243_pointer=" << replay_factor_243_pointer
                    << " factor_244_type=" << replay_factor_244_type
                    << " factor_244_keys=" << replay_factor_244_keys
                    << " factor_244_key_count=" << replay_factor_244_key_count
                    << " factor_244_measured_count="
                    << replay_factor_244_measured_count
                    << " factor_244_is_smart=" << replay_factor_244_is_smart
                    << " factor_244_point_valid=" << replay_factor_244_point_valid
                    << " factor_244_pointer=" << replay_factor_244_pointer
                    << " factor_245_type=" << replay_factor_245_type
                    << " factor_245_keys=" << replay_factor_245_keys
                    << " factor_245_key_count=" << replay_factor_245_key_count
                    << " factor_245_measured_count="
                    << replay_factor_245_measured_count
                    << " factor_245_is_smart=" << replay_factor_245_is_smart
                    << " factor_245_point_valid=" << replay_factor_245_point_valid
                    << " factor_245_pointer=" << replay_factor_245_pointer
                    << " factor_246_type=" << replay_factor_246_type
                    << " factor_246_keys=" << replay_factor_246_keys
                    << " factor_246_key_count=" << replay_factor_246_key_count
                    << " factor_246_measured_count="
                    << replay_factor_246_measured_count
                    << " factor_246_is_smart=" << replay_factor_246_is_smart
                    << " factor_246_point_valid=" << replay_factor_246_point_valid
                    << " factor_246_pointer=" << replay_factor_246_pointer
                    << " factor_244_245_same_keys="
                    << replay_factor_244_245_same_keys
                    << " factor_243_244_same_keys="
                    << replay_factor_243_244_same_keys
                    << " factor_245_246_same_keys="
                    << replay_factor_245_246_same_keys << std::endl;
          std::cerr << std::setprecision(12)
                    << "[CBS][H2ReplayFactorCloneABDiag]"
                    << " curr_kf_id=" << curr_kf_id
                    << " orig_244_succeeded="
                    << (replay_factor_clone_orig_244_succeeded ? 1 : 0)
                    << " orig_244_reason="
                    << (replay_factor_clone_orig_244_reason.empty()
                            ? "none"
                            : replay_factor_clone_orig_244_reason)
                    << " orig_245_succeeded="
                    << (replay_factor_clone_orig_245_succeeded ? 1 : 0)
                    << " orig_245_reason="
                    << (replay_factor_clone_orig_245_reason.empty()
                            ? "none"
                            : replay_factor_clone_orig_245_reason)
                    << " clone_244_attempted="
                    << (replay_factor_clone_clone_244_attempted ? 1 : 0)
                    << " clone_244_succeeded="
                    << (replay_factor_clone_clone_244_succeeded ? 1 : 0)
                    << " clone_244_reason="
                    << (replay_factor_clone_clone_244_reason.empty()
                            ? "none"
                            : replay_factor_clone_clone_244_reason)
                    << " clone_245_attempted="
                    << (replay_factor_clone_clone_245_attempted ? 1 : 0)
                    << " clone_245_succeeded="
                    << (replay_factor_clone_clone_245_succeeded ? 1 : 0)
                    << " clone_245_reason="
                    << (replay_factor_clone_clone_245_reason.empty()
                            ? "none"
                            : replay_factor_clone_clone_245_reason)
                    << " orig_244_245_pair_succeeded="
                    << (replay_factor_clone_orig_244_245_pair_succeeded ? 1 : 0)
                    << " orig_244_245_pair_reason="
                    << (replay_factor_clone_orig_244_245_pair_reason.empty()
                            ? "none"
                            : replay_factor_clone_orig_244_245_pair_reason)
                    << " clone_244_245_pair_attempted="
                    << (replay_factor_clone_clone_244_245_pair_attempted ? 1 : 0)
                    << " clone_244_245_pair_succeeded="
                    << (replay_factor_clone_clone_244_245_pair_succeeded ? 1 : 0)
                    << " clone_244_245_pair_reason="
                    << (replay_factor_clone_clone_244_245_pair_reason.empty()
                            ? "none"
                            : replay_factor_clone_clone_244_245_pair_reason)
                    << std::endl;
          std::cerr << std::setprecision(12)
                    << "[CBS][H2ReplaySmartClusterDiag]"
                    << " curr_kf_id=" << curr_kf_id
                    << " sanitized_bootstrap_graph_size="
                    << sanitized_bootstrap_graph_size
                    << " inspect_indices=242,243,244,245,246"
                    << " factor_diag="
                    << (replay_smart_cluster_factor_diag.empty()
                            ? "none"
                            : replay_smart_cluster_factor_diag)
                    << " factor_244_key_signature="
                    << (replay_smart_cluster_factor_244_key_signature.empty()
                            ? "none"
                            : replay_smart_cluster_factor_244_key_signature)
                    << " factor_245_key_signature="
                    << (replay_smart_cluster_factor_245_key_signature.empty()
                            ? "none"
                            : replay_smart_cluster_factor_245_key_signature)
                    << " factor_244_measure_signature="
                    << (replay_smart_cluster_factor_244_measure_signature.empty()
                            ? "none"
                            : replay_smart_cluster_factor_244_measure_signature)
                    << " factor_245_measure_signature="
                    << (replay_smart_cluster_factor_245_measure_signature.empty()
                            ? "none"
                            : replay_smart_cluster_factor_245_measure_signature)
                    << " factor_244_noise_signature="
                    << (replay_smart_cluster_factor_244_noise_signature.empty()
                            ? "none"
                            : replay_smart_cluster_factor_244_noise_signature)
                    << " factor_245_noise_signature="
                    << (replay_smart_cluster_factor_245_noise_signature.empty()
                            ? "none"
                            : replay_smart_cluster_factor_245_noise_signature)
                    << " factor_244_key_match_count="
                    << replay_smart_cluster_factor_244_key_match_count
                    << " factor_244_key_match_indices="
                    << (replay_smart_cluster_factor_244_key_match_indices.empty()
                            ? "none"
                            : replay_smart_cluster_factor_244_key_match_indices)
                    << " factor_244_measure_match_count="
                    << replay_smart_cluster_factor_244_measure_match_count
                    << " factor_244_measure_match_indices="
                    << (replay_smart_cluster_factor_244_measure_match_indices.empty()
                            ? "none"
                            : replay_smart_cluster_factor_244_measure_match_indices)
                    << " factor_245_key_match_count="
                    << replay_smart_cluster_factor_245_key_match_count
                    << " factor_245_key_match_indices="
                    << (replay_smart_cluster_factor_245_key_match_indices.empty()
                            ? "none"
                            : replay_smart_cluster_factor_245_key_match_indices)
                    << " factor_245_measure_match_count="
                    << replay_smart_cluster_factor_245_measure_match_count
                    << " factor_245_measure_match_indices="
                    << (replay_smart_cluster_factor_245_measure_match_indices.empty()
                            ? "none"
                            : replay_smart_cluster_factor_245_measure_match_indices)
                    << " factor_244_245_same_key_signature="
                    << replay_smart_cluster_244_245_same_key_signature
                    << " factor_244_245_same_measure_signature="
                    << replay_smart_cluster_244_245_same_measure_signature
                    << " factor_244_245_same_noise_signature="
                    << replay_smart_cluster_244_245_same_noise_signature
                    << " window_type_counts="
                    << (replay_smart_cluster_window_type_counts.empty()
                            ? "none"
                            : replay_smart_cluster_window_type_counts)
                    << " window_order_signature="
                    << (replay_smart_cluster_window_order_signature.empty()
                            ? "none"
                            : replay_smart_cluster_window_order_signature)
                    << " window_repeated_key_signature="
                    << replay_smart_cluster_window_repeated_key_signature
                    << " window_repeated_measure_signature="
                    << replay_smart_cluster_window_repeated_measure_signature
                    << std::endl;
          std::cerr << std::setprecision(12)
                    << "[CBS][H2ReplaySmartClusterOrderDiag]"
                    << " curr_kf_id=" << curr_kf_id
                    << " good_prefix_len="
                    << replay_bootstrap_window_good_prefix_len
                    << " fail_idx=" << replay_bootstrap_window_fail_idx
                    << " order_any_success="
                    << replay_smart_cluster_order_any_success
                    << " order_success_mask="
                    << (replay_smart_cluster_order_success_mask.empty()
                            ? "none"
                            : replay_smart_cluster_order_success_mask)
                    << " order_reason_mask="
                    << (replay_smart_cluster_order_reason_mask.empty()
                            ? "none"
                            : replay_smart_cluster_order_reason_mask)
                    << " first_success_variant="
                    << (replay_smart_cluster_order_first_success_variant.empty()
                            ? "none"
                            : replay_smart_cluster_order_first_success_variant)
                    << " first_success_reason="
                    << (replay_smart_cluster_order_first_success_reason.empty()
                            ? "none"
                            : replay_smart_cluster_order_first_success_reason)
                    << std::endl;
          std::cerr << std::setprecision(12)
                    << "[CBS][H2ReplayFamilyAppendDiag]"
                    << " curr_kf_id=" << curr_kf_id
                    << " good_prefix_len="
                    << replay_bootstrap_window_good_prefix_len
                    << " factor_244_key_signature="
                    << (replay_smart_cluster_factor_244_key_signature.empty()
                            ? "none"
                            : replay_smart_cluster_factor_244_key_signature)
                    << " factor_245_key_signature="
                    << (replay_smart_cluster_factor_245_key_signature.empty()
                            ? "none"
                            : replay_smart_cluster_factor_245_key_signature)
                    << " family244_single_append_success_mask="
                    << (replay_family244_single_append_success_mask.empty()
                            ? "none"
                            : replay_family244_single_append_success_mask)
                    << " family244_single_append_reason_mask="
                    << (replay_family244_single_append_reason_mask.empty()
                            ? "none"
                            : replay_family244_single_append_reason_mask)
                    << " family245_single_append_success_mask="
                    << (replay_family245_single_append_success_mask.empty()
                            ? "none"
                            : replay_family245_single_append_success_mask)
                    << " family245_single_append_reason_mask="
                    << (replay_family245_single_append_reason_mask.empty()
                            ? "none"
                            : replay_family245_single_append_reason_mask)
                    << " family244_standalone_success_mask="
                    << (replay_family244_standalone_success_mask.empty()
                            ? "none"
                            : replay_family244_standalone_success_mask)
                    << " family244_standalone_reason_mask="
                    << (replay_family244_standalone_reason_mask.empty()
                            ? "none"
                            : replay_family244_standalone_reason_mask)
                    << " family245_standalone_success_mask="
                    << (replay_family245_standalone_success_mask.empty()
                            ? "none"
                            : replay_family245_standalone_success_mask)
                    << " family245_standalone_reason_mask="
                    << (replay_family245_standalone_reason_mask.empty()
                            ? "none"
                            : replay_family245_standalone_reason_mask)
                    << " family244_any_append_success="
                    << replay_family244_any_append_success
                    << " family245_any_append_success="
                    << replay_family245_any_append_success
                    << " family244_all_append_fail="
                    << replay_family244_all_append_fail
                    << " family245_all_append_fail="
                    << replay_family245_all_append_fail
                    << " family244_first_append_success_idx="
                    << replay_family244_first_append_success_idx
                    << " family245_first_append_success_idx="
                    << replay_family245_first_append_success_idx
                    << " family244_first_append_fail_idx="
                    << replay_family244_first_append_fail_idx
                    << " family245_first_append_fail_idx="
                    << replay_family245_first_append_fail_idx
                    << std::endl;
          std::cerr << std::setprecision(12)
                    << "[CBS][H2ReplayAppendFrontierDiag]"
                    << " curr_kf_id=" << curr_kf_id
                    << " good_prefix_len="
                    << replay_bootstrap_window_good_prefix_len
                    << " frontier_idx=" << replay_append_frontier_idx
                    << " idx0=" << replay_append_frontier_idx0
                    << " idx1=" << replay_append_frontier_idx1
                    << " idx2=" << replay_append_frontier_idx2
                    << " idx0_type="
                    << (replay_append_frontier_idx0_type.empty()
                            ? "none"
                            : replay_append_frontier_idx0_type)
                    << " idx1_type="
                    << (replay_append_frontier_idx1_type.empty()
                            ? "none"
                            : replay_append_frontier_idx1_type)
                    << " idx2_type="
                    << (replay_append_frontier_idx2_type.empty()
                            ? "none"
                            : replay_append_frontier_idx2_type)
                    << " idx0_keysig="
                    << (replay_append_frontier_idx0_keysig.empty()
                            ? "none"
                            : replay_append_frontier_idx0_keysig)
                    << " idx1_keysig="
                    << (replay_append_frontier_idx1_keysig.empty()
                            ? "none"
                            : replay_append_frontier_idx1_keysig)
                    << " idx2_keysig="
                    << (replay_append_frontier_idx2_keysig.empty()
                            ? "none"
                            : replay_append_frontier_idx2_keysig)
                    << " idx0_meassig="
                    << (replay_append_frontier_idx0_meassig.empty()
                            ? "none"
                            : replay_append_frontier_idx0_meassig)
                    << " idx1_meassig="
                    << (replay_append_frontier_idx1_meassig.empty()
                            ? "none"
                            : replay_append_frontier_idx1_meassig)
                    << " idx2_meassig="
                    << (replay_append_frontier_idx2_meassig.empty()
                            ? "none"
                            : replay_append_frontier_idx2_meassig)
                    << " idx0_pointer="
                    << (replay_append_frontier_idx0_pointer.empty()
                            ? "none"
                            : replay_append_frontier_idx0_pointer)
                    << " idx1_pointer="
                    << (replay_append_frontier_idx1_pointer.empty()
                            ? "none"
                            : replay_append_frontier_idx1_pointer)
                    << " idx2_pointer="
                    << (replay_append_frontier_idx2_pointer.empty()
                            ? "none"
                            : replay_append_frontier_idx2_pointer)
                    << " idx0_measured_count="
                    << replay_append_frontier_idx0_measured_count
                    << " idx1_measured_count="
                    << replay_append_frontier_idx1_measured_count
                    << " idx2_measured_count="
                    << replay_append_frontier_idx2_measured_count
                    << " idx0_point_valid="
                    << replay_append_frontier_idx0_point_valid
                    << " idx1_point_valid="
                    << replay_append_frontier_idx1_point_valid
                    << " idx2_point_valid="
                    << replay_append_frontier_idx2_point_valid
                    << " idx0_idx1_same_keys="
                    << replay_append_frontier_idx0_idx1_same_keys
                    << " idx1_idx2_same_keys="
                    << replay_append_frontier_idx1_idx2_same_keys
                    << " idx0_idx2_same_keys="
                    << replay_append_frontier_idx0_idx2_same_keys
                    << " idx0_idx1_same_measure="
                    << replay_append_frontier_idx0_idx1_same_measure
                    << " idx1_idx2_same_measure="
                    << replay_append_frontier_idx1_idx2_same_measure
                    << " idx0_idx2_same_measure="
                    << replay_append_frontier_idx0_idx2_same_measure
                    << " append_idx0_succeeded="
                    << (replay_append_frontier_append_idx0_succeeded ? 1 : 0)
                    << " append_idx0_reason="
                    << (replay_append_frontier_append_idx0_reason.empty()
                            ? "none"
                            : replay_append_frontier_append_idx0_reason)
                    << " append_idx1_succeeded="
                    << (replay_append_frontier_append_idx1_succeeded ? 1 : 0)
                    << " append_idx1_reason="
                    << (replay_append_frontier_append_idx1_reason.empty()
                            ? "none"
                            : replay_append_frontier_append_idx1_reason)
                    << " append_idx2_succeeded="
                    << (replay_append_frontier_append_idx2_succeeded ? 1 : 0)
                    << " append_idx2_reason="
                    << (replay_append_frontier_append_idx2_reason.empty()
                            ? "none"
                            : replay_append_frontier_append_idx2_reason)
                    << " append_idx0_idx1_succeeded="
                    << (replay_append_frontier_append_idx0_idx1_succeeded ? 1 : 0)
                    << " append_idx0_idx1_reason="
                    << (replay_append_frontier_append_idx0_idx1_reason.empty()
                            ? "none"
                            : replay_append_frontier_append_idx0_idx1_reason)
                    << " append_idx1_idx0_succeeded="
                    << (replay_append_frontier_append_idx1_idx0_succeeded ? 1 : 0)
                    << " append_idx1_idx0_reason="
                    << (replay_append_frontier_append_idx1_idx0_reason.empty()
                            ? "none"
                            : replay_append_frontier_append_idx1_idx0_reason)
                    << " append_idx0_idx1_idx2_succeeded="
                    << (replay_append_frontier_append_idx0_idx1_idx2_succeeded ? 1
                                                                               : 0)
                    << " append_idx0_idx1_idx2_reason="
                    << (replay_append_frontier_append_idx0_idx1_idx2_reason.empty()
                            ? "none"
                            : replay_append_frontier_append_idx0_idx1_idx2_reason)
                    << " append_idx1_idx2_succeeded="
                    << (replay_append_frontier_append_idx1_idx2_succeeded ? 1 : 0)
                    << " append_idx1_idx2_reason="
                    << (replay_append_frontier_append_idx1_idx2_reason.empty()
                            ? "none"
                            : replay_append_frontier_append_idx1_idx2_reason)
                    << " append_idx0_idx2_succeeded="
                    << (replay_append_frontier_append_idx0_idx2_succeeded ? 1 : 0)
                    << " append_idx0_idx2_reason="
                    << (replay_append_frontier_append_idx0_idx2_reason.empty()
                            ? "none"
                            : replay_append_frontier_append_idx0_idx2_reason)
                    << " standalone_idx0_succeeded="
                    << (replay_append_frontier_standalone_idx0_succeeded ? 1 : 0)
                    << " standalone_idx0_reason="
                    << (replay_append_frontier_standalone_idx0_reason.empty()
                            ? "none"
                            : replay_append_frontier_standalone_idx0_reason)
                    << " standalone_idx1_succeeded="
                    << (replay_append_frontier_standalone_idx1_succeeded ? 1 : 0)
                    << " standalone_idx1_reason="
                    << (replay_append_frontier_standalone_idx1_reason.empty()
                            ? "none"
                            : replay_append_frontier_standalone_idx1_reason)
                    << " standalone_idx2_succeeded="
                    << (replay_append_frontier_standalone_idx2_succeeded ? 1 : 0)
                    << " standalone_idx2_reason="
                    << (replay_append_frontier_standalone_idx2_reason.empty()
                            ? "none"
                            : replay_append_frontier_standalone_idx2_reason)
                    << std::endl;
          std::cerr << std::setprecision(12)
                    << "[CBS][H2ReplayFrontierBoundaryDiag]"
                    << " curr_kf_id=" << curr_kf_id
                    << " good_prefix_len="
                    << replay_bootstrap_window_good_prefix_len
                    << " frontier_idx=" << replay_append_frontier_idx
                    << " idx0=" << replay_append_frontier_idx0
                    << " idx1=" << replay_append_frontier_idx1
                    << " idx2=" << replay_append_frontier_idx2
                    << " idx0_type="
                    << (replay_append_frontier_idx0_type.empty()
                            ? "none"
                            : replay_append_frontier_idx0_type)
                    << " idx1_type="
                    << (replay_append_frontier_idx1_type.empty()
                            ? "none"
                            : replay_append_frontier_idx1_type)
                    << " idx2_type="
                    << (replay_append_frontier_idx2_type.empty()
                            ? "none"
                            : replay_append_frontier_idx2_type)
                    << " idx0_factor_keys="
                    << (replay_append_frontier_idx0_factor_keys.empty()
                            ? "none"
                            : replay_append_frontier_idx0_factor_keys)
                    << " idx1_factor_keys="
                    << (replay_append_frontier_idx1_factor_keys.empty()
                            ? "none"
                            : replay_append_frontier_idx1_factor_keys)
                    << " idx2_factor_keys="
                    << (replay_append_frontier_idx2_factor_keys.empty()
                            ? "none"
                            : replay_append_frontier_idx2_factor_keys)
                    << " idx0_key_presence="
                    << (replay_append_frontier_idx0_key_presence_mask.empty()
                            ? "none"
                            : replay_append_frontier_idx0_key_presence_mask)
                    << " idx1_key_presence="
                    << (replay_append_frontier_idx1_key_presence_mask.empty()
                            ? "none"
                            : replay_append_frontier_idx1_key_presence_mask)
                    << " idx2_key_presence="
                    << (replay_append_frontier_idx2_key_presence_mask.empty()
                            ? "none"
                            : replay_append_frontier_idx2_key_presence_mask)
                    << " idx0_introduces_new_key="
                    << replay_append_frontier_idx0_introduces_new_key
                    << " idx1_introduces_new_key="
                    << replay_append_frontier_idx1_introduces_new_key
                    << " idx2_introduces_new_key="
                    << replay_append_frontier_idx2_introduces_new_key
                    << " idx0_all_keys_in_good_prefix="
                    << replay_append_frontier_idx0_all_keys_in_good_prefix_values
                    << " idx1_all_keys_in_good_prefix="
                    << replay_append_frontier_idx1_all_keys_in_good_prefix_values
                    << " idx2_all_keys_in_good_prefix="
                    << replay_append_frontier_idx2_all_keys_in_good_prefix_values
                    << " boundary_pair_union_keys="
                    << (replay_append_frontier_pair_union_key_summary.empty()
                            ? "none"
                            : replay_append_frontier_pair_union_key_summary)
                    << " prefix_touch_count_idx0="
                    << replay_append_frontier_prefix_touch_count_idx0
                    << " prefix_touch_count_idx1="
                    << replay_append_frontier_prefix_touch_count_idx1
                    << " prefix_touch_sample_idx0="
                    << (replay_append_frontier_prefix_touch_sample_idx0.empty()
                            ? "none"
                            : replay_append_frontier_prefix_touch_sample_idx0)
                    << " prefix_touch_sample_idx1="
                    << (replay_append_frontier_prefix_touch_sample_idx1.empty()
                            ? "none"
                            : replay_append_frontier_prefix_touch_sample_idx1)
                    << " prefix_has_imu_like_idx0="
                    << replay_append_frontier_prefix_has_imu_like_idx0
                    << " prefix_has_between_like_idx1="
                    << replay_append_frontier_prefix_has_between_like_idx1
                    << " append_idx0_succeeded="
                    << (replay_append_frontier_append_idx0_succeeded ? 1 : 0)
                    << " append_idx0_reason="
                    << (replay_append_frontier_append_idx0_reason.empty()
                            ? "none"
                            : replay_append_frontier_append_idx0_reason)
                    << " append_idx1_succeeded="
                    << (replay_append_frontier_append_idx1_succeeded ? 1 : 0)
                    << " append_idx1_reason="
                    << (replay_append_frontier_append_idx1_reason.empty()
                            ? "none"
                            : replay_append_frontier_append_idx1_reason)
                    << " append_idx2_succeeded="
                    << (replay_append_frontier_append_idx2_succeeded ? 1 : 0)
                    << " append_idx2_reason="
                    << (replay_append_frontier_append_idx2_reason.empty()
                            ? "none"
                            : replay_append_frontier_append_idx2_reason)
                    << " standalone_idx0_succeeded="
                    << (replay_append_frontier_standalone_idx0_succeeded ? 1 : 0)
                    << " standalone_idx0_reason="
                    << (replay_append_frontier_standalone_idx0_reason.empty()
                            ? "none"
                            : replay_append_frontier_standalone_idx0_reason)
                    << " standalone_idx1_succeeded="
                    << (replay_append_frontier_standalone_idx1_succeeded ? 1 : 0)
                    << " standalone_idx1_reason="
                    << (replay_append_frontier_standalone_idx1_reason.empty()
                            ? "none"
                            : replay_append_frontier_standalone_idx1_reason)
                    << " standalone_idx2_succeeded="
                    << (replay_append_frontier_standalone_idx2_succeeded ? 1 : 0)
                    << " standalone_idx2_reason="
                    << (replay_append_frontier_standalone_idx2_reason.empty()
                            ? "none"
                            : replay_append_frontier_standalone_idx2_reason)
                    << std::endl;
          std::cerr << std::setprecision(12)
                    << "[CBS][H2ReplayPrefixSmartContextDiag]"
                    << " curr_kf_id=" << curr_kf_id
                    << " good_prefix_len="
                    << replay_bootstrap_window_good_prefix_len
                    << " frontier_idx=" << replay_append_frontier_idx
                    << " idx0=" << replay_append_frontier_idx0
                    << " idx1=" << replay_append_frontier_idx1
                    << " idx2=" << replay_append_frontier_idx2
                    << " idx0_keys="
                    << (replay_append_frontier_idx0_factor_keys.empty()
                            ? "none"
                            : replay_append_frontier_idx0_factor_keys)
                    << " idx1_keys="
                    << (replay_append_frontier_idx1_factor_keys.empty()
                            ? "none"
                            : replay_append_frontier_idx1_factor_keys)
                    << " prefix_smart_touch_idx0_count="
                    << replay_prefix_smart_touch_idx0_count
                    << " prefix_smart_touch_idx1_count="
                    << replay_prefix_smart_touch_idx1_count
                    << " prefix_smart_touch_both_count="
                    << replay_prefix_smart_touch_both_count
                    << " prefix_smart_touch_idx0_samples="
                    << (replay_prefix_smart_touch_idx0_samples.empty()
                            ? "none"
                            : replay_prefix_smart_touch_idx0_samples)
                    << " prefix_smart_touch_idx1_samples="
                    << (replay_prefix_smart_touch_idx1_samples.empty()
                            ? "none"
                            : replay_prefix_smart_touch_idx1_samples)
                    << " prefix_smart_touch_both_samples="
                    << (replay_prefix_smart_touch_both_samples.empty()
                            ? "none"
                            : replay_prefix_smart_touch_both_samples)
                    << " prefix_same_keysig_as_idx0_count="
                    << replay_prefix_same_keysig_as_idx0_count
                    << " prefix_same_keysig_as_idx0_indices="
                    << (replay_prefix_same_keysig_as_idx0_indices.empty()
                            ? "none"
                            : replay_prefix_same_keysig_as_idx0_indices)
                    << " prefix_same_keysig_as_idx1_count="
                    << replay_prefix_same_keysig_as_idx1_count
                    << " prefix_same_keysig_as_idx1_indices="
                    << (replay_prefix_same_keysig_as_idx1_indices.empty()
                            ? "none"
                            : replay_prefix_same_keysig_as_idx1_indices)
                    << " prefix_same_meassig_as_idx0_count="
                    << replay_prefix_same_meassig_as_idx0_count
                    << " prefix_same_meassig_as_idx0_indices="
                    << (replay_prefix_same_meassig_as_idx0_indices.empty()
                            ? "none"
                            : replay_prefix_same_meassig_as_idx0_indices)
                    << " prefix_same_meassig_as_idx1_count="
                    << replay_prefix_same_meassig_as_idx1_count
                    << " prefix_same_meassig_as_idx1_indices="
                    << (replay_prefix_same_meassig_as_idx1_indices.empty()
                            ? "none"
                            : replay_prefix_same_meassig_as_idx1_indices)
                    << " prefix_prev_m3="
                    << (replay_prefix_prev_m3_context.empty()
                            ? "none"
                            : replay_prefix_prev_m3_context)
                    << " prefix_prev_m2="
                    << (replay_prefix_prev_m2_context.empty()
                            ? "none"
                            : replay_prefix_prev_m2_context)
                    << " prefix_prev_m1="
                    << (replay_prefix_prev_m1_context.empty()
                            ? "none"
                            : replay_prefix_prev_m1_context)
                    << " idx0_idx1_same_keysig="
                    << replay_append_frontier_idx0_idx1_same_keys
                    << " idx0_idx1_same_meassig="
                    << replay_append_frontier_idx0_idx1_same_measure
                    << " append_idx0_succeeded="
                    << (replay_append_frontier_append_idx0_succeeded ? 1 : 0)
                    << " append_idx0_reason="
                    << (replay_append_frontier_append_idx0_reason.empty()
                            ? "none"
                            : replay_append_frontier_append_idx0_reason)
                    << " append_idx1_succeeded="
                    << (replay_append_frontier_append_idx1_succeeded ? 1 : 0)
                    << " append_idx1_reason="
                    << (replay_append_frontier_append_idx1_reason.empty()
                            ? "none"
                            : replay_append_frontier_append_idx1_reason)
                    << " standalone_idx0_succeeded="
                    << (replay_append_frontier_standalone_idx0_succeeded ? 1 : 0)
                    << " standalone_idx0_reason="
                    << (replay_append_frontier_standalone_idx0_reason.empty()
                            ? "none"
                            : replay_append_frontier_standalone_idx0_reason)
                    << " standalone_idx1_succeeded="
                    << (replay_append_frontier_standalone_idx1_succeeded ? 1 : 0)
                    << " standalone_idx1_reason="
                    << (replay_append_frontier_standalone_idx1_reason.empty()
                            ? "none"
                            : replay_append_frontier_standalone_idx1_reason)
                    << std::endl;

          struct ReplayProbeStageResult {
            bool bootstrap_succeeded = false;
            std::string bootstrap_reason = "not_run";
            bool update_succeeded = false;
            std::string update_reason = "not_run";
          };

          auto run_replay_subset_probe_stages =
              [&](const gtsam::NonlinearFactorGraph& replay_factors)
              -> ReplayProbeStageResult {
            ReplayProbeStageResult stage_result;
            if (!replay_probe_base_available) {
              stage_result.bootstrap_reason = "probe_clone_unavailable";
              stage_result.update_reason = "probe_bootstrap_not_run";
              return stage_result;
            }

            std::shared_ptr<cbs::BPSAM> probe_sidecar;
            try {
              probe_sidecar = make_probe_sidecar();
              cbs::BPSAM::UpdateParams bootstrap_params;
              probe_sidecar->update(use_sanitized_bootstrap_for_stage_probes
                                        ? replay_bootstrap_audit
                                              .sanitized_bootstrap_graph
                                        : replay_probe_base_graph,
                                    replay_probe_base_values,
                                    bootstrap_params);
              stage_result.bootstrap_succeeded = true;
              stage_result.bootstrap_reason = "none";
            } catch (const std::exception& e) {
              stage_result.bootstrap_reason =
                  e.what() ? std::string(e.what())
                           : std::string("probe_bootstrap_std_exception");
              stage_result.update_reason = "probe_bootstrap_failed";
              return stage_result;
            } catch (...) {
              stage_result.bootstrap_reason =
                  "probe_bootstrap_unknown_exception";
              stage_result.update_reason = "probe_bootstrap_failed";
              return stage_result;
            }

            try {
              cbs::BPSAM::UpdateParams probe_params;
              probe_sidecar->update(replay_factors,
                                    packet.local_values,
                                    probe_params);
              stage_result.update_succeeded = true;
              stage_result.update_reason = "none";
            } catch (const std::exception& e) {
              stage_result.update_reason =
                  e.what() ? std::string(e.what())
                           : std::string("probe_update_std_exception");
            } catch (...) {
              stage_result.update_reason = "probe_update_unknown_exception";
            }
            return stage_result;
          };

          auto populate_subset_from_stage =
              [&](const ReplayProbeStageResult& stage_result,
                  bool* subset_succeeded,
                  std::string* subset_reason,
                  bool* bootstrap_succeeded,
                  std::string* bootstrap_reason,
                  bool* update_succeeded,
                  std::string* update_reason) {
                CHECK_NOTNULL(subset_succeeded);
                CHECK_NOTNULL(subset_reason);
                CHECK_NOTNULL(bootstrap_succeeded);
                CHECK_NOTNULL(bootstrap_reason);
                CHECK_NOTNULL(update_succeeded);
                CHECK_NOTNULL(update_reason);
                *bootstrap_succeeded = stage_result.bootstrap_succeeded;
                *bootstrap_reason = stage_result.bootstrap_reason.empty()
                                        ? "none"
                                        : stage_result.bootstrap_reason;
                *update_succeeded = stage_result.update_succeeded;
                *update_reason = stage_result.update_reason.empty()
                                     ? "none"
                                     : stage_result.update_reason;
                *subset_succeeded =
                    stage_result.bootstrap_succeeded &&
                    stage_result.update_succeeded;
                if (*subset_succeeded) {
                  *subset_reason = "none";
                } else if (!stage_result.bootstrap_succeeded) {
                  *subset_reason = *bootstrap_reason;
                } else {
                  *subset_reason = *update_reason;
                }
              };

          const gtsam::NonlinearFactorGraph empty_replay_factors;
          const ReplayProbeStageResult empty_stage_result =
              run_replay_subset_probe_stages(empty_replay_factors);
          const ReplayProbeStageResult non_smart_stage_result =
              run_replay_subset_probe_stages(no_smart_any_factors);
          const ReplayProbeStageResult smart_stage_result =
              run_replay_subset_probe_stages(smart_only_factors);
          const ReplayProbeStageResult imu_only_stage_result =
              run_replay_subset_probe_stages(imu_only_factors);
          const ReplayProbeStageResult between_only_stage_result =
              run_replay_subset_probe_stages(between_only_factors);
          const ReplayProbeStageResult full_stage_result =
              run_replay_subset_probe_stages(packet.local_factors);

          populate_subset_from_stage(
              empty_stage_result,
              &stats->h2_replay_subset_empty_succeeded,
              &stats->h2_replay_subset_empty_reason,
              &stats->h2_replay_probe_bootstrap_empty_succeeded,
              &stats->h2_replay_probe_bootstrap_empty_reason,
              &stats->h2_replay_probe_update_empty_succeeded,
              &stats->h2_replay_probe_update_empty_reason);
          populate_subset_from_stage(
              non_smart_stage_result,
              &stats->h2_replay_subset_non_smart_succeeded,
              &stats->h2_replay_subset_non_smart_reason,
              &stats->h2_replay_probe_bootstrap_non_smart_succeeded,
              &stats->h2_replay_probe_bootstrap_non_smart_reason,
              &stats->h2_replay_probe_update_non_smart_succeeded,
              &stats->h2_replay_probe_update_non_smart_reason);
          populate_subset_from_stage(
              smart_stage_result,
              &stats->h2_replay_subset_smart_succeeded,
              &stats->h2_replay_subset_smart_reason,
              &stats->h2_replay_probe_bootstrap_smart_succeeded,
              &stats->h2_replay_probe_bootstrap_smart_reason,
              &stats->h2_replay_probe_update_smart_succeeded,
              &stats->h2_replay_probe_update_smart_reason);
          populate_subset_from_stage(
              imu_only_stage_result,
              &stats->h2_replay_subset_imu_only_succeeded,
              &stats->h2_replay_subset_imu_only_reason,
              &stats->h2_replay_probe_bootstrap_imu_only_succeeded,
              &stats->h2_replay_probe_bootstrap_imu_only_reason,
              &stats->h2_replay_probe_update_imu_only_succeeded,
              &stats->h2_replay_probe_update_imu_only_reason);
          populate_subset_from_stage(
              between_only_stage_result,
              &stats->h2_replay_subset_between_only_succeeded,
              &stats->h2_replay_subset_between_only_reason,
              &stats->h2_replay_probe_bootstrap_between_only_succeeded,
              &stats->h2_replay_probe_bootstrap_between_only_reason,
              &stats->h2_replay_probe_update_between_only_succeeded,
              &stats->h2_replay_probe_update_between_only_reason);
          populate_subset_from_stage(
              full_stage_result,
              &stats->h2_replay_subset_full_probe_succeeded,
              &stats->h2_replay_subset_full_probe_reason,
              &stats->h2_replay_probe_bootstrap_full_succeeded,
              &stats->h2_replay_probe_bootstrap_full_reason,
              &stats->h2_replay_probe_update_full_succeeded,
              &stats->h2_replay_probe_update_full_reason);

          std::cerr << std::setprecision(12)
                    << "[CBS][H2ReplayProbeStageDiag]"
                    << " curr_kf_id=" << curr_kf_id
                    << " probe_clone_available="
                    << (stats->h2_replay_subset_probe_clone_available ? 1 : 0)
                    << " probe_clone_setup_reason="
                    << (stats->h2_replay_subset_probe_clone_setup_reason.empty()
                            ? "none"
                            : stats->h2_replay_subset_probe_clone_setup_reason)
                    << " bootstrap_empty_succeeded="
                    << (stats->h2_replay_probe_bootstrap_empty_succeeded ? 1 : 0)
                    << " bootstrap_non_smart_succeeded="
                    << (stats->h2_replay_probe_bootstrap_non_smart_succeeded ? 1
                                                                              : 0)
                    << " bootstrap_smart_succeeded="
                    << (stats->h2_replay_probe_bootstrap_smart_succeeded ? 1 : 0)
                    << " bootstrap_imu_only_succeeded="
                    << (stats->h2_replay_probe_bootstrap_imu_only_succeeded ? 1
                                                                             : 0)
                    << " bootstrap_between_only_succeeded="
                    << (stats->h2_replay_probe_bootstrap_between_only_succeeded
                            ? 1
                            : 0)
                    << " bootstrap_full_succeeded="
                    << (stats->h2_replay_probe_bootstrap_full_succeeded ? 1 : 0)
                    << " update_empty_succeeded="
                    << (stats->h2_replay_probe_update_empty_succeeded ? 1 : 0)
                    << " update_non_smart_succeeded="
                    << (stats->h2_replay_probe_update_non_smart_succeeded ? 1 : 0)
                    << " update_smart_succeeded="
                    << (stats->h2_replay_probe_update_smart_succeeded ? 1 : 0)
                    << " update_imu_only_succeeded="
                    << (stats->h2_replay_probe_update_imu_only_succeeded ? 1 : 0)
                    << " update_between_only_succeeded="
                    << (stats->h2_replay_probe_update_between_only_succeeded ? 1
                                                                             : 0)
                    << " update_full_succeeded="
                    << (stats->h2_replay_probe_update_full_succeeded ? 1 : 0)
                    << " bootstrap_empty_reason="
                    << (stats->h2_replay_probe_bootstrap_empty_reason.empty()
                            ? "none"
                            : stats->h2_replay_probe_bootstrap_empty_reason)
                    << " bootstrap_non_smart_reason="
                    << (stats->h2_replay_probe_bootstrap_non_smart_reason.empty()
                            ? "none"
                            : stats->h2_replay_probe_bootstrap_non_smart_reason)
                    << " bootstrap_smart_reason="
                    << (stats->h2_replay_probe_bootstrap_smart_reason.empty()
                            ? "none"
                            : stats->h2_replay_probe_bootstrap_smart_reason)
                    << " bootstrap_imu_only_reason="
                    << (stats->h2_replay_probe_bootstrap_imu_only_reason.empty()
                            ? "none"
                            : stats->h2_replay_probe_bootstrap_imu_only_reason)
                    << " bootstrap_between_only_reason="
                    << (stats->h2_replay_probe_bootstrap_between_only_reason.empty()
                            ? "none"
                            : stats->h2_replay_probe_bootstrap_between_only_reason)
                    << " bootstrap_full_reason="
                    << (stats->h2_replay_probe_bootstrap_full_reason.empty()
                            ? "none"
                            : stats->h2_replay_probe_bootstrap_full_reason)
                    << " update_empty_reason="
                    << (stats->h2_replay_probe_update_empty_reason.empty()
                            ? "none"
                            : stats->h2_replay_probe_update_empty_reason)
                    << " update_non_smart_reason="
                    << (stats->h2_replay_probe_update_non_smart_reason.empty()
                            ? "none"
                            : stats->h2_replay_probe_update_non_smart_reason)
                    << " update_smart_reason="
                    << (stats->h2_replay_probe_update_smart_reason.empty()
                            ? "none"
                            : stats->h2_replay_probe_update_smart_reason)
                    << " update_imu_only_reason="
                    << (stats->h2_replay_probe_update_imu_only_reason.empty()
                            ? "none"
                            : stats->h2_replay_probe_update_imu_only_reason)
                    << " update_between_only_reason="
                    << (stats->h2_replay_probe_update_between_only_reason.empty()
                            ? "none"
                            : stats->h2_replay_probe_update_between_only_reason)
                    << " update_full_reason="
                    << (stats->h2_replay_probe_update_full_reason.empty()
                            ? "none"
                            : stats->h2_replay_probe_update_full_reason)
                    << std::endl;

          std::cerr << std::setprecision(12)
                    << "[CBS][H2ReplayAddOnlyABDiag]"
                    << " curr_kf_id=" << curr_kf_id
                    << " full_replay_attempted=1"
                    << " full_replay_reason="
                    << (replay_reason.empty() ? "none" : replay_reason)
                    << " probe_clone_available="
                    << (stats->h2_replay_subset_probe_clone_available ? 1 : 0)
                    << " probe_clone_setup_reason="
                    << (stats->h2_replay_subset_probe_clone_setup_reason.empty()
                            ? "none"
                            : stats->h2_replay_subset_probe_clone_setup_reason)
                    << " smart_count=" << stats->h2_replay_subset_smart_count
                    << " non_smart_count="
                    << stats->h2_replay_subset_non_smart_count
                    << " imu_count=" << stats->h2_replay_subset_imu_count
                    << " between_count=" << stats->h2_replay_subset_between_count
                    << " empty_succeeded="
                    << (stats->h2_replay_subset_empty_succeeded ? 1 : 0)
                    << " non_smart_succeeded="
                    << (stats->h2_replay_subset_non_smart_succeeded ? 1 : 0)
                    << " smart_succeeded="
                    << (stats->h2_replay_subset_smart_succeeded ? 1 : 0)
                    << " imu_only_succeeded="
                    << (stats->h2_replay_subset_imu_only_succeeded ? 1 : 0)
                    << " between_only_succeeded="
                    << (stats->h2_replay_subset_between_only_succeeded ? 1 : 0)
                    << " full_probe_succeeded="
                    << (stats->h2_replay_subset_full_probe_succeeded ? 1 : 0)
                    << " empty_reason="
                    << (stats->h2_replay_subset_empty_reason.empty()
                            ? "none"
                            : stats->h2_replay_subset_empty_reason)
                    << " non_smart_reason="
                    << (stats->h2_replay_subset_non_smart_reason.empty()
                            ? "none"
                            : stats->h2_replay_subset_non_smart_reason)
                    << " smart_reason="
                    << (stats->h2_replay_subset_smart_reason.empty()
                            ? "none"
                            : stats->h2_replay_subset_smart_reason)
                    << " imu_only_reason="
                    << (stats->h2_replay_subset_imu_only_reason.empty()
                            ? "none"
                            : stats->h2_replay_subset_imu_only_reason)
                    << " between_only_reason="
                    << (stats->h2_replay_subset_between_only_reason.empty()
                            ? "none"
                            : stats->h2_replay_subset_between_only_reason)
                    << " full_probe_reason="
                    << (stats->h2_replay_subset_full_probe_reason.empty()
                            ? "none"
                            : stats->h2_replay_subset_full_probe_reason)
                    << std::endl;

          const bool decisive_non_smart_only =
              stats->h2_replay_subset_non_smart_succeeded &&
              !stats->h2_replay_subset_smart_succeeded;
          if (decisive_non_smart_only && replay_probe_base_available) {
            try {
              std::shared_ptr<cbs::BPSAM> restored_sidecar = make_probe_sidecar();
              cbs::BPSAM::UpdateParams bootstrap_params;
              restored_sidecar->update(replay_probe_base_graph,
                                       replay_probe_base_values,
                                       bootstrap_params);
              cbs_local_cov_sidecar_ = restored_sidecar;
            } catch (const std::exception& e) {
              const std::string restore_reason =
                  e.what()
                      ? std::string(e.what())
                      : std::string("h2_repair_epoch_partial_replay_restore_std_exception");
              return fail_and_hard_reset(
                  "h2_repair_epoch_partial_replay_restore_failed:" +
                  restore_reason);
            } catch (...) {
              return fail_and_hard_reset(
                  "h2_repair_epoch_partial_replay_restore_failed:"
                  "h2_repair_epoch_partial_replay_restore_unknown_exception");
            }

            gtsam::ISAM2Result non_smart_replay_result;
            try {
              cbs::BPSAM::UpdateParams non_smart_replay_params;
              non_smart_replay_result = cbs_local_cov_sidecar_->update(
                  no_smart_any_factors,
                  packet.local_values,
                  non_smart_replay_params);
            } catch (const std::exception& e) {
              const std::string partial_reason =
                  e.what()
                      ? std::string(e.what())
                      : std::string("h2_repair_epoch_partial_replay_std_exception");
              return fail_and_hard_reset(
                  "h2_repair_epoch_partial_replay_non_smart_only_failed:" +
                  partial_reason);
            } catch (...) {
              return fail_and_hard_reset(
                  "h2_repair_epoch_partial_replay_non_smart_only_failed:"
                  "h2_repair_epoch_partial_replay_unknown_exception");
            }

            resolved_sidecar_new_indices.assign(
                non_smart_replay_result.newFactorsIndices.begin(),
                non_smart_replay_result.newFactorsIndices.end());
            resolved_heart_new_factor_positions.clear();
            resolved_heart_new_factor_positions.reserve(
                non_smart_local_factor_positions.size());
            for (const size_t local_factor_pos : non_smart_local_factor_positions) {
              if (local_factor_pos >= packet.heart_new_factor_positions.size()) {
                std::ostringstream oss;
                oss << "h2_repair_epoch_partial_replay_non_smart_pos_oob pos="
                    << local_factor_pos
                    << " heart_positions="
                    << packet.heart_new_factor_positions.size();
                return fail_and_hard_reset(oss.str());
              }
              resolved_heart_new_factor_positions.push_back(
                  packet.heart_new_factor_positions[local_factor_pos]);
            }

            if (resolved_sidecar_new_indices.size() !=
                resolved_heart_new_factor_positions.size()) {
              std::ostringstream oss;
              oss << "h2_repair_epoch_partial_replay_non_smart_only_mismatch"
                  << " sidecar=" << resolved_sidecar_new_indices.size()
                  << " non_smart_packet="
                  << resolved_heart_new_factor_positions.size();
              return fail_and_hard_reset(oss.str());
            }

            stats->h2_partial_replay_non_smart_only_applied = true;
            stats->h2_partial_replay_reason =
                "h2_repair_epoch_partial_replay_non_smart_only";
          } else if (!stats->h2_replay_subset_empty_succeeded &&
                     !stats->h2_replay_subset_non_smart_succeeded &&
                     !stats->h2_replay_subset_smart_succeeded &&
                     !stats->h2_replay_subset_imu_only_succeeded &&
                     !stats->h2_replay_subset_between_only_succeeded &&
                     !stats->h2_replay_subset_full_probe_succeeded) {
            LOG(WARNING)
                << "H2 repair-epoch replay probes all failed (including empty); "
                << "repaired sidecar state appears poisoned before replay.";
          }
        }

        if (!stats->h2_partial_replay_non_smart_only_applied) {
          return fail_and_hard_reset("h2_sidecar_replay_add_only_failed:" +
                                     replay_reason);
        }
      } else {
        stats->h2_replay_add_only_succeeded = true;
        stats->h2_replay_add_only_new_index_count =
            replay_add_only_result.newFactorsIndices.size();
        resolved_sidecar_new_indices.assign(
            replay_add_only_result.newFactorsIndices.begin(),
            replay_add_only_result.newFactorsIndices.end());
      }

      stats->sidecar_new_factor_indices_count = resolved_sidecar_new_indices.size();
      stats->sample_sidecar_new_factor_indices =
          indices_to_sample_csv(resolved_sidecar_new_indices);

      if (!stats->h2_partial_replay_non_smart_only_applied &&
          resolved_sidecar_new_indices.size() !=
              expected_sidecar_new_indices_count) {
        stats->h2_replay_add_only_failure_reason =
            "h2_sidecar_replay_new_index_mismatch";
        std::ostringstream oss;
        oss << "h2_sidecar_replay_new_index_mismatch sidecar="
            << resolved_sidecar_new_indices.size()
            << " local_packet=" << expected_sidecar_new_indices_count;
        return fail_and_hard_reset(oss.str());
      }
    }

    gtsam::FactorIndices sampled_heart_slots;
    sampled_heart_slots.reserve(resolved_heart_new_factor_positions.size());
    gtsam::FactorIndices deferred_old_sidecar_slots_from_conflicts;
    deferred_old_sidecar_slots_from_conflicts.reserve(
        resolved_heart_new_factor_positions.size());
    const gtsam::NonlinearFactorGraph& sidecar_graph_after_primary_update =
        cbs_local_cov_sidecar_->getFactorsUnsafe();
    for (size_t i = 0; i < resolved_heart_new_factor_positions.size(); ++i) {
      const size_t heart_packet_pos = resolved_heart_new_factor_positions[i];
      if (heart_packet_pos >= heart_new_factor_indices.size()) {
        std::ostringstream oss;
        oss << "h2_sidecar_heart_index_oob pos=" << heart_packet_pos
            << " heart_size=" << heart_new_factor_indices.size();
        return fail_and_hard_reset(oss.str());
      }
      const gtsam::FactorIndex heart_slot =
          heart_new_factor_indices[heart_packet_pos];
      const gtsam::FactorIndex new_sidecar_slot =
          resolved_sidecar_new_indices[i];
      sampled_heart_slots.push_back(heart_slot);
      const auto existing_map_it =
          cbs_h2_sidecar_heart_to_sidecar_slot_map_.find(heart_slot);
      if (existing_map_it != cbs_h2_sidecar_heart_to_sidecar_slot_map_.end() &&
          existing_map_it->second != new_sidecar_slot) {
        const gtsam::FactorIndex old_sidecar_slot = existing_map_it->second;
        if (!sidecar_graph_after_primary_update.exists(old_sidecar_slot)) {
          record_removed_stale_slot_map_entry(
              heart_slot,
              old_sidecar_slot,
              &stats->stale_slot_map_entries_removed_during_remap);
          if (stats->first_unmapped_heart_slot ==
              std::numeric_limits<gtsam::FactorIndex>::max()) {
            stats->first_unmapped_heart_slot = heart_slot;
          }
          if (stats->first_unmapped_sidecar_slot ==
              std::numeric_limits<gtsam::FactorIndex>::max()) {
            stats->first_unmapped_sidecar_slot = old_sidecar_slot;
          }
          cbs_h2_sidecar_heart_to_sidecar_slot_map_.erase(existing_map_it);
        } else {
          ++stats->remap_existing_heart_slot_conflict_count;
          if (stats->first_conflicting_heart_slot ==
              std::numeric_limits<gtsam::FactorIndex>::max()) {
            stats->first_conflicting_heart_slot = heart_slot;
            stats->first_conflicting_old_sidecar_slot = old_sidecar_slot;
            stats->first_conflicting_new_sidecar_slot = new_sidecar_slot;
          }
          // Primary retry-without-remove can leave old sidecar slots alive.
          // Defer those old slots for explicit post-sync prune instead of
          // losing their only mapping through overwrite below.
          if (stats->retry_without_remove_factor_indices) {
            deferred_old_sidecar_slots_from_conflicts.push_back(
                old_sidecar_slot);
          }
        }
      }
      cbs_h2_sidecar_heart_to_sidecar_slot_map_[heart_slot] = new_sidecar_slot;
    }
    stats->sample_heart_new_factor_indices =
        indices_to_sample_csv(sampled_heart_slots);

    gtsam::FactorIndices post_sync_remove_indices;
    post_sync_remove_indices.reserve(cbs_h2_sidecar_heart_to_sidecar_slot_map_.size());
    const gtsam::NonlinearFactorGraph& heart_graph = smoother_->getFactors();
    const gtsam::NonlinearFactorGraph& sidecar_graph =
        cbs_local_cov_sidecar_->getFactorsUnsafe();
    for (auto it = cbs_h2_sidecar_heart_to_sidecar_slot_map_.begin();
         it != cbs_h2_sidecar_heart_to_sidecar_slot_map_.end();) {
      const gtsam::FactorIndex heart_slot = it->first;
      const gtsam::FactorIndex sidecar_slot = it->second;
      const bool heart_exists = heart_graph.exists(heart_slot);
      const bool sidecar_exists = sidecar_graph.exists(sidecar_slot);
      if (!heart_exists) {
        if (sidecar_exists) {
          post_sync_remove_indices.push_back(sidecar_slot);
        }
        it = cbs_h2_sidecar_heart_to_sidecar_slot_map_.erase(it);
      } else if (!sidecar_exists) {
        it = cbs_h2_sidecar_heart_to_sidecar_slot_map_.erase(it);
      } else {
        ++it;
      }
    }
    if (!deferred_old_sidecar_slots_from_conflicts.empty()) {
      post_sync_remove_indices.insert(post_sync_remove_indices.end(),
                                      deferred_old_sidecar_slots_from_conflicts.begin(),
                                      deferred_old_sidecar_slots_from_conflicts.end());
    }

    if (!post_sync_remove_indices.empty()) {
      std::sort(post_sync_remove_indices.begin(), post_sync_remove_indices.end());
      post_sync_remove_indices.erase(
          std::unique(post_sync_remove_indices.begin(),
                      post_sync_remove_indices.end()),
          post_sync_remove_indices.end());
      stats->post_prune_candidate_count = post_sync_remove_indices.size();
      stats->first_post_prune_sidecar_slot = post_sync_remove_indices.front();

      auto filter_existing_post_prune_slots =
          [&](const gtsam::FactorIndices& input_slots,
              size_t* stale_count) -> gtsam::FactorIndices {
        gtsam::FactorIndices filtered_slots;
        filtered_slots.reserve(input_slots.size());
        const gtsam::NonlinearFactorGraph& sidecar_graph_now =
            cbs_local_cov_sidecar_->getFactorsUnsafe();
        for (const gtsam::FactorIndex slot : input_slots) {
          if (sidecar_graph_now.exists(slot)) {
            filtered_slots.push_back(slot);
          } else if (stale_count) {
            ++(*stale_count);
          }
        }
        if (!filtered_slots.empty()) {
          std::sort(filtered_slots.begin(), filtered_slots.end());
          filtered_slots.erase(
              std::unique(filtered_slots.begin(), filtered_slots.end()),
              filtered_slots.end());
        }
        return filtered_slots;
      };

      size_t stale_count = 0u;
      gtsam::FactorIndices prune_slots =
          filter_existing_post_prune_slots(post_sync_remove_indices,
                                           &stale_count);
      stats->post_prune_filtered_stale_count += stale_count;

      auto try_post_prune_update = [&](const gtsam::FactorIndices& prune_input,
                                       bool* map_at_error) -> bool {
        CHECK_NOTNULL(map_at_error);
        *map_at_error = false;
        cbs::BPSAM::UpdateParams prune_params;
        prune_params.removeFactorIndices = prune_input;
        try {
          cbs_local_cov_sidecar_->update(gtsam::NonlinearFactorGraph(),
                                         gtsam::Values(),
                                         prune_params);
          return true;
        } catch (const std::exception& e) {
          const std::string err_msg =
              e.what() ? std::string(e.what())
                       : std::string("h2_sidecar_post_prune_std_exception");
          if (err_msg.find("map::at") != std::string::npos) {
            *map_at_error = true;
            return false;
          }
          throw;
        }
      };

      if (!prune_slots.empty()) {
        bool prune_map_at = false;
        if (try_post_prune_update(prune_slots, &prune_map_at)) {
          stats->post_prune_remove_count = prune_slots.size();
          stats->remove_count += prune_slots.size();
        } else if (prune_map_at) {
          stats->post_prune_threw_map_at = true;
          size_t retry_stale_count = 0u;
          gtsam::FactorIndices retry_slots =
              filter_existing_post_prune_slots(prune_slots, &retry_stale_count);
          stats->post_prune_filtered_stale_count += retry_stale_count;
          if (retry_slots.size() < prune_slots.size()) {
            stats->post_prune_retry_with_filtered_slots = true;
          }
          if (!retry_slots.empty()) {
            bool retry_map_at = false;
            if (try_post_prune_update(retry_slots, &retry_map_at)) {
              stats->post_prune_remove_count = retry_slots.size();
              stats->remove_count += retry_slots.size();
            } else {
              stats->post_prune_threw_map_at = true;
              stats->post_prune_skipped_deferred = true;
            }
          } else {
            stats->post_prune_skipped_deferred = true;
          }
        }
      } else {
        stats->post_prune_skipped_deferred = true;
      }
    }
    if (curr_kf_id == 51) {
      run_sidecar_self_audit("end_of_epoch_kf51", false);
    }

    stats->hard_reset_count = cbs_h2_sidecar_hard_reset_count_;
    if (stats->h2_partial_replay_non_smart_only_applied) {
      stats->failure_reason = stats->h2_partial_replay_reason.empty()
                                  ? "h2_repair_epoch_partial_replay_non_smart_only"
                                  : stats->h2_partial_replay_reason;
      stats->update_ms = elapsedMs(sync_start, std::chrono::steady_clock::now());
      return false;
    }
    stats->failure_reason = "none";
    stats->update_ms = elapsedMs(sync_start, std::chrono::steady_clock::now());
    return true;
  } catch (const std::exception& e) {
    return fail_and_hard_reset(
        e.what() ? std::string(e.what()) : std::string("h2_sidecar_std_exception"));
  } catch (...) {
    return fail_and_hard_reset("h2_sidecar_unknown_exception");
  }
}
#endif

/* -------------------------------------------------------------------------- */
// TODO remove global variables from optimize, pass them as local
// parameters...
// TODO make changes to global variables to the addVisualInertial blah blah.
// TODO remove timing logging and use Statistics.h instead.
bool VioBackend::optimize(
    const Timestamp& timestamp_kf_nsec,
    const FrameId& cur_id,
    const size_t& max_extra_iterations,
    const gtsam::FactorIndices& extra_factor_slots_to_delete) {
  DCHECK(smoother_) << "Incremental smoother is a null pointer.";

  // Step 4B & 6_5:
  // Robustly consume external priors with:
  // 1) age gating (drop very old),
  // 2) future gating (defer too-future),
  // 3) nearest timestamp match,
  // 4) per-update injection cap.
  const Timestamp kMaxPriorAgeNs =
      FLAGS_external_prior_max_age_ns > 0
          ? static_cast<Timestamp>(FLAGS_external_prior_max_age_ns)
          : static_cast<Timestamp>(20 * 1000 * 1000 * 1000LL);
  const Timestamp kMaxFutureLeadNs =
      FLAGS_external_prior_max_future_lead_ns >= 0
          ? static_cast<Timestamp>(FLAGS_external_prior_max_future_lead_ns)
          : static_cast<Timestamp>(200 * 1000 * 1000LL);
  const size_t kMaxExternalPriorsPerOptimize =
      FLAGS_external_prior_max_per_optimize > 0
          ? static_cast<size_t>(FLAGS_external_prior_max_per_optimize)
          : static_cast<size_t>(400);

  const bool cbs_exchange_active = useCbsBeliefExchange();
  const bool cbs_heart_active = useCbsOptimizerHeart();
#ifdef KIMERA_USE_CBS
  // H2 sidecar filtering only needs external-prior pointers from this epoch's
  // packet construction; clear stale pointers from previous optimize rounds.
  cbs_external_prior_factor_ptrs_.clear();
#endif

  size_t num_external_priors_injected = 0;
  size_t num_external_priors_deferred = 0;
  size_t num_external_priors_dropped_old = 0;
  size_t num_external_priors_dropped_marginalized = 0;
  size_t num_external_priors_dropped_inactive = 0;
  size_t num_external_priors_dropped_disabled_mode = 0;
  size_t num_external_priors_deferred_budget = 0;
  size_t num_external_priors_deferred_no_local_receiver_state = 0;
  size_t num_external_priors_fast_skipped = 0;
  size_t num_external_priors_source_backoff_skipped = 0;
  size_t num_external_priors_considered = 0;

  double ext_queue_scan_filter_ms = 0.0;
  double ext_window_match_ms = 0.0;
  double ext_staging_construct_ms = 0.0;
  double ext_add_beliefs_ms = 0.0;
  double ext_cov_query_ms = 0.0;
  double ext_alignment_ms = 0.0;

  // zy Step 12a
  // In CBS mode, stage/query belief acceptance counters while keeping fixed-lag
  // as optimization heart.
  #ifdef KIMERA_USE_CBS
  size_t num_external_beliefs_received = 0;
  size_t num_external_beliefs_considered = 0;
  size_t num_external_beliefs_staged = 0;
  size_t num_external_beliefs_rejected = 0;
  size_t num_external_beliefs_accepted = 0;
  size_t num_external_beliefs_rejected_keys_touched = 0;
  size_t num_external_beliefs_rejected_factors_touched = 0;
  size_t num_external_beliefs_dropped_bad_noise = 0;
  size_t num_external_beliefs_cov_rejected = 0;
  size_t num_external_beliefs_cov_regularized = 0;
  // Step 31a: avoid silently mixing beliefs from unknown senders into a wrong CBS agent stream.
  size_t num_external_beliefs_dropped_unknown_source = 0;
  // Intuition: avoid self-feedback loops where Kimera re-fuses its own published belief.
  size_t num_external_beliefs_dropped_self_source = 0;
  // zy Step 33b
  constexpr cbs::AgentId kKimeraAgentId = static_cast<cbs::AgentId>('a');
  constexpr cbs::AgentId kLiorfAgentId = static_cast<cbs::AgentId>('b');
  #endif


  auto absDiffNs = [](Timestamp a, Timestamp b) -> Timestamp {
    return (a >= b) ? (a - b) : (b - a);
  };

  Timestamp kDeferredInitialBackoffNs = external_prior_timestamp_tolerance_ns_;
  Timestamp kDeferredMaxBackoffNs = kDeferredInitialBackoffNs;
  Timestamp kRejectedSourceInitialBackoffNs = kDeferredInitialBackoffNs;
  Timestamp kRejectedSourceMaxBackoffNs = kRejectedSourceInitialBackoffNs;
#ifdef KIMERA_USE_CBS
  kDeferredInitialBackoffNs =
      FLAGS_cbs_deferred_prior_initial_backoff_ns > 0
          ? static_cast<Timestamp>(FLAGS_cbs_deferred_prior_initial_backoff_ns)
          : external_prior_timestamp_tolerance_ns_;
  kDeferredMaxBackoffNs =
      std::max(kDeferredInitialBackoffNs,
               FLAGS_cbs_deferred_prior_max_backoff_ns > 0
                   ? static_cast<Timestamp>(FLAGS_cbs_deferred_prior_max_backoff_ns)
                   : kDeferredInitialBackoffNs);
  kRejectedSourceInitialBackoffNs =
      FLAGS_cbs_rejected_source_initial_backoff_ns > 0
          ? static_cast<Timestamp>(FLAGS_cbs_rejected_source_initial_backoff_ns)
          : kDeferredInitialBackoffNs;
  kRejectedSourceMaxBackoffNs =
      std::max(kRejectedSourceInitialBackoffNs,
               FLAGS_cbs_rejected_source_max_backoff_ns > 0
                   ? static_cast<Timestamp>(FLAGS_cbs_rejected_source_max_backoff_ns)
                   : kRejectedSourceInitialBackoffNs);
#endif

  // zy
  // In fixed-lag mode, prune timestamp->key map entries whose pose keys are no
  // longer active in the optimizer and capture the oldest still-active
  // timestamp. Incoming beliefs older than that are guaranteed to target
  // marginalized states and should be dropped early.
  const FrameId oldest_active_frame_id_by_lag =
      computeCbsOldestActiveFrame(cur_id);
  Timestamp oldest_active_pose_timestamp = -1;
  Timestamp newest_active_pose_timestamp = -1;
  size_t num_timestamp_map_pruned = 0;
  pruneTimestampToKeyframeMap(cbs_heart_active,
                              oldest_active_frame_id_by_lag,
                              &num_timestamp_map_pruned,
                              &oldest_active_pose_timestamp,
                              &newest_active_pose_timestamp);
  if (num_timestamp_map_pruned > 0) {
    VLOG(2) << "Pruned " << num_timestamp_map_pruned
            << " marginalized timestamp->key entries. oldest_active_ts[nsec]="
            << oldest_active_pose_timestamp;
  }

  {
    std::deque<ExternalPosePrior> working_queue;
    {
      std::lock_guard<std::mutex> queue_lock(external_pose_priors_queue_mutex_);
      working_queue.swap(external_pose_priors_queue_);
    }
    std::deque<ExternalPosePrior> remaining_queue;
    const auto schedule_deferred_with_backoff =
        [&](ExternalPosePrior* prior) -> Timestamp {
          CHECK_NOTNULL(prior);
          Timestamp backoff_ns = prior->retry_backoff_ns_ > 0
                                     ? prior->retry_backoff_ns_
                                     : kDeferredInitialBackoffNs;
          backoff_ns = std::max(kDeferredInitialBackoffNs, backoff_ns);
          backoff_ns = std::min(kDeferredMaxBackoffNs, backoff_ns);
          prior->next_eligible_timestamp_ns_ = std::max(
              prior->next_eligible_timestamp_ns_, timestamp_kf_nsec + backoff_ns);
          prior->retry_backoff_ns_ = std::min(kDeferredMaxBackoffNs, backoff_ns * 2);
          return backoff_ns;
        };

    for (const auto& queued_prior : working_queue) {
      const auto prior_scan_start = std::chrono::steady_clock::now();
      ExternalPosePrior prior = queued_prior;
#ifdef KIMERA_USE_CBS
      ++num_external_beliefs_received;
#endif
      if (!cbs_exchange_active) {
        ++num_external_priors_dropped_disabled_mode;
        ext_queue_scan_filter_ms +=
            elapsedMs(prior_scan_start, std::chrono::steady_clock::now());
        continue;
      }

      ++num_external_priors_considered;

      if (prior.next_eligible_timestamp_ns_ > timestamp_kf_nsec) {
        remaining_queue.push_back(prior);
        ++num_external_priors_deferred;
        ++num_external_priors_fast_skipped;
        ext_queue_scan_filter_ms +=
            elapsedMs(prior_scan_start, std::chrono::steady_clock::now());
        continue;
      }

#ifdef KIMERA_USE_CBS
      if (cbs_heart_active && !prior.source_.empty()) {
        Timestamp retry_after_ns = std::numeric_limits<Timestamp>::lowest();
        {
          std::lock_guard<std::mutex> backoff_lock(external_source_backoff_mutex_);
          const auto it = external_source_retry_after_ns_.find(prior.source_);
          if (it != external_source_retry_after_ns_.end()) {
            retry_after_ns = it->second;
          }
        }
        if (retry_after_ns > timestamp_kf_nsec) {
          prior.next_eligible_timestamp_ns_ = std::max(
              prior.next_eligible_timestamp_ns_, retry_after_ns);
          remaining_queue.push_back(prior);
          ++num_external_priors_deferred;
          ++num_external_priors_fast_skipped;
          ++num_external_priors_source_backoff_skipped;
          ext_queue_scan_filter_ms +=
              elapsedMs(prior_scan_start, std::chrono::steady_clock::now());
          continue;
        }
      }
#endif

      // Drop priors that are older than the oldest pose still active in the
      // optimizer window (fixed-lag behavior).
      if (oldest_active_pose_timestamp > 0 &&
          prior.timestamp_kf_nsec_ + external_prior_timestamp_tolerance_ns_ <
              oldest_active_pose_timestamp) {
        ++num_external_priors_dropped_marginalized;
        ext_queue_scan_filter_ms +=
            elapsedMs(prior_scan_start, std::chrono::steady_clock::now());
        continue;
      }

      // Drop priors that are too old w.r.t current backend timestamp.
      if (prior.timestamp_kf_nsec_ + kMaxPriorAgeNs < timestamp_kf_nsec) {
        ++num_external_priors_dropped_old;
        ext_queue_scan_filter_ms +=
            elapsedMs(prior_scan_start, std::chrono::steady_clock::now());
        continue;
      }

      // Keep priors that are too far in the future; they may match later frames.
      if (prior.timestamp_kf_nsec_ > timestamp_kf_nsec + kMaxFutureLeadNs) {
        prior.next_eligible_timestamp_ns_ = std::max(
            prior.next_eligible_timestamp_ns_,
            prior.timestamp_kf_nsec_ - kMaxFutureLeadNs);
        remaining_queue.push_back(prior);
        ++num_external_priors_deferred;
        ext_queue_scan_filter_ms +=
            elapsedMs(prior_scan_start, std::chrono::steady_clock::now());
        continue;
      }

      bool matched = false;
      FrameId matched_frame_id = -1;

      const auto match_start = std::chrono::steady_clock::now();
      {
        std::lock_guard<std::mutex> map_lock(timestamp_to_kf_id_map_mutex_);
        if (!timestamp_to_kf_id_map_.empty()) {
          auto it = timestamp_to_kf_id_map_.lower_bound(prior.timestamp_kf_nsec_);

          std::map<Timestamp, FrameId>::const_iterator best_it =
              timestamp_to_kf_id_map_.end();
          Timestamp best_dt = std::numeric_limits<Timestamp>::max();

          if (it != timestamp_to_kf_id_map_.end()) {
            const Timestamp dt = absDiffNs(it->first, prior.timestamp_kf_nsec_);
            best_it = it;
            best_dt = dt;
          }
          if (it != timestamp_to_kf_id_map_.begin()) {
            auto prev_it = std::prev(it);
            const Timestamp dt = absDiffNs(prev_it->first, prior.timestamp_kf_nsec_);
            if (dt < best_dt) {
              best_it = prev_it;
              best_dt = dt;
            }
          }

          if (best_it != timestamp_to_kf_id_map_.end() &&
              best_dt <= external_prior_timestamp_tolerance_ns_) {
            matched = true;
            matched_frame_id = best_it->second;
          }
        }
      }
      ext_window_match_ms +=
          elapsedMs(match_start, std::chrono::steady_clock::now());

      if (!matched) {
        // zy
        // Fixed-lag policy: if a prior cannot be matched to any active key now,
        // discard it instead of deferring indefinitely. Future priors were
        // already handled by the future-gate above.
        if (!cbs_heart_active) {
          ++num_external_priors_dropped_marginalized;
          ext_queue_scan_filter_ms +=
              elapsedMs(prior_scan_start, std::chrono::steady_clock::now());
          continue;
        }

        // zy
        // If backend has already progressed past this prior (even after
        // tolerance) and we still cannot match it to an active key, discard it
        // instead of deferring indefinitely.
        if (newest_active_pose_timestamp > 0 &&
            prior.timestamp_kf_nsec_ + external_prior_timestamp_tolerance_ns_ <
                newest_active_pose_timestamp) {
          ++num_external_priors_dropped_marginalized;
          ext_queue_scan_filter_ms +=
              elapsedMs(prior_scan_start, std::chrono::steady_clock::now());
          continue;
        }
        schedule_deferred_with_backoff(&prior);
        remaining_queue.push_back(prior);
        ++num_external_priors_deferred;
        ext_queue_scan_filter_ms +=
            elapsedMs(prior_scan_start, std::chrono::steady_clock::now());
        continue;
      }

      // Avoid overloading a single optimize() step.
      if (num_external_priors_injected >= kMaxExternalPriorsPerOptimize) {
        schedule_deferred_with_backoff(&prior);
        remaining_queue.push_back(prior);
        ++num_external_priors_deferred_budget;
        ext_queue_scan_filter_ms +=
            elapsedMs(prior_scan_start, std::chrono::steady_clock::now());
        continue;
      }

      const gtsam::Symbol pose_symbol(kPoseSymbolChar, matched_frame_id);
#ifdef KIMERA_USE_CBS
      if (cbs_exchange_active && cbs_heart_active &&
          !isReceiverLocalBeliefReadyForCbs(pose_symbol)) {
        schedule_deferred_with_backoff(&prior);
        remaining_queue.push_back(prior);
        ++num_external_priors_deferred;
        ++num_external_priors_deferred_no_local_receiver_state;
        VLOG(2) << "Deferring external prior: receiver LOCAL CBS state not "
                   "ready yet. source="
                << prior.source_ << ", seq=" << prior.source_seq_
                << ", ts[nsec]=" << prior.timestamp_kf_nsec_
                << ", matched_frame_id=" << matched_frame_id;
        ext_queue_scan_filter_ms +=
            elapsedMs(prior_scan_start, std::chrono::steady_clock::now());
        continue;
      }
#endif
      const bool pose_key_is_active = isPoseKeyActiveInOptimizer(pose_symbol);

      if (pose_key_is_active) {
#ifdef KIMERA_USE_CBS
        if (cbs_exchange_active) {
          ++num_external_beliefs_considered;
          const auto stage_start = std::chrono::steady_clock::now();
          double stage_cov_query_ms = 0.0;
          double stage_alignment_ms = 0.0;
          double stage_add_beliefs_ms = 0.0;
          // Use CBS belief gate for acceptance/rejection. In CBS-heart mode,
          // accepted beliefs remain inside BPSAM message passing. In legacy
          // mode, accepted beliefs are injected as PriorFactor into the
          // legacy fixed-lag smoother graph.
          CHECK(cbs_optimizer_) << "CBS optimizer flag is ON but cbs_optimizer_ is null.";

          bool accepted_by_cbs = true;
          gtsam::Matrix66 cov = gtsam::Matrix66::Zero();
          bool have_cov = false;

          if (auto gaussian_model =
                  boost::dynamic_pointer_cast<gtsam::noiseModel::Gaussian>(
                      prior.noise_model_)) {
            cov = gaussian_model->covariance();
            have_cov = true;
          } else if (auto robust_model =
                         boost::dynamic_pointer_cast<gtsam::noiseModel::Robust>(
                             prior.noise_model_)) {
            auto wrapped =
                boost::dynamic_pointer_cast<gtsam::noiseModel::Gaussian>(
                    robust_model->noise());
            if (wrapped) {
              cov = wrapped->covariance();
              have_cov = true;
            }
          }

          if (!have_cov || !cov.allFinite()) {
            ++num_external_beliefs_dropped_bad_noise;
            ++num_external_beliefs_cov_rejected;
            accepted_by_cbs = false;
            VLOG(2) << "Dropping external belief with unsupported/non-finite noise. "
                    << "source=" << prior.source_
                    << ", seq=" << prior.source_seq_
                    << ", ts[nsec]=" << prior.timestamp_kf_nsec_;
          }

          // Validate/regularize external covariance before creating CBS belief.
          if (accepted_by_cbs) {
            std::string cov_reason = "unknown";
            const PoseCovarianceStatus cov_status =
                sanitizePoseCovariance(&cov, &cov_reason);
            if (cov_status == PoseCovarianceStatus::kRejected) {
              ++num_external_beliefs_dropped_bad_noise;
              ++num_external_beliefs_cov_rejected;
              accepted_by_cbs = false;
              VLOG(2) << "Dropping external belief with invalid covariance. "
                      << "reason=" << cov_reason
                      << "source=" << prior.source_
                      << ", seq=" << prior.source_seq_
                      << ", ts[nsec]=" << prior.timestamp_kf_nsec_;
            } else if (cov_status == PoseCovarianceStatus::kRegularized) {
              ++num_external_beliefs_cov_regularized;
              VLOG(2) << "Regularized external belief covariance before CBS "
                         "injection. source="
                      << prior.source_ << ", seq=" << prior.source_seq_
                      << ", ts[nsec]=" << prior.timestamp_kf_nsec_;
            }
          }

          cbs::AgentId sender_id = kKimeraAgentId;
          bool known_source = true;
          if (prior.source_ == "kimera" || prior.source_ == "self") {
            sender_id = kKimeraAgentId;
          } else if (prior.source_ == "liorf" || prior.source_ == "liosam") {
            sender_id = kLiorfAgentId;
          } else {
            known_source = false;
          }

          if (accepted_by_cbs && !known_source) {
            ++num_external_beliefs_dropped_unknown_source;
            accepted_by_cbs = false;
            VLOG(2) << "Dropping external belief with unknown source tag. "
                    << "source=" << prior.source_
                    << ", seq=" << prior.source_seq_
                    << ", ts[nsec]=" << prior.timestamp_kf_nsec_;
          }

          if (accepted_by_cbs && sender_id == kKimeraAgentId) {
            ++num_external_beliefs_dropped_self_source;
            accepted_by_cbs = false;
            VLOG(2) << "Dropping self-source belief to prevent feedback loop. "
                    << "source=" << prior.source_
                    << ", seq=" << prior.source_seq_
                    << ", ts[nsec]=" << prior.timestamp_kf_nsec_;
          }

          if (accepted_by_cbs) {
            const gtsam::Pose3 incoming_pose_raw = prior.W_Pose_B_;
            gtsam::Pose3 incoming_pose_aligned = incoming_pose_raw;
            bool alignment_mode_enabled = FLAGS_cbs_diag_align_incoming_mean;
            bool alignment_prev_available = false;
            bool alignment_applied = false;
            gtsam::Pose3 alignment_prev = gtsam::Pose3();
            if (alignment_mode_enabled) {
              const auto alignment_start = std::chrono::steady_clock::now();
              std::lock_guard<std::mutex> lock(external_mean_alignment_mutex_);
              const auto it = external_mean_alignment_by_source_.find(prior.source_);
              if (it != external_mean_alignment_by_source_.end()) {
                alignment_prev_available = true;
                alignment_prev = it->second;
                incoming_pose_aligned = alignment_prev.compose(incoming_pose_raw);
                alignment_applied = true;
              }
              stage_alignment_ms +=
                  elapsedMs(alignment_start, std::chrono::steady_clock::now());
            }

            bool have_pose_delta = false;
            double pose_delta_rot_rad = std::numeric_limits<double>::quiet_NaN();
            double pose_delta_trans_m = std::numeric_limits<double>::quiet_NaN();
            double pose_delta_norm = std::numeric_limits<double>::quiet_NaN();
            double pose_delta_rot_rad_before_align =
                std::numeric_limits<double>::quiet_NaN();
            double pose_delta_trans_m_before_align =
                std::numeric_limits<double>::quiet_NaN();
            double pose_delta_norm_before_align =
                std::numeric_limits<double>::quiet_NaN();
            double pose_delta_rot_rad_after_align =
                std::numeric_limits<double>::quiet_NaN();
            double pose_delta_trans_m_after_align =
                std::numeric_limits<double>::quiet_NaN();
            double pose_delta_norm_after_align =
                std::numeric_limits<double>::quiet_NaN();
            double div_hell_local_incoming_before_align =
                std::numeric_limits<double>::quiet_NaN();
            double div_hell_local_incoming_after_align =
                std::numeric_limits<double>::quiet_NaN();
            gtsam::Pose3 current_pose_estimate;
            bool have_current_pose_estimate = false;
            double receiver_local_tx = std::numeric_limits<double>::quiet_NaN();
            double receiver_local_ty = std::numeric_limits<double>::quiet_NaN();
            double receiver_local_tz = std::numeric_limits<double>::quiet_NaN();
            double receiver_local_qx = std::numeric_limits<double>::quiet_NaN();
            double receiver_local_qy = std::numeric_limits<double>::quiet_NaN();
            double receiver_local_qz = std::numeric_limits<double>::quiet_NaN();
            double receiver_local_qw = std::numeric_limits<double>::quiet_NaN();
            const gtsam::Quaternion incoming_q_raw =
                incoming_pose_raw.rotation().toQuaternion();
            const gtsam::Quaternion incoming_q_aligned =
                incoming_pose_aligned.rotation().toQuaternion();

            if (new_values_.exists(pose_symbol)) {
              current_pose_estimate = new_values_.at<gtsam::Pose3>(pose_symbol);
              have_current_pose_estimate = true;
            } else if (state_.exists(pose_symbol)) {
              current_pose_estimate = state_.at<gtsam::Pose3>(pose_symbol);
              have_current_pose_estimate = true;
            } else if (cbs_optimizer_->valueExists(pose_symbol)) {
              current_pose_estimate =
                  cbs_optimizer_->calculateEstimate<gtsam::Pose3>(pose_symbol);
              have_current_pose_estimate = true;
            }

            gtsam::Matrix66 receiver_local_cov_for_diag = gtsam::Matrix66::Zero();
            bool have_receiver_local_cov_for_diag = false;
            if (FLAGS_cbs_query_receiver_local_cov_for_diag) {
              const auto cov_query_start = std::chrono::steady_clock::now();
              if (have_current_pose_estimate && cbs_optimizer_ &&
                  cbs_optimizer_->valueExists(pose_symbol)) {
                try {
                  const gtsam::Matrix local_cov_dynamic =
                      cbs_optimizer_->marginalCovariance(
                          pose_symbol, cbs::BPSAM::MarginalizationType::LOCAL);
                  if (local_cov_dynamic.rows() >= 6 &&
                      local_cov_dynamic.cols() >= 6 &&
                      local_cov_dynamic.block<6, 6>(0, 0).allFinite()) {
                    receiver_local_cov_for_diag =
                        local_cov_dynamic.block<6, 6>(0, 0);
                    have_receiver_local_cov_for_diag = true;
                  }
                } catch (...) {
                  have_receiver_local_cov_for_diag = false;
                }
              }
              stage_cov_query_ms +=
                  elapsedMs(cov_query_start, std::chrono::steady_clock::now());
            }

            if (have_current_pose_estimate) {
              const gtsam::Quaternion local_q =
                  current_pose_estimate.rotation().toQuaternion();
              receiver_local_tx = current_pose_estimate.x();
              receiver_local_ty = current_pose_estimate.y();
              receiver_local_tz = current_pose_estimate.z();
              receiver_local_qx = local_q.x();
              receiver_local_qy = local_q.y();
              receiver_local_qz = local_q.z();
              receiver_local_qw = local_q.w();
              const gtsam::Vector6 pose_delta_vec_before =
                  gtsam::traits<gtsam::Pose3>::Logmap(
                      current_pose_estimate.between(incoming_pose_raw));
              pose_delta_rot_rad_before_align = pose_delta_vec_before.head<3>().norm();
              pose_delta_trans_m_before_align =
                  pose_delta_vec_before.tail<3>().norm();
              pose_delta_norm_before_align = pose_delta_vec_before.norm();

              const gtsam::Vector6 pose_delta_vec_after =
                  gtsam::traits<gtsam::Pose3>::Logmap(
                      current_pose_estimate.between(incoming_pose_aligned));
              pose_delta_rot_rad_after_align = pose_delta_vec_after.head<3>().norm();
              pose_delta_trans_m_after_align =
                  pose_delta_vec_after.tail<3>().norm();
              pose_delta_norm_after_align = pose_delta_vec_after.norm();
              pose_delta_rot_rad = pose_delta_rot_rad_after_align;
              pose_delta_trans_m = pose_delta_trans_m_after_align;
              pose_delta_norm = pose_delta_norm_after_align;
              have_pose_delta = true;

              if (have_receiver_local_cov_for_diag) {
                const Vec6 mu_local =
                    gtsam::traits<gtsam::Pose3>::Logmap(current_pose_estimate);
                const Vec6 mu_incoming_before =
                    gtsam::traits<gtsam::Pose3>::Logmap(incoming_pose_raw);
                const Vec6 mu_incoming_after =
                    gtsam::traits<gtsam::Pose3>::Logmap(incoming_pose_aligned);
                const Mat6 sigma_local = receiver_local_cov_for_diag;
                const Mat6 sigma_incoming = cov;
                div_hell_local_incoming_before_align = hellingerDistance6(
                    mu_local, sigma_local, mu_incoming_before, sigma_incoming);
                div_hell_local_incoming_after_align = hellingerDistance6(
                    mu_local, sigma_local, mu_incoming_after, sigma_incoming);
              }
            }

            bool alignment_updated = false;
            gtsam::Pose3 alignment_update = gtsam::Pose3();
            if (alignment_mode_enabled && have_current_pose_estimate) {
              const auto alignment_start = std::chrono::steady_clock::now();
              alignment_update = current_pose_estimate.compose(
                  incoming_pose_raw.inverse());
              {
                std::lock_guard<std::mutex> lock(external_mean_alignment_mutex_);
                external_mean_alignment_by_source_[prior.source_] = alignment_update;
              }
              alignment_updated = true;
              stage_alignment_ms +=
                  elapsedMs(alignment_start, std::chrono::steady_clock::now());
            }
            const gtsam::Quaternion alignment_prev_q =
                alignment_prev.rotation().toQuaternion();
            const gtsam::Quaternion alignment_update_q =
                alignment_update.rotation().toQuaternion();

            const gtsam::Vector6 mu =
                gtsam::traits<gtsam::Pose3>::Logmap(incoming_pose_aligned);
            gbp::Gaussian belief(pose_symbol, mu, cov, 1);
            std::map<gtsam::Key, std::vector<std::pair<cbs::AgentId, gbp::Gaussian>>>
                single_belief;
            single_belief[pose_symbol].emplace_back(sender_id, belief);
            ++num_external_beliefs_staged;
            const auto add_beliefs_start = std::chrono::steady_clock::now();
            const size_t rejected_count =
                static_cast<size_t>(cbs_optimizer_->addBeliefs(single_belief));
            stage_add_beliefs_ms +=
                elapsedMs(add_beliefs_start, std::chrono::steady_clock::now());
            num_external_beliefs_rejected += rejected_count;
            if (rejected_count > 0u) {
              num_external_beliefs_rejected_keys_touched += single_belief.size();
              size_t touched_factors = 0u;
              for (const auto& key_beliefs : single_belief) {
                touched_factors += key_beliefs.second.size();
              }
              num_external_beliefs_rejected_factors_touched += touched_factors;
            }
            accepted_by_cbs = (rejected_count == 0u);

            LOG(INFO) << "[CBS][KimeraPrior] key=" << pose_symbol.key()
                      << ", matched_frame_id=" << matched_frame_id
                      << ", source=" << prior.source_
                      << ", seq=" << prior.source_seq_
                      << ", ts[nsec]=" << prior.timestamp_kf_nsec_
                      << ", incoming_raw_mean_semantic=world_to_body_pose"
                      << ", incoming_raw_mean_tx=" << incoming_pose_raw.x()
                      << ", incoming_raw_mean_ty=" << incoming_pose_raw.y()
                      << ", incoming_raw_mean_tz=" << incoming_pose_raw.z()
                      << ", incoming_raw_mean_qx=" << incoming_q_raw.x()
                      << ", incoming_raw_mean_qy=" << incoming_q_raw.y()
                      << ", incoming_raw_mean_qz=" << incoming_q_raw.z()
                      << ", incoming_raw_mean_qw=" << incoming_q_raw.w()
                      << ", incoming_mean_semantic=world_to_body_pose"
                      << ", incoming_mean_tx=" << incoming_pose_aligned.x()
                      << ", incoming_mean_ty=" << incoming_pose_aligned.y()
                      << ", incoming_mean_tz=" << incoming_pose_aligned.z()
                      << ", incoming_mean_qx=" << incoming_q_aligned.x()
                      << ", incoming_mean_qy=" << incoming_q_aligned.y()
                      << ", incoming_mean_qz=" << incoming_q_aligned.z()
                      << ", incoming_mean_qw=" << incoming_q_aligned.w()
                      << ", receiver_local_mean_semantic=world_to_body_pose"
                      << ", receiver_local_mean_available="
                      << (have_current_pose_estimate ? 1 : 0)
                      << ", receiver_local_mean_tx=" << receiver_local_tx
                      << ", receiver_local_mean_ty=" << receiver_local_ty
                      << ", receiver_local_mean_tz=" << receiver_local_tz
                      << ", receiver_local_mean_qx=" << receiver_local_qx
                      << ", receiver_local_mean_qy=" << receiver_local_qy
                      << ", receiver_local_mean_qz=" << receiver_local_qz
                      << ", receiver_local_mean_qw=" << receiver_local_qw
                      << ", mean_semantics_match=1"
                      << ", receiver_extrinsic_applied_to_mean=0"
                      << ", mean_alignment_mode_enabled="
                      << (alignment_mode_enabled ? 1 : 0)
                      << ", mean_alignment_prev_available="
                      << (alignment_prev_available ? 1 : 0)
                      << ", mean_alignment_applied=" << (alignment_applied ? 1 : 0)
                      << ", mean_alignment_prev_tx=" << alignment_prev.x()
                      << ", mean_alignment_prev_ty=" << alignment_prev.y()
                      << ", mean_alignment_prev_tz=" << alignment_prev.z()
                      << ", mean_alignment_prev_qx=" << alignment_prev_q.x()
                      << ", mean_alignment_prev_qy=" << alignment_prev_q.y()
                      << ", mean_alignment_prev_qz=" << alignment_prev_q.z()
                      << ", mean_alignment_prev_qw=" << alignment_prev_q.w()
                      << ", mean_alignment_updated=" << (alignment_updated ? 1 : 0)
                      << ", mean_alignment_update_tx=" << alignment_update.x()
                      << ", mean_alignment_update_ty=" << alignment_update.y()
                      << ", mean_alignment_update_tz=" << alignment_update.z()
                      << ", mean_alignment_update_qx=" << alignment_update_q.x()
                      << ", mean_alignment_update_qy=" << alignment_update_q.y()
                      << ", mean_alignment_update_qz=" << alignment_update_q.z()
                      << ", mean_alignment_update_qw=" << alignment_update_q.w()
                      << ", div_hell_local_incoming_before_align="
                      << div_hell_local_incoming_before_align
                      << ", div_hell_local_incoming_after_align="
                      << div_hell_local_incoming_after_align
                      << ", cbs_result="
                      << (accepted_by_cbs ? "accepted" : "rejected")
                      << ", cbs_rejected_count=" << rejected_count
                      << ", have_pose_delta=" << (have_pose_delta ? 1 : 0)
                      << ", delta_trans_m=" << pose_delta_trans_m
                      << ", delta_rot_rad=" << pose_delta_rot_rad
                      << ", delta_norm=" << pose_delta_norm
                      << ", delta_local_incoming_trans_m_before_align="
                      << pose_delta_trans_m_before_align
                      << ", delta_local_incoming_rot_rad_before_align="
                      << pose_delta_rot_rad_before_align
                      << ", delta_local_incoming_norm_before_align="
                      << pose_delta_norm_before_align
                      << ", delta_local_incoming_trans_m_after_align="
                      << pose_delta_trans_m_after_align
                      << ", delta_local_incoming_rot_rad_after_align="
                      << pose_delta_rot_rad_after_align
                      << ", delta_local_incoming_norm_after_align="
                      << pose_delta_norm_after_align;
          }

          const double stage_total_ms =
              elapsedMs(stage_start, std::chrono::steady_clock::now());
          ext_staging_construct_ms +=
              std::max(0.0, stage_total_ms - stage_cov_query_ms -
                                stage_alignment_ms - stage_add_beliefs_ms);
          ext_cov_query_ms += stage_cov_query_ms;
          ext_alignment_ms += stage_alignment_ms;
          ext_add_beliefs_ms += stage_add_beliefs_ms;

          if (!accepted_by_cbs) {
#ifdef KIMERA_USE_CBS
            if (cbs_heart_active && !prior.source_.empty()) {
              std::lock_guard<std::mutex> backoff_lock(
                  external_source_backoff_mutex_);
              Timestamp backoff_ns = kRejectedSourceInitialBackoffNs;
              const auto it = external_source_retry_backoff_ns_.find(prior.source_);
              if (it != external_source_retry_backoff_ns_.end()) {
                backoff_ns = std::max(backoff_ns, it->second);
              }
              backoff_ns = std::min(kRejectedSourceMaxBackoffNs, backoff_ns);
              external_source_retry_after_ns_[prior.source_] =
                  timestamp_kf_nsec + backoff_ns;
              external_source_retry_backoff_ns_[prior.source_] =
                  std::min(kRejectedSourceMaxBackoffNs, backoff_ns * 2);
            }
#endif
            ext_queue_scan_filter_ms +=
                elapsedMs(prior_scan_start, std::chrono::steady_clock::now());
            continue;
          }

          ++num_external_priors_injected;
#ifdef KIMERA_USE_CBS
          if (cbs_heart_active && !prior.source_.empty()) {
            std::lock_guard<std::mutex> backoff_lock(
                external_source_backoff_mutex_);
            external_source_retry_after_ns_.erase(prior.source_);
            external_source_retry_backoff_ns_.erase(prior.source_);
          }
#endif
          if (cbs_heart_active) {
            VLOG(2) << "Accepted external belief routed to CBS-heart only. source="
                    << prior.source_ << ", seq=" << prior.source_seq_
                    << ", ts[nsec]=" << prior.timestamp_kf_nsec_
                    << ", matched_frame_id=" << matched_frame_id;
          } else {
            addExternalPosePrior(
                matched_frame_id, prior.W_Pose_B_, prior.noise_model_);
            VLOG(2) << "Injected external prior factor from CBS-accepted belief. source="
                    << prior.source_ << ", seq=" << prior.source_seq_
                    << ", ts[nsec]=" << prior.timestamp_kf_nsec_
                    << ", matched_frame_id=" << matched_frame_id;
          }
          ext_queue_scan_filter_ms +=
              elapsedMs(prior_scan_start, std::chrono::steady_clock::now());
        } else {
          ++num_external_priors_dropped_disabled_mode;
          VLOG(2) << "Dropping external prior because CBS belief exchange is OFF. source="
                  << prior.source_ << ", seq=" << prior.source_seq_
                  << ", ts[nsec]=" << prior.timestamp_kf_nsec_;
          ext_queue_scan_filter_ms +=
              elapsedMs(prior_scan_start, std::chrono::steady_clock::now());
        }
#else
        ++num_external_priors_dropped_disabled_mode;
        VLOG(2) << "Dropping external prior because CBS support is not compiled.";
        ext_queue_scan_filter_ms +=
            elapsedMs(prior_scan_start, std::chrono::steady_clock::now());
#endif
      } else {

        ++num_external_priors_dropped_inactive;
        VLOG(2) << "Matched external prior but pose key inactive. source="
                << prior.source_ << ", seq=" << prior.source_seq_
                << ", ts[nsec]=" << prior.timestamp_kf_nsec_
                << ", frame_id=" << matched_frame_id;
        ext_queue_scan_filter_ms +=
            elapsedMs(prior_scan_start, std::chrono::steady_clock::now());
      } // zy when something behaves oddly, you can trace exact upstream message through Kimera.
    }

    {
      std::lock_guard<std::mutex> queue_lock(external_pose_priors_queue_mutex_);
      if (!external_pose_priors_queue_.empty()) {
        remaining_queue.insert(
            remaining_queue.end(),
            std::make_move_iterator(external_pose_priors_queue_.begin()),
            std::make_move_iterator(external_pose_priors_queue_.end()));
      }
      external_pose_priors_queue_.swap(remaining_queue);
    }
  }

  size_t external_queue_size_now = 0u;
  {
    std::lock_guard<std::mutex> queue_lock(external_pose_priors_queue_mutex_);
    external_queue_size_now = external_pose_priors_queue_.size();
  }

  bool cbs_no_external_effect_epoch = false;
  bool cbs_allow_heavy_maintenance = true;

  // zy Step 12c
  #ifdef KIMERA_USE_CBS
  if (cbs_exchange_active &&
      (num_external_beliefs_staged > 0 ||
       num_external_beliefs_dropped_bad_noise > 0 ||
       num_external_beliefs_cov_rejected > 0 ||
       num_external_beliefs_cov_regularized > 0 ||
       num_external_beliefs_dropped_unknown_source > 0 ||
       num_external_beliefs_dropped_self_source > 0)) {
    num_external_beliefs_accepted =
        (num_external_beliefs_staged >= num_external_beliefs_rejected)
            ? (num_external_beliefs_staged - num_external_beliefs_rejected)
            : 0u;
    LOG_EVERY_N(INFO, 20) << "CBS addBeliefs: staged="
                          << num_external_beliefs_staged
                          << ", rejected=" << num_external_beliefs_rejected
                          << ", accepted=" << num_external_beliefs_accepted
                          << ", bad_noise="
                          << num_external_beliefs_dropped_bad_noise
                          << ", cov_rejected="
                          << num_external_beliefs_cov_rejected
                          << ", cov_regularized="
                          << num_external_beliefs_cov_regularized
                          << ", unknown_source="
                          << num_external_beliefs_dropped_unknown_source
                          << ", self_source="
                          << num_external_beliefs_dropped_self_source;

  }
  #endif

  if (num_external_priors_injected > 0 || num_external_priors_dropped_old > 0 ||
      num_external_priors_dropped_marginalized > 0 ||
      num_external_priors_dropped_inactive > 0 ||
      num_external_priors_dropped_disabled_mode > 0 ||
      num_external_priors_deferred_budget > 0) {
    LOG_EVERY_N(INFO, 20)
        << "External prior stats: injected=" << num_external_priors_injected
        << ", deferred=" << num_external_priors_deferred
        << ", dropped_old=" << num_external_priors_dropped_old
        << ", dropped_marginalized=" << num_external_priors_dropped_marginalized
        << ", dropped_inactive=" << num_external_priors_dropped_inactive
        << ", dropped_disabled_mode=" << num_external_priors_dropped_disabled_mode
        << ", deferred_budget=" << num_external_priors_deferred_budget
        << ", deferred_no_local_receiver_state="
        << num_external_priors_deferred_no_local_receiver_state
        << ", queue_size_now=" << external_queue_size_now;
  }

#ifdef KIMERA_USE_CBS
  if (cbs_exchange_active) {
    num_external_beliefs_accepted =
        (num_external_beliefs_staged >= num_external_beliefs_rejected)
            ? (num_external_beliefs_staged - num_external_beliefs_rejected)
            : 0u;
    std::cerr << std::setprecision(12)
              << "[CBS][ExternalPriorDiag] timestamp_ns=" << timestamp_kf_nsec
              << " cbs_heart_active=" << (cbs_heart_active ? 1 : 0)
              << " use_cbs_optimizer=" << (FLAGS_use_cbs_optimizer ? 1 : 0)
              << " cbs_replace_fixed_lag_optimizer="
              << (FLAGS_cbs_replace_fixed_lag_optimizer ? 1 : 0)
              << " fixed_lag_states=" << backend_params_.nr_states_
              << " injected=" << num_external_priors_injected
              << " deferred=" << num_external_priors_deferred
              << " deferred_budget=" << num_external_priors_deferred_budget
              << " dropped_old=" << num_external_priors_dropped_old
              << " dropped_marginalized="
              << num_external_priors_dropped_marginalized
              << " dropped_inactive=" << num_external_priors_dropped_inactive
              << " dropped_disabled_mode="
              << num_external_priors_dropped_disabled_mode
              << " deferred_no_local_receiver_state="
              << num_external_priors_deferred_no_local_receiver_state
              << " beliefs_staged=" << num_external_beliefs_staged
              << " beliefs_accepted=" << num_external_beliefs_accepted
              << " beliefs_rejected=" << num_external_beliefs_rejected
              << " beliefs_bad_noise=" << num_external_beliefs_dropped_bad_noise
              << " beliefs_cov_rejected="
              << num_external_beliefs_cov_rejected
              << " beliefs_cov_regularized="
              << num_external_beliefs_cov_regularized
              << " beliefs_unknown_source="
              << num_external_beliefs_dropped_unknown_source
              << " beliefs_self_source="
              << num_external_beliefs_dropped_self_source
              << " queue_size_now=" << external_queue_size_now << std::endl;

    const double ext_queue_total_ms = ext_queue_scan_filter_ms;
    const double ext_queue_filter_only_ms = std::max(
        0.0,
        ext_queue_total_ms - ext_window_match_ms - ext_staging_construct_ms -
            ext_add_beliefs_ms - ext_cov_query_ms - ext_alignment_ms);
    std::cerr
        << std::setprecision(12)
        << "[CBS][EpochTimingDiag] timestamp_ns=" << timestamp_kf_nsec
        << " curr_kf_id=" << cur_id
        << " ext_queue_total_ms=" << ext_queue_total_ms
        << " ext_queue_filter_only_ms=" << ext_queue_filter_only_ms
        << " ext_window_match_ms=" << ext_window_match_ms
        << " ext_stage_construct_ms=" << ext_staging_construct_ms
        << " ext_add_beliefs_ms=" << ext_add_beliefs_ms
        << " ext_cov_query_ms=" << ext_cov_query_ms
        << " ext_alignment_ms=" << ext_alignment_ms
        << " beliefs_received=" << num_external_beliefs_received
        << " beliefs_considered=" << num_external_beliefs_considered
        << " beliefs_staged=" << num_external_beliefs_staged
        << " beliefs_rejected=" << num_external_beliefs_rejected
        << " beliefs_accepted=" << num_external_beliefs_accepted
        << " priors_injected=" << num_external_priors_injected
        << " priors_deferred=" << num_external_priors_deferred
        << " priors_deferred_budget=" << num_external_priors_deferred_budget
        << " priors_fast_skipped=" << num_external_priors_fast_skipped
        << " priors_source_backoff_skipped="
        << num_external_priors_source_backoff_skipped
        << " priors_considered=" << num_external_priors_considered
        << " rejected_keys_touched="
        << num_external_beliefs_rejected_keys_touched
        << " rejected_factors_touched="
        << num_external_beliefs_rejected_factors_touched
        << " queue_size_now=" << external_queue_size_now << std::endl;
  }

  cbs_no_external_effect_epoch =
      cbs_exchange_active && cbs_heart_active &&
      num_external_beliefs_accepted == 0u && num_external_priors_injected == 0u;
  // Keep true fixed-lag maintenance active during normal keyframe epochs.
  // Only allow the no-external fast path to skip remove-index work in epochs
  // that do not add new state (otherwise the active window would grow
  // unbounded).
  const bool cbs_skip_remove_requested =
      FLAGS_cbs_skip_remove_indices_when_no_external_effect &&
      cbs_no_external_effect_epoch;
  const bool cbs_force_no_external_fast_path_active =
      FLAGS_cbs_diag_force_no_external_fast_path_when_no_external_effect &&
      cbs_no_external_effect_epoch;
  const bool cbs_startup_warmup_done =
      cur_id >= static_cast<FrameId>(backend_params_.nr_states_);
  const bool cbs_epoch_adds_new_state = !new_values_.empty();
  const bool cbs_skip_remove_active = cbs_force_no_external_fast_path_active ||
                                      (cbs_skip_remove_requested &&
                                       cbs_startup_warmup_done &&
                                       !cbs_epoch_adds_new_state);
  cbs_allow_heavy_maintenance = !cbs_skip_remove_active;

  {
    std::lock_guard<std::mutex> effect_lock(cbs_external_effect_state_mutex_);
    cbs_last_epoch_no_external_effect_ = cbs_no_external_effect_epoch;
    cbs_last_epoch_beliefs_accepted_ = num_external_beliefs_accepted;
    cbs_last_epoch_priors_injected_ = num_external_priors_injected;
    cbs_last_epoch_timestamp_ns_ = timestamp_kf_nsec;
  }
#endif



  // Only for statistics and debugging.
  // Store start time to calculate absolute total time taken.
  const auto& total_start_time = utils::Timer::tic();
  // Store start time to calculate per module total time.
  auto start_time = total_start_time;
  // Reset all timing infupdateSmoother
  /////////////////////// BOOKKEEPING ////////////////////////////////////
  size_t new_smart_factors_size = new_smart_factors_.size();
  size_t cbs_smart_factor_replacements = 0u;
  size_t cbs_smart_factor_new_insertions = 0u;
  // We need to remove all previous smart factors in the factor graph
  // for which we have new observations.
  // The following is just to update the vector delete_slots with those
  // slots in the factor graph that correspond to smart factors for which
  // we've got new observations.
  // We initialize delete_slots with Extra factor slots to delete contains
  // potential factors that we want to delete, it is typically an empty
  // vector, and is only used to give flexibility to subclasses (regular
  // vio).
  gtsam::FactorIndices delete_slots = extra_factor_slots_to_delete;
  size_t cbs_delete_slots_total = delete_slots.size();
  size_t cbs_delete_slots_from_smart_replacement = 0u;
  size_t cbs_delete_slots_from_cheirality_cleanup = 0u;
  size_t cbs_delete_slots_from_other_cleanup = delete_slots.size();
  size_t cbs_smart_replacement_same_support_count = 0u;
  size_t cbs_smart_replacement_same_pose_key_set_count = 0u;
  size_t cbs_smart_replacement_small_support_delta_count = 0u;
  size_t cbs_smart_replacement_large_support_delta_count = 0u;
  size_t cbs_smart_replacement_due_to_material_support_change = 0u;
  size_t cbs_smart_replacement_due_to_pose_key_set_change = 0u;
  size_t cbs_smart_replacement_due_to_invalid_to_valid_transition = 0u;
  size_t cbs_smart_replacement_due_to_valid_to_invalid_transition = 0u;
  size_t cbs_smart_replacement_due_to_degenerate_factor = 0u;
  size_t cbs_smart_replacement_due_to_correctness_threshold = 0u;
  size_t cbs_smart_replacement_skipped_small_change = 0u;
  size_t cbs_smart_replacement_skipped_keep_existing = 0u;
  std::unordered_set<LandmarkId> cbs_landmarks_touched_for_replacement;
  std::unordered_set<LandmarkId> cbs_landmarks_touched_for_new_insertion;

  // zy Step 22a: pick the currently active optimizer graph (CBS or legacy) so debug/bookkeeping reads the correct factor graph.
  const gtsam::NonlinearFactorGraph* active_factor_graph = nullptr;
#ifdef KIMERA_USE_CBS
  if (useCbsOptimizerHeart()) {
    CHECK(cbs_optimizer_) << "CBS optimizer flag is ON but cbs_optimizer_ is null.";
    active_factor_graph = &cbs_optimizer_->getFactorsUnsafe();
  } else
#endif
  {
    active_factor_graph = &smoother_->getFactors();
  }
  CHECK_NOTNULL(active_factor_graph);


  // TODO we know the actual end size... but I am not sure how to use factor
  // graph API for appending factors without copying or re-allocation...
  std::vector<LandmarkId> lmk_ids_of_new_smart_factors_tmp;
  lmk_ids_of_new_smart_factors_tmp.reserve(new_smart_factors_size);
  gtsam::NonlinearFactorGraph new_factors_tmp;
  new_factors_tmp.reserve(new_smart_factors_size +
                          new_imu_prior_and_other_factors_.size() +
                          new_external_prior_factors_.size());
// zy cancelled it  for (const auto& new_smart_factor : new_smart_factors_) {
//     // Push back the smart factor to the list of new factors to add to the
//     // graph. // Smart factor, so same address right?
//     LandmarkId lmk_id = new_smart_factor.first;  // don't use &

//     // Find smart factor and slot in old_smart_factors_ corresponding to
//     // the lmk with id of the new smart factor.
//     const auto& old_smart_factor_it = old_smart_factors_.find(lmk_id);
//     CHECK(old_smart_factor_it != old_smart_factors_.end())
//         << "Lmk with id: " << lmk_id
//         << " could not be found in old_smart_factors_.";

//     // Slot slot = old_smart_factor_it->second.second; (zy cancelled it)
//     // zy 19a
//     Slot slot = old_smart_factor_it->second.second;
//     if (slot != -1) {
//       DCHECK_GE(slot, 0);

//       bool slot_is_active = false;
// #ifdef KIMERA_USE_CBS
//       if (useCbsOptimizerHeart()) {
//         // Intuition: when CBS is active, slot validity must be checked against the CBS factor graph.
//         CHECK(cbs_optimizer_) << "CBS optimizer flag is ON but cbs_optimizer_ is null.";
//         slot_is_active = cbs_optimizer_->getFactorsUnsafe().exists(slot);
//       } else
// #endif
//       {
//         // Intuition: preserve original fixed-lag slot check when CBS mode is disabled.
//         slot_is_active = smoother_->getFactors().exists(slot);
//       }

//       if (slot_is_active) {
//         // Intuition: replace stale smart factor with its refreshed version for this landmark.
//         delete_slots.push_back(slot);
//         new_factors_tmp.push_back(new_smart_factor.second);
//         lmk_ids_of_new_smart_factors_tmp.push_back(lmk_id);
//       } else {
//         // Intuition: if the previous slot no longer exists, drop stale bookkeeping to keep horizon state consistent.
//         old_smart_factors_.erase(old_smart_factor_it);
//         CHECK(deleteLmkFromFeatureTracks(lmk_id));
//       }
//     } else {
//       // Intuition: slot -1 means this smart factor has never been inserted yet, so add it now.
//       new_factors_tmp.push_back(new_smart_factor.second);
//       lmk_ids_of_new_smart_factors_tmp.push_back(lmk_id);
//     }


//       bool slot_is_active = false;
// #ifdef KIMERA_USE_CBS
//       if (useCbsOptimizerHeart()) {
//         // Intuition: in CBS mode, validate smart-factor slots against CBS factor graph, not the legacy smoother graph.
//         CHECK(cbs_optimizer_) << "CBS optimizer flag is ON but cbs_optimizer_ is null.";
//         slot_is_active = cbs_optimizer_->getFactorsUnsafe().exists(slot);
//       } else
// #endif
//       {
//         // Intuition: preserve original fixed-lag behavior when CBS mode is disabled.
//         slot_is_active = smoother_->getFactors().exists(slot);
//       }

//       if (slot_is_active) {
//       // Smart factor Slot is different than -1, therefore the factor should be
//       // already in the factor graph.
//       DCHECK_GE(slot, 0);
//       if (smoother_->getFactors().exists(slot)) {
//         // Confirmed, the factor is in the graph.
//         // We must delete the old smart factor from the graph.
//         // TODO what happens if delete_slots has repeated elements?
//         delete_slots.push_back(slot);
//         // And we must add the new smart factor to the graph.
//         new_factors_tmp.push_back(new_smart_factor.second);
//         // Store lmk id of the smart factor to add to the graph.
//         lmk_ids_of_new_smart_factors_tmp.push_back(lmk_id);
//       } else {
//         // This should not happen, unless feature tracks are so long
//         // (longer than factor graph's time horizon), than the factor has been
//         // removed from the optimization.
//         // Erase this factor and feature track, as it has gone past the horizon.
//         // TODO(marcus): check with toni if this needs a warning
//         old_smart_factors_.erase(old_smart_factor_it);
//         CHECK(deleteLmkFromFeatureTracks(lmk_id));
//         // TODO(Toni): we should as well remove it from new_smart_factors_!!
//       }
//     } else {
//       // We just add the new smart factor to the graph, as it has never been
//       // there before.
//       new_factors_tmp.push_back(new_smart_factor.second);
//       // Store lmk id of the smart factor to add to the graph.
//       lmk_ids_of_new_smart_factors_tmp.push_back(lmk_id);
//     }
//   }
  // zy step 20
  const size_t cbs_material_support_delta_threshold =
      static_cast<size_t>(std::max(1, FLAGS_cbs_smart_replace_material_support_delta));
  const size_t cbs_material_pose_key_delta_threshold = static_cast<size_t>(
      std::max(1, FLAGS_cbs_smart_replace_material_pose_key_delta));
  const auto sorted_unique_keys =
      [](const SmartStereoFactor::shared_ptr& factor) -> std::vector<gtsam::Key> {
    std::vector<gtsam::Key> keys;
    if (!factor) {
      return keys;
    }
    keys.reserve(factor->keys().size());
    for (const gtsam::Key key : factor->keys()) {
      keys.push_back(key);
    }
    std::sort(keys.begin(), keys.end());
    keys.erase(std::unique(keys.begin(), keys.end()), keys.end());
    return keys;
  };
  const auto symmetric_difference_size = [](const std::vector<gtsam::Key>& a,
                                            const std::vector<gtsam::Key>& b)
      -> size_t {
    std::vector<gtsam::Key> diff;
    diff.reserve(a.size() + b.size());
    std::set_symmetric_difference(a.begin(),
                                  a.end(),
                                  b.begin(),
                                  b.end(),
                                  std::back_inserter(diff));
    return diff.size();
  };
  for (const auto& new_smart_factor : new_smart_factors_) {
    // Push back the smart factor to the list of new factors to add to the graph.
    const LandmarkId lmk_id = new_smart_factor.first;
    const SmartStereoFactor::shared_ptr& candidate_factor = new_smart_factor.second;

    // Find smart factor and slot in old_smart_factors_ corresponding to this landmark.
    auto old_smart_factor_it = old_smart_factors_.find(lmk_id);
    CHECK(old_smart_factor_it != old_smart_factors_.end())
        << "Lmk with id: " << lmk_id
        << " could not be found in old_smart_factors_.";

    Slot slot = old_smart_factor_it->second.second;
    if (slot != -1) {
      DCHECK_GE(slot, 0);

      bool slot_is_active = false;
#ifdef KIMERA_USE_CBS
      if (useCbsOptimizerHeart()) {
        // Intuition: in CBS mode, validate smart-factor slots against CBS graph ownership.
        CHECK(cbs_optimizer_) << "CBS optimizer flag is ON but cbs_optimizer_ is null.";
        slot_is_active = cbs_optimizer_->getFactorsUnsafe().exists(slot);
      } else
#endif
      {
        // Intuition: in legacy mode, keep slot validation against the fixed-lag smoother graph.
        slot_is_active = smoother_->getFactors().exists(slot);
      }

      if (slot_is_active) {
        const SmartStereoFactor::shared_ptr& old_factor =
            old_smart_factor_it->second.first;
        const std::vector<gtsam::Key> old_keys = sorted_unique_keys(old_factor);
        const std::vector<gtsam::Key> new_keys = sorted_unique_keys(candidate_factor);
        const size_t old_support_size = old_keys.size();
        const size_t new_support_size = new_keys.size();
        const bool same_support_size = old_support_size == new_support_size;
        const bool same_pose_key_set = old_keys == new_keys;
        const size_t pose_key_set_delta = symmetric_difference_size(old_keys, new_keys);
        const size_t support_delta_abs =
            (new_support_size >= old_support_size)
                ? (new_support_size - old_support_size)
                : (old_support_size - new_support_size);
        const bool small_support_delta = support_delta_abs <= 1u;
        const bool support_changed_materially =
            support_delta_abs >= cbs_material_support_delta_threshold;
        const bool pose_key_set_changed_materially =
            pose_key_set_delta >= cbs_material_pose_key_delta_threshold;

        bool old_valid = false;
        bool new_valid = false;
        if (old_factor) {
          try {
            old_valid = old_factor->point().valid();
          } catch (...) {
            old_valid = false;
          }
        }
        if (candidate_factor) {
          try {
            new_valid = candidate_factor->point().valid();
          } catch (...) {
            new_valid = false;
          }
        }
        const bool due_to_invalid_to_valid_transition = !old_valid && new_valid;
        const bool due_to_valid_to_invalid_transition = old_valid && !new_valid;
        const bool due_to_degenerate_factor =
            (!old_valid && !new_valid) || old_support_size < 2u ||
            new_support_size < 2u;
        const bool due_to_correctness_threshold =
            old_support_size < 2u && new_support_size >= 2u;
        const bool has_material_shape_change =
            support_changed_materially || pose_key_set_changed_materially;
        const bool has_correctness_upgrade =
            due_to_invalid_to_valid_transition || due_to_correctness_threshold;
        // Conservative gate: keep existing active smart factor unless there is
        // a material support/pose-key change or a clear correctness upgrade.
        const bool should_replace_default =
            has_material_shape_change || has_correctness_upgrade;
        const bool should_replace =
            should_replace_default &&
            !(useCbsOptimizerHeart() &&
              FLAGS_cbs_diag_disable_smart_factor_replacements);

        if (!should_replace) {
          ++cbs_smart_replacement_skipped_keep_existing;
          const bool small_change_case =
              !has_material_shape_change &&
              !due_to_invalid_to_valid_transition &&
              !due_to_valid_to_invalid_transition &&
              !due_to_degenerate_factor &&
              !due_to_correctness_threshold;
          if (small_change_case) {
            ++cbs_smart_replacement_skipped_small_change;
          }
          continue;
        }

        // Intuition: replace previous smart factor for this landmark with the refreshed factor.
        delete_slots.push_back(slot);
        new_factors_tmp.push_back(candidate_factor);
        lmk_ids_of_new_smart_factors_tmp.push_back(lmk_id);
        ++cbs_smart_factor_replacements;
        ++cbs_delete_slots_from_smart_replacement;
        if (same_support_size) {
          ++cbs_smart_replacement_same_support_count;
        }
        if (same_pose_key_set) {
          ++cbs_smart_replacement_same_pose_key_set_count;
        }
        if (small_support_delta) {
          ++cbs_smart_replacement_small_support_delta_count;
        } else {
          ++cbs_smart_replacement_large_support_delta_count;
        }
        if (support_changed_materially) {
          ++cbs_smart_replacement_due_to_material_support_change;
        }
        if (pose_key_set_changed_materially) {
          ++cbs_smart_replacement_due_to_pose_key_set_change;
        }
        if (due_to_invalid_to_valid_transition) {
          ++cbs_smart_replacement_due_to_invalid_to_valid_transition;
        }
        if (due_to_valid_to_invalid_transition) {
          ++cbs_smart_replacement_due_to_valid_to_invalid_transition;
        }
        if (due_to_degenerate_factor) {
          ++cbs_smart_replacement_due_to_degenerate_factor;
        }
        if (due_to_correctness_threshold) {
          ++cbs_smart_replacement_due_to_correctness_threshold;
        }
        cbs_landmarks_touched_for_replacement.insert(lmk_id);
      } else {
        // Intuition: if old slot vanished, drop stale bookkeeping so horizon state stays consistent.
        old_smart_factors_.erase(old_smart_factor_it);
        CHECK(deleteLmkFromFeatureTracks(lmk_id));
      }
    } else {
      // Intuition: slot -1 means first insertion of this smart factor into the graph.
      new_factors_tmp.push_back(candidate_factor);
      lmk_ids_of_new_smart_factors_tmp.push_back(lmk_id);
      ++cbs_smart_factor_new_insertions;
      cbs_landmarks_touched_for_new_insertion.insert(lmk_id);
    }
  }
  cbs_delete_slots_total = delete_slots.size();
  if (cbs_delete_slots_total >=
      cbs_delete_slots_from_smart_replacement +
          cbs_delete_slots_from_cheirality_cleanup) {
    cbs_delete_slots_from_other_cleanup =
        cbs_delete_slots_total - cbs_delete_slots_from_smart_replacement -
        cbs_delete_slots_from_cheirality_cleanup;
  } else {
    cbs_delete_slots_from_other_cleanup = 0u;
  }

  // Add also other factors (imu, priors).
  // SMART FACTORS MUST BE FIRST, otherwise when recovering the slots
  // for the smart factors we will mess up.
  // push back many factors with an iterator over shared_ptr
  // (factors are not copied)
  new_factors_tmp.push_back(new_imu_prior_and_other_factors_.begin(),
                            new_imu_prior_and_other_factors_.end());
  const size_t local_heart_factor_count = new_factors_tmp.size();
#ifndef KIMERA_USE_CBS
  (void)local_heart_factor_count;
#endif
  // External priors are appended only for heart optimization.
  new_factors_tmp.push_back(new_external_prior_factors_.begin(),
                            new_external_prior_factors_.end());

  //////////////////////////////////////////////////////////////////////////////

  if (VLOG_IS_ON(10) || log_output_) {
    debug_info_.factorsAndSlotsTime_ =
        utils::Timer::toc<std::chrono::seconds>(start_time).count();
    start_time = utils::Timer::tic();
  }

  if (VLOG_IS_ON(10)) {
    // Get state before optimization to compute error.
    debug_info_.stateBeforeOpt = gtsam::Values(state_);
    for (const auto& key_value : new_values_) {
      debug_info_.stateBeforeOpt.insert(key_value.key, key_value.value);
    }
  }

  // if (VLOG_IS_ON(10)) {
  //   printSmootherInfo(new_factors_tmp,
  //                     delete_slots,
  //                     "Smoother status before update:",
  //                     VLOG_IS_ON(10));
  // } (zy cancelled it)

  // zy Step 22b
  if (VLOG_IS_ON(10)) {
#ifdef KIMERA_USE_CBS
    if (useCbsOptimizerHeart()) {
      // Intuition: avoid dumping stale fixed-lag internals when CBS is the active optimization heart.
      VLOG(10) << "CBS mode: skipping legacy printSmootherInfo() dump.";
    } else {
      printSmootherInfo(new_factors_tmp,
                        delete_slots,
                        "Smoother status before update:",
                        VLOG_IS_ON(10));
    }
#else
    printSmootherInfo(new_factors_tmp,
                      delete_slots,
                      "Smoother status before update:",
                      VLOG_IS_ON(10));
#endif
  }


  // // Recreate the graph before marginalization.
  // if (VLOG_IS_ON(10) && FLAGS_debug_graph_before_opt) {
  //   debug_info_.graphBeforeOpt = smoother_->getFactors();
  //   debug_info_.graphToBeDeleted = gtsam::NonlinearFactorGraph();
  //   debug_info_.graphToBeDeleted.resize(delete_slots.size());
  //   for (size_t i = 0u; i < delete_slots.size(); i++) {
  //     // If the factor is to be deleted, store it as graph to be deleted.
  //     CHECK(smoother_->getFactors().exists(delete_slots.at(i)));
  //     debug_info_.graphToBeDeleted.at(i) =
  //         smoother_->getFactors().at(delete_slots.at(i));
  //   }
  // } (zy cancelled it)

  // zy Step 22c 
  if (VLOG_IS_ON(10) && FLAGS_debug_graph_before_opt) {
    // Intuition: snapshot/delete-debug must track the active optimizer graph to keep slot references valid.
    debug_info_.graphBeforeOpt = *active_factor_graph;
    debug_info_.graphToBeDeleted = gtsam::NonlinearFactorGraph();
    debug_info_.graphToBeDeleted.resize(delete_slots.size());
    for (size_t i = 0u; i < delete_slots.size(); i++) {
      if (active_factor_graph->exists(delete_slots.at(i))) {
        debug_info_.graphToBeDeleted.at(i) =
            active_factor_graph->at(delete_slots.at(i));
      } else {
        VLOG(2) << "Delete slot " << delete_slots.at(i)
                << " not found in active factor graph.";
      }
    }
  }




  // Use current timestamp for each new value. This timestamp will be used
  // to determine if the variable should be marginalized.
  // Needs to use DOUBLE because gtsam works with that, but we
  // are actually counting the number of states in the smoother.
  std::map<Key, double> key_frame_count;
  for (const auto& key_value : new_values_) {
    key_frame_count[key_value.key] = cur_id;
  }
  DCHECK_EQ(key_frame_count.size(), new_values_.size());

  // Store time before iSAM update.
  if (VLOG_IS_ON(10) || log_output_) {
    debug_info_.updateTime_ =
        utils::Timer::toc<std::chrono::seconds>(start_time).count();
    start_time = utils::Timer::tic();
  }

#ifdef KIMERA_USE_CBS
  H2LocalCovSidecarPacket h2_sidecar_packet;
  bool h2_sidecar_packet_ready = false;
  std::string h2_sidecar_packet_failure_reason;
  std::string h2_sidecar_packet_failure_key;
  if (useCbsH2LocalCovSidecar() && !useCbsOptimizerHeart()) {
    h2_sidecar_packet.smart_factor_replacements = cbs_smart_factor_replacements;
    h2_sidecar_packet.filtered_external_factor_count =
        (new_factors_tmp.size() > local_heart_factor_count)
            ? (new_factors_tmp.size() - local_heart_factor_count)
            : 0u;
    h2_sidecar_packet.local_factors.reserve(local_heart_factor_count);
    h2_sidecar_packet.heart_new_factor_positions.reserve(
        local_heart_factor_count);
    std::unordered_set<gtsam::Key> required_local_keys;
    for (size_t factor_pos = 0u; factor_pos < local_heart_factor_count;
         ++factor_pos) {
      const auto& factor = new_factors_tmp.at(factor_pos);
      if (!factor) {
        continue;
      }
      // Mirror only epoch-local factors. Sharing immutable factor pointers
      // avoids clone() gaps for factor types that do not implement cloning.
      h2_sidecar_packet.local_factors.push_back(factor);
      h2_sidecar_packet.heart_new_factor_positions.push_back(factor_pos);
      for (const gtsam::Key key : factor->keys()) {
        required_local_keys.insert(key);
      }
    }

    auto set_packet_failure = [&](const std::string& reason_prefix,
                                  const gtsam::Key key) {
      const gtsam::Symbol symbol(key);
      std::ostringstream oss;
      oss << symbol.chr() << symbol.index();
      h2_sidecar_packet_failure_key = oss.str();
      h2_sidecar_packet_failure_reason = reason_prefix + ":" + oss.str();
    };

    if (h2_sidecar_packet_failure_reason.empty()) {
      for (const gtsam::Key key : required_local_keys) {
        const gtsam::Symbol symbol(key);
        try {
          if (symbol.chr() == kPoseSymbolChar) {
            if (new_values_.exists(symbol)) {
              h2_sidecar_packet.local_values.insert(
                  symbol, new_values_.at<gtsam::Pose3>(symbol));
            } else if (state_.exists(symbol)) {
              h2_sidecar_packet.local_values.insert(
                  symbol, state_.at<gtsam::Pose3>(symbol));
            } else {
              set_packet_failure("missing_local_value_for_key", key);
              break;
            }
            ++h2_sidecar_packet.pose_values_count;
          } else if (symbol.chr() == kVelocitySymbolChar) {
            if (new_values_.exists(symbol)) {
              h2_sidecar_packet.local_values.insert(
                  symbol, new_values_.at<gtsam::Vector3>(symbol));
            } else if (state_.exists(symbol)) {
              h2_sidecar_packet.local_values.insert(
                  symbol, state_.at<gtsam::Vector3>(symbol));
            } else {
              set_packet_failure("missing_local_value_for_key", key);
              break;
            }
            ++h2_sidecar_packet.vel_values_count;
          } else if (symbol.chr() == kImuBiasSymbolChar) {
            if (new_values_.exists(symbol)) {
              h2_sidecar_packet.local_values.insert(
                  symbol, new_values_.at<gtsam::imuBias::ConstantBias>(symbol));
            } else if (state_.exists(symbol)) {
              h2_sidecar_packet.local_values.insert(
                  symbol, state_.at<gtsam::imuBias::ConstantBias>(symbol));
            } else {
              set_packet_failure("missing_local_value_for_key", key);
              break;
            }
            ++h2_sidecar_packet.bias_values_count;
          } else {
            set_packet_failure("unsupported_key_type", key);
            break;
          }
        } catch (...) {
          set_packet_failure("wrong_value_type_for_key", key);
          break;
        }
      }
    }

    if (h2_sidecar_packet_failure_reason.empty()) {
      h2_sidecar_packet_ready = true;
    }
  }
#endif

  // Compute iSAM update.
  VLOG(10) << "iSAM2 update with " << new_factors_tmp.size() << " new factors "
           << ", " << new_values_.size() << " new values "
           << ", and " << delete_slots.size() << " deleted factors.";
  Smoother::Result result;
  VLOG(10) << "Starting first update.";
  bool is_smoother_ok = updateSmoother(
      &result,
      new_factors_tmp,
      new_values_,
      key_frame_count,
      delete_slots,
      cbs_allow_heavy_maintenance,
      cbs_smart_factor_replacements,
      cbs_smart_factor_new_insertions,
      cbs_delete_slots_total,
      cbs_delete_slots_from_smart_replacement,
      cbs_delete_slots_from_cheirality_cleanup,
      cbs_delete_slots_from_other_cleanup,
      cbs_smart_replacement_same_support_count,
      cbs_smart_replacement_same_pose_key_set_count,
      cbs_smart_replacement_small_support_delta_count,
      cbs_smart_replacement_large_support_delta_count,
      cbs_smart_replacement_due_to_material_support_change,
      cbs_smart_replacement_due_to_pose_key_set_change,
      cbs_smart_replacement_due_to_invalid_to_valid_transition,
      cbs_smart_replacement_due_to_valid_to_invalid_transition,
      cbs_smart_replacement_due_to_degenerate_factor,
      cbs_smart_replacement_due_to_correctness_threshold,
      cbs_smart_replacement_skipped_small_change,
      cbs_smart_replacement_skipped_keep_existing,
      cbs_landmarks_touched_for_replacement.size(),
      cbs_landmarks_touched_for_new_insertion.size(),
      new_imu_prior_and_other_factors_.size());
  VLOG(10) << "Finished first update.";

  // Store time after iSAM update.
  if (VLOG_IS_ON(10) || log_output_) {
    debug_info_.updateTime_ =
        utils::Timer::toc<std::chrono::seconds>(start_time).count();
    start_time = utils::Timer::tic();
  }

  /////////////////////////// BOOKKEEPING //////////////////////////////////////
  if (is_smoother_ok) {
    // Reset everything for next round.
    // TODO what about the old_smart_factors_?
    VLOG(10) << "Clearing new_smart_factors_!";
    new_smart_factors_.clear();

    // Reset list of new imu, prior and other factors to be added.
    // TODO could this be used to check whether we are repeating factors?
    new_imu_prior_and_other_factors_.resize(0);
    new_external_prior_factors_.resize(0);

    // Clear values.
    new_values_.clear();

    // Update slots of smart factors:.
    // TODO(Toni): shouldn't we be doing this after each updateSmoother call?
    VLOG(10) << "Starting to find smart factors slots.";
    updateNewSmartFactorsSlots(lmk_ids_of_new_smart_factors_tmp,
                               &old_smart_factors_);
    VLOG(10) << "Finished to find smart factors slots.";

    if (VLOG_IS_ON(5) || log_output_) {
      debug_info_.updateSlotTime_ =
          utils::Timer::toc<std::chrono::seconds>(start_time).count();
      start_time = utils::Timer::tic();
    }

    ////////////////////////////////////////////////////////////////////////////

    // // Do some more optimization iterations.
    // for (size_t n_iter = 1; n_iter < max_extra_iterations && is_smoother_ok;
    //      ++n_iter) {
    //   VLOG(10) << "Doing extra iteration nr: " << n_iter;
    //   is_smoother_ok = updateSmoother(&result);
    // } (zy cancelled it)

    // ZY Step 21: CBS already performs its own incremental update; repeating empty legacy-style iterations adds no value.
#ifdef KIMERA_USE_CBS
    const bool run_extra_iterations = !useCbsOptimizerHeart();
#else
    const bool run_extra_iterations = true;
#endif
    if (run_extra_iterations) {
      for (size_t n_iter = 1; n_iter < max_extra_iterations && is_smoother_ok;
           ++n_iter) {
        VLOG(10) << "Doing extra iteration nr: " << n_iter;
        is_smoother_ok = updateSmoother(&result);
      }
    }


    if (VLOG_IS_ON(5) || log_output_) {
      debug_info_.extraIterationsTime_ =
          utils::Timer::toc<std::chrono::seconds>(start_time).count();
      start_time = utils::Timer::tic();
    }

    // Update states we need for next iteration, if smoother is ok.
    if (is_smoother_ok) {
      updateStates(cur_id);
#ifdef KIMERA_USE_CBS
      if (useCbsH2LocalCovSidecar()) {
        H2LocalCovSidecarSyncStats h2_stats;
        bool h2_sync_ok = false;
        if (h2_sidecar_packet_ready) {
          h2_sync_ok = refreshH2LocalCovariancePassiveSnapshot(
              h2_sidecar_packet, cur_id, &h2_stats);
        } else if (useCbsOptimizerHeart()) {
          h2_stats.failure_reason = "heart_is_cbs";
        } else if (!h2_sidecar_packet_failure_reason.empty()) {
          h2_stats.failure_reason = h2_sidecar_packet_failure_reason;
          h2_stats.first_failure_reason = h2_sidecar_packet_failure_reason;
          h2_stats.first_failure_key = h2_sidecar_packet_failure_key;
          h2_stats.filtered_external_factor_count =
              h2_sidecar_packet.filtered_external_factor_count;
          h2_stats.unmapped_remove_slot_count =
              h2_sidecar_packet.unmapped_remove_slot_count;
          h2_stats.packet_remove_count =
              h2_sidecar_packet.sidecar_remove_factor_indices.size();
          h2_stats.packet_mapped_heart_remove_slots_count =
              h2_sidecar_packet.mapped_heart_remove_factor_slots.size();
          h2_stats.packet_heart_new_factor_positions_count =
              h2_sidecar_packet.heart_new_factor_positions.size();
          h2_stats.packet_pose_values_count =
              h2_sidecar_packet.pose_values_count;
          h2_stats.packet_vel_values_count =
              h2_sidecar_packet.vel_values_count;
          h2_stats.packet_bias_values_count =
              h2_sidecar_packet.bias_values_count;
          h2_stats.snapshot_refresh_ms = 0.0;
          h2_local_graph_snapshot_.resize(0);
          h2_local_values_snapshot_.clear();
          h2_local_snapshot_timestamp_ns_ = -1;
          h2_local_snapshot_frame_id_ = 0;
          h2_local_snapshot_valid_ = false;
          ++cbs_h2_sidecar_hard_reset_count_;
          h2_stats.hard_reset_count = cbs_h2_sidecar_hard_reset_count_;
        } else {
          h2_stats.failure_reason = "h2_sidecar_packet_unavailable";
          h2_stats.filtered_external_factor_count =
              h2_sidecar_packet.filtered_external_factor_count;
          h2_stats.snapshot_refresh_ms = 0.0;
          h2_stats.hard_reset_count = cbs_h2_sidecar_hard_reset_count_;
          h2_local_graph_snapshot_.resize(0);
          h2_local_values_snapshot_.clear();
          h2_local_snapshot_timestamp_ns_ = -1;
          h2_local_snapshot_frame_id_ = 0;
          h2_local_snapshot_valid_ = false;
        }
        cbs_h2_sidecar_sync_ok_ = h2_sync_ok;
        cbs_h2_sidecar_update_ms_last_epoch_ = 0.0;
        cbs_h2_local_snapshot_refresh_ms_last_epoch_ =
            h2_stats.snapshot_refresh_ms;
        cbs_h2_sidecar_local_factor_count_last_epoch_ =
            h2_stats.local_factor_count;
        cbs_h2_sidecar_filtered_external_count_last_epoch_ =
            h2_stats.filtered_external_factor_count;
        cbs_h2_sidecar_remove_count_last_epoch_ = h2_stats.remove_count;
        cbs_h2_sidecar_values_add_count_last_epoch_ = h2_stats.values_add_count;
        cbs_h2_sidecar_hard_reset_count_ = h2_stats.hard_reset_count;
        cbs_h2_sidecar_smart_factor_replacements_last_epoch_ =
            h2_stats.smart_factor_replacements;
        cbs_h2_sidecar_value_type_mismatch_count_last_epoch_ =
            h2_stats.value_type_mismatch_count;
        cbs_h2_sidecar_missing_value_count_last_epoch_ =
            h2_stats.missing_value_count;
        cbs_h2_sidecar_unsupported_key_type_count_last_epoch_ =
            h2_stats.unsupported_key_type_count;
        cbs_h2_sidecar_unmapped_remove_slot_count_last_epoch_ =
            h2_stats.unmapped_remove_slot_count;
        cbs_h2_sidecar_post_prune_remove_count_last_epoch_ =
            h2_stats.post_prune_remove_count;
        cbs_h2_sidecar_packet_pose_values_count_last_epoch_ =
            h2_stats.packet_pose_values_count;
        cbs_h2_sidecar_packet_vel_values_count_last_epoch_ =
            h2_stats.packet_vel_values_count;
        cbs_h2_sidecar_packet_bias_values_count_last_epoch_ =
            h2_stats.packet_bias_values_count;
        cbs_h2_sidecar_first_failure_key_last_epoch_ =
            h2_stats.first_failure_key;
        cbs_h2_sidecar_first_failure_reason_last_epoch_ =
            h2_stats.first_failure_reason;
        cbs_h2_sidecar_last_failure_reason_ = h2_stats.failure_reason;
        if (h2_sync_ok) {
          cbs_h2_sidecar_desync_streak_ = 0u;
        } else {
          ++cbs_h2_sidecar_desync_streak_;
        }
        if (!h2_sync_ok && !cbs_h2_first_bad_epoch_logged_) {
          cbs_h2_first_bad_epoch_logged_ = true;
          std::cerr << std::setprecision(12)
                    << "[CBS][H2FirstBadEpoch]"
                    << " curr_kf_id=" << cur_id
                    << " reason="
                    << (h2_stats.failure_reason.empty() ? "none"
                                                       : h2_stats.failure_reason)
                    << " first_failure_key="
                    << (h2_stats.first_failure_key.empty()
                            ? "none"
                            : h2_stats.first_failure_key)
                    << " packet_local_factors=" << h2_stats.local_factor_count
                    << " packet_remove_count=" << h2_stats.packet_remove_count
                    << " translated_remove_count="
                    << h2_stats.translated_remove_count
                    << " first_translated_sidecar_remove_slot="
                    << (h2_stats.first_translated_sidecar_remove_slot ==
                                std::numeric_limits<gtsam::FactorIndex>::max()
                            ? -1
                            : static_cast<long long>(
                                  h2_stats.first_translated_sidecar_remove_slot))
                    << " filtered_stale_remove_count="
                    << h2_stats.filtered_stale_remove_count
                    << " map_size_before="
                    << h2_stats.sidecar_slot_map_size_before_update
                    << " graph_size_before="
                    << h2_stats.sidecar_graph_size_before_update
                    << " heart_new_factor_positions="
                    << h2_stats.packet_heart_new_factor_positions_count
                    << " sidecar_new_factor_indices="
                    << h2_stats.sidecar_new_factor_indices_count
                    << " h2_repair_epoch_detected="
                    << (h2_stats.h2_repair_epoch_detected ? 1 : 0)
                    << " h2_repair_epoch_original_new_index_count="
                    << h2_stats.h2_repair_epoch_original_new_index_count
                    << " h2_replay_add_only_attempted="
                    << (h2_stats.h2_replay_add_only_attempted ? 1 : 0)
                    << " h2_replay_add_only_succeeded="
                    << (h2_stats.h2_replay_add_only_succeeded ? 1 : 0)
                    << " h2_replay_add_only_new_index_count="
                    << h2_stats.h2_replay_add_only_new_index_count
                    << " h2_replay_add_only_failure_reason="
                    << (h2_stats.h2_replay_add_only_failure_reason.empty()
                            ? "none"
                            : h2_stats.h2_replay_add_only_failure_reason)
                    << " h2_replay_subset_probe_attempted="
                    << (h2_stats.h2_replay_subset_probe_attempted ? 1 : 0)
                    << " h2_replay_subset_probe_clone_available="
                    << (h2_stats.h2_replay_subset_probe_clone_available ? 1 : 0)
                    << " h2_replay_subset_probe_clone_setup_reason="
                    << (h2_stats.h2_replay_subset_probe_clone_setup_reason.empty()
                            ? "none"
                            : h2_stats.h2_replay_subset_probe_clone_setup_reason)
                    << " h2_replay_subset_empty_succeeded="
                    << (h2_stats.h2_replay_subset_empty_succeeded ? 1 : 0)
                    << " h2_replay_subset_non_smart_succeeded="
                    << (h2_stats.h2_replay_subset_non_smart_succeeded ? 1 : 0)
                    << " h2_replay_subset_smart_succeeded="
                    << (h2_stats.h2_replay_subset_smart_succeeded ? 1 : 0)
                    << " h2_replay_subset_imu_only_succeeded="
                    << (h2_stats.h2_replay_subset_imu_only_succeeded ? 1 : 0)
                    << " h2_replay_subset_between_only_succeeded="
                    << (h2_stats.h2_replay_subset_between_only_succeeded ? 1 : 0)
                    << " h2_replay_subset_full_probe_succeeded="
                    << (h2_stats.h2_replay_subset_full_probe_succeeded ? 1 : 0)
                    << " h2_partial_replay_non_smart_only_applied="
                    << (h2_stats.h2_partial_replay_non_smart_only_applied ? 1 : 0)
                    << " h2_partial_replay_reason="
                    << (h2_stats.h2_partial_replay_reason.empty()
                            ? "none"
                            : h2_stats.h2_partial_replay_reason)
                    << " remap_existing_heart_slot_conflict_count="
                    << h2_stats.remap_existing_heart_slot_conflict_count
                    << " first_conflicting_heart_slot="
                    << (h2_stats.first_conflicting_heart_slot ==
                                std::numeric_limits<gtsam::FactorIndex>::max()
                            ? -1
                            : static_cast<long long>(
                                  h2_stats.first_conflicting_heart_slot))
                    << " first_conflicting_old_sidecar_slot="
                    << (h2_stats.first_conflicting_old_sidecar_slot ==
                                std::numeric_limits<gtsam::FactorIndex>::max()
                            ? -1
                            : static_cast<long long>(
                                  h2_stats.first_conflicting_old_sidecar_slot))
                    << " first_conflicting_new_sidecar_slot="
                    << (h2_stats.first_conflicting_new_sidecar_slot ==
                                std::numeric_limits<gtsam::FactorIndex>::max()
                            ? -1
                            : static_cast<long long>(
                                  h2_stats.first_conflicting_new_sidecar_slot))
                    << " sample_remove_indices="
                    << (h2_stats.sample_remove_indices.empty()
                            ? "none"
                            : h2_stats.sample_remove_indices)
                    << " sample_heart_new_factor_positions="
                    << (h2_stats.sample_heart_new_factor_positions.empty()
                            ? "none"
                            : h2_stats.sample_heart_new_factor_positions)
                    << " sample_heart_new_factor_indices="
                    << (h2_stats.sample_heart_new_factor_indices.empty()
                            ? "none"
                            : h2_stats.sample_heart_new_factor_indices)
                    << " sample_sidecar_new_factor_indices="
                    << (h2_stats.sample_sidecar_new_factor_indices.empty()
                            ? "none"
                            : h2_stats.sample_sidecar_new_factor_indices)
                    << " smart_factor_replacements="
                    << h2_stats.smart_factor_replacements
                    << " post_prune_candidate_count="
                    << h2_stats.post_prune_candidate_count
                    << " first_post_prune_sidecar_slot="
                    << (h2_stats.first_post_prune_sidecar_slot ==
                                std::numeric_limits<gtsam::FactorIndex>::max()
                            ? -1
                            : static_cast<long long>(
                                  h2_stats.first_post_prune_sidecar_slot))
                    << " post_prune_filtered_stale_count="
                    << h2_stats.post_prune_filtered_stale_count
                    << " retry_without_remove="
                    << (h2_stats.retry_without_remove_factor_indices ? 1 : 0)
                    << " primary_remove_attempted="
                    << (h2_stats.primary_remove_attempted ? 1 : 0)
                    << " primary_remove_deferred_due_to_stale_map="
                    << (h2_stats.primary_remove_deferred_due_to_stale_map ? 1 : 0)
                    << " primary_retry_attempted="
                    << (h2_stats.primary_retry_attempted ? 1 : 0)
                    << " primary_retry_succeeded="
                    << (h2_stats.primary_retry_succeeded ? 1 : 0)
                    << " post_prune_retry_with_filtered_slots="
                    << (h2_stats.post_prune_retry_with_filtered_slots ? 1 : 0)
                    << " post_prune_threw_map_at="
                    << (h2_stats.post_prune_threw_map_at ? 1 : 0)
                    << " post_prune_skipped_deferred="
                    << (h2_stats.post_prune_skipped_deferred ? 1 : 0)
                    << " post_prune_remove_count="
                    << h2_stats.post_prune_remove_count
                    << " first_unmapped_heart_slot="
                    << (h2_stats.first_unmapped_heart_slot ==
                                std::numeric_limits<gtsam::FactorIndex>::max()
                            ? -1
                            : static_cast<long long>(
                                  h2_stats.first_unmapped_heart_slot))
                    << " first_unmapped_sidecar_slot="
                    << (h2_stats.first_unmapped_sidecar_slot ==
                                std::numeric_limits<gtsam::FactorIndex>::max()
                            ? -1
                            : static_cast<long long>(
                                  h2_stats.first_unmapped_sidecar_slot))
                    << " packet_local_values_exact_required="
                    << (h2_stats.packet_local_values_exact_required ? 1 : 0)
                    << " packet_missing_required_local_values_count="
                    << h2_stats.packet_missing_required_local_values_count
                    << " packet_extra_local_values_count="
                    << h2_stats.packet_extra_local_values_count
                    << " packet_local_value_keys="
                    << (h2_stats.packet_local_value_keys.empty()
                            ? "none"
                            : h2_stats.packet_local_value_keys)
                    << " packet_missing_required_local_value_keys="
                    << (h2_stats.packet_missing_required_local_value_keys.empty()
                            ? "none"
                            : h2_stats.packet_missing_required_local_value_keys)
                    << std::endl;
        }
        std::cerr << std::setprecision(12)
                  << "[CBS][H2SidecarDiag] timestamp_ns=" << timestamp_kf_nsec
                  << " curr_kf_id=" << cur_id
                  << " h2_mode=" << (useCbsH2LocalCovSidecar() ? 1 : 0)
                  << " h2_sidecar_mode=" << cbsH2SidecarModeName()
                  << " h2_sidecar_sync_ok=" << (h2_sync_ok ? 1 : 0)
                  << " h2_sidecar_update_ms=0"
                  << " h2_local_snapshot_refresh_ms="
                  << h2_stats.snapshot_refresh_ms
                  << " h2_local_factor_count=" << h2_stats.local_factor_count
                  << " h2_filtered_external_factor_count="
                  << h2_stats.filtered_external_factor_count
                  << " h2_sidecar_remove_count=" << h2_stats.remove_count
                  << " h2_sidecar_values_add_count="
                  << h2_stats.values_add_count
                  << " h2_sidecar_hard_reset_count=" << h2_stats.hard_reset_count
                  << " h2_sidecar_smart_factor_replacements="
                  << h2_stats.smart_factor_replacements
                  << " h2_sidecar_value_type_mismatch_count="
                  << h2_stats.value_type_mismatch_count
                  << " h2_sidecar_missing_value_count="
                  << h2_stats.missing_value_count
                  << " h2_sidecar_unsupported_key_type_count="
                  << h2_stats.unsupported_key_type_count
                  << " h2_sidecar_unmapped_remove_slot_count="
                  << h2_stats.unmapped_remove_slot_count
                  << " h2_sidecar_post_prune_remove_count="
                  << h2_stats.post_prune_remove_count
                  << " h2_sidecar_packet_pose_values_count="
                  << h2_stats.packet_pose_values_count
                  << " h2_sidecar_packet_vel_values_count="
                  << h2_stats.packet_vel_values_count
                  << " h2_sidecar_packet_bias_values_count="
                  << h2_stats.packet_bias_values_count
                  << " h2_packet_remove_count="
                  << h2_stats.packet_remove_count
                  << " h2_packet_mapped_heart_remove_slots_count="
                  << h2_stats.packet_mapped_heart_remove_slots_count
                  << " h2_packet_heart_new_factor_positions_count="
                  << h2_stats.packet_heart_new_factor_positions_count
                  << " h2_sidecar_slot_map_size_before_update="
                  << h2_stats.sidecar_slot_map_size_before_update
                  << " h2_sidecar_graph_size_before_update="
                  << h2_stats.sidecar_graph_size_before_update
                  << " h2_translated_remove_count="
                  << h2_stats.translated_remove_count
                  << " h2_first_translated_sidecar_remove_slot="
                  << (h2_stats.first_translated_sidecar_remove_slot ==
                              std::numeric_limits<gtsam::FactorIndex>::max()
                          ? -1
                          : static_cast<long long>(
                                h2_stats.first_translated_sidecar_remove_slot))
                  << " h2_filtered_stale_remove_count="
                  << h2_stats.filtered_stale_remove_count
                  << " h2_sidecar_new_factor_indices_count="
                  << h2_stats.sidecar_new_factor_indices_count
                  << " h2_repair_epoch_detected="
                  << (h2_stats.h2_repair_epoch_detected ? 1 : 0)
                  << " h2_repair_epoch_original_new_index_count="
                  << h2_stats.h2_repair_epoch_original_new_index_count
                  << " h2_replay_add_only_attempted="
                  << (h2_stats.h2_replay_add_only_attempted ? 1 : 0)
                  << " h2_replay_add_only_succeeded="
                  << (h2_stats.h2_replay_add_only_succeeded ? 1 : 0)
                  << " h2_replay_add_only_new_index_count="
                  << h2_stats.h2_replay_add_only_new_index_count
                  << " h2_replay_add_only_failure_reason="
                  << (h2_stats.h2_replay_add_only_failure_reason.empty()
                          ? "none"
                          : h2_stats.h2_replay_add_only_failure_reason)
                  << " h2_replay_subset_probe_attempted="
                  << (h2_stats.h2_replay_subset_probe_attempted ? 1 : 0)
                  << " h2_replay_subset_probe_clone_available="
                  << (h2_stats.h2_replay_subset_probe_clone_available ? 1 : 0)
                  << " h2_replay_subset_probe_clone_setup_reason="
                  << (h2_stats.h2_replay_subset_probe_clone_setup_reason.empty()
                          ? "none"
                          : h2_stats.h2_replay_subset_probe_clone_setup_reason)
                  << " h2_replay_subset_empty_succeeded="
                  << (h2_stats.h2_replay_subset_empty_succeeded ? 1 : 0)
                  << " h2_replay_subset_non_smart_succeeded="
                  << (h2_stats.h2_replay_subset_non_smart_succeeded ? 1 : 0)
                  << " h2_replay_subset_smart_succeeded="
                  << (h2_stats.h2_replay_subset_smart_succeeded ? 1 : 0)
                  << " h2_replay_subset_imu_only_succeeded="
                  << (h2_stats.h2_replay_subset_imu_only_succeeded ? 1 : 0)
                  << " h2_replay_subset_between_only_succeeded="
                  << (h2_stats.h2_replay_subset_between_only_succeeded ? 1 : 0)
                  << " h2_replay_subset_full_probe_succeeded="
                  << (h2_stats.h2_replay_subset_full_probe_succeeded ? 1 : 0)
                  << " h2_partial_replay_non_smart_only_applied="
                  << (h2_stats.h2_partial_replay_non_smart_only_applied ? 1 : 0)
                  << " h2_partial_replay_reason="
                  << (h2_stats.h2_partial_replay_reason.empty()
                          ? "none"
                          : h2_stats.h2_partial_replay_reason)
                  << " h2_remap_existing_heart_slot_conflict_count="
                  << h2_stats.remap_existing_heart_slot_conflict_count
                  << " h2_first_conflicting_heart_slot="
                  << (h2_stats.first_conflicting_heart_slot ==
                              std::numeric_limits<gtsam::FactorIndex>::max()
                          ? -1
                          : static_cast<long long>(
                                h2_stats.first_conflicting_heart_slot))
                  << " h2_first_conflicting_old_sidecar_slot="
                  << (h2_stats.first_conflicting_old_sidecar_slot ==
                              std::numeric_limits<gtsam::FactorIndex>::max()
                          ? -1
                          : static_cast<long long>(
                                h2_stats.first_conflicting_old_sidecar_slot))
                  << " h2_first_conflicting_new_sidecar_slot="
                  << (h2_stats.first_conflicting_new_sidecar_slot ==
                              std::numeric_limits<gtsam::FactorIndex>::max()
                          ? -1
                          : static_cast<long long>(
                                h2_stats.first_conflicting_new_sidecar_slot))
                  << " h2_post_prune_candidate_count="
                  << h2_stats.post_prune_candidate_count
                  << " h2_first_post_prune_sidecar_slot="
                  << (h2_stats.first_post_prune_sidecar_slot ==
                              std::numeric_limits<gtsam::FactorIndex>::max()
                          ? -1
                          : static_cast<long long>(
                                h2_stats.first_post_prune_sidecar_slot))
                  << " h2_post_prune_filtered_stale_count="
                  << h2_stats.post_prune_filtered_stale_count
                  << " h2_retry_without_remove_factor_indices="
                  << (h2_stats.retry_without_remove_factor_indices ? 1 : 0)
                  << " h2_primary_remove_attempted="
                  << (h2_stats.primary_remove_attempted ? 1 : 0)
                  << " h2_primary_remove_deferred_due_to_stale_map="
                  << (h2_stats.primary_remove_deferred_due_to_stale_map ? 1 : 0)
                  << " h2_primary_retry_attempted="
                  << (h2_stats.primary_retry_attempted ? 1 : 0)
                  << " h2_primary_retry_succeeded="
                  << (h2_stats.primary_retry_succeeded ? 1 : 0)
                  << " h2_post_prune_retry_with_filtered_slots="
                  << (h2_stats.post_prune_retry_with_filtered_slots ? 1 : 0)
                  << " h2_post_prune_threw_map_at="
                  << (h2_stats.post_prune_threw_map_at ? 1 : 0)
                  << " h2_post_prune_skipped_deferred="
                  << (h2_stats.post_prune_skipped_deferred ? 1 : 0)
                  << " h2_first_unmapped_heart_slot="
                  << (h2_stats.first_unmapped_heart_slot ==
                              std::numeric_limits<gtsam::FactorIndex>::max()
                          ? -1
                          : static_cast<long long>(
                                h2_stats.first_unmapped_heart_slot))
                  << " h2_first_unmapped_sidecar_slot="
                  << (h2_stats.first_unmapped_sidecar_slot ==
                              std::numeric_limits<gtsam::FactorIndex>::max()
                          ? -1
                          : static_cast<long long>(
                                h2_stats.first_unmapped_sidecar_slot))
                  << " h2_packet_local_values_exact_required="
                  << (h2_stats.packet_local_values_exact_required ? 1 : 0)
                  << " h2_packet_missing_required_local_values_count="
                  << h2_stats.packet_missing_required_local_values_count
                  << " h2_packet_extra_local_values_count="
                  << h2_stats.packet_extra_local_values_count
                  << " h2_required_local_key_count="
                  << h2_stats.required_local_key_count
                  << " h2_required_packet_key_count="
                  << h2_stats.required_packet_key_count
                  << " h2_sample_remove_indices="
                  << (h2_stats.sample_remove_indices.empty()
                          ? "none"
                          : h2_stats.sample_remove_indices)
                  << " h2_sample_heart_new_factor_positions="
                  << (h2_stats.sample_heart_new_factor_positions.empty()
                          ? "none"
                          : h2_stats.sample_heart_new_factor_positions)
                  << " h2_sample_heart_new_factor_indices="
                  << (h2_stats.sample_heart_new_factor_indices.empty()
                          ? "none"
                          : h2_stats.sample_heart_new_factor_indices)
                  << " h2_sample_sidecar_new_factor_indices="
                  << (h2_stats.sample_sidecar_new_factor_indices.empty()
                          ? "none"
                          : h2_stats.sample_sidecar_new_factor_indices)
                  << " h2_sidecar_first_failure_key="
                  << (h2_stats.first_failure_key.empty()
                          ? "none"
                          : h2_stats.first_failure_key)
                  << " h2_sidecar_first_failure_reason="
                  << (h2_stats.first_failure_reason.empty()
                          ? "none"
                          : h2_stats.first_failure_reason)
                  << " h2_sidecar_desync_streak="
                  << cbs_h2_sidecar_desync_streak_
                  << " h2_sidecar_failure_reason="
                  << (h2_stats.failure_reason.empty() ? "none"
                                                     : h2_stats.failure_reason)
                  << std::endl;
      } else {
        cbs_h2_sidecar_sync_ok_ = false;
        cbs_h2_sidecar_update_ms_last_epoch_ = 0.0;
        cbs_h2_local_snapshot_refresh_ms_last_epoch_ = 0.0;
        cbs_h2_sidecar_local_factor_count_last_epoch_ = 0u;
        cbs_h2_sidecar_filtered_external_count_last_epoch_ = 0u;
        cbs_h2_sidecar_remove_count_last_epoch_ = 0u;
        cbs_h2_sidecar_values_add_count_last_epoch_ = 0u;
        cbs_h2_sidecar_smart_factor_replacements_last_epoch_ = 0u;
        cbs_h2_sidecar_value_type_mismatch_count_last_epoch_ = 0u;
        cbs_h2_sidecar_missing_value_count_last_epoch_ = 0u;
        cbs_h2_sidecar_unsupported_key_type_count_last_epoch_ = 0u;
        cbs_h2_sidecar_unmapped_remove_slot_count_last_epoch_ = 0u;
        cbs_h2_sidecar_post_prune_remove_count_last_epoch_ = 0u;
        cbs_h2_sidecar_packet_pose_values_count_last_epoch_ = 0u;
        cbs_h2_sidecar_packet_vel_values_count_last_epoch_ = 0u;
        cbs_h2_sidecar_packet_bias_values_count_last_epoch_ = 0u;
        cbs_h2_sidecar_first_failure_key_last_epoch_.clear();
        cbs_h2_sidecar_first_failure_reason_last_epoch_.clear();
        cbs_h2_sidecar_last_failure_reason_.clear();
        h2_local_graph_snapshot_.resize(0);
        h2_local_values_snapshot_.clear();
        h2_local_snapshot_timestamp_ns_ = -1;
        h2_local_snapshot_frame_id_ = 0;
        h2_local_snapshot_valid_ = false;
      }
#endif
      // ---- zy
      // zy Step 29: track active optimization size per cycle to verify CBS full-replacement behavior and detect growth regressions.
      size_t num_pose_keys = 0;
      size_t num_vel_keys = 0;
      size_t num_bias_keys = 0;
      for (const auto& key_value : state_) {
        const gtsam::Symbol sym(key_value.key);
        if (sym.chr() == kPoseSymbolChar) {
          ++num_pose_keys;
        } else if (sym.chr() == kVelocitySymbolChar) {
          ++num_vel_keys;
        } else if (sym.chr() == kImuBiasSymbolChar) {
          ++num_bias_keys;
        }
      }

      // Intuition: emit one consistent optimizer status line regardless of CBS/legacy mode.
      size_t num_factors_active = 0;
      const char* optimizer_mode = "LEGACY";
      size_t active_root_count = 0;
      size_t active_tree_cliques = 0;
      size_t active_root_frontals_total = 0;
      size_t active_root_frontals_max = 0;
      size_t active_root_parents_total = 0;
      size_t active_root_rows_total = 0;
      size_t active_root_rows_max = 0;
      size_t active_root_cols_total = 0;
      size_t active_root_cols_max = 0;
#ifdef KIMERA_USE_CBS
      if (useCbsOptimizerHeart()) {
        CHECK(cbs_optimizer_) << "CBS optimizer flag is ON but cbs_optimizer_ is null.";
        optimizer_mode = "CBS";
        num_factors_active = cbs_optimizer_->getFactorsUnsafe().size();
        active_tree_cliques = cbs_optimizer_->size();
        const auto& roots = cbs_optimizer_->roots();
        active_root_count = roots.size();
        for (const auto& root : roots) {
          if (!root || !root->conditional()) {
            continue;
          }
          const auto& conditional = root->conditional();
          const size_t nr_frontals =
              static_cast<size_t>(conditional->nrFrontals());
          const size_t nr_parents =
              static_cast<size_t>(conditional->nrParents());
          const size_t nr_rows = static_cast<size_t>(conditional->rows());
          const size_t nr_cols = static_cast<size_t>(conditional->cols());
          active_root_frontals_total += nr_frontals;
          active_root_frontals_max =
              std::max(active_root_frontals_max, nr_frontals);
          active_root_parents_total += nr_parents;
          active_root_rows_total += nr_rows;
          active_root_rows_max = std::max(active_root_rows_max, nr_rows);
          active_root_cols_total += nr_cols;
          active_root_cols_max = std::max(active_root_cols_max, nr_cols);
        }
      } else
#endif
      {
        CHECK(smoother_);
        num_factors_active = smoother_->getFactors().size();
        const gtsam::ISAM2& isam = smoother_->getISAM2();
        active_tree_cliques = isam.size();
        const auto& roots = isam.roots();
        active_root_count = roots.size();
        for (const auto& root : roots) {
          if (!root || !root->conditional()) {
            continue;
          }
          const auto& conditional = root->conditional();
          const size_t nr_frontals =
              static_cast<size_t>(conditional->nrFrontals());
          const size_t nr_parents =
              static_cast<size_t>(conditional->nrParents());
          const size_t nr_rows = static_cast<size_t>(conditional->rows());
          const size_t nr_cols = static_cast<size_t>(conditional->cols());
          active_root_frontals_total += nr_frontals;
          active_root_frontals_max =
              std::max(active_root_frontals_max, nr_frontals);
          active_root_parents_total += nr_parents;
          active_root_rows_total += nr_rows;
          active_root_rows_max = std::max(active_root_rows_max, nr_rows);
          active_root_cols_total += nr_cols;
          active_root_cols_max = std::max(active_root_cols_max, nr_cols);
        }
      }

      VLOG(1) << "Optimize status [" << optimizer_mode
              << "]: factors=" << num_factors_active
              << ", x=" << num_pose_keys
              << ", v=" << num_vel_keys
              << ", b=" << num_bias_keys
              << ", root_count=" << active_root_count
              << ", root_frontals_max=" << active_root_frontals_max
              << ", root_rows_max=" << active_root_rows_max
              << ", cur_kf=" << cur_id;

      const double backend_total_ms =
          utils::Timer::toc<std::chrono::milliseconds>(total_start_time)
              .count();
      std::cerr << std::setprecision(12)
                << "[CBS][OptimizeDiag] timestamp_ns=" << timestamp_kf_nsec
                << " curr_kf_id=" << cur_id
                << " optimizer_mode=" << optimizer_mode
                << " backend_total_ms=" << backend_total_ms
                << " active_x=" << num_pose_keys
                << " active_v=" << num_vel_keys
                << " active_b=" << num_bias_keys
                << " active_factors=" << num_factors_active
                << " active_tree_cliques=" << active_tree_cliques
                << " active_root_count=" << active_root_count
                << " active_root_frontals_total="
                << active_root_frontals_total
                << " active_root_frontals_max=" << active_root_frontals_max
                << " active_root_parents_total="
                << active_root_parents_total
                << " active_root_rows_total=" << active_root_rows_total
                << " active_root_rows_max=" << active_root_rows_max
                << " active_root_cols_total=" << active_root_cols_total
                << " active_root_cols_max=" << active_root_cols_max
                << " use_cbs_optimizer=" << (FLAGS_use_cbs_optimizer ? 1 : 0)
                << " cbs_replace_fixed_lag_optimizer="
                << (FLAGS_cbs_replace_fixed_lag_optimizer ? 1 : 0)
                << " cbs_heart_active=" << (cbs_heart_active ? 1 : 0)
                << " fixed_lag_states=" << backend_params_.nr_states_
                << " ext_injected=" << num_external_priors_injected
                << " ext_deferred=" << num_external_priors_deferred
                << " ext_dropped_old=" << num_external_priors_dropped_old
                << " ext_dropped_marginalized="
                << num_external_priors_dropped_marginalized
                << " ext_dropped_inactive="
                << num_external_priors_dropped_inactive
                << " ext_deferred_budget="
                << num_external_priors_deferred_budget
                << " ext_fast_skipped=" << num_external_priors_fast_skipped
                << " ext_source_backoff_skipped="
                << num_external_priors_source_backoff_skipped
                << " ext_no_external_effect_epoch="
                << (cbs_no_external_effect_epoch ? 1 : 0)
                << " cbs_allow_heavy_maintenance="
                << (cbs_allow_heavy_maintenance ? 1 : 0)
                << " ext_queue_size=" << external_queue_size_now
#ifdef KIMERA_USE_CBS
                << " h2_mode=" << (useCbsH2LocalCovSidecar() ? 1 : 0)
                << " h2_sidecar_mode=" << cbsH2SidecarModeName()
                << " h2_sidecar_sync_ok=" << (cbs_h2_sidecar_sync_ok_ ? 1 : 0)
                << " h2_sidecar_update_ms="
                << cbs_h2_sidecar_update_ms_last_epoch_
                << " h2_local_snapshot_refresh_ms="
                << cbs_h2_local_snapshot_refresh_ms_last_epoch_
                << " h2_local_factor_count="
                << cbs_h2_sidecar_local_factor_count_last_epoch_
                << " h2_filtered_external_factor_count="
                << cbs_h2_sidecar_filtered_external_count_last_epoch_
                << " h2_sidecar_remove_count="
                << cbs_h2_sidecar_remove_count_last_epoch_
                << " h2_sidecar_values_add_count="
                << cbs_h2_sidecar_values_add_count_last_epoch_
                << " h2_sidecar_hard_reset_count="
                << cbs_h2_sidecar_hard_reset_count_
                << " h2_sidecar_smart_factor_replacements="
                << cbs_h2_sidecar_smart_factor_replacements_last_epoch_
                << " h2_sidecar_value_type_mismatch_count="
                << cbs_h2_sidecar_value_type_mismatch_count_last_epoch_
                << " h2_sidecar_missing_value_count="
                << cbs_h2_sidecar_missing_value_count_last_epoch_
                << " h2_sidecar_unsupported_key_type_count="
                << cbs_h2_sidecar_unsupported_key_type_count_last_epoch_
                << " h2_sidecar_unmapped_remove_slot_count="
                << cbs_h2_sidecar_unmapped_remove_slot_count_last_epoch_
                << " h2_sidecar_post_prune_remove_count="
                << cbs_h2_sidecar_post_prune_remove_count_last_epoch_
                << " h2_sidecar_packet_pose_values_count="
                << cbs_h2_sidecar_packet_pose_values_count_last_epoch_
                << " h2_sidecar_packet_vel_values_count="
                << cbs_h2_sidecar_packet_vel_values_count_last_epoch_
                << " h2_sidecar_packet_bias_values_count="
                << cbs_h2_sidecar_packet_bias_values_count_last_epoch_
                << " h2_sidecar_first_failure_key="
                << (cbs_h2_sidecar_first_failure_key_last_epoch_.empty()
                        ? "none"
                        : cbs_h2_sidecar_first_failure_key_last_epoch_)
                << " h2_sidecar_first_failure_reason="
                << (cbs_h2_sidecar_first_failure_reason_last_epoch_.empty()
                        ? "none"
                        : cbs_h2_sidecar_first_failure_reason_last_epoch_)
                << " h2_sidecar_desync_streak=" << cbs_h2_sidecar_desync_streak_
                << " h2_sidecar_fallback_cov_epochs="
                << cbs_h2_sidecar_fallback_cov_epochs_
#endif
                << std::endl;
// ----- zy

      // TODO: Add Update latest covariance --> move flag
      if (FLAGS_compute_state_covariance) {
        computeStateCovariance();
      }

      // Debug.
      postDebug(total_start_time, start_time);
    } else {
      LOG(ERROR) << "Smoother is not ok! Not updating Backend state.";
    }
  }
  return is_smoother_ok;
}
/// Private methods.
/* -------------------------------------------------------------------------- */
void VioBackend::addInitialPriorFactors(const FrameId& frame_id) {
  // Set initial covariance for inertial factors
  // W_Pose_Blkf_ set by motion capture to start with
  Matrix3 B_Rot_W = W_Pose_B_lkf_from_state_.rotation().matrix().transpose();

  // Set initial pose uncertainty: constrain mainly position and global yaw.
  // roll and pitch is observable, therefore low variance.
  Matrix6 pose_prior_covariance = Matrix6::Zero();
  pose_prior_covariance.diagonal()[0] = backend_params_.initialRollPitchSigma_ *
                                        backend_params_.initialRollPitchSigma_;
  pose_prior_covariance.diagonal()[1] = backend_params_.initialRollPitchSigma_ *
                                        backend_params_.initialRollPitchSigma_;
  pose_prior_covariance.diagonal()[2] =
      backend_params_.initialYawSigma_ * backend_params_.initialYawSigma_;
  pose_prior_covariance.diagonal()[3] = backend_params_.initialPositionSigma_ *
                                        backend_params_.initialPositionSigma_;
  pose_prior_covariance.diagonal()[4] = backend_params_.initialPositionSigma_ *
                                        backend_params_.initialPositionSigma_;
  pose_prior_covariance.diagonal()[5] = backend_params_.initialPositionSigma_ *
                                        backend_params_.initialPositionSigma_;

  // Rotate initial uncertainty into local frame, where the uncertainty is
  // specified.
  pose_prior_covariance.topLeftCorner(3, 3) =
      B_Rot_W * pose_prior_covariance.topLeftCorner(3, 3) * B_Rot_W.transpose();

  // Add pose prior.
  // TODO(Toni): Make this noise model a member constant.
  gtsam::SharedNoiseModel noise_init_pose =
      gtsam::noiseModel::Gaussian::Covariance(pose_prior_covariance);
  new_imu_prior_and_other_factors_
      .emplace_shared<gtsam::PriorFactor<gtsam::Pose3>>(
          gtsam::Symbol(kPoseSymbolChar, frame_id),
          W_Pose_B_lkf_from_state_,
          noise_init_pose);

  // Add initial velocity priors.
  // TODO(Toni): Make this noise model a member constant.
  gtsam::SharedNoiseModel noise_init_vel_prior =
      gtsam::noiseModel::Isotropic::Sigma(
          3, backend_params_.initialVelocitySigma_);
  new_imu_prior_and_other_factors_
      .emplace_shared<gtsam::PriorFactor<gtsam::Vector3>>(
          gtsam::Symbol(kVelocitySymbolChar, frame_id),
          W_Vel_B_lkf_,
          noise_init_vel_prior);

  // Add initial bias priors:
  Vector6 prior_biasSigmas;
  prior_biasSigmas.head<3>().setConstant(backend_params_.initialAccBiasSigma_);
  prior_biasSigmas.tail<3>().setConstant(backend_params_.initialGyroBiasSigma_);
  // TODO(Toni): Make this noise model a member constant.
  gtsam::SharedNoiseModel imu_bias_prior_noise =
      gtsam::noiseModel::Diagonal::Sigmas(prior_biasSigmas);
  if (VLOG_IS_ON(10)) {
    LOG(INFO) << "Imu bias for Backend prior:";
    imu_bias_lkf_.print();
  }
  new_imu_prior_and_other_factors_
      .emplace_shared<gtsam::PriorFactor<gtsam::imuBias::ConstantBias>>(
          gtsam::Symbol(kImuBiasSymbolChar, frame_id),
          imu_bias_lkf_,
          imu_bias_prior_noise);

  VLOG(2) << "Added initial priors for frame " << frame_id;
}

/* -------------------------------------------------------------------------- */
void VioBackend::addConstantVelocityFactor(const FrameId& from_id,
                                           const FrameId& to_id) {
  VLOG(10) << "Adding constant velocity factor.";
  new_imu_prior_and_other_factors_
      .emplace_shared<gtsam::BetweenFactor<gtsam::Vector3>>(
          gtsam::Symbol(kVelocitySymbolChar, from_id),
          gtsam::Symbol(kVelocitySymbolChar, to_id),
          gtsam::Vector3::Zero(),
          constant_velocity_prior_noise_);

  // Log number of added constant velocity factors.
  debug_info_.numAddedConstantVelF_++;
}

/* -------------------------------- UPDATE ---------------------------------- */
void VioBackend::updateStates(const FrameId& cur_id) {
  // zy Step 11c, edited the original
  // ---
  VLOG(10) << "Starting to calculate estimate.";
  try {
#ifdef KIMERA_USE_CBS
    if (useCbsOptimizerHeart()) {
      CHECK(cbs_optimizer_)
          << "CBS optimizer flag is ON but cbs_optimizer_ is null.";
      state_ = cbs_optimizer_->calculateEstimate();
    } else {
      state_ = smoother_->calculateEstimate();
    }
#else
    state_ = smoother_->calculateEstimate();
#endif
  } catch (const gtsam::ValuesKeyDoesNotExist& e) {
    LOG(ERROR) << "updateStates(): calculateEstimate missing key: " << e.what()
               << ". Skipping this backend cycle.";
    return;
  } catch (const std::exception& e) {
    LOG(ERROR) << "updateStates(): calculateEstimate failed: " << e.what()
               << ". Skipping this backend cycle.";
    return;
  } catch (...) {
    LOG(ERROR) << "updateStates(): calculateEstimate failed with unknown "
                  "exception. Skipping this backend cycle.";
    return;
  }
  VLOG(10) << "Finished to calculate estimate.";


  const gtsam::Symbol pose_key(kPoseSymbolChar, cur_id);
  const gtsam::Symbol vel_key(kVelocitySymbolChar, cur_id);
  const gtsam::Symbol bias_key(kImuBiasSymbolChar, cur_id);
  if (!state_.exists(pose_key) || !state_.exists(vel_key) ||
      !state_.exists(bias_key)) {
    LOG(ERROR) << "updateStates(): missing current state key(s) at frame "
               << cur_id << " [pose=" << state_.exists(pose_key)
               << ", vel=" << state_.exists(vel_key)
               << ", bias=" << state_.exists(bias_key)
               << "]. Skipping this backend cycle.";
    return;
  }

  gtsam::Pose3 W_Pose_B_kf;
  try {
    W_Pose_B_kf = state_.at<gtsam::Pose3>(pose_key);
  } catch (const gtsam::ValuesKeyDoesNotExist& e) {
    LOG(ERROR) << "updateStates(): pose access failed: " << e.what()
               << ". Skipping this backend cycle.";
    return;
  }
  gtsam::Pose3 W_Pose_B_lkf = gtsam::Pose3();
  gtsam::Pose3 B_lkf_Pose_kf = gtsam::Pose3();

  // If we have an available pose at cur_id - 1 we use it, otw identity
  // gives us W_Pose_B_lkf as our current pose estimate.
  // if (cur_id > 0) {
  //   DCHECK(state_.find(gtsam::Symbol(kPoseSymbolChar, cur_id - 1)) !=
  //          state_.end());
  //   W_Pose_B_lkf =
  //       state_.at<gtsam::Pose3>(gtsam::Symbol(kPoseSymbolChar, cur_id - 1));

  //   // Compute relative pose as odometry to append to pose estimate trajectory
  //   B_lkf_Pose_kf = W_Pose_B_lkf.between(W_Pose_B_kf);
  // } (zy cancelled it)

  // zy Step 32a: Make updateStates() robust if x(k-1) is temporarily missing (can happen in edge CBS graph maintenance cases).
  // This prevents crashes and keeps the backend running.
  if (cur_id > 0) {
    const gtsam::Symbol prev_pose_key(kPoseSymbolChar, cur_id - 1);
    if (state_.exists(prev_pose_key)) {
      try {
        W_Pose_B_lkf = state_.at<gtsam::Pose3>(prev_pose_key);
        // Compute relative pose as odometry to append to pose estimate trajectory.
        B_lkf_Pose_kf = W_Pose_B_lkf.between(W_Pose_B_kf);
      } catch (const gtsam::ValuesKeyDoesNotExist& e) {
        LOG(ERROR) << "updateStates(): prev pose access failed: " << e.what()
                   << ". Using identity increment for frame " << cur_id;
      }
    } else {
      // Intuition: if x(k-1) is temporarily unavailable, keep backend alive with identity increment instead of hard-failing.
      VLOG(2) << "Previous pose key missing in state: " << prev_pose_key
              << ". Using identity increment for frame " << cur_id;
    }
  }


  // Update latest state estimate
  W_Pose_B_lkf_from_state_ = W_Pose_B_kf;
  try {
    W_Vel_B_lkf_ = state_.at<Vector3>(vel_key);
    imu_bias_lkf_ = state_.at<gtsam::imuBias::ConstantBias>(bias_key);
  } catch (const gtsam::ValuesKeyDoesNotExist& e) {
    LOG(ERROR) << "updateStates(): velocity/bias access failed: " << e.what()
               << ". Skipping this backend cycle.";
    return;
  }

  // Update output estimate by chaining relative motion estimates
  W_Pose_B_lkf_from_increments_ =
      W_Pose_B_lkf_from_increments_.compose(B_lkf_Pose_kf);

  VLOG(1) << "Backend: Update IMU Bias.";
  CHECK(imu_bias_update_callback_) << "Did you forget to register the IMU bias "
                                      "update callback for at least the "
                                      "Frontend? Do so by using "
                                      "registerImuBiasUpdateCallback function";
  imu_bias_update_callback_(imu_bias_lkf_);
}

bool VioBackend::updateSmoother(Smoother::Result* result,
                                const gtsam::NonlinearFactorGraph& new_factors,
                                const gtsam::Values& new_values,
                                const std::map<Key, double>& timestamps,
                                const gtsam::FactorIndices& delete_slots,
                                const bool cbs_allow_heavy_maintenance,
                                const size_t cbs_smart_factor_replacements,
                                const size_t cbs_smart_factor_new_insertions,
                                const size_t cbs_delete_slots_total,
                                const size_t cbs_delete_slots_from_smart_replacement,
                                const size_t cbs_delete_slots_from_cheirality_cleanup,
                                const size_t cbs_delete_slots_from_other_cleanup,
                                const size_t cbs_smart_replacement_same_support_count,
                                const size_t cbs_smart_replacement_same_pose_key_set_count,
                                const size_t cbs_smart_replacement_small_support_delta_count,
                                const size_t cbs_smart_replacement_large_support_delta_count,
                                const size_t cbs_smart_replacement_due_to_material_support_change,
                                const size_t cbs_smart_replacement_due_to_pose_key_set_change,
                                const size_t cbs_smart_replacement_due_to_invalid_to_valid_transition,
                                const size_t cbs_smart_replacement_due_to_valid_to_invalid_transition,
                                const size_t cbs_smart_replacement_due_to_degenerate_factor,
                                const size_t cbs_smart_replacement_due_to_correctness_threshold,
                                const size_t cbs_smart_replacement_skipped_small_change,
                                const size_t cbs_smart_replacement_skipped_keep_existing,
                                const size_t cbs_landmarks_touched_for_replacement,
                                const size_t cbs_landmarks_touched_for_new_insertion,
                                const size_t cbs_other_new_factor_count) {
  CHECK_NOTNULL(result);
  // zy Step 16a
#ifdef KIMERA_USE_CBS
  if (useCbsOptimizerHeart()) {
    // Intuition: in CBS mode, BPSAM is the single optimization heart, so we skip the legacy smoother update path.
    CHECK(cbs_optimizer_) << "CBS optimizer flag is ON but cbs_optimizer_ is null.";

    const auto cbs_epoch_start = std::chrono::steady_clock::now();
    double cbs_lag_window_build_ms = 0.0;
    double cbs_anchor_prior_ms = 0.0;
    double cbs_remove_index_discovery_ms = 0.0;
    double cbs_first_update_ms = 0.0;
    double cbs_inner_round_ms = 0.0;
    size_t cbs_remove_indices_requested = 0u;
    size_t cbs_remove_indices_applied = 0u;
    size_t cbs_remove_indices_dropped_stale = 0u;
    size_t cbs_remove_delete_slots_requested = 0u;
    size_t cbs_remove_lag_slots_requested = 0u;
    size_t cbs_remove_from_delete_only = 0u;
    size_t cbs_remove_from_lag_only = 0u;
    size_t cbs_remove_from_both = 0u;
    size_t cbs_remove_smart_factor_slots = 0u;
    size_t cbs_remove_non_smart_slots = 0u;
    size_t cbs_remove_prev_epoch_size = 0u;
    size_t cbs_remove_curr_epoch_size = 0u;
    size_t cbs_remove_repeated_from_prev_epoch = 0u;
    bool cbs_no_external_fast_path_applied = false;
    size_t cbs_anchor_priors_added = 0u;
    // B1 runtime profiling (lightweight, per-epoch).
    double cbs_b1_estimate_extract_ms = 0.0;
    double cbs_b1_boundary_extract_ms = 0.0;
    double cbs_b1_pose_cov_query_ms = 0.0;
    double cbs_b1_vel_cov_query_ms = 0.0;
    double cbs_b1_bias_cov_query_ms = 0.0;
    double cbs_b1_prior_factor_emit_ms = 0.0;
    size_t cbs_b1_pose_cov_queries = 0u;
    size_t cbs_b1_vel_cov_queries = 0u;
    size_t cbs_b1_bias_cov_queries = 0u;
    bool cbs_b1_cov_refresh_this_epoch = false;

    CbsFixedLagWindowState lag_window_state;
    if (cbs_allow_heavy_maintenance) {
      const auto lag_build_start = std::chrono::steady_clock::now();
      lag_window_state = buildCbsFixedLagWindowState(timestamps);
      cbs_lag_window_build_ms +=
          elapsedMs(lag_build_start, std::chrono::steady_clock::now());
    } else {
      cbs_no_external_fast_path_applied = true;
      lag_window_state.newest_frame_id = curr_kf_id_;
      lag_window_state.oldest_active_frame_id =
          computeCbsOldestActiveFrame(curr_kf_id_);
      lag_window_state.prev_oldest_active_frame_id =
          cbs_has_prev_oldest_active_frame_id_
              ? cbs_prev_oldest_active_frame_id_
              : lag_window_state.oldest_active_frame_id;
      lag_window_state.eviction_start_frame_id = lag_window_state.oldest_active_frame_id;
      lag_window_state.eviction_end_frame_id = lag_window_state.oldest_active_frame_id;
      lag_window_state.eviction_incremental = true;
      lag_window_state.eviction_full_rescan = false;
      lag_window_state.eviction_frames = 0u;
      cbs_prev_oldest_active_frame_id_ = lag_window_state.oldest_active_frame_id;
      cbs_has_prev_oldest_active_frame_id_ = true;
    }

    // CBS-heart currently approximates fixed-lag factor removal but does not
    // yet implement a full marginalization prior. Add a boundary anchor once
    // per oldest-active frame to reduce gauge drift/underdetermined windows.
    gtsam::NonlinearFactorGraph cbs_update_factors = new_factors;
    const bool cbs_use_marginalization_prior_bridge =
        FLAGS_cbs_use_marginalization_prior_bridge;
    const FrameId anchor_frame_id = lag_window_state.oldest_active_frame_id;
    if (cbs_allow_heavy_maintenance &&
        !FLAGS_cbs_diag_disable_lag_boundary_anchor_priors &&
        anchor_frame_id > 0u &&
        cbs_last_window_anchor_frame_id_ != anchor_frame_id) {
      const auto anchor_start = std::chrono::steady_clock::now();
      try {
        const auto estimate_extract_start = std::chrono::steady_clock::now();
        const gtsam::Values cbs_values = cbs_optimizer_->calculateEstimate();
        cbs_b1_estimate_extract_ms +=
            elapsedMs(estimate_extract_start, std::chrono::steady_clock::now());
        size_t anchors_added = 0u;
        const bool b1_boundary_changed_this_epoch =
            lag_window_state.prev_oldest_active_frame_id !=
            lag_window_state.oldest_active_frame_id;
        const bool b1_refresh_cov_this_anchor =
            cbs_use_marginalization_prior_bridge &&
            (!FLAGS_cbs_b1_refresh_cov_only_on_boundary_change ||
             b1_boundary_changed_this_epoch);
        if (b1_refresh_cov_this_anchor) {
          cbs_b1_cov_refresh_this_epoch = true;
        }

        const gtsam::Symbol pose_key(kPoseSymbolChar, anchor_frame_id);
        if (cbs_values.exists(pose_key.key())) {
          const auto pose_extract_start = std::chrono::steady_clock::now();
          const gtsam::Pose3 anchor_pose = cbs_values.at<gtsam::Pose3>(pose_key);
          cbs_b1_boundary_extract_ms +=
              elapsedMs(pose_extract_start, std::chrono::steady_clock::now());
          gtsam::SharedNoiseModel pose_noise;
          bool pose_cov_ok = false;
          if (b1_refresh_cov_this_anchor) {
            const auto pose_cov_query_start = std::chrono::steady_clock::now();
            ++cbs_b1_pose_cov_queries;
            try {
              const gtsam::Matrix pose_cov_raw = cbs_optimizer_->marginalCovariance(
                  pose_key, cbs::BPSAM::MarginalizationType::LOCAL);
              cbs_b1_pose_cov_query_ms +=
                  elapsedMs(pose_cov_query_start, std::chrono::steady_clock::now());
              if (pose_cov_raw.rows() >= 6 && pose_cov_raw.cols() >= 6 &&
                  pose_cov_raw.allFinite()) {
                gtsam::Matrix6 pose_cov =
                    pose_cov_raw.block<6, 6>(0, 0);
                pose_cov = 0.5 * (pose_cov + pose_cov.transpose());
                constexpr double kMinVar = 1e-8;
                for (int i = 0; i < 6; ++i) {
                  if (!std::isfinite(pose_cov(i, i)) || pose_cov(i, i) < kMinVar) {
                    pose_cov(i, i) = kMinVar;
                  }
                }
                pose_noise = gtsam::noiseModel::Gaussian::Covariance(pose_cov);
                pose_cov_ok = true;
              }
            } catch (const std::exception& e) {
              cbs_b1_pose_cov_query_ms +=
                  elapsedMs(pose_cov_query_start, std::chrono::steady_clock::now());
              VLOG(1) << "CBS B1 pose covariance query failed, using fixed anchor "
                         "sigmas: "
                      << e.what();
            }
          }
          if (!pose_cov_ok) {
            gtsam::Vector6 sigmas;
            sigmas.head<3>().setConstant(0.05);   // rad
            sigmas.tail<3>().setConstant(0.25);   // m
            pose_noise = gtsam::noiseModel::Diagonal::Sigmas(sigmas);
          }
          const auto pose_emit_start = std::chrono::steady_clock::now();
          cbs_update_factors.emplace_shared<gtsam::PriorFactor<gtsam::Pose3>>(
              pose_key, anchor_pose, pose_noise);
          cbs_b1_prior_factor_emit_ms +=
              elapsedMs(pose_emit_start, std::chrono::steady_clock::now());
          ++anchors_added;
        }

        const gtsam::Symbol vel_key(kVelocitySymbolChar, anchor_frame_id);
        if (cbs_values.exists(vel_key.key())) {
          const auto vel_extract_start = std::chrono::steady_clock::now();
          const gtsam::Vector3 anchor_vel = cbs_values.at<gtsam::Vector3>(vel_key);
          cbs_b1_boundary_extract_ms +=
              elapsedMs(vel_extract_start, std::chrono::steady_clock::now());
          gtsam::SharedNoiseModel vel_noise;
          bool vel_cov_ok = false;
          if (b1_refresh_cov_this_anchor &&
              !FLAGS_cbs_b1_pose_only_cov_refresh) {
            const auto vel_cov_query_start = std::chrono::steady_clock::now();
            ++cbs_b1_vel_cov_queries;
            try {
              const gtsam::Matrix vel_cov_raw = cbs_optimizer_->marginalCovariance(
                  vel_key, cbs::BPSAM::MarginalizationType::LOCAL);
              cbs_b1_vel_cov_query_ms +=
                  elapsedMs(vel_cov_query_start, std::chrono::steady_clock::now());
              if (vel_cov_raw.rows() >= 3 && vel_cov_raw.cols() >= 3 &&
                  vel_cov_raw.allFinite()) {
                gtsam::Matrix3 vel_cov =
                    vel_cov_raw.block<3, 3>(0, 0);
                vel_cov = 0.5 * (vel_cov + vel_cov.transpose());
                constexpr double kMinVar = 1e-8;
                for (int i = 0; i < 3; ++i) {
                  if (!std::isfinite(vel_cov(i, i)) || vel_cov(i, i) < kMinVar) {
                    vel_cov(i, i) = kMinVar;
                  }
                }
                vel_noise = gtsam::noiseModel::Gaussian::Covariance(vel_cov);
                vel_cov_ok = true;
              }
            } catch (const std::exception& e) {
              cbs_b1_vel_cov_query_ms +=
                  elapsedMs(vel_cov_query_start, std::chrono::steady_clock::now());
              VLOG(1) << "CBS B1 velocity covariance query failed, using fixed "
                         "anchor sigmas: "
                      << e.what();
            }
          }
          if (!vel_cov_ok) {
            gtsam::Vector3 vel_sigmas;
            vel_sigmas.setConstant(0.5);  // m/s
            vel_noise = gtsam::noiseModel::Diagonal::Sigmas(vel_sigmas);
          }
          const auto vel_emit_start = std::chrono::steady_clock::now();
          cbs_update_factors.emplace_shared<gtsam::PriorFactor<gtsam::Vector3>>(
              vel_key, anchor_vel, vel_noise);
          cbs_b1_prior_factor_emit_ms +=
              elapsedMs(vel_emit_start, std::chrono::steady_clock::now());
          ++anchors_added;
        }

        const gtsam::Symbol bias_key(kImuBiasSymbolChar, anchor_frame_id);
        if (cbs_values.exists(bias_key.key())) {
          const auto bias_extract_start = std::chrono::steady_clock::now();
          const gtsam::imuBias::ConstantBias anchor_bias =
              cbs_values.at<gtsam::imuBias::ConstantBias>(bias_key);
          cbs_b1_boundary_extract_ms +=
              elapsedMs(bias_extract_start, std::chrono::steady_clock::now());
          gtsam::SharedNoiseModel bias_noise;
          bool bias_cov_ok = false;
          if (b1_refresh_cov_this_anchor &&
              !FLAGS_cbs_b1_pose_only_cov_refresh) {
            const auto bias_cov_query_start = std::chrono::steady_clock::now();
            ++cbs_b1_bias_cov_queries;
            try {
              const gtsam::Matrix bias_cov_raw = cbs_optimizer_->marginalCovariance(
                  bias_key, cbs::BPSAM::MarginalizationType::LOCAL);
              cbs_b1_bias_cov_query_ms +=
                  elapsedMs(bias_cov_query_start, std::chrono::steady_clock::now());
              if (bias_cov_raw.rows() >= 6 && bias_cov_raw.cols() >= 6 &&
                  bias_cov_raw.allFinite()) {
                gtsam::Matrix6 bias_cov =
                    bias_cov_raw.block<6, 6>(0, 0);
                bias_cov = 0.5 * (bias_cov + bias_cov.transpose());
                constexpr double kMinVar = 1e-10;
                for (int i = 0; i < 6; ++i) {
                  if (!std::isfinite(bias_cov(i, i)) || bias_cov(i, i) < kMinVar) {
                    bias_cov(i, i) = kMinVar;
                  }
                }
                bias_noise = gtsam::noiseModel::Gaussian::Covariance(bias_cov);
                bias_cov_ok = true;
              }
            } catch (const std::exception& e) {
              cbs_b1_bias_cov_query_ms +=
                  elapsedMs(bias_cov_query_start, std::chrono::steady_clock::now());
              VLOG(1) << "CBS B1 bias covariance query failed, using fixed anchor "
                         "sigmas: "
                      << e.what();
            }
          }
          if (!bias_cov_ok) {
            gtsam::Vector6 bias_sigmas;
            bias_sigmas.head<3>().setConstant(
                std::max(backend_params_.initialAccBiasSigma_, 1e-4));
            bias_sigmas.tail<3>().setConstant(
                std::max(backend_params_.initialGyroBiasSigma_, 1e-4));
            bias_noise = gtsam::noiseModel::Diagonal::Sigmas(bias_sigmas);
          }
          const auto bias_emit_start = std::chrono::steady_clock::now();
          cbs_update_factors.emplace_shared<
              gtsam::PriorFactor<gtsam::imuBias::ConstantBias>>(
              bias_key,
              anchor_bias,
              bias_noise);
          cbs_b1_prior_factor_emit_ms +=
              elapsedMs(bias_emit_start, std::chrono::steady_clock::now());
          ++anchors_added;
        }

        if (anchors_added > 0u) {
          cbs_last_window_anchor_frame_id_ = anchor_frame_id;
          VLOG(1) << "CBS-heart added lag-boundary anchor priors on frame "
                  << anchor_frame_id << " (count=" << anchors_added << ")";
        }
        cbs_anchor_priors_added = anchors_added;
      } catch (const std::exception& e) {
        LOG(WARNING) << "CBS-heart could not add lag-boundary anchor priors for "
                     << "frame " << anchor_frame_id << ": " << e.what();
      }
      cbs_anchor_prior_ms +=
          elapsedMs(anchor_start, std::chrono::steady_clock::now());
    }

  const auto build_cbs_first_update_params =
        [&](const bool enable_remove_factor_indices) {
          cbs::BPSAM::UpdateParams params;
          if (!enable_remove_factor_indices) {
            VLOG(1) << "CBS-heart running current epoch with "
                       "removeFactorIndices disabled.";
            cbs_remove_indices_requested = 0u;
            cbs_remove_indices_applied = 0u;
            cbs_remove_indices_dropped_stale = 0u;
            cbs_remove_delete_slots_requested = 0u;
            cbs_remove_lag_slots_requested = 0u;
            cbs_remove_from_delete_only = 0u;
            cbs_remove_from_lag_only = 0u;
            cbs_remove_from_both = 0u;
            cbs_remove_smart_factor_slots = 0u;
            cbs_remove_non_smart_slots = 0u;
            cbs_remove_prev_epoch_size = cbs_prev_epoch_remove_factor_indices_.size();
            cbs_remove_curr_epoch_size = 0u;
            cbs_remove_repeated_from_prev_epoch = 0u;
            cbs_prev_epoch_remove_factor_indices_.clear();
            return params;
          }

          const auto remove_discovery_start = std::chrono::steady_clock::now();
          std::unordered_set<size_t> delete_slot_set(delete_slots.begin(),
                                                     delete_slots.end());
          std::unordered_set<size_t> lag_slot_set(
              lag_window_state.remove_factor_indices.begin(),
              lag_window_state.remove_factor_indices.end());
          if (FLAGS_cbs_diag_disable_lag_eviction_remove_candidates) {
            lag_slot_set.clear();
          }
          cbs_remove_delete_slots_requested = delete_slot_set.size();
          cbs_remove_lag_slots_requested = lag_slot_set.size();

          std::unordered_set<size_t> raw_remove_set;
          const bool use_lag_only_remove =
              cbs_use_marginalization_prior_bridge ||
              FLAGS_cbs_diag_disable_merge_delete_slots_into_remove_factor_indices;
          if (use_lag_only_remove) {
            raw_remove_set = lag_slot_set;
          } else {
            raw_remove_set = delete_slot_set;
            raw_remove_set.insert(lag_slot_set.begin(), lag_slot_set.end());
          }
          cbs_remove_indices_requested = raw_remove_set.size();

          cbs_remove_from_delete_only = 0u;
          cbs_remove_from_lag_only = 0u;
          cbs_remove_from_both = 0u;
          for (const size_t slot : raw_remove_set) {
            const bool in_delete = delete_slot_set.count(slot) > 0u;
            const bool in_lag = lag_slot_set.count(slot) > 0u;
            if (in_delete && in_lag) {
              ++cbs_remove_from_both;
            } else if (in_delete) {
              ++cbs_remove_from_delete_only;
            } else if (in_lag) {
              ++cbs_remove_from_lag_only;
            }
          }

          std::vector<size_t> raw_remove_indices(raw_remove_set.begin(),
                                                 raw_remove_set.end());
          std::sort(raw_remove_indices.begin(), raw_remove_indices.end());

          const gtsam::NonlinearFactorGraph& cbs_factors =
              cbs_optimizer_->getFactorsUnsafe();
          size_t dropped_remove_slots = 0u;
          params.removeFactorIndices.reserve(raw_remove_indices.size());
          for (const size_t slot : raw_remove_indices) {
            if (cbs_factors.exists(slot)) {
              params.removeFactorIndices.push_back(slot);
            } else {
              ++dropped_remove_slots;
            }
          }
          cbs_remove_indices_dropped_stale = dropped_remove_slots;
          cbs_remove_prev_epoch_size = cbs_prev_epoch_remove_factor_indices_.size();
          cbs_remove_repeated_from_prev_epoch = 0u;
          if (FLAGS_cbs_diag_disable_repeated_remove_from_prev_epoch) {
            auto& remove_indices = params.removeFactorIndices;
            remove_indices.erase(
                std::remove_if(remove_indices.begin(),
                               remove_indices.end(),
                               [&](const size_t slot) {
                                 const bool repeated =
                                     cbs_prev_epoch_remove_factor_indices_.count(slot) >
                                     0u;
                                 if (repeated) {
                                   ++cbs_remove_repeated_from_prev_epoch;
                                 }
                                 return repeated;
                               }),
                remove_indices.end());
          } else {
            for (const size_t slot : params.removeFactorIndices) {
              if (cbs_prev_epoch_remove_factor_indices_.count(slot) > 0u) {
                ++cbs_remove_repeated_from_prev_epoch;
              }
            }
          }
          cbs_remove_curr_epoch_size = params.removeFactorIndices.size();
          cbs_remove_indices_applied = params.removeFactorIndices.size();

          cbs_remove_smart_factor_slots = 0u;
          cbs_remove_non_smart_slots = 0u;
          for (const size_t slot : params.removeFactorIndices) {
            if (!cbs_factors.exists(slot) || !cbs_factors.at(slot)) {
              continue;
            }
            const auto smart_ptr =
                dynamic_cast<const SmartStereoFactor*>(cbs_factors.at(slot).get());
            if (smart_ptr) {
              ++cbs_remove_smart_factor_slots;
            } else {
              ++cbs_remove_non_smart_slots;
            }
          }

          cbs_prev_epoch_remove_factor_indices_.clear();
          cbs_prev_epoch_remove_factor_indices_.insert(
              params.removeFactorIndices.begin(), params.removeFactorIndices.end());

          if (dropped_remove_slots > 0u) {
            LOG(WARNING) << "CBS-heart dropped " << dropped_remove_slots
                         << " stale removeFactorIndices before update.";
          }
          cbs_remove_index_discovery_ms += elapsedMs(
              remove_discovery_start, std::chrono::steady_clock::now());
          return params;
        };
    if (!lag_window_state.remove_factor_indices.empty()) {
      VLOG(2) << "CBS-heart fixed-lag eviction: newest_frame_id="
              << lag_window_state.newest_frame_id
              << ", prev_oldest_active_frame_id="
              << lag_window_state.prev_oldest_active_frame_id
              << ", oldest_active_frame_id="
              << lag_window_state.oldest_active_frame_id
              << ", eviction_start_frame_id="
              << lag_window_state.eviction_start_frame_id
              << ", eviction_end_frame_id="
              << lag_window_state.eviction_end_frame_id
              << ", eviction_incremental="
              << (lag_window_state.eviction_incremental ? 1 : 0)
              << ", eviction_full_rescan="
              << (lag_window_state.eviction_full_rescan ? 1 : 0)
              << ", eviction_frames=" << lag_window_state.eviction_frames
              << ", stale_state_keys=" << lag_window_state.stale_state_keys
              << ", stale_pose_keys=" << lag_window_state.stale_pose_keys
              << ", removed_factor_slots="
              << lag_window_state.remove_factor_indices.size();
    }

    auto attempt_cbs_recovery =
        [&](const std::optional<size_t>& failed_index_opt,
            const std::string& trigger_reason) -> bool {
      LOG(ERROR) << "CBS recovery triggered. reason=" << trigger_reason;
      gtsam::Values cbs_values;
      try {
        cbs_values = cbs_optimizer_->calculateEstimate();
      } catch (const std::exception& est_e) {
        LOG(ERROR) << "CBS recovery failed to query current estimate: "
                   << est_e.what();
      }

      auto add_stabilizing_prior =
          [&](const gtsam::Symbol& key,
              gtsam::NonlinearFactorGraph* graph_to_update) -> bool {
        CHECK_NOTNULL(graph_to_update);
        switch (key.chr()) {
          case 'x': {
            gtsam::Pose3 pose;
            if (cbs_values.exists(key)) {
              pose = cbs_values.at<gtsam::Pose3>(key);
            } else {
              return false;
            }
            gtsam::Vector6 sigmas;
            sigmas.head<3>().setConstant(0.01);  // rotation
            sigmas.tail<3>().setConstant(0.1);   // translation
            const gtsam::SharedNoiseModel noise =
                gtsam::noiseModel::Diagonal::Sigmas(sigmas);
            graph_to_update->emplace_shared<gtsam::PriorFactor<gtsam::Pose3>>(
                key, pose, noise);
            return true;
          }
          case 'b': {
            gtsam::imuBias::ConstantBias bias;
            if (cbs_values.exists(key)) {
              bias = cbs_values.at<gtsam::imuBias::ConstantBias>(key);
            } else {
              return false;
            }
            gtsam::Vector6 sigmas;
            sigmas.head<3>().setConstant(backend_params_.initialAccBiasSigma_);
            sigmas.tail<3>().setConstant(backend_params_.initialGyroBiasSigma_);
            const gtsam::SharedNoiseModel noise =
                gtsam::noiseModel::Diagonal::Sigmas(sigmas);
            graph_to_update->emplace_shared<
                gtsam::PriorFactor<gtsam::imuBias::ConstantBias>>(key, bias, noise);
            return true;
          }
          case 'v': {
            gtsam::Vector3 vel;
            if (cbs_values.exists(key)) {
              vel = cbs_values.at<gtsam::Vector3>(key);
            } else {
              return false;
            }
            gtsam::Vector3 sigmas;
            sigmas.setConstant(0.1);
            const gtsam::SharedNoiseModel noise =
                gtsam::noiseModel::Diagonal::Sigmas(sigmas);
            graph_to_update->emplace_shared<gtsam::PriorFactor<gtsam::Vector3>>(
                key, vel, noise);
            return true;
          }
          default:
            return false;
        }
      };

      std::unordered_set<gtsam::Key> prior_keys_set;
      const auto add_state_triplet_for_index = [&](const size_t index) {
        const gtsam::Symbol pose_key('x', index);
        const gtsam::Symbol vel_key('v', index);
        const gtsam::Symbol bias_key('b', index);
        if (cbs_values.exists(pose_key.key())) {
          prior_keys_set.insert(pose_key.key());
        }
        if (cbs_values.exists(vel_key.key())) {
          prior_keys_set.insert(vel_key.key());
        }
        if (cbs_values.exists(bias_key.key())) {
          prior_keys_set.insert(bias_key.key());
        }
      };
      if (failed_index_opt) {
        add_state_triplet_for_index(*failed_index_opt);
      }
      add_state_triplet_for_index(curr_kf_id_);
      for (const auto& key_value : new_values) {
        const gtsam::Symbol candidate(key_value.key);
        if (candidate.chr() == 'x' || candidate.chr() == 'v' ||
            candidate.chr() == 'b') {
          if (cbs_values.exists(candidate.key())) {
            prior_keys_set.insert(candidate.key());
          }
        }
      }

      std::vector<gtsam::Key> prior_keys(prior_keys_set.begin(),
                                         prior_keys_set.end());
      std::sort(prior_keys.begin(), prior_keys.end());

      gtsam::NonlinearFactorGraph recovery_priors;
      size_t priors_added = 0u;
      for (const gtsam::Key key_raw : prior_keys) {
        const gtsam::Symbol key(key_raw);
        if (add_stabilizing_prior(key, &recovery_priors)) {
          ++priors_added;
          LOG(ERROR) << "CBS recovery added stabilizing prior on key "
                     << key.chr() << key.index();
        }
      }

      if (priors_added == 0u) {
        LOG(ERROR) << "CBS recovery could not add any stabilizing priors.";
        return false;
      }

      try {
        LOG(ERROR) << "Attempting CBS recovery update with " << priors_added
                   << " stabilizing priors (priors-only).";
        cbs::BPSAM::UpdateParams recovery_params;
        gtsam::ISAM2Result recovery_result = cbs_optimizer_->update(
            recovery_priors, gtsam::Values(), recovery_params);
        result->iterations = 1;
        result->intermediateSteps = 0;
        result->nonlinearVariables = recovery_result.variablesRelinearized;
        result->linearVariables = recovery_result.variablesReeliminated;
        result->error =
            recovery_result.errorAfter ? *recovery_result.errorAfter : 0.0;
        cbs_last_update_result_ = recovery_result;
        cbs_has_last_update_result_ = true;
        LOG(ERROR) << "CBS recovery update succeeded.";
        return true;
      } catch (const std::exception& recovery_e) {
        LOG(ERROR) << "CBS recovery update failed: " << recovery_e.what();
        try {
          LOG(ERROR)
              << "Retrying CBS recovery with stabilizing priors + new values.";
          cbs::BPSAM::UpdateParams recovery_params;
          const gtsam::ISAM2Result recovery_result =
              cbs_optimizer_->update(recovery_priors, new_values, recovery_params);
          result->iterations = 1;
          result->intermediateSteps = 0;
          result->nonlinearVariables = recovery_result.variablesRelinearized;
          result->linearVariables = recovery_result.variablesReeliminated;
          result->error =
              recovery_result.errorAfter ? *recovery_result.errorAfter : 0.0;
          cbs_last_update_result_ = recovery_result;
          cbs_has_last_update_result_ = true;
          LOG(ERROR) << "CBS priors-only recovery update succeeded.";
          return true;
        } catch (const std::exception& priors_only_e) {
          LOG(ERROR) << "CBS priors-only recovery failed: " << priors_only_e.what();
          return false;
        }
      } catch (...) {
        LOG(ERROR) << "CBS recovery update failed with unknown exception.";
        return false;
      }
    };

    const auto run_cbs_update_epoch =
        [&](const bool enable_remove_factor_indices,
            bool* should_retry_without_remove) -> bool {
          CHECK_NOTNULL(should_retry_without_remove);
          *should_retry_without_remove = false;

          try {
            // zy Step 40f
            // Run CBS pose-sharing inner rounds per epoch:
            // round 0 uses incoming factors/values, later rounds run
            // belief-only updates.
            const int max_pose_rounds =
                std::max(1, FLAGS_cbs_pose_rounds_per_epoch);
            const double abs_eps =
                std::max(0.0, FLAGS_cbs_pose_convergence_abs_residual);
            const double rel_eps =
                std::max(0.0, FLAGS_cbs_pose_convergence_rel_residual);
            auto compute_cbs_residual = [&]() -> std::optional<double> {
              try {
                const auto estimate = cbs_optimizer_->calculateEstimate();
                return cbs_optimizer_->getFactorsUnsafe().error(estimate);
              } catch (...) {
                return std::nullopt;
              }
            };
            struct ActiveStateCounts {
              size_t x = 0u;
              size_t v = 0u;
              size_t b = 0u;
            };
            const auto count_active_state_keys = [&]() {
              ActiveStateCounts counts;
              const auto& variable_index = cbs_optimizer_->getVariableIndex();
              for (const auto& key_and_slots : variable_index) {
                const gtsam::Symbol key_symbol(key_and_slots.first);
                if (key_symbol.chr() == kPoseSymbolChar) {
                  ++counts.x;
                } else if (key_symbol.chr() == kVelocitySymbolChar) {
                  ++counts.v;
                } else if (key_symbol.chr() == kImuBiasSymbolChar) {
                  ++counts.b;
                }
              }
              return counts;
            };

            const cbs::BPSAM::UpdateParams cbs_first_update_params =
                build_cbs_first_update_params(enable_remove_factor_indices);
            const size_t removed_factor_indices_applied =
                cbs_first_update_params.removeFactorIndices.size();
            const ActiveStateCounts active_pre = count_active_state_keys();
            const auto first_update_start = std::chrono::steady_clock::now();
            gtsam::ISAM2Result cbs_result = cbs_optimizer_->update(
                cbs_update_factors, new_values, cbs_first_update_params);
            cbs_first_update_ms +=
                elapsedMs(first_update_start, std::chrono::steady_clock::now());
            // Only the first round may delete factor slots; belief-only rounds
            // must not re-apply the same removals.
            cbs::BPSAM::UpdateParams cbs_inner_round_params;
            int rounds_executed = 1;
            std::optional<double> prev_residual = compute_cbs_residual();

            for (int round = 1; round < max_pose_rounds; ++round) {
              const auto inner_round_start = std::chrono::steady_clock::now();
              cbs_result = cbs_optimizer_->update(gtsam::NonlinearFactorGraph(),
                                                  gtsam::Values(),
                                                  cbs_inner_round_params);
              cbs_inner_round_ms += elapsedMs(inner_round_start,
                                              std::chrono::steady_clock::now());
              ++rounds_executed;

              const std::optional<double> curr_residual = compute_cbs_residual();
              if (curr_residual && prev_residual) {
                const double abs_change =
                    std::fabs(*curr_residual - *prev_residual);
                const double rel_change =
                    abs_change / std::max(std::fabs(*prev_residual), 1e-12);
                if (abs_change <= abs_eps || rel_change <= rel_eps) {
                  VLOG(2) << "CBS pose rounds converged early at round "
                          << rounds_executed << "/" << max_pose_rounds
                          << " (abs=" << abs_change << ", rel=" << rel_change
                          << ")";
                  break;
                }
              }

              if (curr_residual) {
                prev_residual = curr_residual;
              }
            }

            // Intuition: keep FixedLagSmoother API contract by populating a
            // compatible summary result in CBS mode.
            result->iterations = rounds_executed;
            result->intermediateSteps = 0;
            result->nonlinearVariables = cbs_result.variablesRelinearized;
            result->linearVariables = cbs_result.variablesReeliminated;
            result->error = cbs_result.errorAfter ? *cbs_result.errorAfter : 0.0;

            // Intuition: refresh smart-factor slot cache only when this update
            // inserted factors.
            if (!cbs_update_factors.empty()) {
              cbs_last_update_result_ = cbs_result;
              cbs_has_last_update_result_ = true;
            }
            const ActiveStateCounts active_post = count_active_state_keys();
            const size_t cbs_new_factors_total = cbs_update_factors.size();
            std::cerr
                << std::setprecision(12)
                << "[CBS][HeartEpochDiag] curr_kf_id=" << curr_kf_id_
                << " remove_factor_indices_applied="
                << removed_factor_indices_applied
                << " remove_factor_indices_requested="
                << cbs_remove_indices_requested
                << " remove_factor_indices_dropped_stale="
                << cbs_remove_indices_dropped_stale
                << " remove_delete_slots_requested="
                << cbs_remove_delete_slots_requested
                << " remove_lag_slots_requested="
                << cbs_remove_lag_slots_requested
                << " remove_from_delete_only="
                << cbs_remove_from_delete_only
                << " remove_from_lag_only="
                << cbs_remove_from_lag_only
                << " remove_from_both="
                << cbs_remove_from_both
                << " remove_smart_factor_slots="
                << cbs_remove_smart_factor_slots
                << " remove_non_smart_slots="
                << cbs_remove_non_smart_slots
                << " remove_prev_epoch_size="
                << cbs_remove_prev_epoch_size
                << " remove_curr_epoch_size="
                << cbs_remove_curr_epoch_size
                << " remove_repeated_from_prev_epoch="
                << cbs_remove_repeated_from_prev_epoch
                << " remove_factor_indices_enabled="
                << (enable_remove_factor_indices ? 1 : 0)
                << " diag_disable_remove_factor_indices="
                << (FLAGS_cbs_diag_disable_remove_factor_indices ? 1 : 0)
                << " diag_disable_lag_eviction_remove_candidates="
                << (FLAGS_cbs_diag_disable_lag_eviction_remove_candidates ? 1 : 0)
                << " diag_disable_merge_delete_slots_into_remove_factor_indices="
                << (FLAGS_cbs_diag_disable_merge_delete_slots_into_remove_factor_indices
                        ? 1
                        : 0)
                << " cbs_use_marginalization_prior_bridge="
                << (FLAGS_cbs_use_marginalization_prior_bridge ? 1 : 0)
                << " cbs_b1_refresh_cov_only_on_boundary_change="
                << (FLAGS_cbs_b1_refresh_cov_only_on_boundary_change ? 1 : 0)
                << " cbs_b1_pose_only_cov_refresh="
                << (FLAGS_cbs_b1_pose_only_cov_refresh ? 1 : 0)
                << " cbs_b1_cov_refresh_this_epoch="
                << (cbs_b1_cov_refresh_this_epoch ? 1 : 0)
                << " diag_disable_repeated_remove_from_prev_epoch="
                << (FLAGS_cbs_diag_disable_repeated_remove_from_prev_epoch ? 1 : 0)
                << " diag_disable_first_attempt_remove_factor_indices="
                << (FLAGS_cbs_diag_disable_first_attempt_remove_factor_indices ? 1
                                                                                : 0)
                << " cbs_allow_heavy_maintenance="
                << (cbs_allow_heavy_maintenance ? 1 : 0)
                << " cbs_no_external_fast_path_applied="
                << (cbs_no_external_fast_path_applied ? 1 : 0)
                << " diag_force_no_external_fast_path="
                << (FLAGS_cbs_diag_force_no_external_fast_path_when_no_external_effect
                        ? 1
                        : 0)
                << " lag_prev_oldest_active_frame_id="
                << lag_window_state.prev_oldest_active_frame_id
                << " lag_oldest_active_frame_id="
                << lag_window_state.oldest_active_frame_id
                << " lag_eviction_start_frame_id="
                << lag_window_state.eviction_start_frame_id
                << " lag_eviction_end_frame_id="
                << lag_window_state.eviction_end_frame_id
                << " lag_eviction_incremental="
                << (lag_window_state.eviction_incremental ? 1 : 0)
                << " lag_eviction_full_rescan="
                << (lag_window_state.eviction_full_rescan ? 1 : 0)
                << " lag_eviction_frames="
                << lag_window_state.eviction_frames
                << " lag_stale_state_keys="
                << lag_window_state.stale_state_keys
                << " lag_stale_pose_keys="
                << lag_window_state.stale_pose_keys
                << " rounds_executed=" << rounds_executed
                << " new_factors_count=" << cbs_new_factors_total
                << " new_factors_anchor_priors=" << cbs_anchor_priors_added
                << " diag_disable_lag_boundary_anchor_priors="
                << (FLAGS_cbs_diag_disable_lag_boundary_anchor_priors ? 1 : 0)
                << " diag_disable_smart_factor_replacements="
                << (FLAGS_cbs_diag_disable_smart_factor_replacements ? 1 : 0)
                << " new_factors_smart_replacements="
                << cbs_smart_factor_replacements
                << " new_factors_smart_new_insertions="
                << cbs_smart_factor_new_insertions
                << " delete_slots_total="
                << cbs_delete_slots_total
                << " delete_slots_from_smart_replacement="
                << cbs_delete_slots_from_smart_replacement
                << " delete_slots_from_cheirality_cleanup="
                << cbs_delete_slots_from_cheirality_cleanup
                << " delete_slots_from_other_cleanup="
                << cbs_delete_slots_from_other_cleanup
                << " smart_factors_replaced_count="
                << cbs_smart_factor_replacements
                << " smart_factors_new_insertions_count="
                << cbs_smart_factor_new_insertions
                << " smart_replacement_same_support_count="
                << cbs_smart_replacement_same_support_count
                << " smart_replacement_same_pose_key_set_count="
                << cbs_smart_replacement_same_pose_key_set_count
                << " smart_replacement_small_support_delta_count="
                << cbs_smart_replacement_small_support_delta_count
                << " smart_replacement_large_support_delta_count="
                << cbs_smart_replacement_large_support_delta_count
                << " smart_replacement_due_to_material_support_change="
                << cbs_smart_replacement_due_to_material_support_change
                << " smart_replacement_due_to_pose_key_set_change="
                << cbs_smart_replacement_due_to_pose_key_set_change
                << " smart_replacement_due_to_invalid_to_valid_transition="
                << cbs_smart_replacement_due_to_invalid_to_valid_transition
                << " smart_replacement_due_to_valid_to_invalid_transition="
                << cbs_smart_replacement_due_to_valid_to_invalid_transition
                << " smart_replacement_due_to_degenerate_factor="
                << cbs_smart_replacement_due_to_degenerate_factor
                << " smart_replacement_due_to_correctness_threshold="
                << cbs_smart_replacement_due_to_correctness_threshold
                << " smart_replacement_skipped_small_change="
                << cbs_smart_replacement_skipped_small_change
                << " smart_replacement_skipped_keep_existing="
                << cbs_smart_replacement_skipped_keep_existing
                << " smart_replacement_material_support_delta_threshold="
                << static_cast<size_t>(std::max(
                       1, FLAGS_cbs_smart_replace_material_support_delta))
                << " smart_replacement_material_pose_key_delta_threshold="
                << static_cast<size_t>(std::max(
                       1, FLAGS_cbs_smart_replace_material_pose_key_delta))
                << " landmarks_touched_for_replacement="
                << cbs_landmarks_touched_for_replacement
                << " landmarks_touched_for_new_insertion="
                << cbs_landmarks_touched_for_new_insertion
                << " new_factors_other_vio="
                << cbs_other_new_factor_count
                << " new_values_count=" << new_values.size()
                << " active_x_pre=" << active_pre.x
                << " active_v_pre=" << active_pre.v
                << " active_b_pre=" << active_pre.b
                << " active_x_post=" << active_post.x
                << " active_v_post=" << active_post.v
                << " active_b_post=" << active_post.b
                << " fixed_lag_states=" << backend_params_.nr_states_
                << " cbs_heart_active=" << (useCbsOptimizerHeart() ? 1 : 0)
                << " use_cbs_optimizer=" << (FLAGS_use_cbs_optimizer ? 1 : 0)
                << " cbs_replace_fixed_lag_optimizer="
                << (FLAGS_cbs_replace_fixed_lag_optimizer ? 1 : 0)
                << std::endl;
            const double cbs_epoch_total_ms =
                elapsedMs(cbs_epoch_start, std::chrono::steady_clock::now());
            std::cerr << std::setprecision(12)
                      << "[CBS][HeartTimingDiag] curr_kf_id=" << curr_kf_id_
                      << " cbs_epoch_total_ms=" << cbs_epoch_total_ms
                      << " lag_window_build_ms=" << cbs_lag_window_build_ms
                      << " anchor_prior_ms=" << cbs_anchor_prior_ms
                      << " b1_estimate_extract_ms=" << cbs_b1_estimate_extract_ms
                      << " b1_boundary_extract_ms=" << cbs_b1_boundary_extract_ms
                      << " b1_pose_cov_query_ms=" << cbs_b1_pose_cov_query_ms
                      << " b1_vel_cov_query_ms=" << cbs_b1_vel_cov_query_ms
                      << " b1_bias_cov_query_ms=" << cbs_b1_bias_cov_query_ms
                      << " b1_cov_query_total_ms="
                      << (cbs_b1_pose_cov_query_ms + cbs_b1_vel_cov_query_ms +
                          cbs_b1_bias_cov_query_ms)
                      << " b1_prior_factor_emit_ms=" << cbs_b1_prior_factor_emit_ms
                      << " b1_pose_cov_queries=" << cbs_b1_pose_cov_queries
                      << " b1_vel_cov_queries=" << cbs_b1_vel_cov_queries
                      << " b1_bias_cov_queries=" << cbs_b1_bias_cov_queries
                      << " remove_index_discovery_ms="
                      << cbs_remove_index_discovery_ms
                      << " remove_factor_indices_requested="
                      << cbs_remove_indices_requested
                      << " remove_factor_indices_applied="
                      << cbs_remove_indices_applied
                      << " remove_factor_indices_dropped_stale="
                      << cbs_remove_indices_dropped_stale
                      << " remove_delete_slots_requested="
                      << cbs_remove_delete_slots_requested
                      << " remove_lag_slots_requested="
                      << cbs_remove_lag_slots_requested
                      << " remove_from_delete_only="
                      << cbs_remove_from_delete_only
                      << " remove_from_lag_only="
                      << cbs_remove_from_lag_only
                      << " remove_from_both="
                      << cbs_remove_from_both
                      << " remove_smart_factor_slots="
                      << cbs_remove_smart_factor_slots
                      << " remove_non_smart_slots="
                      << cbs_remove_non_smart_slots
                      << " remove_prev_epoch_size="
                      << cbs_remove_prev_epoch_size
                      << " remove_curr_epoch_size="
                      << cbs_remove_curr_epoch_size
                      << " remove_repeated_from_prev_epoch="
                      << cbs_remove_repeated_from_prev_epoch
                      << " diag_disable_remove_factor_indices="
                      << (FLAGS_cbs_diag_disable_remove_factor_indices ? 1 : 0)
                      << " diag_disable_lag_eviction_remove_candidates="
                      << (FLAGS_cbs_diag_disable_lag_eviction_remove_candidates ? 1
                                                                                 : 0)
                      << " diag_disable_merge_delete_slots_into_remove_factor_indices="
                      << (FLAGS_cbs_diag_disable_merge_delete_slots_into_remove_factor_indices
                              ? 1
                              : 0)
                      << " cbs_use_marginalization_prior_bridge="
                      << (FLAGS_cbs_use_marginalization_prior_bridge ? 1 : 0)
                      << " cbs_b1_refresh_cov_only_on_boundary_change="
                      << (FLAGS_cbs_b1_refresh_cov_only_on_boundary_change ? 1 : 0)
                      << " cbs_b1_pose_only_cov_refresh="
                      << (FLAGS_cbs_b1_pose_only_cov_refresh ? 1 : 0)
                      << " cbs_b1_cov_refresh_this_epoch="
                      << (cbs_b1_cov_refresh_this_epoch ? 1 : 0)
                      << " diag_disable_repeated_remove_from_prev_epoch="
                      << (FLAGS_cbs_diag_disable_repeated_remove_from_prev_epoch ? 1
                                                                                  : 0)
                      << " diag_disable_first_attempt_remove_factor_indices="
                      << (FLAGS_cbs_diag_disable_first_attempt_remove_factor_indices
                              ? 1
                              : 0)
                      << " diag_force_no_external_fast_path="
                      << (FLAGS_cbs_diag_force_no_external_fast_path_when_no_external_effect
                              ? 1
                              : 0)
                      << " optimizer_first_update_ms=" << cbs_first_update_ms
                      << " optimizer_inner_round_ms=" << cbs_inner_round_ms
                      << " optimizer_total_ms="
                      << (cbs_first_update_ms + cbs_inner_round_ms)
                      << " lag_prev_oldest_active_frame_id="
                      << lag_window_state.prev_oldest_active_frame_id
                      << " lag_oldest_active_frame_id="
                      << lag_window_state.oldest_active_frame_id
                      << " lag_eviction_start_frame_id="
                      << lag_window_state.eviction_start_frame_id
                      << " lag_eviction_end_frame_id="
                      << lag_window_state.eviction_end_frame_id
                      << " lag_eviction_incremental="
                      << (lag_window_state.eviction_incremental ? 1 : 0)
                      << " lag_eviction_full_rescan="
                      << (lag_window_state.eviction_full_rescan ? 1 : 0)
                      << " lag_eviction_frames="
                      << lag_window_state.eviction_frames
                      << " lag_stale_state_keys="
                      << lag_window_state.stale_state_keys
                      << " lag_stale_pose_keys="
                      << lag_window_state.stale_pose_keys
                      << " new_factors_count=" << cbs_new_factors_total
                      << " new_factors_anchor_priors="
                      << cbs_anchor_priors_added
                      << " diag_disable_lag_boundary_anchor_priors="
                      << (FLAGS_cbs_diag_disable_lag_boundary_anchor_priors ? 1 : 0)
                      << " diag_disable_smart_factor_replacements="
                      << (FLAGS_cbs_diag_disable_smart_factor_replacements ? 1 : 0)
                      << " new_factors_smart_replacements="
                      << cbs_smart_factor_replacements
                      << " new_factors_smart_new_insertions="
                      << cbs_smart_factor_new_insertions
                      << " delete_slots_total="
                      << cbs_delete_slots_total
                      << " delete_slots_from_smart_replacement="
                      << cbs_delete_slots_from_smart_replacement
                      << " delete_slots_from_cheirality_cleanup="
                      << cbs_delete_slots_from_cheirality_cleanup
                      << " delete_slots_from_other_cleanup="
                      << cbs_delete_slots_from_other_cleanup
                      << " smart_factors_replaced_count="
                      << cbs_smart_factor_replacements
                      << " smart_factors_new_insertions_count="
                      << cbs_smart_factor_new_insertions
                      << " smart_replacement_same_support_count="
                      << cbs_smart_replacement_same_support_count
                      << " smart_replacement_same_pose_key_set_count="
                      << cbs_smart_replacement_same_pose_key_set_count
                      << " smart_replacement_small_support_delta_count="
                      << cbs_smart_replacement_small_support_delta_count
                      << " smart_replacement_large_support_delta_count="
                      << cbs_smart_replacement_large_support_delta_count
                      << " smart_replacement_due_to_material_support_change="
                      << cbs_smart_replacement_due_to_material_support_change
                      << " smart_replacement_due_to_pose_key_set_change="
                      << cbs_smart_replacement_due_to_pose_key_set_change
                      << " smart_replacement_due_to_invalid_to_valid_transition="
                      << cbs_smart_replacement_due_to_invalid_to_valid_transition
                      << " smart_replacement_due_to_valid_to_invalid_transition="
                      << cbs_smart_replacement_due_to_valid_to_invalid_transition
                      << " smart_replacement_due_to_degenerate_factor="
                      << cbs_smart_replacement_due_to_degenerate_factor
                      << " smart_replacement_due_to_correctness_threshold="
                      << cbs_smart_replacement_due_to_correctness_threshold
                      << " smart_replacement_skipped_small_change="
                      << cbs_smart_replacement_skipped_small_change
                      << " smart_replacement_skipped_keep_existing="
                      << cbs_smart_replacement_skipped_keep_existing
                      << " smart_replacement_material_support_delta_threshold="
                      << static_cast<size_t>(std::max(
                             1, FLAGS_cbs_smart_replace_material_support_delta))
                      << " smart_replacement_material_pose_key_delta_threshold="
                      << static_cast<size_t>(std::max(
                             1, FLAGS_cbs_smart_replace_material_pose_key_delta))
                      << " landmarks_touched_for_replacement="
                      << cbs_landmarks_touched_for_replacement
                      << " landmarks_touched_for_new_insertion="
                      << cbs_landmarks_touched_for_new_insertion
                      << " new_factors_other_vio="
                      << cbs_other_new_factor_count
                      << " cbs_allow_heavy_maintenance="
                      << (cbs_allow_heavy_maintenance ? 1 : 0)
                      << " cbs_no_external_fast_path_applied="
                      << (cbs_no_external_fast_path_applied ? 1 : 0)
                      << std::endl;
            return true;
          } catch (const gtsam::IndeterminantLinearSystemException& e) {
            const gtsam::Symbol failed_symbol(e.nearbyVariable());
            std::ostringstream reason;
            reason << e.what() << " failed_symbol=" << failed_symbol.chr()
                   << failed_symbol.index();
            LOG(ERROR) << "CBS BPSAM indeterminant system: " << reason.str();
            return attempt_cbs_recovery(failed_symbol.index(), reason.str());
          } catch (const std::exception& e) {
            const std::string err_msg =
                e.what() ? std::string(e.what()) : std::string();
            const bool has_map_at = err_msg.find("map::at") != std::string::npos;
            const bool recoverable_signature =
                err_msg.find("IndeterminantLinearSystemException") !=
                    std::string::npos ||
                err_msg.find("invalid noise model") != std::string::npos ||
                err_msg.find("nearbyVariable is not a robot key") !=
                    std::string::npos ||
                has_map_at;

            if (has_map_at && enable_remove_factor_indices) {
              LOG(ERROR) << "CBS-heart update hit map::at while applying "
                            "removeFactorIndices. Retrying this epoch without "
                            "removeFactorIndices.";
              *should_retry_without_remove = true;
              return false;
            }

            if (recoverable_signature) {
              LOG(ERROR) << "CBS BPSAM update failed with recoverable signature: "
                         << err_msg;
              if (attempt_cbs_recovery(std::nullopt, err_msg)) {
                return true;
              }
            }
            LOG(ERROR) << "CBS BPSAM update failed: " << err_msg;
            return false;
          } catch (...) {
            LOG(ERROR) << "CBS BPSAM update failed with unknown exception.";
            return false;
          }
        };

    bool retry_without_remove = false;
    const bool has_remove_candidates =
        !delete_slots.empty() || !lag_window_state.remove_factor_indices.empty();
    const bool enable_remove_factor_indices_first_attempt =
        cbs_allow_heavy_maintenance && has_remove_candidates &&
        !FLAGS_cbs_diag_disable_remove_factor_indices &&
        !FLAGS_cbs_diag_disable_first_attempt_remove_factor_indices;
    if (run_cbs_update_epoch(enable_remove_factor_indices_first_attempt,
                             &retry_without_remove)) {
      return true;
    }

    if (retry_without_remove && enable_remove_factor_indices_first_attempt) {
      bool ignored_retry_flag = false;
      if (run_cbs_update_epoch(/*enable_remove_factor_indices=*/false,
                               &ignored_retry_flag)) {
        return true;
      }
    }

    return false;
  }
#endif

  // Store smoother as backup.
  CHECK(smoother_);
  // This is not doing a full deep copy: it is keeping same shared_ptrs for
  // factors but copying the isam result.
  Smoother smoother_backup(*smoother_);

  bool got_cheirality_exception = false;
  gtsam::Symbol lmk_symbol_cheirality;
  try {
    // Update smoother.
    VLOG(10) << "Starting update of smoother_...";
    *result =
        smoother_->update(new_factors, new_values, timestamps, delete_slots);
    VLOG(10) << "Finished update of smoother_.";
    if (debug_smoother_) {
      printSmootherInfo(new_factors, delete_slots, "CATCHING EXCEPTION", false);
      debug_smoother_ = false;
    }
  } catch (const gtsam::IndeterminantLinearSystemException& e) {
    // zy: GEODE mono robustness path.
    // zy: When the linear system becomes underconstrained, we add
    // zy: stabilizing priors on x/b/v (failure-near + new states) and try a
    // zy: priors-only fallback update instead of hard-stopping the backend.
    LOG(ERROR) << e.what();
    const gtsam::Key& var = e.nearbyVariable();
    gtsam::Symbol symb(var);
    LOG(ERROR) << "ERROR: Variable has type '" << symb.chr() << "' "
               << "and index " << symb.index() << std::endl;

    if (VLOG_IS_ON(1)) {
      smoother_->getFactors().print("Smoother's factors:\n[\n\t");
      LOG(INFO) << " ]";
      state_.print("State values\n[\n\t");
      LOG(INFO) << " ]";
      printSmootherInfo(new_factors, delete_slots);
    }

    // Add priors on all variables to fix indeterminant linear system
    gtsam::Values values = smoother_->calculateEstimate();

    // Add priors on pose/velocity/bias keys near the failure and on newly
    // introduced states for this update.
    std::vector<unsigned char> key_prefixes_to_prior = {'x', 'b', 'v'};
    gtsam::Symbol first_key = values.keys().at(0);
    std::unordered_set<gtsam::Key> prior_keys_set;
    for (const auto& prefix : key_prefixes_to_prior) {
      prior_keys_set.insert(gtsam::Symbol(prefix, symb.index()));
      prior_keys_set.insert(gtsam::Symbol(prefix, first_key.index()));
    }
    for (const auto& key_value : new_values) {
      const gtsam::Symbol candidate(key_value.key);
      if (candidate.chr() == 'x' || candidate.chr() == 'b' ||
          candidate.chr() == 'v') {
        prior_keys_set.insert(candidate.key());
      }
    }
    std::vector<gtsam::Key> prior_keys(prior_keys_set.begin(),
                                       prior_keys_set.end());
    std::sort(prior_keys.begin(), prior_keys.end());
    gtsam::NonlinearFactorGraph nfg;

    auto add_stabilizing_prior =
        [&](const gtsam::Symbol& key,
            gtsam::NonlinearFactorGraph* graph_to_update) -> bool {
      CHECK_NOTNULL(graph_to_update);
      switch (key.chr()) {
        case 'x': {
          gtsam::Pose3 pose;
          if (values.exists(key)) {
            pose = values.at<gtsam::Pose3>(key);
          } else if (new_values.exists(key)) {
            pose = new_values.at<gtsam::Pose3>(key);
          } else {
            return false;
          }
          gtsam::Vector6 sigmas;
          sigmas.head<3>().setConstant(0.01);  // rotation
          sigmas.tail<3>().setConstant(0.1);   // translation
          gtsam::SharedNoiseModel noise =
              gtsam::noiseModel::Diagonal::Sigmas(sigmas);
          graph_to_update->emplace_shared<gtsam::PriorFactor<gtsam::Pose3>>(
              key, pose, noise);
          return true;
        }
        case 'b': {
          gtsam::imuBias::ConstantBias bias;
          if (values.exists(key)) {
            bias = values.at<gtsam::imuBias::ConstantBias>(key);
          } else if (new_values.exists(key)) {
            bias = new_values.at<gtsam::imuBias::ConstantBias>(key);
          } else {
            return false;
          }
          gtsam::Vector6 sigmas;
          sigmas.head<3>().setConstant(backend_params_.initialAccBiasSigma_);
          sigmas.tail<3>().setConstant(backend_params_.initialGyroBiasSigma_);
          gtsam::SharedNoiseModel noise =
              gtsam::noiseModel::Diagonal::Sigmas(sigmas);
          graph_to_update
              ->emplace_shared<gtsam::PriorFactor<gtsam::imuBias::ConstantBias>>(
                  key, bias, noise);
          return true;
        }
        case 'v': {
          gtsam::Vector3 vel;
          if (values.exists(key)) {
            vel = values.at<gtsam::Vector3>(key);
          } else if (new_values.exists(key)) {
            vel = new_values.at<gtsam::Vector3>(key);
          } else {
            return false;
          }
          gtsam::Vector3 sigmas;
          sigmas.setConstant(0.1);
          gtsam::SharedNoiseModel noise =
              gtsam::noiseModel::Diagonal::Sigmas(sigmas);
          graph_to_update->emplace_shared<gtsam::PriorFactor<gtsam::Vector3>>(
              key, vel, noise);
          return true;
        }
        default:
          return false;
      }
    };

    // Only add priors on first state and the state nearest the failure
    for (const gtsam::Key& raw_key : prior_keys) {
      const gtsam::Symbol key(raw_key);
      if (add_stabilizing_prior(key, &nfg)) {
        LOG(ERROR) << "Adding prior on key: " << key.chr() << key.index();
      } else {
        VLOG(2) << "Skipping stabilizing prior on unavailable key: "
                << key.chr() << key.index();
      }
    }
    gtsam::NonlinearFactorGraph new_factors_mutable;
    new_factors_mutable.push_back(new_factors.begin(), new_factors.end());
    new_factors_mutable.push_back(nfg.begin(), nfg.end());

    // Update with graph and GN optimized values
    try {
      // Update smoother
      LOG(ERROR) << "Attempting to update smoother with added prior factors";
      *smoother_ = smoother_backup;  // reset isam to backup
      *result = smoother_->update(
          new_factors_mutable, new_values, timestamps, delete_slots);
    } catch (...) {
      // Catch the rest of exceptions.
      LOG(ERROR) << "Smoother recovery failed. Most likely, the additional "
                    "prior factors were insufficient to keep the system from "
                    "becoming indeterminant.";
      // Final fallback: keep this backend cycle alive by applying only
      // stabilizing priors plus new values.
      *smoother_ = smoother_backup;
      gtsam::NonlinearFactorGraph fallback_priors;
      for (const gtsam::Key& raw_key : prior_keys) {
        const gtsam::Symbol key(raw_key);
        add_stabilizing_prior(key, &fallback_priors);
      }
      try {
        LOG(ERROR) << "Attempting priors-only fallback update after failed "
                      "indeterminant recovery.";
        *result = smoother_->update(
            fallback_priors, new_values, timestamps, delete_slots);
        LOG(ERROR) << "Priors-only fallback update succeeded.";
      } catch (...) {
        LOG(ERROR) << "Priors-only fallback update also failed.";
        return false;
      }
    }
  } catch (const gtsam::InvalidNoiseModel& e) {
    LOG(ERROR) << e.what();
    printSmootherInfo(new_factors, delete_slots);
    return false;
  } catch (const gtsam::InvalidMatrixBlock& e) {
    LOG(ERROR) << e.what();
    printSmootherInfo(new_factors, delete_slots);
    return false;
  } catch (const gtsam::InvalidDenseElimination& e) {
    LOG(ERROR) << e.what();
    printSmootherInfo(new_factors, delete_slots);
    return false;
  } catch (const gtsam::InvalidArgumentThreadsafe& e) {
    LOG(ERROR) << e.what();
    printSmootherInfo(new_factors, delete_slots);
    return false;
  } catch (const gtsam::ValuesKeyDoesNotExist& e) {
    LOG(ERROR) << e.what();
    constexpr size_t kMaxMissingKeyRecoveryAttempts = 5u;
    std::unordered_set<gtsam::Key> missing_keys;
    missing_keys.insert(e.key());

    for (size_t attempt = 0u; attempt < kMaxMissingKeyRecoveryAttempts;
         ++attempt) {
      *smoother_ = smoother_backup;

      gtsam::NonlinearFactorGraph recovery_new_factors;
      size_t dropped_new_factor_count = 0u;
      for (const auto& factor : new_factors) {
        if (!factor) {
          continue;
        }
        bool has_missing_key = false;
        for (const gtsam::Key key : factor->keys()) {
          if (missing_keys.count(key) > 0u) {
            has_missing_key = true;
            break;
          }
        }
        if (has_missing_key) {
          ++dropped_new_factor_count;
          continue;
        }
        recovery_new_factors.push_back(factor);
      }

      gtsam::FactorIndices recovery_delete_slots = delete_slots;
      const auto& factors = smoother_->getFactors();
      for (size_t slot_idx = 0; slot_idx < factors.size(); ++slot_idx) {
        const auto& factor = factors.at(slot_idx);
        if (!factor) {
          continue;
        }
        for (const gtsam::Key key : factor->keys()) {
          if (missing_keys.count(key) > 0u) {
            recovery_delete_slots.push_back(slot_idx);
            break;
          }
        }
      }

      std::sort(recovery_delete_slots.begin(), recovery_delete_slots.end());
      recovery_delete_slots.erase(std::unique(recovery_delete_slots.begin(),
                                              recovery_delete_slots.end()),
                                  recovery_delete_slots.end());
      recovery_delete_slots.erase(
          std::remove_if(
              recovery_delete_slots.begin(),
              recovery_delete_slots.end(),
              [&factors](const gtsam::FactorIndex idx) {
                return idx >= static_cast<gtsam::FactorIndex>(factors.size());
              }),
          recovery_delete_slots.end());

      if (recovery_delete_slots.empty() && dropped_new_factor_count == 0u) {
        LOG(ERROR) << "Missing-key recovery attempt " << (attempt + 1)
                   << " found no removable stale slots/factors; skipping this "
                      "backend update cycle.";
        if (VLOG_IS_ON(1)) {
          printSmootherInfo(new_factors, delete_slots);
        }
        *result = Smoother::Result();
        return true;
      }

      try {
        LOG(ERROR) << "Retrying smoother update after missing-key cleanup "
                   << "(attempt " << (attempt + 1) << "): dropped_new_factors="
                   << dropped_new_factor_count
                   << ", delete_slots=" << recovery_delete_slots.size();
        *result = smoother_->update(
            recovery_new_factors, new_values, timestamps, recovery_delete_slots);
        LOG(ERROR) << "Recovered smoother update after missing-key cleanup.";
        return true;
      } catch (const gtsam::ValuesKeyDoesNotExist& nested_missing) {
        const gtsam::Key nested_missing_key = nested_missing.key();
        const gtsam::Symbol nested_missing_sym(nested_missing_key);
        LOG(ERROR) << "Missing-key recovery attempt " << (attempt + 1)
                   << " still failed with key " << nested_missing_sym.chr()
                   << nested_missing_sym.index() << ": "
                   << nested_missing.what();
        missing_keys.insert(nested_missing_key);
        continue;
      } catch (const std::exception& recovery_e) {
        LOG(ERROR) << "Missing-key recovery failed: " << recovery_e.what();
        if (VLOG_IS_ON(1)) {
          printSmootherInfo(new_factors, recovery_delete_slots);
        }
        return false;
      } catch (...) {
        LOG(ERROR) << "Missing-key recovery failed with unknown exception.";
        if (VLOG_IS_ON(1)) {
          printSmootherInfo(new_factors, recovery_delete_slots);
        }
        return false;
      }
    }

    LOG(ERROR) << "Exhausted missing-key recovery attempts.";
    if (VLOG_IS_ON(1)) {
      printSmootherInfo(new_factors, delete_slots);
    }
    return false;
  } catch (const gtsam::CholeskyFailed& e) {
    LOG(ERROR) << e.what();
    printSmootherInfo(new_factors, delete_slots);
    return false;
  } catch (const gtsam::CheiralityException& e) {
    LOG(ERROR) << e.what();
    const gtsam::Key& lmk_key = e.nearbyVariable();
    lmk_symbol_cheirality = gtsam::Symbol(lmk_key);
    LOG(ERROR) << "ERROR: Variable has type '" << lmk_symbol_cheirality.chr()
               << "' "
               << "and index " << lmk_symbol_cheirality.index();
    printSmootherInfo(new_factors, delete_slots);
    got_cheirality_exception = true;
  } catch (const gtsam::StereoCheiralityException& e) {
    LOG(ERROR) << e.what();
    const gtsam::Key& lmk_key = e.nearbyVariable();
    lmk_symbol_cheirality = gtsam::Symbol(lmk_key);
    LOG(ERROR) << "ERROR: Variable has type '" << lmk_symbol_cheirality.chr()
               << "' "
               << "and index " << lmk_symbol_cheirality.index();
    printSmootherInfo(new_factors, delete_slots);
    got_cheirality_exception = true;
  } catch (const gtsam::RuntimeErrorThreadsafe& e) {
    LOG(ERROR) << e.what();
    printSmootherInfo(new_factors, delete_slots);
    return false;
  } catch (const gtsam::OutOfRangeThreadsafe& e) {
    LOG(ERROR) << e.what();
    printSmootherInfo(new_factors, delete_slots);
    return false;
  } catch (const std::out_of_range& e) {
    LOG(ERROR) << e.what();
    printSmootherInfo(new_factors, delete_slots);
    return false;
  } catch (const std::exception& e) {
    // Catch anything thrown within try block that derives from
    // std::exception.
    LOG(ERROR) << e.what();
    printSmootherInfo(new_factors, delete_slots);
    return false;
  } catch (...) {
    // Catch the rest of exceptions.
    LOG(ERROR) << "Unrecognized exception.";
    printSmootherInfo(new_factors, delete_slots);
    return false;
  }

  if (FLAGS_process_cheirality) {
    if (got_cheirality_exception) {
      LOG(WARNING) << "Starting processing cheirality exception # "
                   << counter_of_exceptions_;
      counter_of_exceptions_++;

      // Restore smoother as it was before failure.
      *smoother_ = smoother_backup;

      // Limit the number of cheirality exceptions per run.
      CHECK_LE(counter_of_exceptions_,
               FLAGS_max_number_of_cheirality_exceptions);

      // Check that we have a landmark.
      CHECK_EQ(lmk_symbol_cheirality.chr(), 'l');

      // Now that we know the lmk id, delete all factors attached to it!
      gtsam::NonlinearFactorGraph new_factors_tmp_cheirality;
      gtsam::Values new_values_cheirality;
      std::map<Key, double> timestamps_cheirality;
      gtsam::FactorIndices delete_slots_cheirality;
      const gtsam::NonlinearFactorGraph& graph = smoother_->getFactors();
      VLOG(10) << "Starting cleanCheiralityLmk...";
      cleanCheiralityLmk(lmk_symbol_cheirality,
                         &new_factors_tmp_cheirality,
                         &new_values_cheirality,
                         &timestamps_cheirality,
                         &delete_slots_cheirality,
                         graph,
                         new_factors,
                         new_values,
                         timestamps,
                         delete_slots);
      VLOG(10) << "Finished cleanCheiralityLmk.";

      // Recreate the graph before marginalization.
      if (VLOG_IS_ON(5) && FLAGS_debug_graph_before_opt) {
        debug_info_.graphBeforeOpt = graph;
        debug_info_.graphToBeDeleted = gtsam::NonlinearFactorGraph();
        debug_info_.graphToBeDeleted.resize(delete_slots_cheirality.size());
        for (size_t i = 0; i < delete_slots_cheirality.size(); i++) {
          // If the factor is to be deleted, store it as graph to be
          // deleted.
          CHECK(graph.exists(delete_slots_cheirality.at(i)))
              << "Slot # " << delete_slots_cheirality.at(i)
              << "does not exist in smoother graph.";
          // TODO here we can get the right slot that we are going to
          // delete, extend graphToBeDeleted to have both the factor and the
          // slot.
          debug_info_.graphToBeDeleted.at(i) =
              graph.at(delete_slots_cheirality.at(i));
        }
      }

      // Try again to optimize. This is a recursive call.
      LOG(WARNING) << "Starting updateSmoother after handling "
                      "cheirality exception.";
      bool status = updateSmoother(result,
                                   new_factors_tmp_cheirality,
                                   new_values_cheirality,
                                   timestamps_cheirality,
                                   delete_slots_cheirality);
      LOG(WARNING) << "Finished updateSmoother after handling "
                      "cheirality exception";
      return status;
    } else {
      counter_of_exceptions_ = 0;
    }
  }

    return true;
}

/* -------------------------------------------------------------------------- */
void VioBackend::cleanCheiralityLmk(
    const gtsam::Symbol& lmk_symbol,
    gtsam::NonlinearFactorGraph* new_factors_tmp_cheirality,
    gtsam::Values* new_values_cheirality,
    std::map<Key, double>* timestamps_cheirality,
    gtsam::FactorIndices* delete_slots_cheirality,
    const gtsam::NonlinearFactorGraph& graph,
    const gtsam::NonlinearFactorGraph& new_factors_tmp,
    const gtsam::Values& new_values,
    const std::map<Key, double>& timestamps,
    const gtsam::FactorIndices& delete_slots) {
  CHECK_NOTNULL(new_factors_tmp_cheirality);
  CHECK_NOTNULL(new_values_cheirality);
  CHECK_NOTNULL(timestamps_cheirality);
  CHECK_NOTNULL(delete_slots_cheirality);
  const gtsam::Key& lmk_key = lmk_symbol.key();

  // Delete from new factors.
  VLOG(10) << "Starting delete from new factors...";
  deleteAllFactorsWithKeyFromFactorGraph(
      lmk_key, new_factors_tmp, new_factors_tmp_cheirality);
  VLOG(10) << "Finished delete from new factors.";

  // Delete from new values.
  VLOG(10) << "Starting delete from new values...";
  bool is_deleted_from_values =
      deleteKeyFromValues(lmk_key, new_values, new_values_cheirality);
  VLOG(10) << "Finished delete from new values."; // zy

  // Delete from new values.
  VLOG(10) << "Starting delete from timestamps...";
  bool is_deleted_from_timestamps =
      deleteKeyFromTimestamps(lmk_key, timestamps, timestamps_cheirality);
  VLOG(10) << "Finished delete from timestamps.";

  // Check that if we deleted from values, we should have deleted as well
  // from timestamps.
  CHECK_EQ(is_deleted_from_values, is_deleted_from_timestamps);

  // Delete slots in current graph.
  VLOG(10) << "Starting delete from current graph...";
  *delete_slots_cheirality = delete_slots;
  std::vector<size_t> slots_of_extra_factors_to_delete;
  // Achtung: This has the chance to make the plane underconstrained, if
  // we delete too many point_plane factors.
  findSlotsOfFactorsWithKey(lmk_key, graph, &slots_of_extra_factors_to_delete);
  delete_slots_cheirality->insert(delete_slots_cheirality->end(),
                                  slots_of_extra_factors_to_delete.begin(),
                                  slots_of_extra_factors_to_delete.end());
  VLOG(10) << "Finished delete from current graph.";

  //////////////////////////// BOOKKEEPING
  ////////////////////////////////////////
  const LandmarkId& lmk_id = lmk_symbol.index();

  // Delete from feature tracks.
  VLOG(10) << "Starting delete from feature tracks...";
  CHECK(deleteLmkFromFeatureTracks(lmk_id));
  VLOG(10) << "Finished delete from feature tracks.";

  // Delete from extra structures (for derived classes).
  VLOG(10) << "Starting delete from extra structures...";
  deleteLmkFromExtraStructures(lmk_id);
  VLOG(10) << "Finished delete from extra structures.";
  //////////////////////////////////////////////////////////////////////////////
}

void VioBackend::deleteLmkFromExtraStructures(const LandmarkId& lmk_id) {
  LOG(ERROR) << "There is nothing to delete for lmk with id: " << lmk_id;
  return;
}

/* -------------------------------------------------------------------------- */
// BOOKKEEPING: updates the SlotIdx in the old_smart_factors such that
// this idx points to the updated slots in the graph after optimization.
// for next iteration to know which slots have to be deleted
// before adding the new smart factors.
// void VioBackend::updateNewSmartFactorsSlots(
//     const std::vector<LandmarkId>& lmk_ids_of_new_smart_factors,
//     SmartFactorMap* old_smart_factors) {
//   CHECK_NOTNULL(old_smart_factors);

//   // Get result.
//   // const gtsam::ISAM2Result& result = smoother_->getISAM2Result(); (zy cancelled it)

//   // zy Step 18d choose update indices + factor graph from the active optimizer so slot remapping stays correct in CBS and legacy modes.
//   const gtsam::ISAM2Result* isam2_result = nullptr;
//   const gtsam::NonlinearFactorGraph* factor_graph = nullptr;

// #ifdef KIMERA_USE_CBS
//   if (useCbsOptimizerHeart()) {
//     CHECK(cbs_optimizer_) << "CBS optimizer flag is ON but cbs_optimizer_ is null.";
//     if (!cbs_has_last_update_result_) {
//       VLOG(2) << "CBS smart-factor slot update skipped: no cached CBS update result yet.";
//       return;
//     }
//     isam2_result = &cbs_last_update_result_;
//     factor_graph = &cbs_optimizer_->getFactorsUnsafe();
//   } else
// #endif
//   {
//     isam2_result = &smoother_->getISAM2Result();
//     factor_graph = &smoother_->getFactors();
//   }

//   CHECK_NOTNULL(isam2_result);
//   CHECK_NOTNULL(factor_graph);


//   // Simple version of find smart factors.
//   for (size_t i = 0u; i < lmk_ids_of_new_smart_factors.size(); ++i) {
//     // DCHECK(i < result.newFactorsIndices.size()) (zy cancelled it)
//     // zy Step 18e: bound-check against the active optimizer's update result (CBS or legacy).
//     DCHECK(i < isam2_result->newFactorsIndices.size())


//         << "There are more new smart factors than new factors added to the "
//            "graph.";
//     // Get new slot in the graph for the newly added smart factor.
//     // const size_t& slot = result.newFactorsIndices.at(i); (zy cancelled it)
//     const size_t& slot = isam2_result->newFactorsIndices.at(i); // zy Step 18e

//     // TODO this will not work if there are non-smart factors!!!
//     // Update slot using isam2 indices.
//     // ORDER of inclusion of factors in the ISAM2::update() function
//     // matters, as these indices have a 1-to-1 correspondence with the
//     // factors.

//     // BOOKKEEPING, for next iteration to know which slots have to be
//     // deleted before adding the new smart factors. Find the entry in
//     // old_smart_factors_.
//     const auto& it =
//         old_smart_factors->find(lmk_ids_of_new_smart_factors.at(i));

//     DCHECK(it != old_smart_factors->end())
//         << "Trying to access unavailable factor.";
//     // CHECK that the factor in the graph at slot position is a smart
//     // factor.
//     // const auto sptr = dynamic_cast<const SmartStereoFactor*>(
//         // smoother_->getFactors().at(slot).get()); (zy cancelled it)
//     DCHECK(factor_graph->exists(slot));
//     const auto sptr = dynamic_cast<const SmartStereoFactor*>(
//         factor_graph->at(slot).get()); // zy Step 18e

//     DCHECK(sptr);
//     // CHECK that shared ptrs point to the same smart factor.
//     // make sure no one is cloning SmartSteroFactors.
//     DCHECK_EQ(it->second.first.get(), sptr)
//         << "Non-matching addresses for same factors for lmk with id: "
//         << lmk_ids_of_new_smart_factors.at(i) << " in old_smart_factors_ "
//         << "VS factor in graph at slot: " << slot
//         << ". Slot previous to update was: " << it->second.second;

//     // Update slot number in old_smart_factors_.
//     it->second.second = slot;
//   }
// } (zy cancelled it)

// zy Step 24: In legacy Kimera, newFactorsIndices order matches the factors we inserted.
// In CBS mode, BPSAM may inject belief factors internally, so those indices can shift.
// If we keep index-based remap in CBS mode, smart-factor slot bookkeeping can silently break.
void VioBackend::updateNewSmartFactorsSlots(
    const std::vector<LandmarkId>& lmk_ids_of_new_smart_factors,
    SmartFactorMap* old_smart_factors) {
  CHECK_NOTNULL(old_smart_factors);

#ifdef KIMERA_USE_CBS
  if (useCbsOptimizerHeart()) {
    // Intuition: CBS may inject extra factors internally, so index-based remap is unsafe; remap by pointer identity instead.
    CHECK(cbs_optimizer_) << "CBS optimizer flag is ON but cbs_optimizer_ is null.";
    if (!cbs_has_last_update_result_) {
      VLOG(2) << "CBS smart-factor slot update skipped: no cached CBS update result yet.";
      return;
    }

    const gtsam::NonlinearFactorGraph& factor_graph =
        cbs_optimizer_->getFactorsUnsafe();

    for (const LandmarkId& lmk_id : lmk_ids_of_new_smart_factors) {
      const auto it = old_smart_factors->find(lmk_id);
      DCHECK(it != old_smart_factors->end())
          << "Trying to access unavailable factor.";

      const SmartStereoFactor* target_factor = it->second.first.get();
      DCHECK(target_factor);

      Slot found_slot = -1;
      for (size_t slot = 0u; slot < factor_graph.size(); ++slot) {
        if (!factor_graph.exists(slot)) {
          continue;
        }
        if (factor_graph.at(slot).get() == target_factor) {
          found_slot = static_cast<Slot>(slot);
          break;
        }
      }

      if (found_slot < 0) {
        LOG(WARNING) << "CBS smart-factor slot remap failed for lmk id: "
                     << lmk_id;
        continue;
      }

      it->second.second = found_slot;
    }
    return;
  }
#endif

  // Intuition: keep original fast index-based remap for legacy smoother path.
  const gtsam::ISAM2Result& result = smoother_->getISAM2Result();
  const gtsam::NonlinearFactorGraph& factor_graph = smoother_->getFactors();
  const size_t expected_new_smart_factors = lmk_ids_of_new_smart_factors.size();
  const size_t inserted_new_factors = result.newFactorsIndices.size();
  const size_t matched_factors =
      std::min(expected_new_smart_factors, inserted_new_factors);

  if (inserted_new_factors < expected_new_smart_factors) {
    LOG(WARNING) << "Smart-factor slot remap: expected "
                 << expected_new_smart_factors
                 << " new smart factors but smoother reported only "
                 << inserted_new_factors
                 << " new factor slots. This can happen after backend recovery; "
                    "dropping unmatched smart-factor bookkeeping entries.";
  }

  for (size_t i = 0u; i < matched_factors; ++i) {
    const LandmarkId lmk_id = lmk_ids_of_new_smart_factors.at(i);
    const size_t slot = result.newFactorsIndices.at(i);

    auto it = old_smart_factors->find(lmk_id);
    if (it == old_smart_factors->end()) {
      LOG(WARNING) << "Smart-factor slot remap: missing bookkeeping entry for "
                   << "lmk id " << lmk_id << ".";
      continue;
    }

    if (slot >= factor_graph.size() || !factor_graph.exists(slot)) {
      LOG(WARNING) << "Smart-factor slot remap: invalid slot " << slot
                   << " for lmk id " << lmk_id
                   << ". Dropping stale bookkeeping entry.";
      old_smart_factors->erase(it);
      continue;
    }

    const auto sptr = dynamic_cast<const SmartStereoFactor*>(
        factor_graph.at(slot).get());
    if (!sptr) {
      LOG(WARNING) << "Smart-factor slot remap: slot " << slot
                   << " is not a SmartStereoFactor for lmk id " << lmk_id
                   << ". Dropping stale bookkeeping entry.";
      old_smart_factors->erase(it);
      continue;
    }

    if (it->second.first.get() != sptr) {
      LOG(WARNING) << "Smart-factor slot remap mismatch for lmk id " << lmk_id
                   << ": expected factor pointer " << it->second.first.get()
                   << " but slot " << slot << " points to " << sptr
                   << ". Dropping stale bookkeeping entry.";
      old_smart_factors->erase(it);
      continue;
    }

    it->second.second = static_cast<Slot>(slot);
  }

  for (size_t i = matched_factors; i < expected_new_smart_factors; ++i) {
    const LandmarkId lmk_id = lmk_ids_of_new_smart_factors.at(i);
    auto it = old_smart_factors->find(lmk_id);
    if (it != old_smart_factors->end()) {
      old_smart_factors->erase(it);
    }
  }
}


void VioBackend::setFactorsParams(
    const BackendParams& vio_params,
    gtsam::SharedNoiseModel* smart_noise,
    gtsam::SmartStereoProjectionParams* smart_factors_params,
    gtsam::SharedNoiseModel* no_motion_prior_noise,
    gtsam::SharedNoiseModel* zero_velocity_prior_noise,
    gtsam::SharedNoiseModel* constant_velocity_prior_noise) {
  CHECK_NOTNULL(smart_noise);
  CHECK_NOTNULL(smart_factors_params);
  CHECK_NOTNULL(no_motion_prior_noise);
  CHECK_NOTNULL(zero_velocity_prior_noise);
  CHECK_NOTNULL(constant_velocity_prior_noise);
  setSmartStereoFactorsNoiseModel(vio_params.smartNoiseSigma_, smart_noise);
  setSmartStereoFactorsParams(vio_params.rankTolerance_,
                              vio_params.landmarkDistanceThreshold_,
                              vio_params.retriangulationThreshold_,
                              vio_params.outlierRejection_,
                              smart_factors_params);

  setNoMotionFactorsParams(vio_params.no_motion_position_precision_,
                           vio_params.no_motion_rotation_precision_,
                           no_motion_prior_noise);

  // Zero velocity factors settings
  gtsam::Vector3 zero_velocity_precisions;
  zero_velocity_precisions.setConstant(vio_params.zero_velocity_precision_);
  *zero_velocity_prior_noise =
      gtsam::noiseModel::Diagonal::Precisions(zero_velocity_precisions);

  // Constant velocity factors settings
  gtsam::Vector3 constant_velocity_precisions;
  constant_velocity_precisions.setConstant(vio_params.constant_vel_precision_);
  *constant_velocity_prior_noise =
      gtsam::noiseModel::Diagonal::Precisions(constant_velocity_precisions);
}

void VioBackend::setSmartStereoFactorsNoiseModel(
    const double& smart_noise_sigma,
    gtsam::SharedNoiseModel* smart_noise) {
  CHECK_NOTNULL(smart_noise);
  // smart_noise_ = gtsam::noiseModel::Robust::Create(
  //                  gtsam::noiseModel::mEstimator::Huber::Create(1.345),
  //                  model);
  // vio_smart_reprojection_err_thresh / cam_->fx());
  *smart_noise = gtsam::noiseModel::Isotropic::Sigma(3, smart_noise_sigma);
}

void VioBackend::setSmartStereoFactorsParams(
    const double& rank_tolerance,
    const double& landmark_distance_threshold,
    const double& retriangulation_threshold,
    const double& outlier_rejection,
    gtsam::SmartStereoProjectionParams* smart_factors_params) {
  CHECK_NOTNULL(smart_factors_params);
  *smart_factors_params = gtsam::SmartStereoProjectionParams();
  smart_factors_params->setRankTolerance(rank_tolerance);
  smart_factors_params->setLandmarkDistanceThreshold(
      landmark_distance_threshold);
  smart_factors_params->setRetriangulationThreshold(retriangulation_threshold);
  smart_factors_params->setDynamicOutlierRejectionThreshold(outlier_rejection);
  //! EPI: If set to true, will refine triangulation using LM.
  smart_factors_params->setEnableEPI(false);
  smart_factors_params->setLinearizationMode(gtsam::HESSIAN);
  smart_factors_params->setDegeneracyMode(gtsam::ZERO_ON_DEGENERACY);
  smart_factors_params->throwCheirality = false;
  smart_factors_params->verboseCheirality = false;
}

void VioBackend::setNoMotionFactorsParams(
    const double& position_precision,
    const double& rotation_precision,
    gtsam::SharedNoiseModel* no_motion_prior_noise) {
  CHECK_NOTNULL(no_motion_prior_noise);
  gtsam::Vector6 precisions;
  precisions.head<3>().setConstant(rotation_precision);
  precisions.tail<3>().setConstant(position_precision);
  *no_motion_prior_noise = gtsam::noiseModel::Diagonal::Precisions(precisions);
}

void VioBackend::print() const {
  backend_params_.print();

  smoother_->params().print(std::string(10, '.') + "** ISAM2 Parameters **" +
                            std::string(10, '.'));

  LOG(INFO) << "Used stereo calibration in Backend: ";
  if (FLAGS_minloglevel < 1) {
    stereo_cal_->print("\n stereoCal_\n");
  }

  LOG(INFO) << "** Backend Initial Members: \n"
            << "B_Pose_leftCam_: " << B_Pose_leftCamRect_ << '\n'
            << "W_Pose_B_lkf_from_state_: " << W_Pose_B_lkf_from_state_ << '\n'
            << "W_Pose_B_lkf_from_increments_: "
            << W_Pose_B_lkf_from_increments_ << '\n'
            << "W_Vel_B_lkf_ (transpose): " << W_Vel_B_lkf_.transpose() << '\n'
            << "imu_bias_lkf_" << imu_bias_lkf_ << '\n'
            << "imu_bias_prev_kf_" << imu_bias_prev_kf_ << '\n'
            << "last_id_ " << last_kf_id_ << '\n'
            << "cur_id_ " << curr_kf_id_ << '\n'
            << "landmark_count_ " << landmark_count_;
}

void VioBackend::printFeatureTracks() const {
  LOG(INFO) << "---- Feature tracks: --------- ";
  for (const auto& keyTrack_j : feature_tracks_) {
    LOG(INFO) << "Landmark " << keyTrack_j.first << " having ";
    keyTrack_j.second.print();
  }
}

void VioBackend::printSmootherInfo(
    const gtsam::NonlinearFactorGraph& new_factors_tmp,
    const gtsam::FactorIndices& delete_slots,
    const std::string& message,
    const bool& showDetails) const {
  LOG(INFO) << " =============== START:" << message << " =============== ";

  // const std::string* which_graph = nullptr;
  // const gtsam::NonlinearFactorGraph* graph = nullptr;
  // // Pick the graph that makes more sense:
  // // This is code is mostly run post update, when it throws exception,
  // // shouldn't we print the graph before optimization instead?
  // // Yes if available, but if not, then just ask the smoother.
  // static const std::string graph_before_opt = "(graph before optimization)";
  // static const std::string smoother_get_factors = "(smoother getFactors)";
  // if (debug_info_.graphBeforeOpt.size() != 0) {
  //   which_graph = &graph_before_opt;
  //   graph = &(debug_info_.graphBeforeOpt);
  // } else {
  //   which_graph = &smoother_get_factors;
  //   graph = &(smoother_->getFactors());
  // }
  // CHECK_NOTNULL(which_graph);
  // CHECK_NOTNULL(graph); (zy cancelled it)

  // zy Step 25a
  const std::string* which_graph = nullptr;
  const gtsam::NonlinearFactorGraph* graph = nullptr;
  // Pick the graph that makes more sense:
  // This code is mostly run post update, when it throws exception.
  static const std::string graph_before_opt = "(graph before optimization)";
  static const std::string smoother_get_factors = "(smoother getFactors)";
  static const std::string cbs_get_factors = "(cbs getFactorsUnsafe)";

  if (debug_info_.graphBeforeOpt.size() != 0) {
    which_graph = &graph_before_opt;
    graph = &(debug_info_.graphBeforeOpt);
  } else {
#ifdef KIMERA_USE_CBS
    if (useCbsOptimizerHeart()) {
      // Intuition: when CBS is active, debug dumps must reflect CBS graph, not stale smoother graph.
      CHECK(cbs_optimizer_) << "CBS optimizer flag is ON but cbs_optimizer_ is null.";
      which_graph = &cbs_get_factors;
      graph = &(cbs_optimizer_->getFactorsUnsafe());
    } else
#endif
    {
      which_graph = &smoother_get_factors;
      graph = &(smoother_->getFactors());
    }
  }
  CHECK_NOTNULL(which_graph);
  CHECK_NOTNULL(graph);


  static constexpr bool print_smart_factors = true;  // There a lot of these!
  static constexpr bool print_point_plane_factors = true;
  static constexpr bool print_plane_priors = true;
  static constexpr bool print_point_priors = true;
  static constexpr bool print_linear_container_factors = true;
  ////////////////////// Print all factors.
  ///////////////////////////////////////
  LOG(INFO) << "Nr of factors in graph " + *which_graph << ": " << graph->size()
            << ", with factors:" << std::endl;
  LOG(INFO) << "[\n";
  printSelectedGraph(*graph,
                     print_smart_factors,
                     print_point_plane_factors,
                     print_plane_priors,
                     print_point_priors,
                     print_linear_container_factors);
  LOG(INFO) << " ]" << std::endl;

  ///////////// Print factors that were newly added to the optimization.//////
  LOG(INFO) << "Nr of new factors to add: " << new_factors_tmp.size()
            << " with factors:" << std::endl;
  LOG(INFO) << "[\n (slot # wrt to new_factors_tmp graph) \t";
  printSelectedGraph(new_factors_tmp,
                     print_smart_factors,
                     print_point_plane_factors,
                     print_plane_priors,
                     print_point_priors,
                     print_linear_container_factors);
  LOG(INFO) << " ]" << std::endl;

  ////////////////////////////// Print deleted /// slots.///////////////////////
  LOG(INFO) << "Nr deleted slots: " << delete_slots.size()
            << ", with slots:" << std::endl;
  LOG(INFO) << "[\n\t";
  std::stringstream ss;
  if (debug_info_.graphToBeDeleted.size() != 0) {
    // If we are storing the graph to be deleted, then print extended info
    // besides the slot to be deleted.
    CHECK_GE(debug_info_.graphToBeDeleted.size(), delete_slots.size());
    for (size_t i = 0u; i < delete_slots.size(); i++) {
      CHECK(debug_info_.graphToBeDeleted.at(i));
      if (print_point_plane_factors) {
        printSelectedFactors(debug_info_.graphToBeDeleted.at(i).get(),
                             delete_slots.at(i),
                             false,
                             print_point_plane_factors,
                             false,
                             false,
                             false);
      } else {
        ss << "\tSlot # " << delete_slots.at(i) << ":";
        ss << "\t";
        debug_info_.graphToBeDeleted.at(i)->printKeys();
      }
    }
  } else {
    for (size_t i = 0; i < delete_slots.size(); ++i) {
      ss << delete_slots.at(i) << " ";
    }
  }
  LOG(INFO) << ss.str();
  LOG(INFO) << " ]" << std::endl;

  //////////////////////// Print all values in state. ////////////////////////
  LOG(INFO) << "Nr of values in state_ : " << state_.size() << ", with keys:";
  std::stringstream state_ss;
  state_ss << "[\n\t";
  for (const auto& key_value : state_) {
    state_ss << gtsam::DefaultKeyFormatter(key_value.key) << " ";
  }
  LOG(INFO) << state_ss.str();
  LOG(INFO) << " ]";

  // Print only new values.
  LOG(INFO) << "Nr values in new_values_ : " << new_values_.size()
            << ", with keys:";
  std::stringstream new_values_ss;
  new_values_ss << "[\n\t";
  for (const auto& key_value : new_values_) {
    new_values_ss << " " << gtsam::DefaultKeyFormatter(key_value.key) << " ";
  }
  LOG(INFO) << new_values_ss.str();
  LOG(INFO) << " ]";

  if (showDetails) {
    graph->print("isam2 graph:\n");
    new_factors_tmp.print("new_factors_tmp:\n");
    new_values_.print("new values:\n");
    // LOG(INFO) << "new_smart_factors_: "  << std::endl;
    // for (auto& s : new_smart_factors_)
    //	s.second->print();
  }

  LOG(INFO) << " =============== END: " << message << " =============== ";
}

template <typename T>
void printFactorIfValid(const gtsam::NonlinearFactor* factor, size_t slot) {
  const auto derived = dynamic_cast<const T*>(factor);
  if (derived) {
    std::cout << "\tSlot # " << slot << ": "
              << FactorFormatter::format(*derived) << "\n";
  }
}

void VioBackend::printSelectedFactors(
    const gtsam::NonlinearFactor* factor,
    const size_t& slot,
    const bool print_smart_factors,
    const bool print_point_plane_factors,
    const bool print_plane_priors,
    const bool print_point_priors,
    const bool print_linear_container_factors) const {
  if (!factor) {
    return;
  }

  if (print_smart_factors) {
    printFactorIfValid<SmartStereoFactor>(factor, slot);
  }

  if (print_point_plane_factors) {
    printFactorIfValid<gtsam::PointPlaneFactor>(factor, slot);
  }

  if (print_plane_priors) {
    printFactorIfValid<gtsam::PriorFactor<gtsam::OrientedPlane3>>(factor, slot);
  }

  if (print_point_priors) {
    printFactorIfValid<gtsam::PriorFactor<gtsam::Point3>>(factor, slot);
  }

  if (print_linear_container_factors) {
    printFactorIfValid<gtsam::LinearContainerFactor>(factor, slot);
  }
}

void VioBackend::printSelectedGraph(
    const gtsam::NonlinearFactorGraph& graph,
    const bool& print_smart_factors,
    const bool& print_point_plane_factors,
    const bool& print_plane_priors,
    const bool& print_point_priors,
    const bool& print_linear_container_factors) const {
  size_t slot = 0;
  for (const auto& g : graph) {
    printSelectedFactors(g.get(),
                         slot,
                         print_smart_factors,
                         print_point_plane_factors,
                         print_plane_priors,
                         print_point_priors,
                         print_linear_container_factors);
    slot++;
  }
  std::cout << std::endl;
}

/* -------------------------------------------------------------------------- */
void VioBackend::computeSmartFactorStatistics() {
  // Compute number of valid/degenerate
  debug_info_.resetSmartFactorsStatistics();
  // gtsam::NonlinearFactorGraph graph = smoother_->getFactors(); (zy cancelled it)
  // zy Step 25b: smart-factor stats must be computed from the currently active optimizer graph.
  const gtsam::NonlinearFactorGraph* active_factor_graph = nullptr;
#ifdef KIMERA_USE_CBS
  if (useCbsOptimizerHeart()) {
    CHECK(cbs_optimizer_) << "CBS optimizer flag is ON but cbs_optimizer_ is null.";
    active_factor_graph = &cbs_optimizer_->getFactorsUnsafe();
  } else
#endif
  {
    CHECK(smoother_);
    active_factor_graph = &smoother_->getFactors();
  }
  CHECK_NOTNULL(active_factor_graph);
  const gtsam::NonlinearFactorGraph& graph = *active_factor_graph;

  for (const auto& g : graph) {
    if (g) {
      const auto gsf = dynamic_cast<const SmartStereoFactor*>(g.get());
      if (gsf) {
        debug_info_.numSF_ += 1;

        // Check for consecutive Keys: this check is wrong: if there is
        // LOW_DISPARITY at some frame, we do not add the measurement to the
        // smart factor, hence keys are not necessarily consecutive
        // auto keys = g->keys();
        // Key last_key;
        // bool first_key = true;
        // for (Key key : keys)
        //{
        //  if (!first_key && key - last_key != 1){
        //    std::cout << " Last: " << gtsam::DefaultKeyFormatter(last_key)
        //    << " Current: " << gtsam::DefaultKeyFormatter(key) <<
        //    std::endl; for (Key k : keys){ std::cout << " " <<
        //    gtsam::DefaultKeyFormatter(k)
        //    << " "; } throw std::runtime_error("\n
        //    computeSmartFactorStatistics: found nonconsecutive keys in
        //    smart factors \n");
        //  }
        //  last_key = key;
        //  first_key = false;
        //}

        // Check SF status
        const gtsam::TriangulationResult& result = gsf->point();
        if (result) {
          if (result.valid()) {
            debug_info_.numValid_ += 1;
            // Check track length
            size_t trackLength = gsf->keys().size();
            if (trackLength > debug_info_.maxTrackLength_) {
              debug_info_.maxTrackLength_ = trackLength;
            }
            debug_info_.meanTrackLength_ += trackLength;
          }
        } else {
          VLOG(5) << "Triangulation result is not initialized...";
          if (result.degenerate()) debug_info_.numDegenerate_ += 1;
          if (result.farPoint()) debug_info_.numFarPoints_ += 1;
          if (result.outlier()) debug_info_.numOutliers_ += 1;
          if (result.behindCamera()) debug_info_.numCheirality_ += 1;
          debug_info_.numNonInitialized_ += 1;
        }
      }
    }
  }
  if (debug_info_.numValid_ > 0) {
    debug_info_.meanTrackLength_ = debug_info_.meanTrackLength_ /
                                   static_cast<double>(debug_info_.numValid_);
  } else {
    debug_info_.meanTrackLength_ = 0;
  }
}

void VioBackend::computeSparsityStatistics() {
  // gtsam::NonlinearFactorGraph graph = smoother_->getFactors();
  // zy Step 25c: sparsity/hessian diagnostics must use the same graph that produced the current estimate.
  const gtsam::NonlinearFactorGraph* active_factor_graph = nullptr;
#ifdef KIMERA_USE_CBS
  if (useCbsOptimizerHeart()) {
    CHECK(cbs_optimizer_) << "CBS optimizer flag is ON but cbs_optimizer_ is null.";
    active_factor_graph = &cbs_optimizer_->getFactorsUnsafe();
  } else
#endif
  {
    CHECK(smoother_);
    active_factor_graph = &smoother_->getFactors();
  }
  CHECK_NOTNULL(active_factor_graph);
  const gtsam::NonlinearFactorGraph& graph = *active_factor_graph;

  gtsam::GaussianFactorGraph::shared_ptr gfg = graph.linearize(state_);
  gtsam::Matrix Hessian = gfg->hessian().first;
  debug_info_.nrElementsInMatrix_ = Hessian.rows() * Hessian.cols();
  debug_info_.nrZeroElementsInMatrix_ = 0;
  for (int i = 0; i < Hessian.rows(); ++i) {
    for (int j = 0; j < Hessian.cols(); ++j) {
      if (std::fabs(Hessian(i, j)) < 1e-15) {
        debug_info_.nrZeroElementsInMatrix_ += 1;
      }
    }
  }

  CHECK_EQ(Hessian.rows(), Hessian.cols())
      << "computeSparsityStatistics: hessian is not a square matrix?";

  VLOG(10) << "Hessian stats: ===========\n"
           << "rows: " << Hessian.rows() << '\n'
           << "nrElementsInMatrix_: " << debug_info_.nrElementsInMatrix_ << '\n'
           << "nrZeroElementsInMatrix_: "
           << debug_info_.nrZeroElementsInMatrix_;
}

// Debugging post optimization and estimate calculation.
void VioBackend::postDebug(
    const std::chrono::high_resolution_clock::time_point& total_start_time,
    const std::chrono::high_resolution_clock::time_point& start_time) {
  if (log_output_) {
    computeSparsityStatistics();
    computeSmartFactorStatistics();
  }

  if (VLOG_IS_ON(10)) {
    // Print old_smart_factors_
    LOG(INFO) << "Landmarks in old_smart_factors_: "
              << old_smart_factors_.size();
    for (const auto& it : old_smart_factors_) {
      LOG(INFO) << " - Landmark " << it.first << " with slot "
                << it.second.second;
    }

    // Print debug_info_
    debug_info_.print();

    // Print times.
    debug_info_.printTimes();

    // Sanity check timings
    const auto& end_time =
        utils::Timer::toc<std::chrono::seconds>(total_start_time).count();
    const auto& end_time_from_sum = debug_info_.sumAllTimes();
    LOG_IF(ERROR, end_time != end_time_from_sum)
        << "Optimize: time measurement mismatch."
           "The sum of the parts is not equal to the total.";

    // Print error.
    // gtsam::NonlinearFactorGraph graph = gtsam::NonlinearFactorGraph(
    //     smoother_->getFactors());  // clone, expensive but safer! (zy cancelled it)
    // zy Step 25c: error-before/after debug must compare against the active optimizer graph (CBS or legacy).
    const gtsam::NonlinearFactorGraph* active_factor_graph = nullptr;
#ifdef KIMERA_USE_CBS
    if (useCbsOptimizerHeart()) {
      CHECK(cbs_optimizer_) << "CBS optimizer flag is ON but cbs_optimizer_ is null.";
      active_factor_graph = &cbs_optimizer_->getFactorsUnsafe();
    } else
#endif
    {
      CHECK(smoother_);
      active_factor_graph = &smoother_->getFactors();
    }
    CHECK_NOTNULL(active_factor_graph);
    gtsam::NonlinearFactorGraph graph = gtsam::NonlinearFactorGraph(
        *active_factor_graph);  // clone, expensive but safer!

    VLOG(10) << "Optimization Errors:\n"
             << " - Error before :" << graph.error(debug_info_.stateBeforeOpt)
             << '\n'
             << " - Error after  :" << graph.error(state_);
  }
}

// Reset state of debug info.
void VioBackend::resetDebugInfo(DebugVioInfo* debug_info) {
  CHECK_NOTNULL(debug_info);
  debug_info->resetSmartFactorsStatistics();
  debug_info->resetTimes();
  debug_info->resetAddedFactorsStatistics();
  debug_info->nrElementsInMatrix_ = 0;
  debug_info->nrZeroElementsInMatrix_ = 0;
}

void VioBackend::cleanNullPtrsFromGraph(
    gtsam::NonlinearFactorGraph* new_imu_prior_and_other_factors) {
  CHECK_NOTNULL(new_imu_prior_and_other_factors);
  gtsam::NonlinearFactorGraph tmp_graph = *new_imu_prior_and_other_factors;
  new_imu_prior_and_other_factors->resize(0);
  for (const auto& factor : tmp_graph) {
    if (factor != nullptr) {
      new_imu_prior_and_other_factors->push_back(factor);
    }
  }
}

void VioBackend::deleteAllFactorsWithKeyFromFactorGraph(
    const gtsam::Key& key,
    const gtsam::NonlinearFactorGraph& factor_graph,
    gtsam::NonlinearFactorGraph* factor_graph_output) {
  CHECK_NOTNULL(factor_graph_output);
  size_t new_factors_slot = 0;
  *factor_graph_output = factor_graph;
  for (auto it = factor_graph_output->begin();
       it != factor_graph_output->end();) {
    if (*it) {
      if ((*it)->find(key) != (*it)->end()) {
        // We found our lmk in the list of keys of the factor.
        // Sanity check, this lmk has no priors right?
        CHECK(
            !dynamic_cast<const gtsam::PriorFactor<gtsam::Point3>*>(it->get()));
        // We are not deleting a smart factor right?
        // Otherwise we need to update structure:
        // lmk_ids_of_new_smart_factors...
        CHECK(!dynamic_cast<const SmartStereoFactor*>(it->get()));
        // Whatever factor this is, it has our lmk...
        // Delete it.
        LOG(WARNING) << "Delete factor in new_factors at slot # "
                     << new_factors_slot << " of new_factors graph.";
        it = factor_graph_output->erase(it);
      } else {
        it++;
      }
    } else {
      LOG(ERROR) << "*it, which is itself a pointer, is null.";
      it++;
    }
    new_factors_slot++;
  }
}

// Returns if the key in timestamps could be removed or not.
bool VioBackend::deleteKeyFromTimestamps(
    const gtsam::Key& key,
    const std::map<Key, double>& timestamps,
    std::map<Key, double>* timestamps_output) {
  CHECK_NOTNULL(timestamps_output);
  *timestamps_output = timestamps;
  if (timestamps_output->find(key) != timestamps_output->end()) {
    timestamps_output->erase(key);
    return true;
  }
  return false;
}

// Returns if the key in timestamps could be removed or not.
bool VioBackend::deleteKeyFromValues(const gtsam::Key& key,
                                     const gtsam::Values& values,
                                     gtsam::Values* values_output) {
  CHECK_NOTNULL(values_output);
  *values_output = values;
  if (values.find(key) != values.end()) {
    // We found the lmk in new values, delete it.
    LOG(WARNING) << "Delete value in new_values for key "
                 << gtsam::DefaultKeyFormatter(key);
    CHECK(values_output->find(key) != values_output->end());
    try {
      values_output->erase(key);
    } catch (const gtsam::ValuesKeyDoesNotExist& e) {
      LOG(FATAL) << e.what();
    } catch (...) {
      LOG(FATAL) << "Unhandled exception when erasing key"
                    " in new_values_cheirality";
    }
    return true;
  }
  return false;
}

// Returns if the key in timestamps could be removed or not.
void VioBackend::findSlotsOfFactorsWithKey(
    const gtsam::Key& key,
    const gtsam::NonlinearFactorGraph& graph,
    std::vector<size_t>* slots_of_factors_with_key) {
  CHECK_NOTNULL(slots_of_factors_with_key);
  slots_of_factors_with_key->resize(0);
  size_t slot = 0;
  for (const auto& g : graph) {
    if (g) {
      // Found a valid factor.
      if (g->find(key) != g->end()) {
        // Whatever factor this is, it has our lmk...
        // Sanity check, this lmk has no priors right?
        CHECK(!dynamic_cast<const gtsam::LinearContainerFactor*>(g.get()));
        CHECK(!dynamic_cast<const gtsam::PriorFactor<gtsam::Point3>*>(g.get()));
        // Sanity check that we are not deleting a smart factor.
        CHECK(!dynamic_cast<const SmartStereoFactor*>(g.get()));
        // Delete it.
        LOG(WARNING) << "Delete factor in graph at slot # " << slot
                     << " corresponding to lmk with id: "
                     << gtsam::Symbol(key).index();
        CHECK(graph.exists(slot));
        slots_of_factors_with_key->push_back(slot);
      }
    }
    slot++;
  }
}

// Returns if the key in feature tracks could be removed or not.
bool VioBackend::deleteLmkFromFeatureTracks(const LandmarkId& lmk_id) {
  if (feature_tracks_.find(lmk_id) != feature_tracks_.end()) {
    VLOG(2) << "Deleting feature track for lmk with id: " << lmk_id;
    feature_tracks_.erase(lmk_id);
    return true;
  }
  return false;
}

}  // namespace VIO.
