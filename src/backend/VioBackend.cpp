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
#include <map>
#include <string>
#include <utility>  // for make_pair
#include <unordered_set>
#include <vector>
#include <cmath> // zy step 5_c



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
// Default to legacy fixed-lag backend unless explicitly enabling CBS at runtime.
DEFINE_bool(use_cbs_optimizer,
            false,
            "If true (and compiled with KIMERA_USE_CBS), enable CBS belief "
            "exchange (belief validation/merging) while keeping Kimera fixed-lag "
            "smoothing as the optimization heart.");
DEFINE_bool(cbs_replace_fixed_lag_optimizer,
            false,
            "Deprecated: retained for compatibility. Kimera fixed-lag remains "
            "the optimization heart in CBS mode.");
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
#endif

namespace {
inline bool useCbsBeliefExchange() {
#ifdef KIMERA_USE_CBS
  return FLAGS_use_cbs_optimizer;
#else
  return false;
#endif
}

inline bool useCbsOptimizerHeart() {
#ifdef KIMERA_USE_CBS
  // zy Step 41a
  // Keep Kimera fixed-lag as the single optimization heart in CBS mode.
  // CBS remains enabled only for inter-agent belief exchange/validation.
  return false;
#else
  return false;
#endif
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
  if (FLAGS_use_cbs_optimizer && FLAGS_cbs_replace_fixed_lag_optimizer) {
    // zy Step 41b
    // Guardrail: CBS belief exchange is enabled, but Kimera keeps fixed-lag heart.
    LOG(WARNING) << "cbs_replace_fixed_lag_optimizer=true is deprecated. "
                 << "Forcing fixed-lag optimization heart in Kimera.";
    FLAGS_cbs_replace_fixed_lag_optimizer = false;
  }

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
            << (useCbsOptimizerHeart() ? "true" : "false");
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

    if (external_pose_belief_callback_) {
      ExternalPoseBelief belief;
      if (getLatestExternalPoseBelief(&belief)) {
        external_pose_belief_callback_(belief);
      } else {
        VLOG(2) << "External pose belief callback registered, but no valid "
                   "belief is available this cycle.";
      }
    }

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



/* -------------------------------------------------------------------------- */
bool VioBackend::initStateAndSetPriors(
    const VioNavStateTimestamped& vio_nav_state_initial_seed) {
  // Clean state
  new_values_.clear();

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

  const auto& old_factor = old_smart_factors_it->second.first;
  // Clone old factor to keep all previous measurements, now append one.
  SmartStereoFactor::shared_ptr new_factor(new SmartStereoFactor(*old_factor));

  const gtsam::Symbol pose_symbol(kPoseSymbolChar, new_measurement.first);
  const StereoPoint2& measurement = new_measurement.second;
  new_factor->add(measurement, pose_symbol, stereo_cal_);

  // Update the factor
  Slot slot = old_smart_factors_it->second.second;
  if (slot != -1) {
    new_smart_factors_[lmk_id] = new_factor;
  } else {
    // Factor not yet inserted in the graph: keep the queued version updated.
    LOG(WARNING) << "updateLandmarkInGraph: slot == -1 for landmark "
                 << lmk_id << ". Updating queued smart factor.";
    new_smart_factors_[lmk_id] = new_factor;
  }
  old_smart_factors_it->second.first = new_factor;
  VLOG(10) << "updateLandmarkInGraph: added observation to point: " << lmk_id;
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
      const gtsam::Matrix pose_cov = cbs_optimizer_->marginalCovariance(pose_key);
      if (pose_cov.rows() >= 6 && pose_cov.cols() >= 6 && pose_cov.allFinite()) {
        state_covariance_lkf_.block(0, 0, 6, 6) = pose_cov.topLeftCorner(6, 6);
      } else {
        VLOG(2) << "Invalid CBS pose covariance for key: " << pose_key;
      }
    } else {
      VLOG(2) << "CBS pose key not found for covariance: " << pose_key;
    }

    if (cbs_optimizer_->valueExists(vel_key)) {
      const gtsam::Matrix vel_cov = cbs_optimizer_->marginalCovariance(vel_key);
      if (vel_cov.rows() >= 3 && vel_cov.cols() >= 3 && vel_cov.allFinite()) {
        state_covariance_lkf_.block(6, 6, 3, 3) = vel_cov.topLeftCorner(3, 3);
      } else {
        VLOG(2) << "Invalid CBS velocity covariance for key: " << vel_key;
      }
    } else {
      VLOG(2) << "CBS velocity key not found for covariance: " << vel_key;
    }

    if (cbs_optimizer_->valueExists(bias_key)) {
      const gtsam::Matrix bias_cov = cbs_optimizer_->marginalCovariance(bias_key);
      if (bias_cov.rows() >= 6 && bias_cov.cols() >= 6 && bias_cov.allFinite()) {
        state_covariance_lkf_.block(9, 9, 6, 6) = bias_cov.topLeftCorner(6, 6);
      } else {
        VLOG(2) << "Invalid CBS bias covariance for key: " << bias_key;
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


// zy Step 7_b
// When CBS is driving estimates, outgoing belief should carry CBS covariance, not legacy smoother covariance.
// Fallbacks keep behavior robust when CBS marginals are temporarily unavailable.
bool VioBackend::getLatestExternalPoseBelief(
    ExternalPoseBelief* belief) const {
  CHECK_NOTNULL(belief);

  if (backend_state_ == BackendState::Bootstrap) {
    return false;
  }

  belief->timestamp_kf_nsec_ = timestamp_lkf_;
  belief->frame_id_ = curr_kf_id_;
  belief->W_Pose_B_ = W_Pose_B_lkf_from_state_;

  bool covariance_set = false;
  const gtsam::Symbol pose_symbol(kPoseSymbolChar, curr_kf_id_);

#ifdef KIMERA_USE_CBS
  // In CBS mode, export covariance from BPSAM marginals so shared beliefs reflect the CBS state.
  if (useCbsOptimizerHeart()) {
    CHECK(cbs_optimizer_) << "CBS optimizer flag is ON but cbs_optimizer_ is null.";
    bool local_marginals_active = false;
    try {
      // Match CBS pose-sharing stage: export pose covariance from LOCAL
      // marginalization (exclude belief factors from covariance computation).
      cbs_optimizer_->setMarginalizationGraph(
          cbs::BPSAM::MarginalizationType::LOCAL);
      local_marginals_active = true;

      if (cbs_optimizer_->valueExists(pose_symbol)) {
        const gtsam::Matrix cov = cbs_optimizer_->marginalCovariance(pose_symbol);
        if (cov.rows() >= 6 && cov.cols() >= 6 && cov.allFinite()) {
          belief->covariance_ = cov.topLeftCorner(6, 6);
          covariance_set = true;
        } else {
          VLOG(2) << "CBS covariance unavailable/invalid for pose key: "
                  << pose_symbol;
        }
      } else {
        VLOG(2) << "CBS value not found for pose key: " << pose_symbol;
      }
    } catch (const std::exception& e) {
      VLOG(2) << "CBS marginal covariance query failed: " << e.what();
    }

    if (local_marginals_active) {
      try {
        cbs_optimizer_->setMarginalizationGraph(
            cbs::BPSAM::MarginalizationType::FULL);
      } catch (const std::exception& e) {
        VLOG(2) << "Failed to restore CBS FULL marginalization graph: "
                << e.what();
      } catch (...) {
        VLOG(2) << "Failed to restore CBS FULL marginalization graph.";
      }
    }
  }
#endif

  // Optional safe path: only use backend covariance when it has been explicitly
  // computed and validated.
  if (!covariance_set) {
    if (FLAGS_external_pose_belief_safe_covariance_fallback) {
      if (state_covariance_lkf_valid_ && state_covariance_lkf_.rows() >= 6 &&
          state_covariance_lkf_.cols() >= 6) {
        const gtsam::Matrix66 pose_cov =
            state_covariance_lkf_.topLeftCorner<6, 6>();
        if (pose_cov.allFinite()) {
          belief->covariance_ = pose_cov;
          covariance_set = true;
        }
      }
    } else {
      // Legacy path (preserve existing behavior outside explicitly tuned runs).
      if (state_covariance_lkf_.rows() >= 6 && state_covariance_lkf_.cols() >= 6) {
        belief->covariance_ = state_covariance_lkf_.topLeftCorner<6, 6>();
        covariance_set = true;
      }
    }
  }

  // Fallback when covariance is unavailable.
  if (!covariance_set) {
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
  // Intuition: in CBS mode, query arbitrary historical pose beliefs directly from BPSAM marginals.
  if (useCbsOptimizerHeart()) {
    CHECK(cbs_optimizer_) << "CBS optimizer flag is ON but cbs_optimizer_ is null.";

    if (!cbs_optimizer_->valueExists(pose_symbol)) {
      VLOG(2) << "CBS does not contain requested key: " << pose_symbol;
      return false;
    }

    try {
      bool local_marginals_active = false;
      cbs_optimizer_->setMarginalizationGraph(
          cbs::BPSAM::MarginalizationType::LOCAL);
      local_marginals_active = true;

      belief->timestamp_kf_nsec_ = matched_timestamp;
      belief->frame_id_ = matched_frame_id;
      belief->W_Pose_B_ =
          cbs_optimizer_->calculateEstimate<gtsam::Pose3>(pose_symbol);

      const gtsam::Matrix cov = cbs_optimizer_->marginalCovariance(pose_symbol);
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

      if (local_marginals_active) {
        cbs_optimizer_->setMarginalizationGraph(
            cbs::BPSAM::MarginalizationType::FULL);
      }
      return true;
    } catch (const std::exception& e) {
      try {
        cbs_optimizer_->setMarginalizationGraph(
            cbs::BPSAM::MarginalizationType::FULL);
      } catch (...) {
      }
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
  new_imu_prior_and_other_factors_.emplace_shared<gtsam::PriorFactor<gtsam::Pose3>>(
      gtsam::Symbol(kPoseSymbolChar, frame_id),
      W_Pose_B,
      noise_model);

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

  // Symmetrize to avoid tiny asymmetries from serialization / numeric noise.
  gtsam::Matrix6 cov = 0.5 * (covariance + covariance.transpose());

  // Keep covariance numerically well-conditioned.
  // Ordering is [rot, rot, rot, trans, trans, trans].
  constexpr double kMinRotVar = 1e-8;    // rad^2
  constexpr double kMinTransVar = 1e-8;  // m^2
  for (int i = 0; i < 6; ++i) {
    const double min_var = (i < 3) ? kMinRotVar : kMinTransVar;
    if (!std::isfinite(cov(i, i)) || cov(i, i) < min_var) {
      cov(i, i) = min_var;
    }
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

  size_t num_external_priors_injected = 0;
  size_t num_external_priors_deferred = 0;
  size_t num_external_priors_dropped_old = 0;
  size_t num_external_priors_dropped_marginalized = 0;
  size_t num_external_priors_dropped_inactive = 0;
  size_t num_external_priors_dropped_disabled_mode = 0;
  size_t num_external_priors_deferred_budget = 0;

  // zy Step 12a
  // In CBS mode, stage/query belief acceptance counters while keeping fixed-lag
  // as optimization heart.
  #ifdef KIMERA_USE_CBS
  size_t num_external_beliefs_staged = 0;
  size_t num_external_beliefs_rejected = 0;
  size_t num_external_beliefs_dropped_bad_noise = 0;
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

  // zy
  // In fixed-lag mode, prune timestamp->key map entries whose pose keys are no
  // longer active in the optimizer and capture the oldest still-active
  // timestamp. Incoming beliefs older than that are guaranteed to target
  // marginalized states and should be dropped early.
  Timestamp oldest_active_pose_timestamp = -1;
  Timestamp newest_active_pose_timestamp = -1;
  size_t num_timestamp_map_pruned = 0;
  {
    std::lock_guard<std::mutex> map_lock(timestamp_to_kf_id_map_mutex_);
    for (auto it = timestamp_to_kf_id_map_.begin();
         it != timestamp_to_kf_id_map_.end();) {
      const gtsam::Symbol pose_symbol(kPoseSymbolChar, it->second);
      bool pose_key_is_active = false;
#ifdef KIMERA_USE_CBS
      if (useCbsOptimizerHeart()) {
        CHECK(cbs_optimizer_)
            << "CBS optimizer flag is ON but cbs_optimizer_ is null.";
        pose_key_is_active = cbs_optimizer_->valueExists(pose_symbol) ||
                             new_values_.exists(pose_symbol);
      } else
#endif
      {
        pose_key_is_active =
            state_.exists(pose_symbol) || new_values_.exists(pose_symbol);
      }

      if (pose_key_is_active) {
        ++it;
      } else {
        it = timestamp_to_kf_id_map_.erase(it);
        ++num_timestamp_map_pruned;
      }
    }

    if (!timestamp_to_kf_id_map_.empty()) {
      oldest_active_pose_timestamp = timestamp_to_kf_id_map_.begin()->first;
      newest_active_pose_timestamp = timestamp_to_kf_id_map_.rbegin()->first;
    }
  }
  if (num_timestamp_map_pruned > 0) {
    VLOG(2) << "Pruned " << num_timestamp_map_pruned
            << " marginalized timestamp->key entries. oldest_active_ts[nsec]="
            << oldest_active_pose_timestamp;
  }

  {
    std::deque<ExternalPosePrior> remaining_queue;
    std::lock_guard<std::mutex> queue_lock(external_pose_priors_queue_mutex_);

    for (const auto& prior : external_pose_priors_queue_) {
      if (!cbs_exchange_active) {
        ++num_external_priors_dropped_disabled_mode;
        continue;
      }

      // Drop priors that are older than the oldest pose still active in the
      // optimizer window (fixed-lag behavior).
      if (oldest_active_pose_timestamp > 0 &&
          prior.timestamp_kf_nsec_ + external_prior_timestamp_tolerance_ns_ <
              oldest_active_pose_timestamp) {
        ++num_external_priors_dropped_marginalized;
        continue;
      }

      // Drop priors that are too old w.r.t current backend timestamp.
      if (prior.timestamp_kf_nsec_ + kMaxPriorAgeNs < timestamp_kf_nsec) {
        ++num_external_priors_dropped_old;
        continue;
      }

      // Keep priors that are too far in the future; they may match later frames.
      if (prior.timestamp_kf_nsec_ > timestamp_kf_nsec + kMaxFutureLeadNs) {
        remaining_queue.push_back(prior);
        ++num_external_priors_deferred;
        continue;
      }

      bool matched = false;
      FrameId matched_frame_id = -1;

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

      if (!matched) {
        // zy
        // Fixed-lag policy: if a prior cannot be matched to any active key now,
        // discard it instead of deferring indefinitely. Future priors were
        // already handled by the future-gate above.
        if (!cbs_heart_active) {
          ++num_external_priors_dropped_marginalized;
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
          continue;
        }
        remaining_queue.push_back(prior);
        ++num_external_priors_deferred;
        continue;
      }

      // Avoid overloading a single optimize() step.
      if (num_external_priors_injected >= kMaxExternalPriorsPerOptimize) {
        remaining_queue.push_back(prior);
        ++num_external_priors_deferred_budget;
        continue;
      }

      const gtsam::Symbol pose_symbol(kPoseSymbolChar, matched_frame_id);
      // zy Step 12b, edited the original one
      // zy Step 28a: key-availability must be checked against the optimizer that is actually active (CBS or legacy).
      bool pose_key_is_active = false;
#ifdef KIMERA_USE_CBS
      if (cbs_heart_active) {
        CHECK(cbs_optimizer_) << "CBS optimizer flag is ON but cbs_optimizer_ is null.";
        pose_key_is_active =
            cbs_optimizer_->valueExists(pose_symbol) || new_values_.exists(pose_symbol);
      } else
#endif
      {
        pose_key_is_active =
            state_.exists(pose_symbol) || new_values_.exists(pose_symbol);
      }

      if (pose_key_is_active) {
#ifdef KIMERA_USE_CBS
        if (cbs_exchange_active) {
          // zy Step 41c
          // In CBS mode with fixed-lag heart, use CBS belief gate for
          // acceptance/rejection, then inject accepted priors into Kimera graph.
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
            accepted_by_cbs = false;
            VLOG(2) << "Dropping external belief with unsupported/non-finite noise. "
                    << "source=" << prior.source_
                    << ", seq=" << prior.source_seq_
                    << ", ts[nsec]=" << prior.timestamp_kf_nsec_;
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
            const gtsam::Vector6 mu =
                gtsam::traits<gtsam::Pose3>::Logmap(prior.W_Pose_B_);
            gbp::Gaussian belief(pose_symbol, mu, cov, 1);
            std::map<gtsam::Key, std::vector<std::pair<cbs::AgentId, gbp::Gaussian>>>
                single_belief;
            single_belief[pose_symbol].emplace_back(sender_id, belief);
            ++num_external_beliefs_staged;
            const size_t rejected_count =
                static_cast<size_t>(cbs_optimizer_->addBeliefs(single_belief));
            num_external_beliefs_rejected += rejected_count;
            accepted_by_cbs = (rejected_count == 0u);
          }

          if (!accepted_by_cbs) {
            continue;
          }

          // zy Step 41d
          // Inject accepted external belief as a standard PriorFactor so fixed-lag
          // smoothing remains the only optimization heart.
          addExternalPosePrior(matched_frame_id, prior.W_Pose_B_, prior.noise_model_);
          ++num_external_priors_injected;
          VLOG(2) << "Injected external prior factor from CBS-accepted belief. source="
                  << prior.source_ << ", seq=" << prior.source_seq_
                  << ", ts[nsec]=" << prior.timestamp_kf_nsec_
                  << ", matched_frame_id=" << matched_frame_id;
        } else {
          ++num_external_priors_dropped_disabled_mode;
          VLOG(2) << "Dropping external prior because CBS belief exchange is OFF. source="
                  << prior.source_ << ", seq=" << prior.source_seq_
                  << ", ts[nsec]=" << prior.timestamp_kf_nsec_;
        }
#else
        ++num_external_priors_dropped_disabled_mode;
        VLOG(2) << "Dropping external prior because CBS support is not compiled.";
#endif
      } else {

        ++num_external_priors_dropped_inactive;
        VLOG(2) << "Matched external prior but pose key inactive. source="
                << prior.source_ << ", seq=" << prior.source_seq_
                << ", ts[nsec]=" << prior.timestamp_kf_nsec_
                << ", frame_id=" << matched_frame_id;
      } // zy when something behaves oddly, you can trace exact upstream message through Kimera.
    }

    external_pose_priors_queue_.swap(remaining_queue);
  }

  // zy Step 12c
  #ifdef KIMERA_USE_CBS
  if (cbs_exchange_active && num_external_beliefs_staged > 0) {
    const size_t num_external_beliefs_accepted =
        (num_external_beliefs_staged >= num_external_beliefs_rejected)
            ? (num_external_beliefs_staged - num_external_beliefs_rejected)
            : 0u;
    LOG_EVERY_N(INFO, 20) << "CBS addBeliefs: staged="
                          << num_external_beliefs_staged
                          << ", rejected=" << num_external_beliefs_rejected
                          << ", accepted=" << num_external_beliefs_accepted
                          << ", bad_noise="
                          << num_external_beliefs_dropped_bad_noise
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
        << ", queue_size_now=" << external_pose_priors_queue_.size();
  }



  // Only for statistics and debugging.
  // Store start time to calculate absolute total time taken.
  const auto& total_start_time = utils::Timer::tic();
  // Store start time to calculate per module total time.
  auto start_time = total_start_time;
  // Reset all timing infupdateSmoother
  /////////////////////// BOOKKEEPING ////////////////////////////////////
  size_t new_smart_factors_size = new_smart_factors_.size();
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
                          new_imu_prior_and_other_factors_.size());
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
  for (const auto& new_smart_factor : new_smart_factors_) {
    // Push back the smart factor to the list of new factors to add to the graph.
    LandmarkId lmk_id = new_smart_factor.first;

    // Find smart factor and slot in old_smart_factors_ corresponding to this landmark.
    const auto& old_smart_factor_it = old_smart_factors_.find(lmk_id);
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
        // Intuition: replace previous smart factor for this landmark with the refreshed factor.
        delete_slots.push_back(slot);
        new_factors_tmp.push_back(new_smart_factor.second);
        lmk_ids_of_new_smart_factors_tmp.push_back(lmk_id);
      } else {
        // Intuition: if old slot vanished, drop stale bookkeeping so horizon state stays consistent.
        old_smart_factors_.erase(old_smart_factor_it);
        CHECK(deleteLmkFromFeatureTracks(lmk_id));
      }
    } else {
      // Intuition: slot -1 means first insertion of this smart factor into the graph.
      new_factors_tmp.push_back(new_smart_factor.second);
      lmk_ids_of_new_smart_factors_tmp.push_back(lmk_id);
    }
  }

  // Add also other factors (imu, priors).
  // SMART FACTORS MUST BE FIRST, otherwise when recovering the slots
  // for the smart factors we will mess up.
  // push back many factors with an iterator over shared_ptr
  // (factors are not copied)
  new_factors_tmp.push_back(new_imu_prior_and_other_factors_.begin(),
                            new_imu_prior_and_other_factors_.end());

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

  // Compute iSAM update.
  VLOG(10) << "iSAM2 update with " << new_factors_tmp.size() << " new factors "
           << ", " << new_values_.size() << " new values "
           << ", and " << delete_slots.size() << " deleted factors.";
  Smoother::Result result;
  VLOG(10) << "Starting first update.";
  bool is_smoother_ok = updateSmoother(
      &result, new_factors_tmp, new_values_, key_frame_count, delete_slots);
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
#ifdef KIMERA_USE_CBS
      if (useCbsOptimizerHeart()) {
        CHECK(cbs_optimizer_) << "CBS optimizer flag is ON but cbs_optimizer_ is null.";
        optimizer_mode = "CBS";
        num_factors_active = cbs_optimizer_->getFactorsUnsafe().size();
      } else
#endif
      {
        CHECK(smoother_);
        num_factors_active = smoother_->getFactors().size();
      }

      VLOG(1) << "Optimize status [" << optimizer_mode
              << "]: factors=" << num_factors_active
              << ", x=" << num_pose_keys
              << ", v=" << num_vel_keys
              << ", b=" << num_bias_keys
              << ", cur_kf=" << cur_id;
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
                                const gtsam::FactorIndices& delete_slots) {
  CHECK_NOTNULL(result);
  // zy Step 16a
#ifdef KIMERA_USE_CBS
  if (useCbsOptimizerHeart()) {
    // Intuition: in CBS mode, BPSAM is the single optimization heart, so we skip the legacy smoother update path.
    CHECK(cbs_optimizer_) << "CBS optimizer flag is ON but cbs_optimizer_ is null.";

    cbs::BPSAM::UpdateParams cbs_update_params;
    cbs_update_params.removeFactorIndices.insert(
        cbs_update_params.removeFactorIndices.end(),
        delete_slots.begin(),
        delete_slots.end());

      try {
      //zy Step 40f
      // Run CBS pose-sharing inner rounds per epoch:
      // round 0 uses incoming factors/values, later rounds run belief-only updates.
      const int max_pose_rounds = std::max(1, FLAGS_cbs_pose_rounds_per_epoch);
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

      gtsam::ISAM2Result cbs_result =
          cbs_optimizer_->update(new_factors, new_values, cbs_update_params);
      int rounds_executed = 1;
      std::optional<double> prev_residual = compute_cbs_residual();

      for (int round = 1; round < max_pose_rounds; ++round) {
        cbs_result = cbs_optimizer_->update(
            gtsam::NonlinearFactorGraph(), gtsam::Values(), cbs_update_params);
        ++rounds_executed;

        const std::optional<double> curr_residual = compute_cbs_residual();
        if (curr_residual && prev_residual) {
          const double abs_change = std::fabs(*curr_residual - *prev_residual);
          const double rel_change =
              abs_change / std::max(std::fabs(*prev_residual), 1e-12);
          if (abs_change <= abs_eps || rel_change <= rel_eps) {
            VLOG(2) << "CBS pose rounds converged early at round "
                    << rounds_executed << "/" << max_pose_rounds
                    << " (abs=" << abs_change << ", rel=" << rel_change << ")";
            break;
          }
        }

        if (curr_residual) {
          prev_residual = curr_residual;
        }
      }

      // Intuition: keep FixedLagSmoother API contract by populating a compatible summary result in CBS mode.
      result->iterations = rounds_executed;
      result->intermediateSteps = 0;
      result->nonlinearVariables = cbs_result.variablesRelinearized;
      result->linearVariables = cbs_result.variablesReeliminated;
      result->error = cbs_result.errorAfter ? *cbs_result.errorAfter : 0.0;

      // Intuition: refresh smart-factor slot cache only when this update inserted factors.
      if (!new_factors.empty()) {
        cbs_last_update_result_ = cbs_result;
        cbs_has_last_update_result_ = true;
      }
    } catch (const std::exception& e) {
      LOG(ERROR) << "CBS BPSAM update failed: " << e.what();
      return false;
    } catch (...) {
      LOG(ERROR) << "CBS BPSAM update failed with unknown exception.";
      return false;
    }



    return true;
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
