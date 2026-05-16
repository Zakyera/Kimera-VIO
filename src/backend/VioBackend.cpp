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

#include <cbs/bpsam/incremental_fixed_lag_bpsam_smoother.h>
#include <cbs/gbp/contraction/hellinger.h>
#include <cbs/key.h>
#include <cbs/utils/gtsam_compat.h>
#include <gflags/gflags.h>
#include <glog/logging.h>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <exception>
#include <iomanip>
#include <limits>  // for numeric_limits<>
#include <map>
#include <sstream>
#include <string>
#include <unordered_set>
#include <utility>  // for make_pair
#include <vector>

#include "kimera-vio/backend/CbsLocalBeliefCovariance.h"
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
DEFINE_bool(cbs_use_local_smoother_for_belief_covariance,
            true,
            "Use a mirror fixed-lag smoother without external belief factors "
            "to compute local-only covariance for CBS pose belief publishing.");
DEFINE_bool(cbs_use_bpsam_for_local_belief_covariance,
            false,
            "Use a BPSAM snapshot built from the local-sidecar graph/state for "
            "pose belief covariance extraction. Falls back to local smoother "
            "covariance if the BPSAM snapshot path fails.");
DEFINE_bool(cbs_use_persistent_bpsam_sidecar_for_belief_covariance,
            false,
            "Use a persistent BPSAM-backed local sidecar adapter for CBS "
            "pose-belief covariance (experimental scaffold).");
DEFINE_bool(cbs_use_persistent_bpsam_for_main_backend,
            false,
            "Use persistent fixed-lag BPSAM as the main Kimera backend "
            "optimizer (experimental).");
DEFINE_bool(cbs_log_covariance_sanity_diff,
            false,
            "Log debug-only local-vs-fused pose covariance sanity metrics.");
DEFINE_bool(cbs_use_temporary_cbs_linear_factors,
            true,
            "Linearize accepted CBS odometry factors inside the current "
            "iSAM2 update and use them only in a temporary augmented linear "
            "delta solve. They are never inserted into the persistent factor "
            "graph.");
DEFINE_bool(cbs_temporary_linear_already_applied_gate_enable,
            true,
            "In temporary-linear CBS mode, skip near-identical beliefs from "
            "the same sender/key that were already applied.");
DEFINE_double(cbs_temporary_linear_already_applied_metric_threshold,
              0.01,
              "Receiver-side Hellinger threshold for skipping already-applied "
              "temporary-linear CBS beliefs.");
DEFINE_double(cbs_temporary_linear_already_applied_dmu_threshold,
              1e-3,
              "Receiver-side pose delta threshold for skipping already-applied "
              "temporary-linear CBS beliefs.");
DEFINE_double(cbs_temporary_linear_already_applied_cov_rel_threshold,
              1e-3,
              "Receiver-side relative covariance Frobenius threshold for "
              "skipping already-applied temporary-linear CBS beliefs.");
DEFINE_bool(cbs_enable_soft_reset,
            true,
            "Enable GBP soft reset for incoming CBS external belief updates.");
DEFINE_bool(cbs_use_raw_previous_belief_gate,
            false,
            "Use the previous raw incoming belief from the same sender/key for "
            "CBS receiver-side hard reset gating. GBP is still used for "
            "contraction after the raw gate accepts the belief.");
DEFINE_bool(cbs_reject_first_message,
            true,
            "Use the first incoming CBS belief for a sender/key only to "
            "initialize receiver-side GBP state. Disable for one-way "
            "experiments where a sender may publish each key only once.");
DEFINE_double(cbs_d_reset,
              0.1,
              "CBS GBP hard reset Hellinger threshold.");
DEFINE_double(cbs_l2k_odom_factor_covariance_scale,
              1.0,
              "Scale applied only to LiORF-to-Kimera CBS odometry factor "
              "covariance after receiver gating/contraction.");
DEFINE_double(cbs_external_belief_timestamp_tolerance_sec,
              0.2,
              "Max allowed absolute timestamp mismatch (seconds) when matching "
              "incoming external pose beliefs to local keyframes.");
DEFINE_double(cbs_odom_unmatched_retry_max_age_sec,
              5.0,
              "Maximum receiver-local timestamp age for retrying unmatched "
              "incoming CBS odometry beliefs.");
DEFINE_int32(cbs_odom_unmatched_retry_max_beliefs,
             500,
             "Maximum number of unmatched incoming CBS odometry beliefs kept "
             "for retry.");
DEFINE_string(cbs_odom_sender_mode,
              "adjacent_window",
              "Sender-side CBS odometry mode: adjacent_window, latest_edge, "
              "or new_edge_once.");
DEFINE_bool(no_incremental_pose,
            false,
            "Flag to disable incremental pose usage in backend");

namespace VIO {

namespace {

inline FrameId saturatingSubFrameId(const FrameId value, const FrameId delta) {
  return value >= delta ? value - delta : 0u;
}

inline FrameId computeOldestActiveFrameId(const FrameId cur_id,
                                          const BackendParams& backend_params) {
  const FrameId window_size =
      std::max<FrameId>(1u, static_cast<FrameId>(backend_params.nr_states_));
  return saturatingSubFrameId(cur_id, window_size - 1u);
}

inline double wallTimeNowSec() {
  return std::chrono::duration<double>(
             std::chrono::system_clock::now().time_since_epoch())
      .count();
}

template <typename TimerStart>
inline double elapsedSec(const TimerStart& start_time) {
  return utils::Timer::toc<std::chrono::duration<double>>(start_time).count();
}

inline double secToMs(const double seconds) {
  return seconds * 1000.0;
}

inline std::string normalizeCbsOdomSenderMode(std::string mode) {
  std::transform(mode.begin(), mode.end(), mode.begin(), [](unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });
  std::replace(mode.begin(), mode.end(), '-', '_');
  if (mode == "adjacent" || mode == "adjacent_only") {
    return "adjacent_window";
  }
  if (mode == "latest" || mode == "latest_only") {
    return "latest_edge";
  }
  return mode;
}

inline bool isValidCbsOdomSenderMode(const std::string& mode) {
  return mode == "adjacent_window" || mode == "latest_edge" ||
         mode == "new_edge_once";
}

inline std::string formatPoseKeyToken(const uint8_t source_agent,
                                      const uint32_t pose_index) {
  const char agent_char = static_cast<char>(source_agent);
  return std::string("p:") + agent_char + ":" + std::to_string(pose_index);
}

inline std::string sanitizeLogToken(std::string token) {
  std::replace(token.begin(), token.end(), ',', '_');
  std::replace(token.begin(), token.end(), ' ', '_');
  std::replace(token.begin(), token.end(), '\n', '_');
  std::replace(token.begin(), token.end(), '\r', '_');
  return token.empty() ? "na" : token;
}

inline const char* externalBeliefRejectReasonToken(const int reason) {
  switch (reason) {
    case 0:
      return "none";
    case 1:
      return "window";
    case 2:
      return "timestamp";
    case 3:
      return "missing_state";
    case 4:
      return "covariance";
  }
  return "unknown";
}

inline gtsam::Matrix6 poseCovarianceFromMatrix(
    const gtsam::Matrix& state_covariance) {
  gtsam::Matrix6 pose_cov = gtsam::Matrix6::Identity() * 1e-3;
  if (state_covariance.rows() >= 6 && state_covariance.cols() >= 6) {
    pose_cov = gtsam::sub(state_covariance, 0, 6, 0, 6);
  }
  return pose_cov;
}

inline double beliefTraceFromMatrix(const Eigen::MatrixXd& covariance) {
  return covariance.trace();
}

inline gtsam::Vector6 vector6FromArray(const std::array<double, 6>& values) {
  gtsam::Vector6 vector = gtsam::Vector6::Zero();
  for (size_t i = 0u; i < 6u; ++i) {
    vector(i) = values[i];
  }
  return vector;
}

inline gtsam::Matrix6 matrix6FromArray(
    const std::array<double, 36>& values) {
  gtsam::Matrix6 matrix = gtsam::Matrix6::Zero();
  for (size_t r = 0u; r < 6u; ++r) {
    for (size_t c = 0u; c < 6u; ++c) {
      matrix(r, c) = values[r * 6u + c];
    }
  }
  return matrix;
}

inline void vector6ToArray(const gtsam::Vector6& vector,
                           std::array<double, 6>* values) {
  CHECK_NOTNULL(values);
  for (size_t i = 0u; i < 6u; ++i) {
    (*values)[i] = vector(i);
  }
}

inline void matrix6ToArray(const gtsam::Matrix6& matrix,
                           std::array<double, 36>* values) {
  CHECK_NOTNULL(values);
  for (size_t r = 0u; r < 6u; ++r) {
    for (size_t c = 0u; c < 6u; ++c) {
      (*values)[r * 6u + c] = matrix(r, c);
    }
  }
}

inline double safeHellingerDistance(const gtsam::Vector6& mu_a,
                                    const gtsam::Matrix6& cov_a,
                                    const gtsam::Vector6& mu_b,
                                    const gtsam::Matrix6& cov_b) {
  try {
    const gtsam::Pose3 pose_a = gtsam::Pose3::Expmap(mu_a);
    const gtsam::Pose3 pose_b = gtsam::Pose3::Expmap(mu_b);
    const gtsam::Vector6 delta =
        gtsam::Pose3::Logmap(pose_a.inverse() * pose_b);
    return gbp::Hellinger::hellingerDistanceGaussian(delta, cov_a, cov_b);
  } catch (...) {
    return std::numeric_limits<double>::quiet_NaN();
  }
}

inline double safeMahalanobisDistance(const gtsam::Vector6& mu_a,
                                      const gtsam::Matrix6& cov_a,
                                      const gtsam::Vector6& mu_b) {
  try {
    const gtsam::Vector6 delta = mu_b - mu_a;
    const gtsam::Matrix6 cov_inv = cov_a.inverse();
    return static_cast<double>(delta.transpose() * cov_inv * delta);
  } catch (...) {
    return std::numeric_limits<double>::quiet_NaN();
  }
}

inline double safeDeltaNorm(const gtsam::Vector6& mu_a,
                            const gtsam::Vector6& mu_b) {
  return (mu_b - mu_a).norm();
}

}  // namespace

class VioBackend::LocalSmootherPoseBeliefCovarianceSidecarAdapter final
    : public VioBackend::PoseBeliefCovarianceSidecarAdapter {
 public:
  explicit LocalSmootherPoseBeliefCovarianceSidecarAdapter(VioBackend* backend)
      : backend_(CHECK_NOTNULL(backend)) {}

  bool update(const gtsam::NonlinearFactorGraph& new_factors_tmp,
              const gtsam::Values& new_values,
              const std::map<Key, double>& timestamps,
              const gtsam::FactorIndices& delete_slots,
              size_t max_extra_iterations,
              const std::vector<LandmarkId>& lmk_ids_of_new_smart_factors_tmp)
      override {
    return backend_->updatePoseBeliefLocalSidecar(
        new_factors_tmp,
        new_values,
        timestamps,
        delete_slots,
        max_extra_iterations,
        lmk_ids_of_new_smart_factors_tmp);
  }

  bool computePoseBeliefLocalCovariance(const FrameId& cur_id) override {
    return backend_->computePoseBeliefLocalCovarianceFromSidecar(cur_id);
  }

  const char* covarianceSourceTag() const override { return "local_smoother"; }

 private:
  VioBackend* backend_ = nullptr;
};

class VioBackend::PersistentBpsamPoseBeliefCovarianceSidecarAdapter final
    : public VioBackend::PoseBeliefCovarianceSidecarAdapter {
 public:
  PersistentBpsamPoseBeliefCovarianceSidecarAdapter(
      VioBackend* backend,
      const BackendParams& backend_params)
      : backend_(CHECK_NOTNULL(backend)) {
    gtsam::ISAM2Params isam2_params;
    BackendParams::setIsam2Params(backend_params, &isam2_params);
    sidecar_ = std::make_unique<PersistentBpsamLocalCovarianceSidecar>(
        isam2_params, 'k', std::max<size_t>(1u, backend_params.nr_states_));
  }

  bool update(const gtsam::NonlinearFactorGraph& new_factors_tmp,
              const gtsam::Values& new_values,
              const std::map<Key, double>& timestamps,
              const gtsam::FactorIndices& delete_slots,
              size_t /*max_extra_iterations*/,
              const std::vector<LandmarkId>& lmk_ids_of_new_smart_factors_tmp)
      override {
    if (!sidecar_) {
      return false;
    }

    std::vector<size_t> smart_factor_slots;
    const bool update_ok =
        sidecar_->update(new_factors_tmp,
                         new_values,
                         timestamps,
                         delete_slots,
                         lmk_ids_of_new_smart_factors_tmp.size(),
                         &smart_factor_slots);
    if (!update_ok) {
      LOG(WARNING) << "Persistent BPSAM sidecar update failed.";
      return false;
    }

    const size_t n_smart_factors = std::min(
        lmk_ids_of_new_smart_factors_tmp.size(), smart_factor_slots.size());
    for (size_t i = 0u; i < n_smart_factors; ++i) {
      const LandmarkId lmk_id = lmk_ids_of_new_smart_factors_tmp.at(i);
      auto it_local =
          backend_->old_smart_factors_local_belief_cov_.find(lmk_id);
      if (it_local == backend_->old_smart_factors_local_belief_cov_.end()) {
        continue;
      }
      it_local->second.second = static_cast<Slot>(smart_factor_slots.at(i));
    }
    return true;
  }

  bool computePoseBeliefLocalCovariance(const FrameId& cur_id) override {
    if (!sidecar_) {
      return false;
    }
    const auto pose_cov =
        sidecar_->computePoseCovariance(gtsam::Symbol(kPoseSymbolChar, cur_id));
    if (!pose_cov) {
      return false;
    }
    backend_->pose_belief_local_covariance_lkf_ = *pose_cov;
    return true;
  }

  const char* covarianceSourceTag() const override { return "local_bpsam"; }

 private:
  VioBackend* backend_ = nullptr;
  std::unique_ptr<PersistentBpsamLocalCovarianceSidecar> sidecar_;
};

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

  cbs::BPSAM::Params bpsam_params;
  bpsam_params.robot_id = static_cast<cbs::AgentId>('k');
  bpsam_params.sam_params_ = isam_param;
  bpsam_params.gbp_update_params.enable_soft_reset =
      FLAGS_cbs_enable_soft_reset;
  bpsam_params.gbp_update_params.d_reset = FLAGS_cbs_d_reset;
  bpsam_params.use_raw_previous_belief_gate =
      FLAGS_cbs_use_raw_previous_belief_gate;
  bpsam_params.reject_first_message = FLAGS_cbs_reject_first_message;
  bpsam_params.use_temporary_cbs_linear_factors =
      FLAGS_cbs_use_temporary_cbs_linear_factors;
  bpsam_params.temporary_linear_already_applied_gate_enable =
      FLAGS_cbs_temporary_linear_already_applied_gate_enable;
  bpsam_params.temporary_linear_already_applied_metric_threshold =
      FLAGS_cbs_temporary_linear_already_applied_metric_threshold;
  bpsam_params.temporary_linear_already_applied_dmu_threshold =
      FLAGS_cbs_temporary_linear_already_applied_dmu_threshold;
  bpsam_params.temporary_linear_already_applied_cov_rel_threshold =
      FLAGS_cbs_temporary_linear_already_applied_cov_rel_threshold;
  if (std::isfinite(FLAGS_cbs_l2k_odom_factor_covariance_scale) &&
      FLAGS_cbs_l2k_odom_factor_covariance_scale > 0.0) {
    bpsam_params.external_factor_covariance_scale_by_source
        [static_cast<cbs::AgentId>('l')] =
            FLAGS_cbs_l2k_odom_factor_covariance_scale;
  }

  smoother_ =
      std::make_unique<Smoother>(backend_params.nr_states_, bpsam_params);
  if (FLAGS_cbs_use_persistent_bpsam_for_main_backend) {
    main_bpsam_backend_ =
        std::make_unique<PersistentBpsamLocalCovarianceSidecar>(
            isam_param, 'k', std::max<size_t>(1u, backend_params.nr_states_));
  }
  if (FLAGS_cbs_use_local_smoother_for_belief_covariance &&
      !FLAGS_cbs_use_persistent_bpsam_sidecar_for_belief_covariance) {
    local_belief_cov_smoother_ =
        std::make_unique<Smoother>(backend_params.nr_states_, bpsam_params);
  }
#else  // BATCH SMOOTHER
  gtsam::LevenbergMarquardtParams lmParams;
  lmParams.setlambdaInitial(0.0);     // same as GN
  lmParams.setlambdaLowerBound(0.0);  // same as GN
  lmParams.setlambdaUpperBound(0.0);  // same as GN)
  smoother_ = std::make_unique<Smoother>(backend_params.nr_states_, lmParams);
  if (FLAGS_cbs_use_local_smoother_for_belief_covariance &&
      !FLAGS_cbs_use_persistent_bpsam_sidecar_for_belief_covariance) {
    local_belief_cov_smoother_ =
        std::make_unique<Smoother>(backend_params.nr_states_, lmParams);
  }
#endif

  if (std::isfinite(FLAGS_cbs_external_belief_timestamp_tolerance_sec) &&
      FLAGS_cbs_external_belief_timestamp_tolerance_sec > 0.0) {
    external_belief_timestamp_tolerance_sec_ =
        FLAGS_cbs_external_belief_timestamp_tolerance_sec;
  } else {
    LOG(WARNING) << "Invalid --cbs_external_belief_timestamp_tolerance_sec="
                 << FLAGS_cbs_external_belief_timestamp_tolerance_sec
                 << ", keeping default "
                 << external_belief_timestamp_tolerance_sec_ << "s.";
  }
  LOG(INFO) << "CBS external belief matcher tolerance: "
            << external_belief_timestamp_tolerance_sec_ << " s";
  if (std::isfinite(FLAGS_cbs_odom_unmatched_retry_max_age_sec) &&
      FLAGS_cbs_odom_unmatched_retry_max_age_sec >= 0.0) {
    external_odom_unmatched_retry_max_age_sec_ =
        FLAGS_cbs_odom_unmatched_retry_max_age_sec;
  }
  max_unmatched_external_odom_retry_beliefs_ = static_cast<size_t>(
      std::max(0, FLAGS_cbs_odom_unmatched_retry_max_beliefs));
  cbs_odom_sender_mode_ =
      normalizeCbsOdomSenderMode(FLAGS_cbs_odom_sender_mode);
  if (!isValidCbsOdomSenderMode(cbs_odom_sender_mode_)) {
    LOG(WARNING) << "Invalid --cbs_odom_sender_mode="
                 << FLAGS_cbs_odom_sender_mode
                 << ", falling back to adjacent_window.";
    cbs_odom_sender_mode_ = "adjacent_window";
  }
  LOG(INFO) << "CBS outgoing odometry mode: " << cbs_odom_sender_mode_
            << " retry_max_age="
            << external_odom_unmatched_retry_max_age_sec_
            << " retry_max_beliefs="
            << max_unmatched_external_odom_retry_beliefs_;
  LOG(INFO) << "CBS external belief soft reset: "
            << (FLAGS_cbs_enable_soft_reset ? "enabled" : "disabled");
  LOG(INFO) << "CBS reject first message: "
            << (FLAGS_cbs_reject_first_message ? "enabled" : "disabled");
  LOG(INFO) << "CBS temporary linear odometry factors: "
            << (FLAGS_cbs_use_temporary_cbs_linear_factors ? "enabled"
                                                           : "disabled");

  initializePoseBeliefCovarianceSidecarAdapter();

  // Set parameters for all factors.
  setFactorsParams(backend_params,
                   &smart_noise_,
                   &smart_factors_params_,
                   &no_motion_prior_noise_,
                   &zero_velocity_prior_noise_,
                   &constant_velocity_prior_noise_);

  // Reset debug info.
  resetDebugInfo(&debug_info_);

  // Print parameters if verbose
  if (VLOG_IS_ON(1)) print();
}

VioBackend::~VioBackend() { LOG(INFO) << "Backend destructor called."; }

bool VioBackend::usePersistentBpsamMainBackend() const {
  return FLAGS_cbs_use_persistent_bpsam_for_main_backend &&
         static_cast<bool>(main_bpsam_backend_);
}

const gtsam::NonlinearFactorGraph& VioBackend::getMainBackendFactors() const {
  if (usePersistentBpsamMainBackend()) {
    return main_bpsam_backend_->factors();
  }
  CHECK(smoother_);
  return smoother_->getFactors();
}

bool VioBackend::mainBackendFactorExists(size_t slot) const {
  if (usePersistentBpsamMainBackend()) {
    return main_bpsam_backend_->factorExists(slot);
  }
  CHECK(smoother_);
  return smoother_->getFactors().exists(slot);
}

const gtsam::NonlinearFactor::shared_ptr VioBackend::mainBackendFactorAt(
    size_t slot) const {
  if (usePersistentBpsamMainBackend()) {
    return main_bpsam_backend_->factorAt(slot);
  }
  CHECK(smoother_);
  return smoother_->getFactors().at(slot);
}

const gtsam::ISAM2Result& VioBackend::getMainBackendResult() const {
  if (usePersistentBpsamMainBackend()) {
    return main_bpsam_backend_->lastUpdateResult();
  }
  CHECK(smoother_);
  return smoother_->getISAM2Result();
}

gtsam::Values VioBackend::calculateMainBackendEstimate() const {
  if (usePersistentBpsamMainBackend()) {
    return main_bpsam_backend_->calculateEstimate();
  }
  CHECK(smoother_);
  return smoother_->calculateEstimate();
}

void VioBackend::initializePoseBeliefCovarianceSidecarAdapter() {
  pose_belief_cov_sidecar_adapter_.reset();
  if (FLAGS_cbs_use_persistent_bpsam_sidecar_for_belief_covariance) {
    pose_belief_cov_sidecar_adapter_ =
        std::make_unique<PersistentBpsamPoseBeliefCovarianceSidecarAdapter>(
            this, backend_params_);
    return;
  }
  if (!local_belief_cov_smoother_) {
    return;
  }
  pose_belief_cov_sidecar_adapter_ =
      std::make_unique<LocalSmootherPoseBeliefCovarianceSidecarAdapter>(this);
}

/* -------------------------------------------------------------------------- */
BackendOutput::UniquePtr VioBackend::spinOnce(const BackendInput& input) {
  const auto spin_once_start_time = utils::Timer::tic();
  double backend_state_process_time_sec = 0.0;
  double output_landmark_map_time_sec = 0.0;
  double map_update_callback_time_sec = 0.0;
  double backend_output_construct_time_sec = 0.0;
  double backend_logger_output_time_sec = 0.0;

  if (VLOG_IS_ON(10)) {
    input.print();
  }

  if (logger_) {
    logger_->logBackendExtOdom(input);
  }

  bool backend_status = false;
  const BackendState backend_state = backend_state_;
  const auto backend_state_process_start_time = utils::Timer::tic();
  try {
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
    backend_state_process_time_sec = elapsedSec(backend_state_process_start_time);
  } catch (const std::exception& e) {
    LOG(ERROR) << "VioBackend::spinOnce failed while processing backend state "
               << static_cast<int>(backend_state) << ": " << e.what();
    throw;
  } catch (...) {
    LOG(ERROR) << "VioBackend::spinOnce failed while processing backend state "
               << static_cast<int>(backend_state) << ".";
    throw;
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
      const auto output_landmark_map_start_time = utils::Timer::tic();
      // Generate this map only if requested, since costly.
      // Also, if lmk type requested, fill lmk id to lmk type object.
      // WARNING this also cleans the lmks inside the old_smart_factors map!
      try {
        lmk_ids_to_3d_points_in_time_horizon =
            getMapLmkIdsTo3dPointsInTimeHorizon(
                getMainBackendFactors(),
                kOutputLmkTypeMap ? &lmk_id_to_lmk_type_map : nullptr,
                kMinLmkObs);
      } catch (const std::exception& e) {
        LOG(ERROR) << "VioBackend::spinOnce failed while building output "
                   << "landmark map: " << e.what();
        throw;
      } catch (...) {
        LOG(ERROR) << "VioBackend::spinOnce failed while building output "
                   << "landmark map.";
        throw;
      }
      output_landmark_map_time_sec = elapsedSec(output_landmark_map_start_time);
    }

    if (map_update_callback_) {
      const auto map_update_callback_start_time = utils::Timer::tic();
      try {
        map_update_callback_(lmk_ids_to_3d_points_in_time_horizon);
      } catch (const std::exception& e) {
        LOG(ERROR) << "VioBackend::spinOnce map update callback failed: "
                   << e.what();
        throw;
      } catch (...) {
        LOG(ERROR) << "VioBackend::spinOnce map update callback failed.";
        throw;
      }
      map_update_callback_time_sec = elapsedSec(map_update_callback_start_time);
    } else {
      LOG(FATAL) << "Did you forget to register the Map "
                    "Update callback for at least the "
                    "Frontend? Do so by using "
                    "registerMapUpdateCallback function.";
    }

    // Create Backend Output Payload.
    const auto backend_output_construct_start_time = utils::Timer::tic();
    try {
      output_payload = std::make_unique<BackendOutput>(
          VioNavStateTimestamped(
              input.timestamp_,
              (FLAGS_no_incremental_pose ? W_Pose_B_lkf_from_state_
                                         : W_Pose_B_lkf_from_increments_),
              W_Vel_B_lkf_,
              imu_bias_lkf_),
          // TODO(Toni): Make all below optional!!
          state_,
          getMainBackendFactors(),
          getCurrentStateCovariance(),
          curr_kf_id_,
          landmark_count_,
          debug_info_,
          lmk_ids_to_3d_points_in_time_horizon,
          lmk_id_to_lmk_type_map,
          pose_belief_local_covariance_lkf_,
          pose_belief_local_covariance_valid_,
          pose_belief_covariance_source_,
          external_beliefs_received_per_update_,
          external_beliefs_added_per_update_,
          external_beliefs_rejected_first_message_per_update_,
          external_beliefs_rejected_update_status_per_update_,
          external_beliefs_rejected_inactive_window_per_update_,
          external_beliefs_rejected_shape_per_update_,
          external_beliefs_rejected_exception_per_update_,
          optimization_time_sec_per_update_,
          cbs_belief_generation_time_sec_per_update_,
          cbs_marginalization_graph_factor_count_,
          cbs_outgoing_odom_beliefs_);
    } catch (const std::exception& e) {
      LOG(ERROR)
          << "VioBackend::spinOnce failed while creating backend output: "
          << e.what();
      throw;
    } catch (...) {
      LOG(ERROR)
          << "VioBackend::spinOnce failed while creating backend output.";
      throw;
    }
    backend_output_construct_time_sec =
        elapsedSec(backend_output_construct_start_time);

    if (logger_) {
      const auto backend_logger_output_start_time = utils::Timer::tic();
      try {
        logger_->logBackendOutput(*output_payload);
      } catch (const std::exception& e) {
        LOG(ERROR)
            << "VioBackend::spinOnce failed while logging backend output: "
            << e.what();
        throw;
      } catch (...) {
        LOG(ERROR)
            << "VioBackend::spinOnce failed while logging backend output.";
        throw;
      }
      backend_logger_output_time_sec =
          elapsedSec(backend_logger_output_start_time);
    }
  }

  const double spin_once_total_time_sec = elapsedSec(spin_once_start_time);
  LOG(INFO) << "KIMERA_BACKEND_SPINONCE_TIMING_ROW,"
            << curr_kf_id_ << ","
            << secToMs(spin_once_total_time_sec) << ","
            << secToMs(backend_state_process_time_sec) << ","
            << secToMs(output_landmark_map_time_sec) << ","
            << secToMs(map_update_callback_time_sec) << ","
            << secToMs(backend_output_construct_time_sec) << ","
            << secToMs(backend_logger_output_time_sec) << ","
            << static_cast<int>(backend_state) << ","
            << (backend_status ? 1 : 0);

  return output_payload;
}

void VioBackend::saveGraph(const std::string& filepath) const {
  getMainBackendFactors().saveGraph(filepath);
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

void VioBackend::enqueueExternalOdometryBeliefs(
    const std::vector<ExternalOdometryBelief>& beliefs) {
  if (beliefs.empty()) {
    return;
  }
  external_beliefs_received_total_.fetch_add(beliefs.size(),
                                             std::memory_order_relaxed);

  std::lock_guard<std::mutex> lock(external_beliefs_mutex_);
  for (const auto& belief : beliefs) {
    ExternalOdometryBelief queued_belief = belief;
    if (!std::isfinite(queued_belief.received_wall_time_sec) ||
        queued_belief.received_wall_time_sec <= 0.0) {
      queued_belief.received_wall_time_sec = wallTimeNowSec();
    }
    pending_external_odom_beliefs_.push_back(queued_belief);
  }
  const size_t size_before_cap = pending_external_odom_beliefs_.size();
  capPendingExternalOdometryBeliefsLocked();
  const size_t queue_dropped_now =
      size_before_cap > pending_external_odom_beliefs_.size()
          ? size_before_cap - pending_external_odom_beliefs_.size()
          : 0u;
  if (queue_dropped_now > 0u) {
    external_beliefs_queue_dropped_total_.fetch_add(queue_dropped_now,
                                                    std::memory_order_relaxed);
  }
}

std::vector<ExternalOdometryBelief>
VioBackend::popPendingExternalOdometryBeliefs() {
  std::vector<ExternalOdometryBelief> beliefs;
  std::lock_guard<std::mutex> lock(external_beliefs_mutex_);
  beliefs.reserve(pending_external_odom_beliefs_.size());
  while (!pending_external_odom_beliefs_.empty()) {
    beliefs.push_back(pending_external_odom_beliefs_.front());
    pending_external_odom_beliefs_.pop_front();
  }
  return beliefs;
}

void VioBackend::capPendingExternalOdometryBeliefsLocked() {
  while (pending_external_odom_beliefs_.size() >
         max_pending_external_odom_beliefs_) {
    pending_external_odom_beliefs_.pop_front();
  }
}

void VioBackend::requeuePendingExternalOdometryBeliefs(
    const std::vector<ExternalOdometryBelief>& beliefs) {
  if (beliefs.empty() || max_unmatched_external_odom_retry_beliefs_ == 0u) {
    return;
  }

  std::lock_guard<std::mutex> lock(external_beliefs_mutex_);
  size_t remaining_retry_slots =
      max_unmatched_external_odom_retry_beliefs_ >
              pending_external_odom_beliefs_.size()
          ? max_unmatched_external_odom_retry_beliefs_ -
                pending_external_odom_beliefs_.size()
          : 0u;
  for (const auto& belief : beliefs) {
    if (remaining_retry_slots == 0u) {
      break;
    }
    pending_external_odom_beliefs_.push_back(belief);
    --remaining_retry_slots;
  }
}

void VioBackend::updateKeyframeTimestampIndex(
    const FrameId& frame_id,
    const Timestamp& timestamp_kf_nsec) {
  const double stamp_sec = static_cast<double>(timestamp_kf_nsec) * 1e-9;
  keyframe_timestamp_sec_[frame_id] = stamp_sec;

  const FrameId keep_span =
      static_cast<FrameId>(backend_params_.nr_states_ * 3.0);
  const FrameId min_keep_frame = saturatingSubFrameId(frame_id, keep_span);
  auto it = keyframe_timestamp_sec_.begin();
  while (it != keyframe_timestamp_sec_.end() && it->first < min_keep_frame) {
    it = keyframe_timestamp_sec_.erase(it);
  }
}

bool VioBackend::resolveExternalBeliefStamp(
    double stamp_sec,
    const FrameId& cur_id,
    FrameId* local_frame_id,
    ExternalBeliefRejectReason* reject_reason,
    FrameId* best_frame_id,
    double* best_stamp_sec,
    double* best_abs_dt) const {
  CHECK_NOTNULL(local_frame_id);
  if (reject_reason) {
    *reject_reason = ExternalBeliefRejectReason::kNone;
  }
  if (best_frame_id) {
    *best_frame_id = 0u;
  }
  if (best_stamp_sec) {
    *best_stamp_sec = std::numeric_limits<double>::quiet_NaN();
  }
  if (best_abs_dt) {
    *best_abs_dt = std::numeric_limits<double>::quiet_NaN();
  }

  const bool has_valid_stamp =
      std::isfinite(stamp_sec) && stamp_sec > 0.0;

  if (has_valid_stamp) {
    if (keyframe_timestamp_sec_.empty()) {
      if (reject_reason) {
        *reject_reason = ExternalBeliefRejectReason::kWindow;
      }
      return false;
    }

    double best_dt = std::numeric_limits<double>::max();
    FrameId best_frame = 0;
    bool has_best_frame = false;
    for (const auto& [frame_id, frame_stamp_sec] : keyframe_timestamp_sec_) {
      if (frame_id > cur_id) {
        continue;
      }
      const double dt = std::abs(frame_stamp_sec - stamp_sec);
      if (dt < best_dt) {
        best_dt = dt;
        best_frame = frame_id;
        has_best_frame = true;
      }
    }

    if (!has_best_frame) {
      if (reject_reason) {
        *reject_reason = ExternalBeliefRejectReason::kWindow;
      }
      return false;
    }

    const auto best_stamp_it = keyframe_timestamp_sec_.find(best_frame);
    if (best_frame_id) {
      *best_frame_id = best_frame;
    }
    if (best_stamp_sec) {
      *best_stamp_sec = best_stamp_it != keyframe_timestamp_sec_.end()
                            ? best_stamp_it->second
                            : std::numeric_limits<double>::quiet_NaN();
    }
    if (best_abs_dt) {
      *best_abs_dt = best_dt;
    }

    if (best_dt > external_belief_timestamp_tolerance_sec_) {
      if (reject_reason) {
        *reject_reason = ExternalBeliefRejectReason::kTimestamp;
      }
      return false;
    }

    *local_frame_id = best_frame;
    return true;
  }

  if (reject_reason) {
    *reject_reason = ExternalBeliefRejectReason::kTimestamp;
  }
  return false;
}

double VioBackend::latestKeyframeTimestampSec(const FrameId& cur_id) const {
  double latest = std::numeric_limits<double>::quiet_NaN();
  for (const auto& [frame_id, stamp_sec] : keyframe_timestamp_sec_) {
    if (frame_id > cur_id || !std::isfinite(stamp_sec) || stamp_sec <= 0.0) {
      continue;
    }
    if (!std::isfinite(latest) || stamp_sec > latest) {
      latest = stamp_sec;
    }
  }
  return latest;
}

double VioBackend::externalOdomBeliefRetryAgeSec(
    const ExternalOdometryBelief& belief,
    const FrameId& cur_id) const {
  double event_stamp_sec =
      belief.to_stamp_sec > 0.0 ? belief.to_stamp_sec : belief.from_stamp_sec;
  if (!std::isfinite(event_stamp_sec) || event_stamp_sec <= 0.0) {
    event_stamp_sec = belief.sender_timestamp_ns > 0u
                          ? static_cast<double>(belief.sender_timestamp_ns) *
                                1e-9
                          : std::numeric_limits<double>::quiet_NaN();
  }

  const double latest_stamp_sec = latestKeyframeTimestampSec(cur_id);
  if (std::isfinite(latest_stamp_sec) && std::isfinite(event_stamp_sec) &&
      event_stamp_sec > 0.0) {
    return std::max(0.0, latest_stamp_sec - event_stamp_sec);
  }

  if (std::isfinite(belief.received_wall_time_sec) &&
      belief.received_wall_time_sec > 0.0) {
    return std::max(0.0,
                    wallTimeNowSec() - belief.received_wall_time_sec);
  }
  return 0.0;
}

bool VioBackend::shouldRetryExternalOdometryBelief(
    const ExternalOdometryBelief& belief,
    const FrameId& cur_id,
    const std::string& reason) const {
  const double age_sec = externalOdomBeliefRetryAgeSec(belief, cur_id);
  const bool keep = max_unmatched_external_odom_retry_beliefs_ > 0u &&
                    (!std::isfinite(age_sec) ||
                     age_sec <= external_odom_unmatched_retry_max_age_sec_);
  LOG(INFO) << std::fixed << std::setprecision(9)
            << "CBS_ODOM_RETRY_ROW_L2K,"
            << formatPoseKeyToken(belief.source_agent,
                                  belief.sender_from_pose_index)
            << "->"
            << formatPoseKeyToken(belief.source_agent,
                                  belief.sender_to_pose_index)
            << "," << belief.from_stamp_sec << "," << belief.to_stamp_sec
            << "," << age_sec << ","
            << external_odom_unmatched_retry_max_age_sec_ << ","
            << sanitizeLogToken(reason) << ","
            << (keep ? "retry" : "drop_old_unmatched");
  return keep;
}

void VioBackend::collectExternalBeliefFactors(
    const FrameId& cur_id,
    gtsam::FactorIndices* delete_slots,
    gtsam::NonlinearFactorGraph* new_factors_tmp,
    std::vector<ExternalBeliefFactorId>* inserted_external_factor_ids) {
  CHECK_NOTNULL(delete_slots);
  CHECK_NOTNULL(new_factors_tmp);
  CHECK_NOTNULL(inserted_external_factor_ids);
  (void)delete_slots;
  (void)new_factors_tmp;
  (void)inserted_external_factor_ids;
  external_beliefs_received_per_update_ = 0u;
  external_beliefs_added_per_update_ = 0u;
  external_beliefs_rejected_first_message_per_update_ = 0u;
  external_beliefs_rejected_update_status_per_update_ = 0u;
  external_beliefs_rejected_inactive_window_per_update_ = 0u;
  external_beliefs_rejected_shape_per_update_ = 0u;
  external_beliefs_rejected_exception_per_update_ = 0u;

  const auto pending_odom_beliefs = popPendingExternalOdometryBeliefs();
  external_beliefs_received_per_update_ = pending_odom_beliefs.size();
  if (pending_odom_beliefs.empty()) {
    return;
  }

  size_t dropped_unmatched = 0u;
  size_t resolved_count = 0u;
  size_t rejected_window = 0u;
  size_t rejected_timestamp = 0u;
  size_t rejected_missing_state = 0u;
  size_t rejected_covariance = 0u;
  size_t rejected_by_bpsam = 0u;
  size_t rejected_bpsam_first_message = 0u;
  size_t rejected_bpsam_update_status = 0u;
  size_t rejected_bpsam_inactive_window = 0u;
  size_t rejected_bpsam_shape = 0u;
  size_t rejected_bpsam_exception = 0u;
  size_t retried_unmatched = 0u;
  std::vector<ExternalOdometryBelief> retry_beliefs;
  const auto log_match_decision =
      [this](const ExternalOdometryBelief& belief,
             const FrameId from_best_frame_id,
             const double from_best_stamp_sec,
             const double from_best_abs_dt,
             const ExternalBeliefRejectReason from_reject_reason,
             const FrameId to_best_frame_id,
             const double to_best_stamp_sec,
             const double to_best_abs_dt,
             const ExternalBeliefRejectReason to_reject_reason,
             const std::string& decision) {
        const bool has_from_candidate = std::isfinite(from_best_abs_dt);
        const bool has_to_candidate = std::isfinite(to_best_abs_dt);
        const std::string from_token =
            has_from_candidate
                ? formatPoseKeyToken(static_cast<uint8_t>('k'), from_best_frame_id)
                : std::string("na");
        const std::string to_token =
            has_to_candidate
                ? formatPoseKeyToken(static_cast<uint8_t>('k'), to_best_frame_id)
                : std::string("na");
        LOG(INFO) << std::fixed << std::setprecision(9)
                  << "CBS_ODOM_MATCH_ROW_L2K,"
                  << formatPoseKeyToken(belief.source_agent,
                                        belief.sender_from_pose_index)
                  << "->"
                  << formatPoseKeyToken(belief.source_agent,
                                        belief.sender_to_pose_index)
                  << "," << belief.from_stamp_sec << "," << belief.to_stamp_sec
                  << "," << from_token << "->" << to_token << ","
                  << from_best_stamp_sec << "," << to_best_stamp_sec << ","
                  << from_best_abs_dt << "," << to_best_abs_dt << ","
                  << external_belief_timestamp_tolerance_sec_ << ","
                  << externalBeliefRejectReasonToken(
                         static_cast<int>(from_reject_reason))
                  << ","
                  << externalBeliefRejectReasonToken(
                         static_cast<int>(to_reject_reason))
                  << ","
                  << sanitizeLogToken(decision);
      };

  for (const auto& belief : pending_odom_beliefs) {
    FrameId from_frame_id = 0u;
    FrameId to_frame_id = 0u;
    FrameId from_best_frame_id = 0u;
    FrameId to_best_frame_id = 0u;
    double from_best_stamp_sec = std::numeric_limits<double>::quiet_NaN();
    double to_best_stamp_sec = std::numeric_limits<double>::quiet_NaN();
    double from_best_abs_dt = std::numeric_limits<double>::quiet_NaN();
    double to_best_abs_dt = std::numeric_limits<double>::quiet_NaN();
    ExternalBeliefRejectReason from_reject_reason =
        ExternalBeliefRejectReason::kNone;
    ExternalBeliefRejectReason to_reject_reason =
        ExternalBeliefRejectReason::kNone;
    const bool from_resolved =
        resolveExternalBeliefStamp(belief.from_stamp_sec,
                                   cur_id,
                                   &from_frame_id,
                                   &from_reject_reason,
                                   &from_best_frame_id,
                                   &from_best_stamp_sec,
                                   &from_best_abs_dt);
    const bool to_resolved =
        resolveExternalBeliefStamp(belief.to_stamp_sec,
                                   cur_id,
                                   &to_frame_id,
                                   &to_reject_reason,
                                   &to_best_frame_id,
                                   &to_best_stamp_sec,
                                   &to_best_abs_dt);
    if (!from_resolved || !to_resolved) {
      const bool sender_stamps_valid =
          std::isfinite(belief.from_stamp_sec) && belief.from_stamp_sec > 0.0 &&
          std::isfinite(belief.to_stamp_sec) && belief.to_stamp_sec > 0.0;
      if (sender_stamps_valid &&
          shouldRetryExternalOdometryBelief(belief, cur_id, "timestamp_match")) {
        ++retried_unmatched;
        retry_beliefs.push_back(belief);
        log_match_decision(belief,
                           from_best_frame_id,
                           from_best_stamp_sec,
                           from_best_abs_dt,
                           from_reject_reason,
                           to_best_frame_id,
                           to_best_stamp_sec,
                           to_best_abs_dt,
                           to_reject_reason,
                           "retry_timestamp_match");
      } else {
        ++dropped_unmatched;
        if (from_reject_reason == ExternalBeliefRejectReason::kWindow ||
            to_reject_reason == ExternalBeliefRejectReason::kWindow) {
          ++rejected_window;
        } else {
          ++rejected_timestamp;
        }
        log_match_decision(belief,
                           from_best_frame_id,
                           from_best_stamp_sec,
                           from_best_abs_dt,
                           from_reject_reason,
                           to_best_frame_id,
                           to_best_stamp_sec,
                           to_best_abs_dt,
                           to_reject_reason,
                           "dropped_timestamp_match");
      }
      continue;
    }
    ++resolved_count;

    const gtsam::Key from_pose_key =
        gtsam::Symbol(kPoseSymbolChar, from_frame_id);
    const gtsam::Key to_pose_key =
        gtsam::Symbol(kPoseSymbolChar, to_frame_id);

    cbs::BPSAM::CbsOdometryBelief odom_belief;
    odom_belief.source_agent = static_cast<cbs::AgentId>(belief.source_agent);
    odom_belief.from_pose_key = from_pose_key;
    odom_belief.to_pose_key = to_pose_key;
    odom_belief.sender_from_pose_key =
        cbs::toPoseKey(static_cast<cbs::AgentId>(belief.source_agent),
                       belief.sender_from_pose_index);
    odom_belief.sender_to_pose_key =
        cbs::toPoseKey(static_cast<cbs::AgentId>(belief.source_agent),
                       belief.sender_to_pose_index);
    odom_belief.measured_from_to =
        gtsam::Pose3::Expmap(vector6FromArray(belief.relative_mu));
    odom_belief.covariance =
        poseCovarianceFromMatrix(matrix6FromArray(belief.covariance));
    odom_belief.relax_factor = belief.relax_factor;

    try {
      std::vector<cbs::BPSAM::CbsOdometryBelief> single_belief;
      single_belief.push_back(std::move(odom_belief));
      const auto add_result =
          smoother_->addOdometryBeliefsDetailed(std::move(single_belief));
      const bool retry_bpsam_inactive =
          add_result.accepted == 0u &&
          add_result.rejected_inactive_window > 0u &&
          add_result.rejected() == add_result.rejected_inactive_window &&
          shouldRetryExternalOdometryBelief(belief,
                                            cur_id,
                                            "bpsam_inactive_window");
      if (retry_bpsam_inactive) {
        ++retried_unmatched;
        retry_beliefs.push_back(belief);
        log_match_decision(belief,
                           from_best_frame_id,
                           from_best_stamp_sec,
                           from_best_abs_dt,
                           from_reject_reason,
                           to_best_frame_id,
                           to_best_stamp_sec,
                           to_best_abs_dt,
                           to_reject_reason,
                           "retry_bpsam_inactive_window");
        continue;
      }
      external_beliefs_added_per_update_ += add_result.accepted;
      rejected_by_bpsam += add_result.rejected();
      rejected_bpsam_inactive_window += add_result.rejected_inactive_window;
      rejected_bpsam_shape += add_result.rejected_shape;
      rejected_bpsam_exception += add_result.rejected_exception;
      if (add_result.details.empty()) {
        log_match_decision(belief,
                           from_best_frame_id,
                           from_best_stamp_sec,
                           from_best_abs_dt,
                           from_reject_reason,
                           to_best_frame_id,
                           to_best_stamp_sec,
                           to_best_abs_dt,
                           to_reject_reason,
                           "bpsam_no_detail");
      }
      for (const auto& detail : add_result.details) {
        log_match_decision(belief,
                           from_best_frame_id,
                           from_best_stamp_sec,
                           from_best_abs_dt,
                           from_reject_reason,
                           to_best_frame_id,
                           to_best_stamp_sec,
                           to_best_abs_dt,
                           to_reject_reason,
                           detail.message);
        LOG(INFO) << "CBS_BPSAM_ODOM_ADD_ROW_L2K,"
                  << formatPoseKeyToken(belief.source_agent,
                                        belief.sender_from_pose_index)
                  << "->"
                  << formatPoseKeyToken(belief.source_agent,
                                        belief.sender_to_pose_index)
                  << ","
                  << formatPoseKeyToken(static_cast<uint8_t>('k'),
                                        from_frame_id)
                  << "->"
                  << formatPoseKeyToken(static_cast<uint8_t>('k'),
                                        to_frame_id)
                  << "," << static_cast<int>(detail.status) << ","
                  << detail.covariance_trace << ","
                  << sanitizeLogToken(detail.message);
      }
    } catch (const std::exception& e) {
      ++rejected_by_bpsam;
      ++rejected_bpsam_exception;
      log_match_decision(belief,
                         from_best_frame_id,
                         from_best_stamp_sec,
                         from_best_abs_dt,
                         from_reject_reason,
                         to_best_frame_id,
                         to_best_stamp_sec,
                         to_best_abs_dt,
                         to_reject_reason,
                         "bpsam_exception");
      LOG(WARNING) << "BPSAM rejected external CBS odometry "
                   << formatPoseKeyToken(belief.source_agent,
                                         belief.sender_from_pose_index)
                   << "->"
                   << formatPoseKeyToken(belief.source_agent,
                                         belief.sender_to_pose_index)
                   << " for Kimera frames " << from_frame_id << "->"
                   << to_frame_id << ": " << e.what();
    } catch (...) {
      ++rejected_by_bpsam;
      ++rejected_bpsam_exception;
      log_match_decision(belief,
                         from_best_frame_id,
                         from_best_stamp_sec,
                         from_best_abs_dt,
                         from_reject_reason,
                         to_best_frame_id,
                         to_best_stamp_sec,
                         to_best_abs_dt,
                         to_reject_reason,
                         "bpsam_unknown_exception");
      LOG(WARNING) << "BPSAM rejected external CBS odometry "
                   << formatPoseKeyToken(belief.source_agent,
                                         belief.sender_from_pose_index)
                   << "->"
                   << formatPoseKeyToken(belief.source_agent,
                                         belief.sender_to_pose_index)
                   << " for Kimera frames " << from_frame_id << "->"
                   << to_frame_id << ".";
    }
  }

  requeuePendingExternalOdometryBeliefs(retry_beliefs);

  dropped_unmatched += rejected_by_bpsam;
  external_beliefs_rejected_first_message_per_update_ =
      rejected_bpsam_first_message;
  external_beliefs_rejected_update_status_per_update_ =
      rejected_bpsam_update_status;
  external_beliefs_rejected_inactive_window_per_update_ =
      rejected_bpsam_inactive_window;
  external_beliefs_rejected_shape_per_update_ = rejected_bpsam_shape;
  external_beliefs_rejected_exception_per_update_ = rejected_bpsam_exception;
  external_beliefs_inserted_total_.fetch_add(external_beliefs_added_per_update_,
                                             std::memory_order_relaxed);
  external_beliefs_rejected_total_.fetch_add(dropped_unmatched,
                                             std::memory_order_relaxed);
  external_beliefs_rejected_window_total_.fetch_add(rejected_window,
                                                    std::memory_order_relaxed);
  external_beliefs_rejected_timestamp_total_.fetch_add(
      rejected_timestamp, std::memory_order_relaxed);
  external_beliefs_rejected_missing_state_total_.fetch_add(
      rejected_missing_state, std::memory_order_relaxed);
  external_beliefs_rejected_covariance_total_.fetch_add(
      rejected_covariance, std::memory_order_relaxed);
  external_beliefs_rejected_first_message_total_.fetch_add(
      rejected_bpsam_first_message, std::memory_order_relaxed);
  external_beliefs_rejected_update_status_total_.fetch_add(
      rejected_bpsam_update_status, std::memory_order_relaxed);
  external_beliefs_rejected_inactive_window_total_.fetch_add(
      rejected_bpsam_inactive_window, std::memory_order_relaxed);
  external_beliefs_rejected_shape_total_.fetch_add(rejected_bpsam_shape,
                                                   std::memory_order_relaxed);
  external_beliefs_rejected_exception_total_.fetch_add(
      rejected_bpsam_exception, std::memory_order_relaxed);

  external_beliefs_resolved_total_.fetch_add(resolved_count,
                                             std::memory_order_relaxed);
  external_beliefs_selected_total_.fetch_add(resolved_count,
                                             std::memory_order_relaxed);

  LOG(INFO) << "Kimera CBS incoming odometry flow: pending_odom="
            << pending_odom_beliefs.size()
            << " resolved=" << resolved_count
            << " bpsam_added=" << external_beliefs_added_per_update_
            << " rejected=" << dropped_unmatched
            << " retried=" << retried_unmatched
            << " rejected_by(window=" << rejected_window
            << ",timestamp=" << rejected_timestamp
            << ",state=" << rejected_missing_state
            << ",covariance=" << rejected_covariance
            << ",bpsam=" << rejected_by_bpsam
            << ",bpsam_first_message=" << rejected_bpsam_first_message
            << ",bpsam_update_status=" << rejected_bpsam_update_status
            << ",bpsam_inactive_window=" << rejected_bpsam_inactive_window
            << ",bpsam_shape=" << rejected_bpsam_shape
            << ",bpsam_exception=" << rejected_bpsam_exception << ",soft_reset="
            << (FLAGS_cbs_enable_soft_reset ? "true" : "false") << ")";
  return;
}

void VioBackend::refreshCbsOutgoingBeliefs(const FrameId& cur_id) {
  const auto cbs_outgoing_start_time = utils::Timer::tic();
  cbs_outgoing_odom_beliefs_.clear();
  cbs_belief_generation_time_sec_per_update_ = 0.0;
  cbs_outgoing_total_time_sec_per_update_ = 0.0;
  cbs_set_marginalization_graph_time_sec_per_update_ = 0.0;
  cbs_get_odometry_beliefs_time_sec_per_update_ = 0.0;
  cbs_marginalization_graph_factor_count_ = 0u;

  if (!smoother_) {
    return;
  }

  const FrameId oldest_active_frame_id =
      computeOldestActiveFrameId(cur_id, backend_params_);
  gtsam::KeySet request_keys;
  std::vector<std::pair<FrameId, gtsam::Key>> timestamped_pose_keys;
  for (FrameId frame_id = oldest_active_frame_id; frame_id <= cur_id;
       ++frame_id) {
    const gtsam::Key pose_key = gtsam::Symbol(kPoseSymbolChar, frame_id);
    if (smoother_->valueExists(pose_key)) {
      if (cbs_odom_sender_mode_ == "adjacent_window") {
        request_keys.insert(pose_key);
      } else {
        const auto stamp_it = keyframe_timestamp_sec_.find(frame_id);
        if (stamp_it != keyframe_timestamp_sec_.end() &&
            std::isfinite(stamp_it->second) && stamp_it->second > 0.0) {
          timestamped_pose_keys.emplace_back(frame_id, pose_key);
        }
      }
    }
    if (frame_id == std::numeric_limits<FrameId>::max()) {
      break;
    }
  }

  bool latest_pair_valid = false;
  gtsam::Key latest_from_key = 0u;
  gtsam::Key latest_to_key = 0u;
  if (cbs_odom_sender_mode_ == "latest_edge" ||
      cbs_odom_sender_mode_ == "new_edge_once") {
    if (timestamped_pose_keys.size() < 2u) {
      return;
    }
    latest_from_key = timestamped_pose_keys[timestamped_pose_keys.size() - 2u].second;
    latest_to_key = timestamped_pose_keys.back().second;
    latest_pair_valid = true;
    if (cbs_odom_sender_mode_ == "new_edge_once" &&
        last_cbs_outgoing_odom_pair_valid_ &&
        latest_from_key == last_cbs_outgoing_odom_from_key_ &&
        latest_to_key == last_cbs_outgoing_odom_to_key_) {
      return;
    }
    request_keys.insert(latest_from_key);
    request_keys.insert(latest_to_key);
  }

  if (request_keys.empty()) {
    return;
  }

  try {
    const auto set_marginalization_graph_start_time = utils::Timer::tic();
    smoother_->setMarginalizationGraph(cbs::BPSAM::MarginalizationType::LOCAL);
    cbs_set_marginalization_graph_time_sec_per_update_ =
        elapsedSec(set_marginalization_graph_start_time);
    cbs_marginalization_graph_factor_count_ =
        smoother_->marginalizationGraphFactorCount();

    const auto get_beliefs_start = utils::Timer::tic();
    const auto outgoing =
        smoother_->getOdometryBeliefs(request_keys,
                                      static_cast<cbs::AgentId>('l'));
    cbs_belief_generation_time_sec_per_update_ =
        utils::Timer::toc<std::chrono::duration<double>>(get_beliefs_start)
            .count();
    cbs_get_odometry_beliefs_time_sec_per_update_ =
        cbs_belief_generation_time_sec_per_update_;

    for (const auto& odom : outgoing) {
      const gtsam::Symbol from_symbol(odom.from_pose_key);
      const gtsam::Symbol to_symbol(odom.to_pose_key);
      if (from_symbol.chr() != kPoseSymbolChar ||
          to_symbol.chr() != kPoseSymbolChar) {
        continue;
      }

      const FrameId from_frame_id = static_cast<FrameId>(from_symbol.index());
      const FrameId to_frame_id = static_cast<FrameId>(to_symbol.index());
      const auto from_stamp_it = keyframe_timestamp_sec_.find(from_frame_id);
      const auto to_stamp_it = keyframe_timestamp_sec_.find(to_frame_id);
      if (from_stamp_it == keyframe_timestamp_sec_.end() ||
          to_stamp_it == keyframe_timestamp_sec_.end()) {
        continue;
      }

      ExternalOdometryBelief stamped_belief;
      stamped_belief.source_agent = static_cast<uint8_t>('k');
      stamped_belief.from_pose_index = static_cast<uint32_t>(from_frame_id);
      stamped_belief.to_pose_index = static_cast<uint32_t>(to_frame_id);
      stamped_belief.sender_from_pose_index = stamped_belief.from_pose_index;
      stamped_belief.sender_to_pose_index = stamped_belief.to_pose_index;
      stamped_belief.from_stamp_sec = from_stamp_it->second;
      stamped_belief.to_stamp_sec = to_stamp_it->second;
      stamped_belief.sender_timestamp_ns =
          static_cast<uint64_t>(std::llround(stamped_belief.to_stamp_sec * 1e9));
      stamped_belief.sender_frame_id = "odom";
      stamped_belief.received_wall_time_sec = wallTimeNowSec();
      stamped_belief.relax_factor = odom.relax_factor;
      vector6ToArray(gtsam::Pose3::Logmap(odom.measured_from_to),
                     &stamped_belief.relative_mu);
      matrix6ToArray(poseCovarianceFromMatrix(odom.covariance),
                     &stamped_belief.covariance);
      cbs_outgoing_odom_beliefs_.push_back(stamped_belief);
    }
    if (latest_pair_valid && !cbs_outgoing_odom_beliefs_.empty()) {
      last_cbs_outgoing_odom_pair_valid_ = true;
      last_cbs_outgoing_odom_from_key_ = latest_from_key;
      last_cbs_outgoing_odom_to_key_ = latest_to_key;
    }
  } catch (const std::exception& e) {
    LOG(WARNING) << "Kimera CBS getOdometryBeliefs failed: " << e.what();
  } catch (...) {
    LOG(WARNING) << "Kimera CBS getOdometryBeliefs failed.";
  }
  cbs_outgoing_total_time_sec_per_update_ = elapsedSec(cbs_outgoing_start_time);
  LOG(INFO) << "KIMERA_CBS_OUTGOING_TIMING_ROW,"
            << cur_id << ","
            << secToMs(cbs_outgoing_total_time_sec_per_update_) << ","
            << secToMs(cbs_set_marginalization_graph_time_sec_per_update_) << ","
            << secToMs(cbs_get_odometry_beliefs_time_sec_per_update_) << ","
            << cbs_outgoing_odom_beliefs_.size() << ","
            << cbs_marginalization_graph_factor_count_ << ","
            << request_keys.size();
}

void VioBackend::refreshExternalBeliefFactorSlots(
    const size_t num_factors_before_external,
    const std::vector<ExternalBeliefFactorId>& inserted_external_factor_ids) {
  if (inserted_external_factor_ids.empty()) {
    return;
  }

  const auto& new_factor_indices = getMainBackendResult().newFactorsIndices;
  const size_t required_size =
      num_factors_before_external + inserted_external_factor_ids.size();
  if (new_factor_indices.size() < required_size) {
    LOG(WARNING) << "Unexpected iSAM2 result while mapping external belief "
                 << "factor slots: expected at least " << required_size
                 << " entries, got " << new_factor_indices.size() << ".";
    return;
  }

  for (size_t i = 0u; i < inserted_external_factor_ids.size(); ++i) {
    const size_t slot = new_factor_indices[num_factors_before_external + i];
    if (!mainBackendFactorExists(slot)) {
      continue;
    }
    active_external_belief_factor_slots_.push_back(
        ExternalBeliefFactorSlot{inserted_external_factor_ids[i], slot});
  }
}

/* -------------------------------------------------------------------------- */
bool VioBackend::initStateAndSetPriors(
    const VioNavStateTimestamped& vio_nav_state_initial_seed) {
  // Clean state
  new_values_.clear();

  // Update member variables.
  timestamp_lkf_ = vio_nav_state_initial_seed.timestamp_;

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
    const GtsamPreintegrationType& pim,
    std::optional<gtsam::Pose3> odometry_body_pose,
    std::optional<gtsam::Velocity3> odometry_vel) {
  debug_info_.resetAddedFactorsStatistics();

  // Features and IMU line up --> do iSAM update
  last_kf_id_ = curr_kf_id_;
  ++curr_kf_id_;

  VLOG(1) << "VIO: adding keyframe " << curr_kf_id_
          << " at timestamp:" << UtilsNumerical::NsecToSec(timestamp_kf_nsec)
          << " (nsec).";

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
      ft.in_ba_graph_ = true;
      addLandmarkToGraph(lmk_id, ft);
      ++n_new_landmarks;
    } else {
      const std::pair<FrameId, StereoPoint2> obs_kf = ft.obs_.back();

      LOG_IF(FATAL, obs_kf.first != static_cast<FrameId>(curr_kf_id_))
          << "addLandmarksToGraph: last obs is not from the current "
             "keyframe!\n";

      updateLandmarkInGraph(lmk_id, obs_kf);
      ++n_updated_landmarks;
    }
  }

  VLOG(10) << "Added " << n_new_landmarks << " new landmarks\n"
           << "Updated " << n_updated_landmarks << " landmarks in graph";
}

/* -------------------------------------------------------------------------- */
// Adds a landmark to the graph for the first time.
void VioBackend::addLandmarkToGraph(const LandmarkId& lmk_id,
                                    const FeatureTrack& ft) {
  // We use a unit pinhole projection camera for the smart factors to be
  // more efficient.
  SmartStereoFactor::shared_ptr new_factor(new SmartStereoFactor(
      smart_noise_, smart_factors_params_, B_Pose_leftCamRect_));

  VLOG(10) << "Adding landmark with: " << ft.obs_.size()
           << " landmarks to graph, with keys: ";

  // Add observations to smart factor
  if (VLOG_IS_ON(10)) new_factor->print();
  std::stringstream ss;
  for (const std::pair<FrameId, StereoPoint2>& obs : ft.obs_) {
    const FrameId& frame_id = obs.first;
    const gtsam::Symbol& pose_symbol = gtsam::Symbol(kPoseSymbolChar, frame_id);
    const StereoPoint2& measurement = obs.second;
    new_factor->add(measurement, pose_symbol, stereo_cal_);

    if (VLOG_IS_ON(10)) ss << " " << obs.first;
  }
  VLOG(10) << ss.str() << std::endl;

  // add new factor to suitable structures:
  new_smart_factors_.insert(std::make_pair(lmk_id, new_factor));
  old_smart_factors_.insert(
      std::make_pair(lmk_id, std::make_pair(new_factor, -1)));
  if (local_belief_cov_smoother_) {
    old_smart_factors_local_belief_cov_.insert(
        std::make_pair(lmk_id, std::make_pair(new_factor, -1)));
  }
}

/* -------------------------------------------------------------------------- */
// Updates a landmark already in the graph.
void VioBackend::updateLandmarkInGraph(
    const LandmarkId& lmk_id,
    const std::pair<FrameId, StereoPoint2>& new_measurement) {
  // Update existing smart-factor
  auto old_smart_factors_it = old_smart_factors_.find(lmk_id);
  CHECK(old_smart_factors_it != old_smart_factors_.end())
      << "Landmark not found in old_smart_factors_ with id: " << lmk_id;

  const auto& old_factor = old_smart_factors_it->second.first;
  // Clone old factor to keep all previous measurements, now append one.
  SmartStereoFactor::shared_ptr new_factor(new SmartStereoFactor(*old_factor));

  const gtsam::Symbol pose_symbol(kPoseSymbolChar, new_measurement.first);
  const StereoPoint2& measurement = new_measurement.second;
  new_factor->add(measurement, pose_symbol, stereo_cal_);

  // Update the factor
  Slot slot = old_smart_factors_it->second.second;
  if (slot != -1) {
    new_smart_factors_.insert(std::make_pair(lmk_id, new_factor));
  } else {
    // If it's slot in the graph is still -1, it means that the factor has not
    // been inserted yet in the graph...
    LOG(FATAL) << "When updating the smart factor, its slot should not be -1!"
                  " Offensive lmk_id: "
               << lmk_id;
  }
  old_smart_factors_it->second.first = new_factor;
  if (local_belief_cov_smoother_) {
    auto it_local = old_smart_factors_local_belief_cov_.find(lmk_id);
    if (it_local != old_smart_factors_local_belief_cov_.end()) {
      it_local->second.first = new_factor;
    } else {
      old_smart_factors_local_belief_cov_.insert(
          std::make_pair(lmk_id, std::make_pair(new_factor, -1)));
    }
  }
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
    const gtsam::LabeledSymbol labeled_key(key_value.key);
    if (cbs::isRobotKey(labeled_key) ||
        labeled_key.label() == cbs::kPoseLabel) {
      continue;
    }

    const auto lmk_id = key.index();
    DCHECK(points_with_id.find(lmk_id) == points_with_id.end());
    try {
      points_with_id[lmk_id] = key_value.value.cast<gtsam::Point3>();
    } catch (const std::exception& e) {
      VLOG(5) << "Skipping non-Point3 value with landmark-like key "
              << gtsam::DefaultKeyFormatter(key_value.key) << ": " << e.what();
      continue;
    } catch (...) {
      VLOG(5) << "Skipping non-Point3 value with landmark-like key "
              << gtsam::DefaultKeyFormatter(key_value.key) << ".";
      continue;
    }
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

/* -------------------------------------------------------------------------- */
// NOT TESTED (--> There is a UnitTest function in UtilsOpenCV)
void VioBackend::computeStateCovariance() {
  gtsam::Marginals marginals(getMainBackendFactors(),
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
}

void VioBackend::logPoseBeliefCovarianceSanityDiff(
    const FrameId& cur_id) const {
  if (!FLAGS_cbs_log_covariance_sanity_diff) {
    return;
  }

  bool local_valid = pose_belief_local_covariance_valid_;
  bool fused_valid = false;
  gtsam::Matrix6 local_cov = gtsam::Matrix6::Zero();
  gtsam::Matrix6 fused_cov = gtsam::Matrix6::Zero();
  double trace_local = std::numeric_limits<double>::quiet_NaN();
  double trace_fused = std::numeric_limits<double>::quiet_NaN();
  double trace_ratio = std::numeric_limits<double>::quiet_NaN();
  double frob_delta = std::numeric_limits<double>::quiet_NaN();
  double max_abs_delta = std::numeric_limits<double>::quiet_NaN();

  if (local_valid) {
    local_cov = 0.5 * (pose_belief_local_covariance_lkf_ +
                       pose_belief_local_covariance_lkf_.transpose());
    local_valid = local_cov.allFinite();
    if (local_valid) {
      trace_local = local_cov.trace();
    }
  }

  try {
    gtsam::Marginals marginals(getMainBackendFactors(),
                               state_,
                               gtsam::Marginals::Factorization::CHOLESKY);
    gtsam::KeyVector keys;
    keys.push_back(gtsam::Symbol(kPoseSymbolChar, cur_id));
    const gtsam::Matrix fused_cov_dyn =
        marginals.jointMarginalCovariance(keys).fullMatrix();
    if (fused_cov_dyn.rows() == 6 && fused_cov_dyn.cols() == 6) {
      fused_cov = fused_cov_dyn;
      fused_cov = 0.5 * (fused_cov + fused_cov.transpose());
      fused_valid = fused_cov.allFinite();
      if (fused_valid) {
        trace_fused = fused_cov.trace();
      }
    }
  } catch (const std::exception& e) {
    LOG(WARNING) << "CBS covariance sanity: failed fused-pose covariance at "
                 << "frame " << cur_id << " with: " << e.what();
  }

  if (local_valid && fused_valid) {
    const gtsam::Matrix6 delta = local_cov - fused_cov;
    frob_delta = delta.norm();
    max_abs_delta = delta.cwiseAbs().maxCoeff();
    if (std::abs(trace_fused) > 1e-12) {
      trace_ratio = trace_local / trace_fused;
    }
  }

  LOG(INFO) << "CBS covariance sanity frame=" << cur_id
            << " source=" << pose_belief_covariance_source_
            << " local_valid=" << local_valid << " fused_valid=" << fused_valid
            << " trace_local=" << trace_local << " trace_fused=" << trace_fused
            << " trace_ratio=" << trace_ratio << " frob_delta=" << frob_delta
            << " max_abs_delta=" << max_abs_delta;
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
                                const GtsamPreintegrationType& pim,
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
                              const GtsamPreintegrationType& pim) {
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
  const bool use_local_belief_cov_sidecar = false;
  external_beliefs_received_per_update_ = 0u;
  external_beliefs_added_per_update_ = 0u;
  external_beliefs_rejected_first_message_per_update_ = 0u;
  external_beliefs_rejected_update_status_per_update_ = 0u;
  external_beliefs_rejected_inactive_window_per_update_ = 0u;
  external_beliefs_rejected_shape_per_update_ = 0u;
  external_beliefs_rejected_exception_per_update_ = 0u;
  optimization_time_sec_per_update_ = 0.0;
  cbs_belief_generation_time_sec_per_update_ = 0.0;
  cbs_outgoing_total_time_sec_per_update_ = 0.0;
  cbs_set_marginalization_graph_time_sec_per_update_ = 0.0;
  cbs_get_odometry_beliefs_time_sec_per_update_ = 0.0;
  cbs_marginalization_graph_factor_count_ = 0u;
  cbs_outgoing_odom_beliefs_.clear();
  double factor_preparation_time_sec = 0.0;
  double collect_external_beliefs_time_sec = 0.0;
  double delete_slots_sort_time_sec = 0.0;
  double smoother_update_time_sec = 0.0;
  double slot_bookkeeping_time_sec = 0.0;
  double extra_iterations_time_sec = 0.0;
  double update_states_time_sec = 0.0;
  double compute_state_covariance_time_sec = 0.0;
  double post_debug_time_sec = 0.0;

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
  gtsam::FactorIndices local_delete_slots;
  updateKeyframeTimestampIndex(cur_id, timestamp_kf_nsec);

  // TODO we know the actual end size... but I am not sure how to use factor
  // graph API for appending factors without copying or re-allocation...
  std::vector<LandmarkId> lmk_ids_of_new_smart_factors_tmp;
  std::vector<LandmarkId> lmk_ids_of_new_smart_factors_local_belief_cov;
  lmk_ids_of_new_smart_factors_tmp.reserve(new_smart_factors_size);
  lmk_ids_of_new_smart_factors_local_belief_cov.reserve(new_smart_factors_size);
  gtsam::NonlinearFactorGraph new_factors_tmp;
  gtsam::NonlinearFactorGraph new_factors_local_belief_cov;
  new_factors_tmp.reserve(new_smart_factors_size +
                          new_imu_prior_and_other_factors_.size());
  new_factors_local_belief_cov.reserve(new_smart_factors_size +
                                       new_imu_prior_and_other_factors_.size());
  for (const auto& new_smart_factor : new_smart_factors_) {
    // Push back the smart factor to the list of new factors to add to the
    // graph. // Smart factor, so same address right?
    LandmarkId lmk_id = new_smart_factor.first;  // don't use &

    // Find smart factor and slot in old_smart_factors_ corresponding to
    // the lmk with id of the new smart factor.
    const auto& old_smart_factor_it = old_smart_factors_.find(lmk_id);
    CHECK(old_smart_factor_it != old_smart_factors_.end())
        << "Lmk with id: " << lmk_id
        << " could not be found in old_smart_factors_.";

    Slot slot = old_smart_factor_it->second.second;
    if (slot != -1) {
      // Smart factor Slot is different than -1, therefore the factor should be
      // already in the factor graph.
      DCHECK_GE(slot, 0);
      if (mainBackendFactorExists(slot)) {
        // Confirmed, the factor is in the graph.
        // We must delete the old smart factor from the graph.
        // TODO what happens if delete_slots has repeated elements?
        delete_slots.push_back(slot);
        // And we must add the new smart factor to the graph.
        new_factors_tmp.push_back(new_smart_factor.second);
        // Store lmk id of the smart factor to add to the graph.
        lmk_ids_of_new_smart_factors_tmp.push_back(lmk_id);
      } else {
        // This should not happen, unless feature tracks are so long
        // (longer than factor graph's time horizon), than the factor has been
        // removed from the optimization.
        // Erase this factor and feature track, as it has gone past the horizon.
        // TODO(marcus): check with toni if this needs a warning
        old_smart_factors_.erase(old_smart_factor_it);
        CHECK(deleteLmkFromFeatureTracks(lmk_id));
        // TODO(Toni): we should as well remove it from new_smart_factors_!!
      }
    } else {
      // We just add the new smart factor to the graph, as it has never been
      // there before.
      new_factors_tmp.push_back(new_smart_factor.second);
      // Store lmk id of the smart factor to add to the graph.
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

  if (use_local_belief_cov_sidecar) {
    // Keep local-smart-factor bookkeeping aligned with the main map.
    for (auto it = old_smart_factors_local_belief_cov_.begin();
         it != old_smart_factors_local_belief_cov_.end();) {
      if (!old_smart_factors_.exists(it->first)) {
        it = old_smart_factors_local_belief_cov_.erase(it);
      } else {
        ++it;
      }
    }

    // For the local covariance smoother we only remove smart-factor slots that
    // belong to this local graph.
    local_delete_slots.clear();
    for (const auto& new_smart_factor : new_smart_factors_) {
      const LandmarkId lmk_id = new_smart_factor.first;
      auto it_local = old_smart_factors_local_belief_cov_.find(lmk_id);
      if (it_local == old_smart_factors_local_belief_cov_.end()) {
        it_local = old_smart_factors_local_belief_cov_
                       .insert(std::make_pair(
                           lmk_id, std::make_pair(new_smart_factor.second, -1)))
                       .first;
      }

      Slot local_slot = it_local->second.second;
      if (local_slot != -1) {
        if (local_belief_cov_smoother_) {
          if (local_belief_cov_smoother_->getFactors().exists(local_slot)) {
            local_delete_slots.push_back(local_slot);
          }
        } else {
          local_delete_slots.push_back(static_cast<size_t>(local_slot));
        }
      }

      new_factors_local_belief_cov.push_back(new_smart_factor.second);
      lmk_ids_of_new_smart_factors_local_belief_cov.push_back(lmk_id);
      it_local->second.first = new_smart_factor.second;
    }

    new_factors_local_belief_cov.push_back(
        new_imu_prior_and_other_factors_.begin(),
        new_imu_prior_and_other_factors_.end());

    std::sort(local_delete_slots.begin(), local_delete_slots.end());
    local_delete_slots.erase(
        std::unique(local_delete_slots.begin(), local_delete_slots.end()),
        local_delete_slots.end());
  }

  const size_t num_factors_before_external = new_factors_tmp.size();
  std::vector<ExternalBeliefFactorId> inserted_external_factor_ids;
  factor_preparation_time_sec = elapsedSec(total_start_time);
  const auto collect_external_beliefs_start_time = utils::Timer::tic();
  try {
    collectExternalBeliefFactors(
        cur_id, &delete_slots, &new_factors_tmp, &inserted_external_factor_ids);
  } catch (const std::exception& e) {
    LOG(ERROR) << "VioBackend::optimize failed in collectExternalBeliefFactors "
               << "for frame " << cur_id << ": " << e.what();
    throw;
  } catch (...) {
    LOG(ERROR) << "VioBackend::optimize failed in collectExternalBeliefFactors "
               << "for frame " << cur_id << ".";
    throw;
  }
  collect_external_beliefs_time_sec =
      elapsedSec(collect_external_beliefs_start_time);

  // Avoid repeated deletions of the same slot when replacing beliefs and
  // removing stale factors in the same iteration.
  const auto delete_slots_sort_start_time = utils::Timer::tic();
  std::sort(delete_slots.begin(), delete_slots.end());
  delete_slots.erase(std::unique(delete_slots.begin(), delete_slots.end()),
                     delete_slots.end());
  delete_slots_sort_time_sec = elapsedSec(delete_slots_sort_start_time);

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

  if (VLOG_IS_ON(10)) {
    printSmootherInfo(new_factors_tmp,
                      delete_slots,
                      "Smoother status before update:",
                      VLOG_IS_ON(10));
  }

  // Recreate the graph before marginalization.
  if (VLOG_IS_ON(10) && FLAGS_debug_graph_before_opt) {
    debug_info_.graphBeforeOpt = getMainBackendFactors();
    debug_info_.graphToBeDeleted = gtsam::NonlinearFactorGraph();
    debug_info_.graphToBeDeleted.resize(delete_slots.size());
    for (size_t i = 0u; i < delete_slots.size(); i++) {
      // If the factor is to be deleted, store it as graph to be deleted.
      CHECK(mainBackendFactorExists(delete_slots.at(i)));
      debug_info_.graphToBeDeleted.at(i) =
          mainBackendFactorAt(delete_slots.at(i));
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
  gtsam::FixedLagSmoother::Result result;
  VLOG(10) << "Starting first update.";
  const auto optimizer_start_time = utils::Timer::tic();
  bool is_smoother_ok = false;
  try {
    is_smoother_ok = updateSmoother(
        &result, new_factors_tmp, new_values_, key_frame_count, delete_slots);
  } catch (const std::exception& e) {
    LOG(ERROR) << "VioBackend::optimize failed in updateSmoother for frame "
               << cur_id << ": " << e.what();
    throw;
  } catch (...) {
    LOG(ERROR) << "VioBackend::optimize failed in updateSmoother for frame "
               << cur_id << ".";
    throw;
  }
  optimization_time_sec_per_update_ +=
      utils::Timer::toc<std::chrono::duration<double>>(optimizer_start_time)
          .count();
  smoother_update_time_sec = optimization_time_sec_per_update_;
  VLOG(10) << "Finished first update.";

  // Store time after iSAM update.
  if (VLOG_IS_ON(10) || log_output_) {
    debug_info_.updateTime_ =
        utils::Timer::toc<std::chrono::seconds>(start_time).count();
    start_time = utils::Timer::tic();
  }

  /////////////////////////// BOOKKEEPING //////////////////////////////////////
  if (is_smoother_ok) {
    const auto slot_bookkeeping_start_time = utils::Timer::tic();
    pose_belief_local_covariance_valid_ = false;
    pose_belief_covariance_source_ = "unused_bpsam_getBeliefs";

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
    refreshExternalBeliefFactorSlots(num_factors_before_external,
                                     inserted_external_factor_ids);
    slot_bookkeeping_time_sec = elapsedSec(slot_bookkeeping_start_time);
    VLOG(10) << "Finished to find smart factors slots.";

    if (VLOG_IS_ON(5) || log_output_) {
      debug_info_.updateSlotTime_ =
          utils::Timer::toc<std::chrono::seconds>(start_time).count();
      start_time = utils::Timer::tic();
    }

    ////////////////////////////////////////////////////////////////////////////

    // Do some more optimization iterations.
    const auto extra_iterations_start_time = utils::Timer::tic();
    for (size_t n_iter = 1; n_iter < max_extra_iterations && is_smoother_ok;
         ++n_iter) {
      VLOG(10) << "Doing extra iteration nr: " << n_iter;
      const auto extra_iteration_start_time = utils::Timer::tic();
      try {
        is_smoother_ok = updateSmoother(&result);
      } catch (const std::exception& e) {
        LOG(ERROR) << "VioBackend::optimize failed in extra updateSmoother "
                   << "iteration " << n_iter << " for frame " << cur_id << ": "
                   << e.what();
        throw;
      } catch (...) {
        LOG(ERROR) << "VioBackend::optimize failed in extra updateSmoother "
                   << "iteration " << n_iter << " for frame " << cur_id << ".";
        throw;
      }
      optimization_time_sec_per_update_ +=
          utils::Timer::toc<std::chrono::duration<double>>(
              extra_iteration_start_time)
              .count();
    }
    extra_iterations_time_sec = elapsedSec(extra_iterations_start_time);
    smoother_update_time_sec = optimization_time_sec_per_update_;

    if (VLOG_IS_ON(5) || log_output_) {
      debug_info_.extraIterationsTime_ =
          utils::Timer::toc<std::chrono::seconds>(start_time).count();
      start_time = utils::Timer::tic();
    }

    // Update states we need for next iteration, if smoother is ok.
    if (is_smoother_ok) {
      const auto update_states_start_time = utils::Timer::tic();
      try {
        updateStates(cur_id);
      } catch (const std::exception& e) {
        LOG(ERROR) << "VioBackend::optimize failed in updateStates for frame "
                   << cur_id << ": " << e.what();
        throw;
      } catch (...) {
        LOG(ERROR) << "VioBackend::optimize failed in updateStates for frame "
                   << cur_id << ".";
        throw;
      }
      update_states_time_sec = elapsedSec(update_states_start_time);

      try {
        refreshCbsOutgoingBeliefs(cur_id);
      } catch (const std::exception& e) {
        LOG(ERROR)
            << "VioBackend::optimize failed in refreshCbsOutgoingBeliefs "
            << "for frame " << cur_id << ": " << e.what();
        throw;
      } catch (...) {
        LOG(ERROR)
            << "VioBackend::optimize failed in refreshCbsOutgoingBeliefs "
            << "for frame " << cur_id << ".";
        throw;
      }

      // TODO: Add Update latest covariance --> move flag
      if (FLAGS_compute_state_covariance) {
        const auto compute_state_covariance_start_time = utils::Timer::tic();
        computeStateCovariance();
        compute_state_covariance_time_sec =
            elapsedSec(compute_state_covariance_start_time);
      }
      if (FLAGS_cbs_log_covariance_sanity_diff) {
        logPoseBeliefCovarianceSanityDiff(cur_id);
      }

      // Debug.
      const auto post_debug_start_time = utils::Timer::tic();
      postDebug(total_start_time, start_time);
      post_debug_time_sec = elapsedSec(post_debug_start_time);
    } else {
      LOG(ERROR) << "Smoother is not ok! Not updating Backend state.";
    }
  }
  const double optimize_total_time_sec = elapsedSec(total_start_time);
  LOG(INFO) << "KIMERA_OPTIMIZE_TIMING_ROW,"
            << cur_id << ","
            << secToMs(optimize_total_time_sec) << ","
            << secToMs(factor_preparation_time_sec) << ","
            << secToMs(collect_external_beliefs_time_sec) << ","
            << secToMs(delete_slots_sort_time_sec) << ","
            << secToMs(smoother_update_time_sec) << ","
            << secToMs(slot_bookkeeping_time_sec) << ","
            << secToMs(extra_iterations_time_sec) << ","
            << secToMs(update_states_time_sec) << ","
            << secToMs(cbs_outgoing_total_time_sec_per_update_) << ","
            << secToMs(cbs_set_marginalization_graph_time_sec_per_update_) << ","
            << secToMs(cbs_get_odometry_beliefs_time_sec_per_update_) << ","
            << secToMs(compute_state_covariance_time_sec) << ","
            << secToMs(post_debug_time_sec) << ","
            << cbs_outgoing_odom_beliefs_.size() << ","
            << cbs_marginalization_graph_factor_count_ << ","
            << (is_smoother_ok ? 1 : 0);
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
  VLOG(10) << "Starting to calculate estimate.";
  state_ = calculateMainBackendEstimate();
  VLOG(10) << "Finished to calculate estimate.";

  DCHECK(state_.find(gtsam::Symbol(kPoseSymbolChar, cur_id)) != state_.end());
  DCHECK(state_.find(gtsam::Symbol(kVelocitySymbolChar, cur_id)) !=
         state_.end());
  DCHECK(state_.find(gtsam::Symbol(kImuBiasSymbolChar, cur_id)) !=
         state_.end());

  gtsam::Pose3 W_Pose_B_kf =
      state_.at<gtsam::Pose3>(gtsam::Symbol(kPoseSymbolChar, cur_id));
  gtsam::Pose3 W_Pose_B_lkf = gtsam::Pose3();
  gtsam::Pose3 B_lkf_Pose_kf = gtsam::Pose3();

  // If we have an available pose at cur_id - 1 we use it, otw identity
  // gives us W_Pose_B_lkf as our current pose estimate.
  if (cur_id > 0) {
    DCHECK(state_.find(gtsam::Symbol(kPoseSymbolChar, cur_id - 1)) !=
           state_.end());
    W_Pose_B_lkf =
        state_.at<gtsam::Pose3>(gtsam::Symbol(kPoseSymbolChar, cur_id - 1));

    // Compute relative pose as odometry to append to pose estimate trajectory
    B_lkf_Pose_kf = W_Pose_B_lkf.between(W_Pose_B_kf);
  }

  // Update latest state estimate
  W_Pose_B_lkf_from_state_ = W_Pose_B_kf;
  W_Vel_B_lkf_ = state_.at<Vector3>(gtsam::Symbol(kVelocitySymbolChar, cur_id));
  imu_bias_lkf_ = state_.at<gtsam::imuBias::ConstantBias>(
      gtsam::Symbol(kImuBiasSymbolChar, cur_id));

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

bool VioBackend::updateSmoother(gtsam::FixedLagSmoother::Result* result,
                                const gtsam::NonlinearFactorGraph& new_factors,
                                const gtsam::Values& new_values,
                                const std::map<Key, double>& timestamps,
                                const gtsam::FactorIndices& delete_slots) {
  CHECK_NOTNULL(result);
  if (usePersistentBpsamMainBackend()) {
    auto try_main_backend_update =
        [&](const gtsam::NonlinearFactorGraph& factors_to_add,
            const gtsam::Values& values_to_add) {
          std::vector<size_t> ignored_smart_slots;
          return main_bpsam_backend_->update(factors_to_add,
                                             values_to_add,
                                             timestamps,
                                             delete_slots,
                                             0u,
                                             &ignored_smart_slots);
        };

    if (try_main_backend_update(new_factors, new_values)) {
      return true;
    }

    LOG(ERROR) << "Persistent BPSAM main-backend update failed.";
    if (new_values.empty()) {
      return false;
    }

    gtsam::Values estimate;
    try {
      estimate = main_bpsam_backend_->calculateEstimate();
    } catch (const std::exception& e) {
      LOG(ERROR) << "Persistent BPSAM recovery estimate failed: " << e.what();
      return false;
    } catch (...) {
      LOG(ERROR) << "Persistent BPSAM recovery estimate failed with unknown "
                    "exception.";
      return false;
    }

    if (estimate.empty()) {
      LOG(ERROR) << "Persistent BPSAM recovery skipped: estimate is empty.";
      return false;
    }

    // Failure-only recovery: inject soft priors on earliest and latest x/v/b
    // states available in the current estimate (similar intent to native path).
    auto min_max_pose_indices =
        [&estimate]() -> std::optional<std::pair<size_t, size_t>> {
      bool has_pose = false;
      size_t min_idx = 0u;
      size_t max_idx = 0u;
      for (const auto& kv : estimate) {
        const gtsam::Symbol key(kv.key);
        if (key.chr() != kPoseSymbolChar) {
          continue;
        }
        if (!has_pose) {
          min_idx = key.index();
          max_idx = key.index();
          has_pose = true;
          continue;
        }
        min_idx = std::min(min_idx, key.index());
        max_idx = std::max(max_idx, key.index());
      }
      if (!has_pose) {
        return std::nullopt;
      }
      return std::make_pair(min_idx, max_idx);
    };

    const auto idx_pair = min_max_pose_indices();
    if (!idx_pair) {
      LOG(ERROR) << "Persistent BPSAM recovery skipped: no pose keys in "
                    "estimate.";
      return false;
    }

    const std::vector<size_t> frame_indices =
        (idx_pair->first == idx_pair->second)
            ? std::vector<size_t>{idx_pair->first}
            : std::vector<size_t>{idx_pair->first, idx_pair->second};
    gtsam::NonlinearFactorGraph recovery_factors;
    recovery_factors.reserve(new_factors.size() + frame_indices.size() * 3u);
    recovery_factors.push_back(new_factors.begin(), new_factors.end());

    size_t num_recovery_priors = 0u;
    for (const size_t idx : frame_indices) {
      const gtsam::Symbol pose_key(kPoseSymbolChar, idx);
      const gtsam::Symbol vel_key(kVelocitySymbolChar, idx);
      const gtsam::Symbol bias_key(kImuBiasSymbolChar, idx);

      if (estimate.exists(pose_key)) {
        const gtsam::Pose3 pose = estimate.at<gtsam::Pose3>(pose_key);
        gtsam::Vector6 sigmas;
        sigmas.head<3>().setConstant(0.01);
        sigmas.tail<3>().setConstant(0.1);
        const gtsam::SharedNoiseModel noise =
            gtsam::noiseModel::Diagonal::Sigmas(sigmas);
        recovery_factors.emplace_shared<gtsam::PriorFactor<gtsam::Pose3>>(
            pose_key, pose, noise);
        ++num_recovery_priors;
      }

      if (estimate.exists(vel_key)) {
        const gtsam::Vector3 vel = estimate.at<gtsam::Vector3>(vel_key);
        const gtsam::SharedNoiseModel noise =
            gtsam::noiseModel::Diagonal::Sigmas(
                (gtsam::Vector3() << 0.1, 0.1, 0.1).finished());
        recovery_factors.emplace_shared<gtsam::PriorFactor<gtsam::Vector3>>(
            vel_key, vel, noise);
        ++num_recovery_priors;
      }

      if (estimate.exists(bias_key)) {
        const gtsam::imuBias::ConstantBias bias =
            estimate.at<gtsam::imuBias::ConstantBias>(bias_key);
        gtsam::Vector6 sigmas;
        sigmas.head<3>().setConstant(backend_params_.initialAccBiasSigma_);
        sigmas.tail<3>().setConstant(backend_params_.initialGyroBiasSigma_);
        const gtsam::SharedNoiseModel noise =
            gtsam::noiseModel::Diagonal::Sigmas(sigmas);
        recovery_factors
            .emplace_shared<gtsam::PriorFactor<gtsam::imuBias::ConstantBias>>(
                bias_key, bias, noise);
        ++num_recovery_priors;
      }
    }

    if (num_recovery_priors == 0u) {
      LOG(ERROR) << "Persistent BPSAM recovery skipped: no eligible x/v/b keys "
                    "for priors.";
      return false;
    }

    LOG(WARNING) << "Persistent BPSAM recovery retry with "
                 << num_recovery_priors << " temporary priors.";
    const bool recovered =
        try_main_backend_update(recovery_factors, new_values);
    if (!recovered) {
      LOG(ERROR) << "Persistent BPSAM recovery retry failed.";
      return false;
    }
    LOG(WARNING) << "Persistent BPSAM recovery retry succeeded.";
    return true;
  }

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

    // Add priors on keys with these prefixes (pose, imu bias, velocity)
    std::vector<unsigned char> key_prefixes_to_prior = {'x', 'b', 'v'};
    gtsam::Symbol first_key = values.keys().at(0);
    gtsam::KeyVector prior_keys;
    for (const auto& prefix : key_prefixes_to_prior) {
      prior_keys.push_back(gtsam::Symbol(prefix, symb.index()));
      prior_keys.push_back(gtsam::Symbol(prefix, first_key.index()));
    }
    CHECK_EQ(prior_keys.size(), 6u);
    gtsam::NonlinearFactorGraph nfg;

    // Only add priors on first state and the state nearest the failure
    for (const gtsam::Symbol& key : prior_keys) {
      CHECK(values.exists(key));
      LOG(ERROR) << "Adding prior on key: " << key.chr() << key.index();
      switch (key.chr()) {
        case 'x': {
          gtsam::Pose3 pose = values.at<gtsam::Pose3>(key);
          gtsam::Vector6 sigmas;
          sigmas.head<3>().setConstant(0.01);  // rotation
          sigmas.tail<3>().setConstant(0.1);   // translation
          gtsam::SharedNoiseModel noise =
              gtsam::noiseModel::Diagonal::Sigmas(sigmas);
          nfg.emplace_shared<gtsam::PriorFactor<gtsam::Pose3>>(
              key, pose, noise);
          break;
        }
        case 'b': {
          gtsam::imuBias::ConstantBias bias =
              values.at<gtsam::imuBias::ConstantBias>(key);
          gtsam::Vector6 sigmas;
          sigmas.head<3>().setConstant(backend_params_.initialAccBiasSigma_);
          sigmas.tail<3>().setConstant(backend_params_.initialGyroBiasSigma_);
          gtsam::SharedNoiseModel noise =
              gtsam::noiseModel::Diagonal::Sigmas(sigmas);
          nfg.emplace_shared<gtsam::PriorFactor<gtsam::imuBias::ConstantBias>>(
              key, bias, noise);
          break;
        }
        case 'v': {
          gtsam::Vector3 vel = values.at<gtsam::Vector3>(key);
          gtsam::Vector3 sigmas;
          sigmas.setConstant(0.1);
          gtsam::SharedNoiseModel noise =
              gtsam::noiseModel::Diagonal::Sigmas(sigmas);
          nfg.emplace_shared<gtsam::PriorFactor<gtsam::Vector3>>(
              key, vel, noise);
          break;
        }
        default: {
          LOG(FATAL)
              << "Key not recognized in indeterminant exception handling.";
        }
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
      return false;
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
    printSmootherInfo(new_factors, delete_slots);
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
  VLOG(10) << "Finished delete from timestamps.";

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
void VioBackend::updateNewSmartFactorsSlots(
    const std::vector<LandmarkId>& lmk_ids_of_new_smart_factors,
    SmartFactorMap* old_smart_factors) {
  CHECK_NOTNULL(old_smart_factors);

  // Get result.
  const gtsam::ISAM2Result& result = getMainBackendResult();

  // Simple version of find smart factors.
  for (size_t i = 0u; i < lmk_ids_of_new_smart_factors.size(); ++i) {
    DCHECK(i < result.newFactorsIndices.size())
        << "There are more new smart factors than new factors added to the "
           "graph.";
    // Get new slot in the graph for the newly added smart factor.
    const size_t& slot = result.newFactorsIndices.at(i);
    if (!mainBackendFactorExists(slot)) {
      continue;
    }

    // TODO this will not work if there are non-smart factors!!!
    // Update slot using isam2 indices.
    // ORDER of inclusion of factors in the ISAM2::update() function
    // matters, as these indices have a 1-to-1 correspondence with the
    // factors.

    // BOOKKEEPING, for next iteration to know which slots have to be
    // deleted before adding the new smart factors. Find the entry in
    // old_smart_factors_.
    const auto& it =
        old_smart_factors->find(lmk_ids_of_new_smart_factors.at(i));

    DCHECK(it != old_smart_factors->end())
        << "Trying to access unavailable factor.";
    // CHECK that the factor in the graph at slot position is a smart
    // factor.
    const auto sptr =
        dynamic_cast<const SmartStereoFactor*>(mainBackendFactorAt(slot).get());
    DCHECK(sptr);
    // CHECK that shared ptrs point to the same smart factor.
    // make sure no one is cloning SmartSteroFactors.
    DCHECK_EQ(it->second.first.get(), sptr)
        << "Non-matching addresses for same factors for lmk with id: "
        << lmk_ids_of_new_smart_factors.at(i) << " in old_smart_factors_ "
        << "VS factor in graph at slot: " << slot
        << ". Slot previous to update was: " << it->second.second;

    // Update slot number in old_smart_factors_.
    it->second.second = slot;
  }
}

void VioBackend::updateLocalSmartFactorsSlots(
    const std::vector<LandmarkId>& lmk_ids_of_new_smart_factors) {
  if (!local_belief_cov_smoother_) {
    return;
  }

  const gtsam::ISAM2Result& result =
      local_belief_cov_smoother_->getISAM2Result();
  for (size_t i = 0u; i < lmk_ids_of_new_smart_factors.size(); ++i) {
    if (i >= result.newFactorsIndices.size()) {
      break;
    }

    const size_t slot = result.newFactorsIndices.at(i);
    if (!local_belief_cov_smoother_->getFactors().exists(slot)) {
      continue;
    }

    const auto& it_local = old_smart_factors_local_belief_cov_.find(
        lmk_ids_of_new_smart_factors.at(i));
    if (it_local == old_smart_factors_local_belief_cov_.end()) {
      continue;
    }

    const auto* smart_factor = dynamic_cast<const SmartStereoFactor*>(
        local_belief_cov_smoother_->getFactors().at(slot).get());
    if (!smart_factor) {
      continue;
    }
    it_local->second.second = static_cast<Slot>(slot);
  }
}

bool VioBackend::updateLocalBeliefCovarianceSmoother(
    const gtsam::NonlinearFactorGraph& new_factors_tmp,
    const gtsam::Values& new_values,
    const std::map<Key, double>& timestamps,
    const gtsam::FactorIndices& delete_slots,
    size_t max_extra_iterations) {
  if (!local_belief_cov_smoother_) {
    return false;
  }

  Smoother local_backup(*local_belief_cov_smoother_);
  try {
    gtsam::FixedLagSmoother::Result local_result;
    local_result = local_belief_cov_smoother_->update(
        new_factors_tmp, new_values, timestamps, delete_slots);
    for (size_t n_iter = 1; n_iter < max_extra_iterations; ++n_iter) {
      local_result = local_belief_cov_smoother_->update();
    }
    local_belief_cov_state_ = local_belief_cov_smoother_->calculateEstimate();
  } catch (const std::exception& e) {
    *local_belief_cov_smoother_ = local_backup;
    LOG(WARNING) << "Local belief covariance smoother update failed: "
                 << e.what();
    return false;
  } catch (...) {
    *local_belief_cov_smoother_ = local_backup;
    LOG(WARNING)
        << "Local belief covariance smoother update failed with unknown error.";
    return false;
  }
  return true;
}

bool VioBackend::updatePoseBeliefLocalSidecar(
    const gtsam::NonlinearFactorGraph& new_factors_tmp,
    const gtsam::Values& new_values,
    const std::map<Key, double>& timestamps,
    const gtsam::FactorIndices& delete_slots,
    size_t max_extra_iterations,
    const std::vector<LandmarkId>& lmk_ids_of_new_smart_factors) {
  const bool local_update_ok =
      updateLocalBeliefCovarianceSmoother(new_factors_tmp,
                                          new_values,
                                          timestamps,
                                          delete_slots,
                                          max_extra_iterations);
  if (!local_update_ok) {
    return false;
  }
  updateLocalSmartFactorsSlots(lmk_ids_of_new_smart_factors);
  return true;
}

bool VioBackend::computeLocalPoseBeliefCovariance(const FrameId& cur_id) {
  if (!local_belief_cov_smoother_) {
    return false;
  }
  const gtsam::Key pose_key = gtsam::Symbol(kPoseSymbolChar, cur_id);
  if (!local_belief_cov_state_.exists(pose_key)) {
    return false;
  }

  try {
    gtsam::Marginals marginals(local_belief_cov_smoother_->getFactors(),
                               local_belief_cov_state_,
                               gtsam::Marginals::Factorization::CHOLESKY);
    const gtsam::Matrix pose_cov = marginals.marginalCovariance(pose_key);
    if (pose_cov.rows() != 6 || pose_cov.cols() != 6 || !pose_cov.allFinite()) {
      return false;
    }
    pose_belief_local_covariance_lkf_ = pose_cov;
  } catch (const std::exception& e) {
    LOG(WARNING) << "Failed local pose covariance extraction: " << e.what();
    return false;
  } catch (...) {
    LOG(WARNING) << "Failed local pose covariance extraction.";
    return false;
  }
  return true;
}

bool VioBackend::computePoseBeliefLocalCovarianceFromSidecar(
    const FrameId& cur_id) {
  if (!FLAGS_cbs_use_bpsam_for_local_belief_covariance) {
    return computeLocalPoseBeliefCovariance(cur_id);
  }

  if (!local_belief_cov_smoother_) {
    return false;
  }

  const gtsam::Key pose_key = gtsam::Symbol(kPoseSymbolChar, cur_id);
  if (!local_belief_cov_state_.exists(pose_key)) {
    return false;
  }

  gtsam::NonlinearFactorGraph local_graph;
  const auto& raw_factors = local_belief_cov_smoother_->getFactors();
  local_graph.reserve(raw_factors.size());
  for (size_t slot = 0u; slot < raw_factors.size(); ++slot) {
    if (!raw_factors.exists(slot)) {
      continue;
    }
    const auto& factor = raw_factors.at(slot);
    if (!factor) {
      continue;
    }
    local_graph.push_back(factor);
  }
  if (local_graph.empty()) {
    return false;
  }

  gtsam::Values local_values;
  for (const gtsam::Key& key : local_graph.keys()) {
    if (local_belief_cov_state_.exists(key)) {
      cbs::insertOrAssign(local_values, key, local_belief_cov_state_.at(key));
    } else if (state_.exists(key)) {
      cbs::insertOrAssign(local_values, key, state_.at(key));
    } else {
      LOG(WARNING) << "BPSAM local covariance snapshot missing value for key "
                   << gtsam::DefaultKeyFormatter(key) << ".";
      return computeLocalPoseBeliefCovariance(cur_id);
    }
  }

  gtsam::ISAM2Params isam_params;
  BackendParams::setIsam2Params(backend_params_, &isam_params);
  const auto pose_cov = computePoseBeliefCovarianceWithBpsamSnapshot(
      local_graph, local_values, pose_key, isam_params, 'k');
  if (pose_cov) {
    pose_belief_local_covariance_lkf_ = *pose_cov;
    return true;
  }
  LOG(WARNING) << "BPSAM local covariance snapshot failed. Falling back to "
                  "local smoother covariance.";
  return computeLocalPoseBeliefCovariance(cur_id);
}

bool VioBackend::computePoseBeliefCovarianceWithoutExternalFactors(
    const FrameId& cur_id) {
  const gtsam::Key pose_key = gtsam::Symbol(kPoseSymbolChar, cur_id);
  if (!state_.exists(pose_key)) {
    return false;
  }

  const gtsam::NonlinearFactorGraph& full_graph = getMainBackendFactors();
  std::unordered_set<size_t> external_slots;
  external_slots.reserve(active_external_belief_factor_slots_.size());
  for (const auto& active_slot : active_external_belief_factor_slots_) {
    external_slots.insert(active_slot.slot);
  }

  gtsam::NonlinearFactorGraph local_only_graph;
  local_only_graph.reserve(full_graph.size());
  for (size_t slot = 0u; slot < full_graph.size(); ++slot) {
    if (!full_graph.exists(slot)) {
      continue;
    }
    if (external_slots.find(slot) != external_slots.end()) {
      continue;
    }
    local_only_graph.push_back(full_graph.at(slot));
  }

  if (local_only_graph.empty()) {
    return false;
  }

  try {
    gtsam::Marginals marginals(
        local_only_graph, state_, gtsam::Marginals::Factorization::CHOLESKY);
    const gtsam::Matrix pose_cov = marginals.marginalCovariance(pose_key);
    if (pose_cov.rows() != 6 || pose_cov.cols() != 6 || !pose_cov.allFinite()) {
      return false;
    }
    pose_belief_local_covariance_lkf_ = pose_cov;
    return true;
  } catch (const std::exception& e) {
    LOG(WARNING) << "Failed local-only pose covariance extraction: "
                 << e.what();
  } catch (...) {
    LOG(WARNING) << "Failed local-only pose covariance extraction.";
  }
  return false;
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

/* --------------------------- PRINTERS ------------------------------------- */
/// Printers.
void VioBackend::print() const {
  backend_params_.print();
  if (usePersistentBpsamMainBackend()) {
    LOG(INFO) << "Main backend optimizer mode: persistent BPSAM.";
  } else {
    LOG(INFO) << "Main backend optimizer mode: native fixed-lag smoother.";
  }

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

  const std::string* which_graph = nullptr;
  const gtsam::NonlinearFactorGraph* graph = nullptr;
  // Pick the graph that makes more sense:
  // This is code is mostly run post update, when it throws exception,
  // shouldn't we print the graph before optimization instead?
  // Yes if available, but if not, then just ask the smoother.
  static const std::string graph_before_opt = "(graph before optimization)";
  static const std::string backend_factors = "(main backend factors)";
  if (debug_info_.graphBeforeOpt.size() != 0) {
    which_graph = &graph_before_opt;
    graph = &(debug_info_.graphBeforeOpt);
  } else {
    which_graph = &backend_factors;
    graph = &(getMainBackendFactors());
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
    for (size_t i = 0u; i < delete_slots.size(); ++i) {
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
  gtsam::NonlinearFactorGraph graph = getMainBackendFactors();
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
  gtsam::NonlinearFactorGraph graph = getMainBackendFactors();
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
    gtsam::NonlinearFactorGraph graph =
        gtsam::NonlinearFactorGraph(getMainBackendFactors());
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
    old_smart_factors_local_belief_cov_.erase(lmk_id);
    return true;
  }
  return false;
}

}  // namespace VIO.
