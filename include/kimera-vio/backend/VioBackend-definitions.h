/* ----------------------------------------------------------------------------
 * Copyright 2017, Massachusetts Institute of Technology,
 * Cambridge, MA 02139
 * All Rights Reserved
 * Authors: Luca Carlone, et al. (see THANKS for the full author list)
 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */

/**
 * @file   VioBackend-definitions.h
 * @brief  Definitions for VioBackend.
 * @author Antoni Rosinol
 */

#pragma once

#include <glog/logging.h>
#include <gtsam/geometry/Cal3_S2.h>
#include <gtsam/geometry/StereoPoint2.h>
#include <gtsam_unstable/slam/SmartStereoProjectionPoseFactor.h>

#include <array>
#include <cstdint>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include "kimera-vio/common/VioNavState.h"
#include "kimera-vio/common/vio_types.h"
#include "kimera-vio/frontend/StereoVisionImuFrontend-definitions.h"
#include "kimera-vio/frontend/Tracker-definitions.h"
#include "kimera-vio/imu-frontend/ImuFrontend-definitions.h"
#include "kimera-vio/imu-frontend/ImuFrontend.h"
#include "kimera-vio/pipeline/PipelinePayload.h"
#include "kimera-vio/utils/Macros.h"
#include "kimera-vio/utils/UtilsOpenCV.h"

namespace cbs {
class IncrementalFixedLagBpsamSmoother;
}  // namespace cbs

namespace VIO {

// Gtsam types. // TODO remove these!!
using gtsam::Cal3_S2;
using gtsam::Key;
using gtsam::Point2;
using gtsam::Point3;
using gtsam::Pose3;
using gtsam::Rot3;
using gtsam::StereoPoint2;
using StereoCalibPtr = gtsam::Cal3_S2Stereo::shared_ptr;

using SymbolChar = unsigned char;
static constexpr SymbolChar kPoseSymbolChar = 'x';
static constexpr SymbolChar kVelocitySymbolChar = 'v';
static constexpr SymbolChar kImuBiasSymbolChar = 'b';
static constexpr SymbolChar kLandmarkSymbolChar = 'l';

#define INCREMENTAL_SMOOTHER
#ifdef INCREMENTAL_SMOOTHER
typedef cbs::IncrementalFixedLagBpsamSmoother Smoother;
#else
typedef gtsam::BatchFixedLagSmoother Smoother;
#endif

// Backend types
using SmartStereoFactor = gtsam::SmartStereoProjectionPoseFactor;
using SmartFactorParams = gtsam::SmartStereoProjectionParams;
using LandmarkIdSmartFactorMap =
    std::unordered_map<LandmarkId, SmartStereoFactor::shared_ptr>;
using Slot = long int;
using SmartFactorMap =
    gtsam::FastMap<LandmarkId, std::pair<SmartStereoFactor::shared_ptr, Slot>>;

using PointWithId = std::pair<LandmarkId, Landmark>;
using PointsWithId = std::vector<PointWithId>;
// TODO(Toni):  there is the same in vio_types.cpp, replace by that one, since
// now the frontend also has such concept.
using PointsWithIdMap = std::unordered_map<LandmarkId, Landmark>;
using LmkIdToLmkTypeMap = std::unordered_map<LandmarkId, LandmarkType>;

////////////////////////////////////////////////////////////////////////////////
// FeatureTrack
// TODO(Toni): what is this doing here... should be in Frontend at worst.
class FeatureTrack {
  // TODO(Toni): a feature track should have a landmark id...
  // TODO(Toni): a feature track should contain a pixel measurement per frame
  // but allow for multi-frame measurements at a time.
  // TODO(Toni): add getters for feature track length
 public:
  //! Observation: {FrameId, Px-Measurement}
  std::vector<std::pair<FrameId, StereoPoint2>> obs_;

  // Is the lmk in the graph?
  bool in_ba_graph_ = false;

  FeatureTrack(FrameId frame_id, const StereoPoint2& px) {
    obs_.push_back(std::make_pair(frame_id, px));
  }

  void print() const {
    LOG(INFO) << "Feature track with cameras: ";
    for (size_t i = 0u; i < obs_.size(); i++) {
      std::cout << " " << obs_[i].first << " ";
    }
    std::cout << std::endl;
  }
};

// Landmark id to measurements.
// Key is the lmk_id and feature track the collection of pairs of
// frame id and pixel location.
// TODO(Toni): what is this doing here... should be in Frontend at worst.
using FeatureTracks = std::unordered_map<LandmarkId, FeatureTrack>;

struct ExternalOdometryBelief {
  uint8_t source_agent = 0u;
  uint32_t from_pose_index = 0u;
  uint32_t to_pose_index = 0u;
  uint32_t sender_from_pose_index = 0u;
  uint32_t sender_to_pose_index = 0u;
  double from_stamp_sec = 0.0;
  double to_stamp_sec = 0.0;
  uint64_t sender_timestamp_ns = 0u;
  std::string sender_frame_id = "na";
  double received_wall_time_sec = 0.0;
  std::array<double, 6> relative_mu{};
  std::array<double, 36> covariance{};
  double relax_factor = 0.0;

  // Optional diagnostics-only identity assigned by the ROS bridge. These
  // fields are ignored by CBS and every estimator-control path.
  bool shadow_identity_valid = false;
  uint64_t shadow_receipt_sequence = 0u;
  uint32_t shadow_belief_ordinal = 0u;
};

// Immutable diagnostics-only observation of the exact receiver endpoint
// selection made for an incoming external odometry belief. It is emitted only
// when an optional observer is registered and never feeds back into CBS.
struct ExternalBeliefMatchDiagnostic {
  enum class Status : uint8_t {
    Unavailable = 0u,
    Accepted = 1u,
    AlreadyApplied = 2u,
    Rejected = 3u,
    RetryPending = 4u,
    Superseded = 5u,
  };

  bool shadow_identity_valid = false;
  uint64_t shadow_receipt_sequence = 0u;
  uint32_t shadow_belief_ordinal = 0u;
  uint8_t source_agent = 0u;
  uint32_t sender_from_pose_index = 0u;
  uint32_t sender_to_pose_index = 0u;
  double sender_from_stamp_sec = 0.0;
  double sender_to_stamp_sec = 0.0;
  FrameId receiver_from_frame_id = 0u;
  FrameId receiver_to_frame_id = 0u;
  double receiver_from_stamp_sec = std::numeric_limits<double>::quiet_NaN();
  double receiver_to_stamp_sec = std::numeric_limits<double>::quiet_NaN();
  double start_abs_timestamp_error_sec =
      std::numeric_limits<double>::quiet_NaN();
  double end_abs_timestamp_error_sec =
      std::numeric_limits<double>::quiet_NaN();
  double interval_duration_error_sec =
      std::numeric_limits<double>::quiet_NaN();
  uint32_t start_ambiguous_match_count = 0u;
  uint32_t end_ambiguous_match_count = 0u;
  bool receiver_pose_available = false;
  Pose3 receiver_from_pose;
  Pose3 receiver_to_pose;
  Pose3 sender_relative_pose;
  bool sender_relative_pose_available = false;
  gtsam::Matrix6 sender_covariance = gtsam::Matrix6::Zero();
  bool sender_covariance_available = false;
  bool sender_frame_conversion_available = true;
  uint32_t shadow_invalid_reason_mask = 0u;
  Status status = Status::Unavailable;
  bool terminal = false;
  std::string reason;
};

struct CbsPoseMergeDiagnostic {
  bool valid = false;
  FrameId optimization_frame_id = 0u;
  gtsam::Key from_pose_key = 0u;
  gtsam::Key to_pose_key = 0u;
  gtsam::Pose3 local_relative_before;
  gtsam::Pose3 external_relative;
  gtsam::Pose3 final_relative_after;
};

////////////////////////////////////////////////////////////////////////////////
class DebugVioInfo {
 public:
  int numSF_;
  int numValid_;
  int numDegenerate_;
  int numFarPoints_;
  int numOutliers_;
  int numCheirality_;
  int numNonInitialized_;

  gtsam::Rot3 imuR_lkf_kf = gtsam::Rot3();

  gtsam::Values stateBeforeOpt;
  gtsam::NonlinearFactorGraph graphBeforeOpt;
  gtsam::NonlinearFactorGraph graphToBeDeleted;

  double factorsAndSlotsTime_;
  double preUpdateTime_;
  double updateTime_;
  double updateSlotTime_;
  double extraIterationsTime_;

  double meanPixelError_;
  double maxPixelError_;
  double meanTrackLength_;
  size_t maxTrackLength_;

  int numAddedSmartF_;
  int numAddedImuF_;
  int numAddedNoMotionF_;
  int numAddedConstantVelF_;
  int numAddedBetweenStereoF_;

  int nrElementsInMatrix_;
  int nrZeroElementsInMatrix_;

  gtsam::NavState navstate_k_;

  double linearizeTime_;
  double linearSolveTime_;
  double retractTime_;
  double linearizeMarginalizeTime_;
  double marginalizeTime_;

  /* ------------------------------------------------------------------------ */
  void resetSmartFactorsStatistics() {
    numSF_ = 0;
    numValid_ = 0;
    numDegenerate_ = 0;
    numFarPoints_ = 0;
    numOutliers_ = 0;
    numCheirality_ = 0;
    numNonInitialized_ = 0;

    meanPixelError_ = 0;
    maxPixelError_ = 0;
    meanTrackLength_ = 0;
    maxTrackLength_ = 0;
  }

  /* ------------------------------------------------------------------------ */
  void resetTimes() {
    factorsAndSlotsTime_ = 0;
    preUpdateTime_ = 0;
    updateTime_ = 0;
    updateSlotTime_ = 0;
    extraIterationsTime_ = 0;
    linearizeTime_ = 0;
    linearSolveTime_ = 0;
    retractTime_ = 0;
    linearizeMarginalizeTime_ = 0;
    marginalizeTime_ = 0;
  }

  /* ------------------------------------------------------------------------ */
  void resetAddedFactorsStatistics() {
    numAddedSmartF_ = 0;
    numAddedImuF_ = 0;
    numAddedNoMotionF_ = 0;
    numAddedConstantVelF_ = 0;
    numAddedBetweenStereoF_ = 0;
  }

  /* ------------------------------------------------------------------------ */
  double sumAllTimes() const {
    return factorsAndSlotsTime_ + preUpdateTime_ + updateTime_ +
           updateSlotTime_ + extraIterationsTime_;
  }

  /* ------------------------------------------------------------------------ */
  void printTimes() const {
    LOG(INFO) << "-------- VIO Backend Timing --------\n"
              << "Find delete time      :" << factorsAndSlotsTime_ << '\n'
              << "preUpdate time        :" << preUpdateTime_ << '\n'
              << "Update Time time      :" << updateTime_ << '\n'
              << "Update slot time      :" << updateSlotTime_ << '\n'
              << "Extra iterations time :" << extraIterationsTime_;
  }

  /* ------------------------------------------------------------------------ */
  void print() const {
    LOG(INFO) << "-------- DebugVioInfo: --------\n"
              << " numSF: " << numSF_ << '\n'
              << " numValid: " << numValid_ << '\n'
              << " numDegenerate: " << numDegenerate_ << '\n'
              << " numOutliers: " << numOutliers_ << '\n'
              << " numFarPoints: " << numFarPoints_ << '\n'
              << " numCheirality: " << numCheirality_ << '\n'
              << " numNonInitialized: " << numNonInitialized_ << '\n'
              << " meanPixelError: " << meanPixelError_ << '\n'
              << " maxPixelError: " << maxPixelError_ << '\n'
              << " meanTrackLength: " << meanTrackLength_ << '\n'
              << " maxTrackLength: " << maxTrackLength_;
  }
};

////////////////////////////////////////////////////////////////////////////////
struct BackendInput : public PipelinePayload {
 public:
  KIMERA_POINTER_TYPEDEFS(BackendInput);
  KIMERA_DELETE_COPY_CONSTRUCTORS(BackendInput);
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  BackendInput(
      const Timestamp& timestamp_kf_nsec,
      const StatusStereoMeasurementsPtr& status_stereo_measurements_kf,
      const ImuFrontend::PimPtr& pim,
      //! Raw imu msgs for Backend init only
      const ImuAccGyrS& imu_acc_gyrs,
      std::optional<gtsam::Pose3> body_lkf_OdomPose_body_kf = std::nullopt,
      std::optional<gtsam::Velocity3> body_kf_world_OdomVel_body_kf =
          std::nullopt)
      : PipelinePayload(timestamp_kf_nsec),
        status_stereo_measurements_kf_(status_stereo_measurements_kf),
        pim_(pim),
        imu_acc_gyrs_(imu_acc_gyrs),
        body_lkf_OdomPose_body_kf_(body_lkf_OdomPose_body_kf),
        body_kf_world_OdomVel_body_kf_(body_kf_world_OdomVel_body_kf) {}

 public:
  const StatusStereoMeasurementsPtr status_stereo_measurements_kf_;
  ImuFrontend::PimPtr pim_;
  ImuAccGyrS imu_acc_gyrs_;
  // between pose from last keyframe to current according to external odometry
  std::optional<gtsam::Pose3> body_lkf_OdomPose_body_kf_;
  // velocity of the current keyframe body w.r.t. the world frame in the body
  // frame
  std::optional<gtsam::Velocity3> body_kf_world_OdomVel_body_kf_;

 public:
  void print() const {
    LOG(INFO) << "VioBackend Input Payload print:\n"
              << "Timestamp: " << timestamp_;
    if (status_stereo_measurements_kf_) {
      LOG(INFO) << "Status smart stereo measurements: "
                << "\n\t Meas size: "
                << status_stereo_measurements_kf_->second.size();

      LOG(INFO)
          << "Mono Tracking Status: "
          << TrackerStatusSummary::asString(
                 status_stereo_measurements_kf_->first.kfTrackingStatus_mono_);
      status_stereo_measurements_kf_->first.lkf_T_k_mono_.print(
          "\n\t Tracker Pose (mono): ");
      LOG(INFO) << "Stereo Tracking Status: "
                << TrackerStatusSummary::asString(
                       status_stereo_measurements_kf_->first
                           .kfTrackingStatus_stereo_);
      status_stereo_measurements_kf_->first.lkf_T_k_stereo_.print(
          "\n\t Tracker Pose (stereo): ");
    }
    if (pim_) pim_->print("PIM : ");
  }
};

////////////////////////////////////////////////////////////////////////////////
struct BackendOutput : public PipelinePayload {
  KIMERA_POINTER_TYPEDEFS(BackendOutput);
  KIMERA_DELETE_COPY_CONSTRUCTORS(BackendOutput);
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  BackendOutput(
      const Timestamp& timestamp_kf,
      const gtsam::Values& state,
      const gtsam::NonlinearFactorGraph& factor_graph,
      const gtsam::Pose3& W_Pose_Blkf,
      const Vector3& W_Vel_Blkf,
      const ImuBias& imu_bias_lkf,
      const gtsam::Matrix& state_covariance_lkf,
      const FrameId& cur_kf_id,
      const int& landmark_count,
      const DebugVioInfo& debug_info,
      const PointsWithIdMap& landmarks_with_id_map,
      const LmkIdToLmkTypeMap& lmk_id_to_lmk_type_map,
      const gtsam::Matrix& pose_belief_local_covariance_lkf = gtsam::Matrix(),
      bool pose_belief_local_covariance_valid = false,
      const std::string& pose_belief_covariance_source = "unavailable",
      size_t external_beliefs_received_per_update = 0u,
      size_t external_beliefs_added_per_update = 0u,
      size_t external_beliefs_rejected_first_message_per_update = 0u,
      size_t external_beliefs_rejected_update_status_per_update = 0u,
      size_t external_beliefs_rejected_inactive_window_per_update = 0u,
      size_t external_beliefs_rejected_shape_per_update = 0u,
      size_t external_beliefs_rejected_exception_per_update = 0u,
      double optimization_time_sec = 0.0,
      double optimize_total_time_sec = 0.0,
      double collect_external_beliefs_time_sec = 0.0,
      double compute_state_covariance_time_sec = 0.0,
      double cbs_belief_generation_time_sec = 0.0,
      double cbs_outgoing_total_time_sec = 0.0,
      double cbs_set_marginalization_graph_time_sec = 0.0,
      double cbs_get_odometry_beliefs_time_sec = 0.0,
      size_t cbs_marginalization_graph_factor_count = 0u,
      std::vector<ExternalOdometryBelief> cbs_outgoing_odom_beliefs = {},
      gtsam::NonlinearFactorGraph cbs_active_odom_factors = {},
      CbsPoseMergeDiagnostic cbs_pose_merge_diagnostic = {})
      : PipelinePayload(timestamp_kf),
        W_State_Blkf_(timestamp_kf, W_Pose_Blkf, W_Vel_Blkf, imu_bias_lkf),
        state_(state),
        state_covariance_lkf_(state_covariance_lkf),
        factor_graph_(factor_graph),
        cur_kf_id_(cur_kf_id),
        landmark_count_(landmark_count),
        debug_info_(debug_info),
        landmarks_with_id_map_(landmarks_with_id_map),
        lmk_id_to_lmk_type_map_(lmk_id_to_lmk_type_map),
        pose_belief_local_covariance_lkf_(pose_belief_local_covariance_lkf),
        pose_belief_local_covariance_valid_(pose_belief_local_covariance_valid),
        pose_belief_covariance_source_(pose_belief_covariance_source),
        external_beliefs_received_per_update_(
            external_beliefs_received_per_update),
        external_beliefs_added_per_update_(external_beliefs_added_per_update),
        external_beliefs_rejected_first_message_per_update_(
            external_beliefs_rejected_first_message_per_update),
        external_beliefs_rejected_update_status_per_update_(
            external_beliefs_rejected_update_status_per_update),
        external_beliefs_rejected_inactive_window_per_update_(
            external_beliefs_rejected_inactive_window_per_update),
        external_beliefs_rejected_shape_per_update_(
            external_beliefs_rejected_shape_per_update),
        external_beliefs_rejected_exception_per_update_(
            external_beliefs_rejected_exception_per_update),
        optimization_time_sec_(optimization_time_sec),
        optimize_total_time_sec_(optimize_total_time_sec),
        collect_external_beliefs_time_sec_(
            collect_external_beliefs_time_sec),
        compute_state_covariance_time_sec_(compute_state_covariance_time_sec),
        cbs_belief_generation_time_sec_(cbs_belief_generation_time_sec),
        cbs_outgoing_total_time_sec_(cbs_outgoing_total_time_sec),
        cbs_set_marginalization_graph_time_sec_(
            cbs_set_marginalization_graph_time_sec),
        cbs_get_odometry_beliefs_time_sec_(
            cbs_get_odometry_beliefs_time_sec),
        cbs_marginalization_graph_factor_count_(
            cbs_marginalization_graph_factor_count),
        cbs_outgoing_odom_beliefs_(std::move(cbs_outgoing_odom_beliefs)),
        cbs_active_odom_factors_(std::move(cbs_active_odom_factors)),
        cbs_pose_merge_diagnostic_(std::move(cbs_pose_merge_diagnostic)) {}

  BackendOutput(
      const VioNavStateTimestamped& vio_navstate_timestamped,
      const gtsam::Values& state,
      const gtsam::NonlinearFactorGraph& factor_graph,
      const gtsam::Matrix& state_covariance_lkf,
      const FrameId& cur_kf_id,
      const int& landmark_count,
      const DebugVioInfo& debug_info,
      const PointsWithIdMap& landmarks_with_id_map,
      const LmkIdToLmkTypeMap& lmk_id_to_lmk_type_map,
      const gtsam::Matrix& pose_belief_local_covariance_lkf = gtsam::Matrix(),
      bool pose_belief_local_covariance_valid = false,
      const std::string& pose_belief_covariance_source = "unavailable",
      size_t external_beliefs_received_per_update = 0u,
      size_t external_beliefs_added_per_update = 0u,
      size_t external_beliefs_rejected_first_message_per_update = 0u,
      size_t external_beliefs_rejected_update_status_per_update = 0u,
      size_t external_beliefs_rejected_inactive_window_per_update = 0u,
      size_t external_beliefs_rejected_shape_per_update = 0u,
      size_t external_beliefs_rejected_exception_per_update = 0u,
      double optimization_time_sec = 0.0,
      double optimize_total_time_sec = 0.0,
      double collect_external_beliefs_time_sec = 0.0,
      double compute_state_covariance_time_sec = 0.0,
      double cbs_belief_generation_time_sec = 0.0,
      double cbs_outgoing_total_time_sec = 0.0,
      double cbs_set_marginalization_graph_time_sec = 0.0,
      double cbs_get_odometry_beliefs_time_sec = 0.0,
      size_t cbs_marginalization_graph_factor_count = 0u,
      std::vector<ExternalOdometryBelief> cbs_outgoing_odom_beliefs = {},
      gtsam::NonlinearFactorGraph cbs_active_odom_factors = {},
      CbsPoseMergeDiagnostic cbs_pose_merge_diagnostic = {})
      : PipelinePayload(vio_navstate_timestamped.timestamp_),
        W_State_Blkf_(vio_navstate_timestamped),
        state_(state),
        state_covariance_lkf_(state_covariance_lkf),
        factor_graph_(factor_graph),
        cur_kf_id_(cur_kf_id),
        landmark_count_(landmark_count),
        debug_info_(debug_info),
        landmarks_with_id_map_(landmarks_with_id_map),
        lmk_id_to_lmk_type_map_(lmk_id_to_lmk_type_map),
        pose_belief_local_covariance_lkf_(pose_belief_local_covariance_lkf),
        pose_belief_local_covariance_valid_(pose_belief_local_covariance_valid),
        pose_belief_covariance_source_(pose_belief_covariance_source),
        external_beliefs_received_per_update_(
            external_beliefs_received_per_update),
        external_beliefs_added_per_update_(external_beliefs_added_per_update),
        external_beliefs_rejected_first_message_per_update_(
            external_beliefs_rejected_first_message_per_update),
        external_beliefs_rejected_update_status_per_update_(
            external_beliefs_rejected_update_status_per_update),
        external_beliefs_rejected_inactive_window_per_update_(
            external_beliefs_rejected_inactive_window_per_update),
        external_beliefs_rejected_shape_per_update_(
            external_beliefs_rejected_shape_per_update),
        external_beliefs_rejected_exception_per_update_(
            external_beliefs_rejected_exception_per_update),
        optimization_time_sec_(optimization_time_sec),
        optimize_total_time_sec_(optimize_total_time_sec),
        collect_external_beliefs_time_sec_(
            collect_external_beliefs_time_sec),
        compute_state_covariance_time_sec_(compute_state_covariance_time_sec),
        cbs_belief_generation_time_sec_(cbs_belief_generation_time_sec),
        cbs_outgoing_total_time_sec_(cbs_outgoing_total_time_sec),
        cbs_set_marginalization_graph_time_sec_(
            cbs_set_marginalization_graph_time_sec),
        cbs_get_odometry_beliefs_time_sec_(
            cbs_get_odometry_beliefs_time_sec),
        cbs_marginalization_graph_factor_count_(
            cbs_marginalization_graph_factor_count),
        cbs_outgoing_odom_beliefs_(std::move(cbs_outgoing_odom_beliefs)),
        cbs_active_odom_factors_(std::move(cbs_active_odom_factors)),
        cbs_pose_merge_diagnostic_(std::move(cbs_pose_merge_diagnostic)) {}

  const VioNavStateTimestamped W_State_Blkf_;
  const gtsam::Values state_;
  const gtsam::Matrix state_covariance_lkf_;
  const gtsam::NonlinearFactorGraph factor_graph_;
  const FrameId cur_kf_id_;
  const int landmark_count_;
  const DebugVioInfo debug_info_;
  const PointsWithIdMap landmarks_with_id_map_;
  const LmkIdToLmkTypeMap lmk_id_to_lmk_type_map_;
  const gtsam::Matrix pose_belief_local_covariance_lkf_;
  const bool pose_belief_local_covariance_valid_;
  const std::string pose_belief_covariance_source_;
  const size_t external_beliefs_received_per_update_;
  const size_t external_beliefs_added_per_update_;
  const size_t external_beliefs_rejected_first_message_per_update_;
  const size_t external_beliefs_rejected_update_status_per_update_;
  const size_t external_beliefs_rejected_inactive_window_per_update_;
  const size_t external_beliefs_rejected_shape_per_update_;
  const size_t external_beliefs_rejected_exception_per_update_;
  const double optimization_time_sec_;
  const double optimize_total_time_sec_;
  const double collect_external_beliefs_time_sec_;
  const double compute_state_covariance_time_sec_;
  const double cbs_belief_generation_time_sec_;
  const double cbs_outgoing_total_time_sec_;
  const double cbs_set_marginalization_graph_time_sec_;
  const double cbs_get_odometry_beliefs_time_sec_;
  const size_t cbs_marginalization_graph_factor_count_;
  const std::vector<ExternalOdometryBelief> cbs_outgoing_odom_beliefs_;
  const gtsam::NonlinearFactorGraph cbs_active_odom_factors_;
  const CbsPoseMergeDiagnostic cbs_pose_merge_diagnostic_;
};

////////////////////////////////////////////////////////////////////////////////
/** \brief The BackendType enum
 *  - kStereoImu: vanilla Backend type using Stereo and IMU
 *  - kStructuralRegularities: the `regular VIO` Backend, using structural
 * regularities derived from the 3D Mesh.
 */
enum class BackendType { kStereoImu = 0, kStructuralRegularities = 1 };

}  // namespace VIO
