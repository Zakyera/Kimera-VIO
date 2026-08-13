/* ----------------------------------------------------------------------------
 * Copyright 2017, Massachusetts Institute of Technology,
 * Cambridge, MA 02139
 * All Rights Reserved
 * Authors: Luca Carlone, et al. (see THANKS for the full author list)
 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */

/**
 * @file   VioBackend.h
 * @brief  Visual-Inertial Odometry pipeline, as described in these papers:
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

#pragma once

#include <gtsam/geometry/Cal3DS2.h>
#include <gtsam/geometry/Cal3_S2.h>
#include <gtsam/geometry/StereoCamera.h>
#include <gtsam/geometry/StereoPoint2.h>
#include <gtsam/navigation/CombinedImuFactor.h>
#include <gtsam/navigation/ImuFactor.h>
#include <gtsam/nonlinear/ISAM2.h>
#include <gtsam/nonlinear/LinearContainerFactor.h>
#include <gtsam/nonlinear/Marginals.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/slam/PriorFactor.h>
#if GTSAM_VERSION_MAJOR <= 4 && GTSAM_VERSION_MINOR < 3
#include <gtsam_unstable/nonlinear/BatchFixedLagSmoother.h>
#else
#include <gtsam/nonlinear/BatchFixedLagSmoother.h>
#endif
#include <gtsam_unstable/slam/SmartStereoProjectionPoseFactor.h>

#include <atomic>
#include <deque>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include <cbs/utils/health_aware_relative_cbs.h>

#include "kimera-vio/backend/VioBackend-definitions.h"
#include "kimera-vio/backend/VioBackendParams.h"
#include "kimera-vio/factors/PointPlaneFactor.h"
#include "kimera-vio/frontend/OdometryParams.h"
#include "kimera-vio/frontend/StereoVisionImuFrontend-definitions.h"
#include "kimera-vio/imu-frontend/ImuFrontend.h"
#include "kimera-vio/initial/InitializationFromImu.h"
#include "kimera-vio/logging/Logger.h"
#include "kimera-vio/utils/Macros.h"
#include "kimera-vio/utils/ThreadsafeQueue.h"
#include "kimera-vio/utils/UtilsGTSAM.h"
#include "kimera-vio/utils/UtilsOpenCV.h"

namespace VIO {

// Forward-declarations
class VioNavState;
class PersistentBpsamLocalCovarianceSidecar;

class VioBackend {
 public:
  KIMERA_DELETE_COPY_CONSTRUCTORS(VioBackend);
  KIMERA_POINTER_TYPEDEFS(VioBackend);
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  typedef std::function<void(const ImuBias& imu_bias)> ImuBiasCallback;
  typedef std::function<void(const LandmarksMap& map)> MapCallback;
  typedef std::function<void(const ExternalBeliefMatchDiagnostic&)>
      ExternalBeliefMatchDiagnosticCallback;

  /**
   * @brief VioBackend Constructor. Initialization must be done separately.
   * @param B_Pose_leftCam Pose from IMU Body (Backend's reference pose) to
   * left camera.
   * @param stereo_calibration Stereo Calibration data for stereo factors.
   * @param backend_params Parameters for Backend.
   * @param log_output Whether to log to CSV files the Backend output.
   */
  VioBackend(const gtsam::Pose3& B_Pose_leftCamRect,
             const StereoCalibPtr& stereo_calibration,
             const BackendParams& backend_params,
             const ImuParams& imu_params,
             const BackendOutputParams& backend_output_params,
             bool log_output,
             std::optional<OdometryParams> odom_params = std::nullopt);
  virtual ~VioBackend();

 public:
  BackendOutput::UniquePtr spinOnce(const BackendInput& input);

  /**
   * @brief isInitialized Returns whether the Frontend is initializing.
   * Needs to be Thread-Safe! Therefore, frontend_state_ is atomic.
   */
  inline bool isInitialized() const {
    return backend_state_ != BackendState::Bootstrap;
  }

  // Register (and trigger!) callback that will be called as soon as the Backend
  // comes up with a new IMU bias update.
  // The callback is also triggered in this function to update the imu bias for
  // the callee of this function.
  void registerImuBiasUpdateCallback(
      const ImuBiasCallback& imu_bias_update_callback);

  /**
   * @brief registerMapUpdateCallback Register callback that will be called as
   * soon as the Backend optimizes the 3D landmarks.
   * @param map_update_callback Callback to call once backend finishes
   * optimizing.
   */
  void registerMapUpdateCallback(const MapCallback& map_update_callback);

  // Registers a passive observation callback for external-belief timestamp
  // matching. The callback is never required for estimator operation.
  void registerExternalBeliefMatchDiagnosticCallback(
      const ExternalBeliefMatchDiagnosticCallback& callback);

  // Thread-safe ingestion of external odometry beliefs (e.g., from LiDAR backend).
  void enqueueExternalOdometryBeliefs(
      const std::vector<ExternalOdometryBelief>& beliefs);

  // Get valid 3D points - TODO: this copies the graph.
  void get3DPoints(std::vector<gtsam::Point3>* points_3d) const;

  // Get valid 3D points and corresponding lmk id.
  // Warning! it modifies old_smart_factors_!!
  PointsWithIdMap getMapLmkIdsTo3dPointsInTimeHorizon(
      const gtsam::NonlinearFactorGraph& graph,
      LmkIdToLmkTypeMap* lmk_id_to_lmk_type_map = nullptr,
      const size_t& min_age = 2);

  inline gtsam::Matrix getCurrentStateCovariance() const {
    return state_covariance_lkf_;
  }

  // Update covariance matrix using getCurrentStateCovariance()
  // NOT TESTED
  void computeStateCovariance();

  // Set initial state at given pose, velocity and bias.
  bool initStateAndSetPriors(
      const VioNavStateTimestamped& vio_nav_state_initial_seed);

  void initializeBackend(const BackendInput& input) {
    CHECK(backend_state_ == BackendState::Bootstrap);
    switch (backend_params_.autoInitialize_) {
      case 0: {
        initializeFromGt(input);
        break;
      }
      case 1: {
        initializeFromIMU(input);
        break;
      }
      default: {
        LOG(FATAL) << "Wrong initialization mode.";
      }
    }
    // Signal that the Backend has been initialized.
    backend_state_ = BackendState::Nominal;
  }

  bool initializeFromGt(const BackendInput& input) {
    // If the gtNavState is identity, the params provider probably did a
    // mistake, although it can happen that the ground truth initial pose is
    // identity, but this is super unlikely
    CHECK(!backend_params_.initial_ground_truth_state_.equals(VioNavState()))
        << "Requested initialization from Ground-Truth pose but got an "
           "identity pose: did you parse your ground-truth correctly?";
    return initStateAndSetPriors(VioNavStateTimestamped(
        input.timestamp_, backend_params_.initial_ground_truth_state_));
  }

  /**
   * @brief initializeFromIMU
   * Assumes Zero Velocity & upright vehicle. Uses the InitializationFromIMU
   * class.
   * @param input BackendInput that MUST be castable to a
   * BackendInputImuInitialization
   * @return
   */
  bool initializeFromIMU(const BackendInput& input) {
    LOG(INFO) << "------------------- Initialize Pipeline: timestamp = "
              << input.timestamp_ << "--------------------";
    // Guess pose from IMU, assumes vehicle to be static.
    const VioNavState& initial_state_estimate =
        InitializationFromImu::getInitialStateEstimate(
            input.imu_acc_gyrs_,
            imu_params_.n_gravity_,
            backend_params_.roundOnAutoInitialize_);

    // Initialize Backend using IMU data.
    return initStateAndSetPriors(
        VioNavStateTimestamped(input.timestamp_, initial_state_estimate));
  }

  void saveGraph(const std::string& filepath) const;

 protected:
  enum class BackendState {
    Bootstrap = 0u,  //! Initialize Backend
    Nominal = 1u     //! Run Backend
  };
  std::atomic<BackendState> backend_state_;

  // Store stereo frame info into landmarks table:
  // returns landmarks observed in current frame.
  void addStereoMeasurementsToFeatureTracks(
      const int& frameNum,
      const StereoMeasurements& stereoMeasurements_kf,
      LandmarkIds* landmarks_kf);

  // Workhorse that stores data and optimizes at each keyframe.
  // [in] timestamp_kf_nsec, keyframe timestamp.
  // [in] status_smart_stereo_measurements_kf, vision data.
  // [in] stereo_ransac_body_pose, inertial data.
  virtual bool addVisualInertialStateAndOptimize(
      const Timestamp& timestamp_kf_nsec,
      const StatusStereoMeasurements& status_smart_stereo_measurements_kf,
      const GtsamPreintegrationType& pim,
      std::optional<gtsam::Pose3> odometry_body_pose = std::nullopt,
      std::optional<gtsam::Velocity3> odometry_vel = std::nullopt);

  // Uses landmark table to add factors in graph.
  void addLandmarksToGraph(const LandmarkIds& landmarks_kf);

  // Adds a landmark to the graph for the first time.
  void addLandmarkToGraph(const LandmarkId& lm_id, const FeatureTrack& lm);

  void updateLandmarkInGraph(
      const LandmarkId& lmk_id,
      const std::pair<FrameId, StereoPoint2>& new_measurement);

  /**
   * @brief addStateValues Add values for the state: pose, velocity, and imu
   * bias.
   * @param cur_id Current Keyframe ID
   * @param pose Pose of body wrt world
   * @param velocity Velocity of body expressed in world coordinates
   * @param imu_bias Updated biases for gyro and accel
   */
  void addStateValues(const FrameId& cur_id,
                      const gtsam::Pose3& pose,
                      const gtsam::Velocity3& velocity,
                      const ImuBias& imu_bias);
  void addStateValues(
      const FrameId& frame_id,
      const TrackerStatusSummary& tracker_status,
      const GtsamPreintegrationType& pim,
      const std::optional<gtsam::Pose3> odom_pose = std::nullopt,
      std::optional<gtsam::Vector3> odom_vel = std::nullopt);
  void addStateValuesFromNavState(const FrameId& frame_id,
                                  const gtsam::NavState& nav_state);

  void addImuFactor(const FrameId& from_id,
                    const FrameId& to_id,
                    const GtsamPreintegrationType& pim);

  // Add no motion factors in case of low disparity.
  void addZeroVelocityPrior(const FrameId& frame_id);

  void addNoMotionFactor(const FrameId& from_id, const FrameId& to_id);

  void addVelocityPrior(const FrameId& frame_id,
                        const gtsam::Velocity3& vel,
                        const double& precision);

  void addBetweenFactor(const FrameId& from_id,
                        const FrameId& to_id,
                        const gtsam::Pose3& from_id_POSE_to_id,
                        const double& between_rotation_precision,
                        const double& between_translation_precision);

  /**
   * @brief optimize
   * @param timestamp_kf_nsec
   * @param cur_id
   * @param max_iterations
   * @param extra_factor_slots_to_delete
   * @return False if optimization failed, true otherwise
   */
  bool optimize(const Timestamp& timestamp_kf_nsec,
                const FrameId& cur_id,
                const size_t& max_iterations,
                const gtsam::FactorIndices& extra_factor_slots_to_delete =
                    gtsam::FactorIndices());
  /// Printers.
  void printFeatureTracks() const;

  void cleanNullPtrsFromGraph(
      gtsam::NonlinearFactorGraph* new_imu_prior_and_other_factors);

  bool deleteLmkFromFeatureTracks(const LandmarkId& lmk_id);

 private:
  class PoseBeliefCovarianceSidecarAdapter {
   public:
    virtual ~PoseBeliefCovarianceSidecarAdapter() = default;

    virtual bool update(
        const gtsam::NonlinearFactorGraph& new_factors_tmp,
        const gtsam::Values& new_values,
        const std::map<Key, double>& timestamps,
        const gtsam::FactorIndices& delete_slots,
        size_t max_extra_iterations,
        const std::vector<LandmarkId>& lmk_ids_of_new_smart_factors_tmp) = 0;

    virtual bool computePoseBeliefLocalCovariance(const FrameId& cur_id) = 0;

    virtual const char* covarianceSourceTag() const = 0;
  };

  class LocalSmootherPoseBeliefCovarianceSidecarAdapter;
  class PersistentBpsamPoseBeliefCovarianceSidecarAdapter;

  void initializePoseBeliefCovarianceSidecarAdapter();

  bool addVisualInertialStateAndOptimize(const BackendInput& input);

 protected:
  bool usePersistentBpsamMainBackend() const;

  const gtsam::NonlinearFactorGraph& getMainBackendFactors() const;

  bool mainBackendFactorExists(size_t slot) const;

  const gtsam::NonlinearFactor::shared_ptr mainBackendFactorAt(
      size_t slot) const;

  const gtsam::ISAM2Result& getMainBackendResult() const;

  gtsam::Values calculateMainBackendEstimate() const;

 private:
  // Add initial prior factors.
  void addInitialPriorFactors(const FrameId& frame_id);

  void addConstantVelocityFactor(const FrameId& from_id, const FrameId& to_id);

  // Update states.
  void updateStates(const FrameId& cur_id);

  /**
   * @brief updateSmoother
   * @param result
   * @param new_factors_tmp
   * @param new_values
   * @param timestamps
   * @param delete_slots
   * @return False if the update failed, true otw.
   */
  bool updateSmoother(
      gtsam::FixedLagSmoother::Result* result,
      const gtsam::NonlinearFactorGraph& new_factors_tmp =
          gtsam::NonlinearFactorGraph(),
      const gtsam::Values& new_values = gtsam::Values(),
      const std::map<Key, double>& timestamps =
          gtsam::FixedLagSmoother::KeyTimestampMap(),
      const gtsam::FactorIndices& delete_slots = gtsam::FactorIndices());

  void cleanCheiralityLmk(
      const gtsam::Symbol& lmk_symbol,
      gtsam::NonlinearFactorGraph* new_factors_tmp_cheirality,
      gtsam::Values* new_values_cheirality,
      std::map<Key, double>* timestamps_cheirality,
      gtsam::FactorIndices* delete_slots_cheirality,
      const gtsam::NonlinearFactorGraph& graph,
      const gtsam::NonlinearFactorGraph& new_factors_tmp,
      const gtsam::Values& new_values,
      const std::map<Key, double>& timestamps,
      const gtsam::FactorIndices& delete_slots);

  void deleteAllFactorsWithKeyFromFactorGraph(
      const gtsam::Key& key,
      const gtsam::NonlinearFactorGraph& new_factors_tmp,
      gtsam::NonlinearFactorGraph* factor_graph_output);

  // Returns if the key in timestamps could be removed or not.
  bool deleteKeyFromTimestamps(const gtsam::Key& key,
                               const std::map<Key, double>& timestamps,
                               std::map<Key, double>* timestamps_output);

  // Returns if the key in timestamps could be removed or not.
  bool deleteKeyFromValues(const gtsam::Key& key,
                           const gtsam::Values& values,
                           gtsam::Values* values_output);

  // Find all slots of factors that have the given key in the list of keys.
  void findSlotsOfFactorsWithKey(
      const gtsam::Key& key,
      const gtsam::NonlinearFactorGraph& graph,
      std::vector<size_t>* slots_of_factors_with_key);

  virtual void deleteLmkFromExtraStructures(const LandmarkId& lmk_id);

  void updateNewSmartFactorsSlots(
      const std::vector<LandmarkId>& lmk_ids_of_new_smart_factors_tmp,
      SmartFactorMap* old_smart_factors);

  void updateLocalSmartFactorsSlots(
      const std::vector<LandmarkId>& lmk_ids_of_new_smart_factors_tmp);

  bool updateLocalBeliefCovarianceSmoother(
      const gtsam::NonlinearFactorGraph& new_factors_tmp,
      const gtsam::Values& new_values,
      const std::map<Key, double>& timestamps,
      const gtsam::FactorIndices& delete_slots,
      size_t max_extra_iterations);

  virtual bool updatePoseBeliefLocalSidecar(
      const gtsam::NonlinearFactorGraph& new_factors_tmp,
      const gtsam::Values& new_values,
      const std::map<Key, double>& timestamps,
      const gtsam::FactorIndices& delete_slots,
      size_t max_extra_iterations,
      const std::vector<LandmarkId>& lmk_ids_of_new_smart_factors_tmp);

  bool computeLocalPoseBeliefCovariance(const FrameId& cur_id);

  virtual bool computePoseBeliefLocalCovarianceFromSidecar(
      const FrameId& cur_id);

  bool computePoseBeliefCovarianceWithoutExternalFactors(const FrameId& cur_id);
  void logPoseBeliefCovarianceSanityDiff(const FrameId& cur_id) const;

  struct ExternalBeliefFactorId {
    uint8_t source_agent = 0u;
    FrameId local_frame_id = 0;
    bool operator==(const ExternalBeliefFactorId& other) const {
      return source_agent == other.source_agent &&
             local_frame_id == other.local_frame_id;
    }
  };

  struct ExternalBeliefFactorSlot {
    ExternalBeliefFactorId id;
    size_t slot = 0u;
  };

  enum class ExternalBeliefRejectReason {
    kNone = 0,
    kWindow = 1,
    kTimestamp = 2,
    kMissingState = 3,
    kCovariance = 4,
    kDuration = 5,
  };

  void updateKeyframeTimestampIndex(const FrameId& frame_id,
                                    const Timestamp& timestamp_kf_nsec);

  bool resolveExternalBeliefStamp(
      double stamp_sec,
      const FrameId& cur_id,
      FrameId* local_frame_id,
      ExternalBeliefRejectReason* reject_reason,
      FrameId* best_frame_id = nullptr,
      double* best_stamp_sec = nullptr,
      double* best_abs_dt = nullptr) const;
  std::vector<ExternalOdometryBelief> popPendingExternalOdometryBeliefs();
  void requeuePendingExternalOdometryBeliefs(
      const std::vector<ExternalOdometryBelief>& beliefs);
  void capPendingExternalOdometryBeliefsLocked();
  double latestKeyframeTimestampSec(const FrameId& cur_id) const;
  double externalOdomBeliefRetryAgeSec(
      const ExternalOdometryBelief& belief,
      const FrameId& cur_id) const;
  bool shouldRetryExternalOdometryBelief(
      const ExternalOdometryBelief& belief,
      const FrameId& cur_id,
      const std::string& reason) const;
  void collectExternalBeliefFactors(
      const FrameId& cur_id,
      gtsam::FactorIndices* delete_slots,
      gtsam::NonlinearFactorGraph* new_factors_tmp,
      std::vector<ExternalBeliefFactorId>* inserted_external_factor_ids);

  void refreshCbsOutgoingBeliefs(const FrameId& cur_id);

  void refreshExternalBeliefFactorSlots(
      const size_t num_factors_before_external,
      const std::vector<ExternalBeliefFactorId>& inserted_external_factor_ids);

  // Set parameters for all types of factors.
  void setFactorsParams(
      const BackendParams& vio_params,
      gtsam::SharedNoiseModel* smart_noise,
      gtsam::SmartStereoProjectionParams* smart_factors_params,
      gtsam::SharedNoiseModel* no_motion_prior_noise,
      gtsam::SharedNoiseModel* zero_velocity_prior_noise,
      gtsam::SharedNoiseModel* constant_velocity_prior_noise);

  void setSmartStereoFactorsNoiseModel(const double& smart_noise_sigma,
                                       gtsam::SharedNoiseModel* smart_noise);

  // Set parameters for smart factors.
  void setSmartStereoFactorsParams(
      const double& rank_tolerance,
      const double& landmark_distance_threshold,
      const double& retriangulation_threshold,
      const double& outlier_rejection,
      gtsam::SmartStereoProjectionParams* smart_factors_params);

  void setNoMotionFactorsParams(const double& rotation_precision,
                                const double& position_precision,
                                gtsam::SharedNoiseModel* no_motion_prior_noise);

  /// Private printers.
  void print() const;

  void printSmootherInfo(const gtsam::NonlinearFactorGraph& new_factors_tmp,
                         const gtsam::FactorIndices& delete_slots,
                         const std::string& message = "CATCHING EXCEPTION",
                         const bool& showDetails = false) const;

  // Provide a nonlinear factor, which will be casted to any of the selected
  // factors, and then printed.
  // Slot argument, is just to print the slot of the factor if you know it.
  // If slot is -1, there is no slot number printed.
  void printSelectedFactors(
      const gtsam::NonlinearFactor* g,
      const size_t& slot = 0,
      const bool print_smart_factors = true,
      const bool print_point_plane_factors = true,
      const bool print_plane_priors = true,
      const bool print_point_priors = true,
      const bool print_linear_container_factors = true) const;

  void printSelectedGraph(
      const gtsam::NonlinearFactorGraph& graph,
      const bool& print_smart_factors = true,
      const bool& print_point_plane_factors = true,
      const bool& print_plane_priors = true,
      const bool& print_point_priors = true,
      const bool& print_linear_container_factors = true) const;

  /// Debuggers.
  void computeSmartFactorStatistics();

  void computeSparsityStatistics();

  // Debugging post optimization and estimate calculation.
  void postDebug(
      const std::chrono::high_resolution_clock::time_point& total_start_time,
      const std::chrono::high_resolution_clock::time_point& start_time);

  void logFactorGraphAudit(const FrameId& cur_id,
                           const Timestamp& timestamp_kf_nsec,
                           size_t new_factor_count,
                           size_t delete_slot_count,
                           size_t num_factors_before_external,
                           size_t inserted_external_factor_count) const;

  // Reset state of debug info.
  void resetDebugInfo(DebugVioInfo* debug_info);

 public:
  /// Getters
  // Thread-safe methods, but also the returns are const, so no problems.
  inline const BackendParams& getBackendParams() const {
    return backend_params_;
  }

  // Not Thread Safe. The methods below return copies but they are not
  // mutex protected, and these are used in the backend thread
  inline Timestamp getTimestampLkf() const { return timestamp_lkf_; }
  inline ImuBias getLatestImuBias() const { return imu_bias_lkf_; }
  inline ImuBias getImuBiasPrevKf() const { return imu_bias_prev_kf_; }
  inline Vector3 getWVelBLkf() const { return W_Vel_B_lkf_; }
  inline Pose3 getWPoseBLkfFromIncrements() const {
    return W_Pose_B_lkf_from_increments_;
  }
  inline Pose3 getWPoseBLkfFromState() const {
    return W_Pose_B_lkf_from_state_;
  }
  inline gtsam::Matrix getStateCovarianceLkf() const {
    return state_covariance_lkf_;
  }
  inline int getCurrKfId() const { return curr_kf_id_; }
  inline gtsam::Values getState() const { return state_; }
  inline int getLandmarkCount() const { return landmark_count_; }
  inline DebugVioInfo getCurrentDebugVioInfo() const { return debug_info_; }

 protected:
  // Raw, user-specified params.
  const BackendParams backend_params_;
  const ImuParams imu_params_;
  const BackendOutputParams backend_output_params_;
  std::optional<OdometryParams> odom_params_;

  // State estimates.
  // TODO(Toni): bundle these in a VioNavStateTimestamped.
  Timestamp timestamp_lkf_;
  ImuBias imu_bias_lkf_;  //!< Most recent bias estimate..
  Vector3 W_Vel_B_lkf_;   //!< Velocity of body at k-1 in world coordinates
  Pose3 W_Pose_B_lkf_from_increments_;  //!< Body pose at at k-1 in world
                                        //!< coordinates obtained by chaining
                                        //!< relative motion estimates.
  Pose3
      W_Pose_B_lkf_from_state_;  //!< Body pose at at k-1 in world coordinates,
                                 //!< straight from VIO smoother_.

  ImuBias imu_bias_prev_kf_;  //!< bias estimate at previous keyframe

  // State covariance. (initialize to zero)
  gtsam::Matrix state_covariance_lkf_ = Eigen::MatrixXd::Zero(15, 15);
  gtsam::Matrix pose_belief_local_covariance_lkf_ = Eigen::MatrixXd::Zero(6, 6);
  bool pose_belief_local_covariance_valid_ = false;
  std::string pose_belief_covariance_source_ = "unavailable";
  size_t external_beliefs_received_per_update_ = 0u;
  size_t external_beliefs_added_per_update_ = 0u;
  size_t external_beliefs_rejected_first_message_per_update_ = 0u;
  size_t external_beliefs_rejected_update_status_per_update_ = 0u;
  size_t external_beliefs_rejected_inactive_window_per_update_ = 0u;
  size_t external_beliefs_rejected_shape_per_update_ = 0u;
  size_t external_beliefs_rejected_exception_per_update_ = 0u;
  double optimization_time_sec_per_update_ = 0.0;
  double optimize_total_time_sec_per_update_ = 0.0;
  double collect_external_beliefs_time_sec_per_update_ = 0.0;
  double compute_state_covariance_time_sec_per_update_ = 0.0;
  double cbs_belief_generation_time_sec_per_update_ = 0.0;
  double cbs_outgoing_total_time_sec_per_update_ = 0.0;
  double cbs_set_marginalization_graph_time_sec_per_update_ = 0.0;
  double cbs_get_odometry_beliefs_time_sec_per_update_ = 0.0;
  size_t cbs_marginalization_graph_factor_count_ = 0u;
  std::vector<ExternalOdometryBelief> cbs_outgoing_odom_beliefs_;
  CbsPoseMergeDiagnostic pending_cbs_pose_merge_diagnostic_;
  CbsPoseMergeDiagnostic cbs_pose_merge_diagnostic_;
  std::string cbs_odom_sender_mode_ = "adjacent_window";
  uint8_t cbs_odom_requesting_agent_id_ = static_cast<uint8_t>('l');
  bool external_odom_duration_gate_enable_ = false;
  double external_odom_duration_tolerance_sec_ = 0.06;
  double external_odom_duration_ratio_min_ = 0.5;
  double external_odom_duration_ratio_max_ = 2.0;
  double cbs_odom_horizon_sec_ = 0.20;
  double cbs_odom_horizon_tolerance_sec_ = 0.06;
  size_t cbs_odom_max_horizon_pairs_per_update_ = 25u;
  bool last_cbs_outgoing_odom_pair_valid_ = false;
  gtsam::Key last_cbs_outgoing_odom_from_key_ = 0u;
  gtsam::Key last_cbs_outgoing_odom_to_key_ = 0u;
  cbs::health_aware::Params cbs_health_sender_params_;
  cbs::health_aware::Params cbs_health_receiver_params_;
  cbs::health_aware::SenderHealthState cbs_sender_health_state_;

  // Vision params.
  gtsam::SmartStereoProjectionParams smart_factors_params_;
  gtsam::SharedNoiseModel smart_noise_;
  // Pose of the left camera wrt body
  const Pose3 B_Pose_leftCamRect_;
  // Stores calibration, baseline.
  const gtsam::Cal3_S2Stereo::shared_ptr stereo_cal_;

  // State.
  //!< current state of the system.
  gtsam::Values state_;

  // ISAM2 smoother
  std::unique_ptr<Smoother> smoother_;
  std::unique_ptr<PersistentBpsamLocalCovarianceSidecar> main_bpsam_backend_;
  std::unique_ptr<Smoother> local_belief_cov_smoother_;
  std::unique_ptr<PoseBeliefCovarianceSidecarAdapter>
      pose_belief_cov_sidecar_adapter_;
  gtsam::Values local_belief_cov_state_;

  // Values
  //!< new states to be added
  gtsam::Values new_values_;

  // Factors.
  //!< New factors to be added
  gtsam::NonlinearFactorGraph new_imu_prior_and_other_factors_;
  //!< landmarkId -> {SmartFactorPtr}
  LandmarkIdSmartFactorMap new_smart_factors_;
  //!< landmarkId -> {SmartFactorPtr, SlotIndex}
  SmartFactorMap old_smart_factors_;
  SmartFactorMap old_smart_factors_local_belief_cov_;
  // if SlotIndex is -1, means that the factor has not been inserted yet in
  // the graph

  // Data:
  // TODO grows unbounded currently, but it should be limited to time horizon.
  FeatureTracks feature_tracks_;

  // Counters.
  //! Last keyframe id.
  int last_kf_id_;
  //! Current keyframe id.
  int curr_kf_id_;

  // Imu Bias update callback. To be called as soon as we have a new IMU bias
  // update so that the Frontend performs preintegration with the newest bias.
  ImuBiasCallback imu_bias_update_callback_;

  //! Map update callback for the frontend PnP tracker.
  MapCallback map_update_callback_;

  ExternalBeliefMatchDiagnosticCallback
      external_belief_match_diagnostic_callback_;

  // Debug info.
  DebugVioInfo debug_info_;

  // To print smoother info, useful when looking for optimization bugs.
  bool debug_smoother_ = false;

 private:
  //! No motion factors settings.
  gtsam::SharedNoiseModel zero_velocity_prior_noise_;
  gtsam::SharedNoiseModel no_motion_prior_noise_;
  gtsam::SharedNoiseModel constant_velocity_prior_noise_;

  //! Landmark count.
  int landmark_count_;

  //! Number of Cheirality exceptions
  size_t counter_of_exceptions_ = 0;

  // External belief bridge state.
  mutable std::mutex external_beliefs_mutex_;
  std::deque<ExternalOdometryBelief> pending_external_odom_beliefs_;
  std::vector<ExternalBeliefFactorSlot> active_external_belief_factor_slots_;
  std::map<FrameId, double> keyframe_timestamp_sec_;
  size_t max_pending_external_odom_beliefs_ = 800u;
  double external_belief_timestamp_tolerance_sec_ = 0.2;
  double external_odom_unmatched_retry_max_age_sec_ = 5.0;
  size_t max_unmatched_external_odom_retry_beliefs_ = 500u;
  // External-belief flow diagnostics (monotonic counters).
  std::atomic<size_t> external_beliefs_received_total_{0u};
  std::atomic<size_t> external_beliefs_queue_dropped_total_{0u};
  std::atomic<size_t> external_beliefs_resolved_total_{0u};
  std::atomic<size_t> external_beliefs_selected_total_{0u};
  std::atomic<size_t> external_beliefs_inserted_total_{0u};
  std::atomic<size_t> external_beliefs_rejected_total_{0u};
  std::atomic<size_t> external_beliefs_rejected_window_total_{0u};
  std::atomic<size_t> external_beliefs_rejected_timestamp_total_{0u};
  std::atomic<size_t> external_beliefs_rejected_missing_state_total_{0u};
  std::atomic<size_t> external_beliefs_rejected_covariance_total_{0u};
  std::atomic<size_t> external_beliefs_rejected_duration_total_{0u};
  std::atomic<size_t> external_beliefs_rejected_first_message_total_{0u};
  std::atomic<size_t> external_beliefs_rejected_update_status_total_{0u};
  std::atomic<size_t> external_beliefs_rejected_inactive_window_total_{0u};
  std::atomic<size_t> external_beliefs_rejected_shape_total_{0u};
  std::atomic<size_t> external_beliefs_rejected_exception_total_{0u};
  std::atomic<size_t> external_merge_diag_sample_idx_{0u};

  //! Logger.
  const bool log_output_ = {false};
  std::unique_ptr<BackendLogger> logger_;
};

}  // namespace VIO
