/* ----------------------------------------------------------------------------
 * Copyright 2017, Massachusetts Institute of Technology,
 * Cambridge, MA 02139
 * All Rights Reserved
 * Authors: Luca Carlone, et al. (see THANKS for the full author list)
 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */

/**
 * @file   VioBackendModule.h
 * @brief  Pipeline module for the Backend.
 *
 * @author Antoni Rosinol
 */

#pragma once

#include "kimera-vio/backend/RegularVioBackend.h"
#include "kimera-vio/backend/VioBackend-definitions.h"
#include "kimera-vio/backend/VioBackend.h"
#include "kimera-vio/pipeline/PipelineModule.h"
#include "kimera-vio/utils/Macros.h"
#include "kimera-vio/utils/Statistics.h"

namespace VIO {

class VioBackendModule
    : public SIMOPipelineModule<BackendInput, BackendOutput> {
 public:
  KIMERA_DELETE_COPY_CONSTRUCTORS(VioBackendModule);
  KIMERA_POINTER_TYPEDEFS(VioBackendModule);
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  using SIMO = SIMOPipelineModule<BackendInput, BackendOutput>;
  using InputQueue = ThreadsafeQueue<typename PIO::InputUniquePtr>;

  /**
   * @brief VioBackendModule
   * @param input_queue
   * @param output_queue
   * @param parallel_run
   * @param vio_backend
   */
  VioBackendModule(InputQueue* input_queue,
                   bool parallel_run,
                   VioBackend::UniquePtr vio_backend);
  virtual ~VioBackendModule() = default;

  /**
   * @brief spinOnce
   * @param input
   * @return
   */
  virtual OutputUniquePtr spinOnce(BackendInput::UniquePtr input);

 public:
  inline bool isInitialized() const { return vio_backend_->isInitialized(); }

  /**
   * @brief registerImuBiasUpdateCallback Register callback to be called
   * whenever the Backend has a new estimate of the IMU bias.
   * The Frontend needs to register this function for
   * the IMU preintegration to be done wrt the latest IMU bias estimate.
   * @param imu_bias_update_callback function that will be called on a new
   * IMU bias update.
   */
  void registerImuBiasUpdateCallback(
      const VioBackend::ImuBiasCallback& imu_bias_update_callback);

  /**
   * @brief registerMapUpdateCallback Register callback to be called
   * whenever the Backend has a new estimate of landmarks in time-horizon.
   * The Frontend needs to register this function for
   * the PnP tracking to be done wrt to the latest landmark estimates.
   * @param map_update_callback function that will be called on new
   * optimized landmark positions.
   */
  void registerMapUpdateCallback(
      const VioBackend::MapCallback& map_update_callback);

    // zy Step 9a
    // External CBS-style pose belief publication hook.
  void registerExternalPoseBeliefCallback(
      const std::function<void(const VioBackend::ExternalPoseBelief&)>&
          external_pose_belief_callback);

  // External CBS-style pose prior ingestion hooks.
  void enqueueExternalPosePrior(const Timestamp& timestamp_kf_nsec,
                                const gtsam::Pose3& W_Pose_B,
                                const gtsam::SharedNoiseModel& noise_model,
                                const std::string& source = "unknown",
                                uint64_t source_seq = 0);

  bool enqueueExternalPosePriorFromCovariance(
      const Timestamp& timestamp_kf_nsec,
      const gtsam::Pose3& W_Pose_B,
      const gtsam::Matrix6& covariance,
      const std::string& source = "unknown",
      uint64_t source_seq = 0);

  // zy Step 15a
  // forward timestamp-based belief queries through the module API used by Pipeline.
  bool getExternalPoseBeliefAtTimestamp(
      const Timestamp& query_timestamp_kf_nsec,
      VioBackend::ExternalPoseBelief* belief,
      const Timestamp& tolerance_ns = -1) const;


 protected:
  const VioBackend::UniquePtr vio_backend_;
};

}  // namespace VIO
