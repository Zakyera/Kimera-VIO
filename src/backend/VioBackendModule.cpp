/* ----------------------------------------------------------------------------
 * Copyright 2017, Massachusetts Institute of Technology,
 * Cambridge, MA 02139
 * All Rights Reserved
 * Authors: Luca Carlone, et al. (see THANKS for the full author list)
 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */

/**
 * @file   VioBackendModule.cpp
 * @brief  Pipeline module for the Backend.
 *
 * @author Antoni Rosinol
 */

#include "kimera-vio/backend/VioBackendModule.h"

namespace VIO {

VioBackendModule::VioBackendModule(InputQueue* input_queue,
                                   bool parallel_run,
                                   VioBackend::UniquePtr vio_backend)
    : SIMO(input_queue, "VioBackend", parallel_run),
      vio_backend_(std::move(vio_backend)) {
  CHECK(vio_backend_);
}

VioBackendModule::OutputUniquePtr VioBackendModule::spinOnce(
    BackendInput::UniquePtr input) {
  CHECK(input);
  CHECK(vio_backend_);
  OutputUniquePtr output = vio_backend_->spinOnce(*input);
  if (!output) {
    LOG(ERROR) << "Backend did not return an output: shutting down Backend.";
    shutdown();
  }
  return output;
}

void VioBackendModule::registerImuBiasUpdateCallback(
    const VioBackend::ImuBiasCallback& imu_bias_update_callback) {
  CHECK(vio_backend_);
  vio_backend_->registerImuBiasUpdateCallback(imu_bias_update_callback);
}

void VioBackendModule::registerMapUpdateCallback(
    const VioBackend::MapCallback& map_update_callback) {
  CHECK(vio_backend_);
  vio_backend_->registerMapUpdateCallback(map_update_callback);
}

// Step 9b
void VioBackendModule::registerExternalPoseBeliefCallback(
    const std::function<void(const VioBackend::ExternalPoseBelief&)>&
        external_pose_belief_callback) {
  CHECK(vio_backend_);
  vio_backend_->registerExternalPoseBeliefCallback(external_pose_belief_callback);
}

void VioBackendModule::enqueueExternalPosePrior(
    const Timestamp& timestamp_kf_nsec,
    const gtsam::Pose3& W_Pose_B,
    const gtsam::SharedNoiseModel& noise_model,
    const std::string& source,
    uint64_t source_seq) {
  CHECK(vio_backend_);
  vio_backend_->enqueueExternalPosePrior(
      timestamp_kf_nsec, W_Pose_B, noise_model, source, source_seq);
}

bool VioBackendModule::enqueueExternalPosePriorFromCovariance(
    const Timestamp& timestamp_kf_nsec,
    const gtsam::Pose3& W_Pose_B,
    const gtsam::Matrix6& covariance,
    const std::string& source,
    uint64_t source_seq) {
  CHECK(vio_backend_);
  return vio_backend_->enqueueExternalPosePriorFromCovariance(
      timestamp_kf_nsec, W_Pose_B, covariance, source, source_seq);
}


}  // namespace VIO
