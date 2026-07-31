#pragma once

#include <gtsam/base/Matrix.h>
#include <gtsam/inference/Key.h>
#include <gtsam/nonlinear/ISAM2.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>

#include <map>
#include <memory>
#include <optional>
#include <vector>

namespace VIO {

std::optional<gtsam::Matrix> computePoseBeliefCovarianceWithBpsamSnapshot(
    const gtsam::NonlinearFactorGraph& local_graph,
    const gtsam::Values& local_values,
    gtsam::Key pose_key,
    const gtsam::ISAM2Params& isam2_params,
    char robot_id);

class PersistentBpsamLocalCovarianceSidecar {
 public:
  PersistentBpsamLocalCovarianceSidecar(const gtsam::ISAM2Params& isam2_params,
                                        char robot_id,
                                        size_t max_window_size);
  ~PersistentBpsamLocalCovarianceSidecar();

  bool update(const gtsam::NonlinearFactorGraph& new_factors,
              const gtsam::Values& new_values,
              const std::map<gtsam::Key, double>& timestamps,
              const gtsam::FactorIndices& delete_slots,
              size_t num_smart_factors,
              std::vector<size_t>* smart_factor_slots_out,
              bool shadow_acceptance_audit_enable = false);

  std::optional<gtsam::Matrix> computePoseCovariance(gtsam::Key pose_key);

  gtsam::Values calculateEstimate() const;

  const gtsam::NonlinearFactorGraph& factors() const;

  bool factorExists(size_t slot) const;

  gtsam::NonlinearFactor::shared_ptr factorAt(size_t slot) const;

  const gtsam::ISAM2Result& lastUpdateResult() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace VIO
