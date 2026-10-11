#pragma once

#include <algorithm>
#include <map>
#include <mutex>
#include <stdexcept>
#include <vector>

#include "kimera-vio/imu-frontend/ImuFrontend.h"

namespace VIO {

// Experimental nominal-only adapter. It stores the EXACT frontend packet
// intervals, including interpolated image borders. It does not reinterpret
// raw CPT7 sample support and never returns or modifies a measurement covariance.
class CbsNativeImuMeanBridge {
 public:
  struct Segment {
    ImuStamp end;
    gtsam::Vector6 rate;
  };

  void append(const ImuStampS& stamps, const ImuAccGyrS& rates) {
    if (stamps.cols() != rates.cols() || stamps.cols() < 2)
      throw std::runtime_error("CBS mean bridge: invalid native IMU packet");
    std::lock_guard<std::mutex> lock(mutex_);
    // Validate the whole packet before changing history.
    for (int i = 0; i + 1 < stamps.cols(); ++i) {
      if (stamps(i + 1) <= stamps(i) || !rates.col(i).allFinite())
        throw std::runtime_error("CBS mean bridge: invalid native IMU interval");
      auto next = segments_.lower_bound(stamps(i));
      if (next != segments_.end() && next->first < stamps(i + 1))
        throw std::runtime_error("CBS mean bridge: overlapping native packets");
      if (next != segments_.begin() && std::prev(next)->second.end > stamps(i))
        throw std::runtime_error("CBS mean bridge: overlapping native packets");
    }
    if (segments_.size() + stamps.cols() - 1 > 100000)
      throw std::runtime_error("CBS mean bridge: history capacity exceeded");
    for (int i = 0; i + 1 < stamps.cols(); ++i)
      segments_.emplace(stamps(i), Segment{stamps(i + 1), rates.col(i)});
  }

  void pruneBefore(ImuStamp stamp) {
    std::lock_guard<std::mutex> lock(mutex_);
    while (!segments_.empty() && segments_.begin()->second.end <= stamp)
      segments_.erase(segments_.begin());
  }

  // Both endpoints are evaluated on the SAME native-prior-anchored nominal
  // trajectory. Never subtract poses belonging to differently reanchored
  // keyframes. Fractional boundaries retain the original native held rate.
  gtsam::Pose3 between(ImuStamp anchor_stamp, const gtsam::NavState& anchor,
                       const ImuBias& bias, const ImuParams& params,
                       ImuStamp from, ImuStamp to, ImuStamp available_until) const {
    if (std::min(from, to) < anchor_stamp ||
        std::max(from, to) > available_until)
      throw std::runtime_error("CBS mean bridge: noncausal/invalid endpoint");
    if (from == to) return gtsam::Pose3();
    std::vector<std::pair<ImuStamp, Segment>> history;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      auto it = segments_.upper_bound(anchor_stamp);
      if (it != segments_.begin()) --it;
      for (; it != segments_.end() && it->first < std::max(from, to); ++it)
        history.push_back(*it);
    }
    auto pose_at = [&](ImuStamp target) -> gtsam::Pose3 {
      if (target == anchor_stamp) return anchor.pose();
      std::vector<ImuStamp> stamps;
      std::vector<gtsam::Vector6> rates;
      ImuStamp cursor = anchor_stamp;
      for (const auto& item : history) {
        if (item.second.end <= cursor) continue;
        if (item.first > cursor)
          throw std::runtime_error("CBS mean bridge: missing native IMU interval");
        stamps.push_back(cursor);
        rates.push_back(item.second.rate);
        cursor = std::min(target, item.second.end);
        if (cursor == target) break;
      }
      if (cursor != target || rates.empty())
        throw std::runtime_error("CBS mean bridge: insufficient native IMU history");
      stamps.push_back(target);
      rates.push_back(rates.back());  // terminal value is unused by preintegration
      ImuStampS ts(1, stamps.size());
      ImuAccGyrS xs(6, rates.size());
      for (size_t i = 0; i < stamps.size(); ++i) {
        ts(i) = stamps[i];
        xs.col(i) = rates[i];
      }
      ImuFrontend native(params, bias);
      return native.preintegrateImuMeasurements(ts, xs)->predict(anchor, bias).pose();
    };
    return pose_at(from).between(pose_at(to));
  }

  static gtsam::Pose3 align(const gtsam::Pose3& left,
                            const gtsam::Pose3& measurement,
                            const gtsam::Pose3& right) {
    return left.compose(measurement).compose(right);
  }

 private:
  mutable std::mutex mutex_;
  std::map<ImuStamp, Segment> segments_;
};
}  // namespace VIO
