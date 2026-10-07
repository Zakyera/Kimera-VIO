#pragma once

// Passive, opt-in ETH1 visual-health telemetry. Never used by estimation.
#include <algorithm>
#include <array>
#include <cstdlib>
#include <iomanip>
#include <sstream>
#include <string>

#include "kimera-vio/frontend/StereoFrame.h"

namespace VIO {
inline bool VisualHealthAuditEnabled() {
  static const bool enabled = [] {
    const char* value = std::getenv("KIMERA_VISUAL_HEALTH_AUDIT");
    return value != nullptr && std::string(value) == "1";
  }();
  return enabled;
}

// Read actual frame arrays rather than cached DebugTrackerInfo counters.
// grid occupancy is descriptive (4x4 image bins), not a tracking threshold.
inline void LogVisualHealthFrame(const StereoFrame& stereo,
                                 const char* stage,
                                 const Timestamp reference_stamp) {
  if (!VisualHealthAuditEnabled()) return;
  const auto& frame = stereo.left_frame_;
  std::array<size_t, 16> bins{};
  size_t valid = 0;
  size_t age_sum = 0;
  const std::string phase(stage);
  const bool depth_available = !stereo.right_keypoints_rectified_.empty() &&
      (phase == "pre_stereo" || phase == "post_ransac_pre_replenishment" ||
       phase == "keyframe_final");
  int valid_depth = depth_available ? 0 : -1;
  for (size_t i = 0; i < frame.keypoints_.size(); ++i) {
    if (frame.landmarks_.at(i) == -1) continue;
    ++valid;
    age_sum += frame.landmarks_age_.at(i);
    const auto& p = frame.keypoints_.at(i);
    const int x = std::max(0, std::min(3, static_cast<int>(4*p.x/frame.img_.cols)));
    const int y = std::max(0, std::min(3, static_cast<int>(4*p.y/frame.img_.rows)));
    ++bins.at(4*y+x);
    if (depth_available && i < stereo.right_keypoints_rectified_.size() &&
        stereo.right_keypoints_rectified_[i].first == KeypointStatus::VALID) {
      ++valid_depth;
    }
  }
  std::ostringstream row;
  row << std::setprecision(17) << "KIMERA_VISUAL_HEALTH_FRAME_ROW,"
      << stereo.timestamp_ << ',' << reference_stamp << ',' << stereo.id_
      << ',' << stage << ',' << frame.keypoints_.size() << ',' << valid
      << ',' << valid_depth << ',' << age_sum;
  for (const auto count : bins) row << ',' << count;
  LOG(INFO) << row.str();
}

inline void LogVisualHealthMatches(const char* channel,
                                   const Timestamp from_stamp,
                                   const Timestamp to_stamp,
                                   const size_t candidates,
                                   const size_t returned_inliers,
                                   const TrackingStatus status) {
  if (!VisualHealthAuditEnabled()) return;
  // Returned inliers on INVALID are NOT certified geometrically valid.
  LOG(INFO) << "KIMERA_VISUAL_HEALTH_MATCH_ROW," << from_stamp << ','
            << to_stamp << ',' << channel << ',' << candidates << ','
            << returned_inliers << ',' << TrackerStatusSummary::asString(status);
}
}  // namespace VIO
