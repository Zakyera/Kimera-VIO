#include <gtest/gtest.h>
#include "kimera-vio/imu-frontend/CbsNativeImuMeanBridge.h"

namespace VIO {
namespace {
ImuParams params() {
  ImuParams p;
  p.acc_random_walk_ = p.gyro_random_walk_ = 0.001;
  p.acc_noise_density_ = p.gyro_noise_density_ = 0.01;
  p.imu_integration_sigma_ = 0.0001;
  p.n_gravity_.setZero();
  p.imu_preintegration_type_ = ImuPreintegrationType::kPreintegratedCombinedMeasurements;
  return p;
}
constexpr ImuStamp k0 = 1727775008000000000;
ImuStamp t(double s) { return k0 + std::llround(s * 1e9); }
void fill(CbsNativeImuMeanBridge& h, bool variable = false) {
  ImuStampS ts(1, 11);
  ImuAccGyrS rates = ImuAccGyrS::Zero(6, 11);
  for (int i = 0; i < 11; ++i) {
    ts(i) = t(i * .01);
    rates(0, i) = variable ? .2 + .03 * i : 2.;
    rates(5, i) = variable ? .4 - .01 * i : 0.;
  }
  // Multiple consecutive packets, including the repeated boundary sample.
  h.append(ts.leftCols(6), rates.leftCols(6));
  h.append(ts.rightCols(6), rates.rightCols(6));
}
gtsam::NavState seed() {
  return gtsam::NavState(gtsam::Pose3(), gtsam::Vector3(1., 0., 0.));
}
}  // namespace

TEST(CbsNativeImuMeanBridge, ExactFractionalConstantAccelerationAndInverse) {
  CbsNativeImuMeanBridge h;
  fill(h);
  const auto z = h.between(t(0), seed(), ImuBias(), params(), t(.023), t(.071), t(.1));
  const double expected = (.071 + .071 * .071) - (.023 + .023 * .023);
  EXPECT_NEAR(z.x(), expected, 1e-13);
  EXPECT_NEAR(z.y(), 0., 1e-13);
  const auto inverse = h.between(t(0), seed(), ImuBias(), params(), t(.071), t(.023), t(.1));
  EXPECT_LT(gtsam::Pose3::Logmap(z.compose(inverse)).norm(), 1e-13);
}

TEST(CbsNativeImuMeanBridge, MatchesNativePreintegrationAndBias) {
  CbsNativeImuMeanBridge h;
  ImuStampS ts(1, 6);
  ImuAccGyrS xs = ImuAccGyrS::Zero(6, 6);
  for (int i = 0; i < 6; ++i) {
    ts(i) = t(i * .01);
    xs(0, i) = .2 + .03 * i;
    xs(5, i) = .4 - .01 * i;
  }
  const ImuBias bias(gtsam::Vector3(.01, -.02, .03), gtsam::Vector3(.002, 0., .001));
  h.append(ts, xs);
  ImuFrontend f(params(), bias);
  const auto expected = f.preintegrateImuMeasurements(ts, xs)->predict(seed(), bias).pose();
  const auto actual = h.between(t(0), seed(), bias, params(), t(0), t(.05), t(.05));
  EXPECT_LT(gtsam::Pose3::Logmap(expected.between(actual)).norm(), 1e-14);
}

TEST(CbsNativeImuMeanBridge, NoncommutingBothOffsetSignsAndIdentity) {
  CbsNativeImuMeanBridge h;
  fill(h, true);
  auto u = [&](double a, double b) {
    return h.between(t(0), seed(), ImuBias(), params(), t(a), t(b), t(.1));
  };
  for (double offset : {-.008, 0., .009}) {
    auto y = CbsNativeImuMeanBridge::align(u(.02, .02 + offset),
                u(.02 + offset, .07 + offset), u(.07 + offset, .07));
    EXPECT_LT(gtsam::Pose3::Logmap(u(.02, .07).between(y)).norm(), 1e-13);
  }
  EXPECT_LT(gtsam::Pose3::Logmap(u(.05, .05)).norm(), 1e-15);
}

TEST(CbsNativeImuMeanBridge, CausalityMissingHistoryPruningAndTransaction) {
  CbsNativeImuMeanBridge h;
  fill(h, true);
  auto query = [&](double a, double b, double available) {
    return h.between(t(0), seed(), ImuBias(), params(), t(a), t(b), t(available));
  };
  const auto before = query(.02, .04, .04);
  EXPECT_THROW(query(.02, .051, .05), std::runtime_error);
  EXPECT_THROW(query(.02, .11, .12), std::runtime_error);
  ImuStampS overlap(1, 2); overlap << t(.03), t(.04);
  ImuAccGyrS xs = ImuAccGyrS::Zero(6, 2);
  EXPECT_THROW(h.append(overlap, xs), std::runtime_error);
  EXPECT_TRUE(before.equals(query(.02, .04, .04), 0.));
  for (int i = 0; i < 5; ++i)
    EXPECT_TRUE(before.equals(query(.02, .04, .04), 0.));
  h.pruneBefore(t(.05));
  EXPECT_THROW(query(.02, .04, .04), std::runtime_error);
}
}  // namespace VIO
