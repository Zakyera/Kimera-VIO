# GrandTour ETH-1 front-stereo Kimera shared-CPT7 baseline

This validated profile uses `n_gravity = [0, 0, +9.81]`, matching the
stationary CPT7 accelerometer convention in the downloaded ETH-1 bag.
It uses Kimera's metric stereo relative pose as the backend pose guess,
instead of the IMU-only prediction.

- Mission: `2024-10-01-11-29-55`
- Body/IMU: `/boxi/cpt7/imu` (`cpt7_imu`, about 100 Hz)
- Left: `/boxi/alphasense/front_left/image_raw/compressed`
- Right: `/boxi/alphasense/front_right/image_raw/compressed`
- Cameras: 1440 x 1080, 10 Hz, equidistant distortion
- Baseline: 0.11008469252975077 m
- CBS: disabled for the native-estimator baseline

`T_BS` is `T_cpt7_imu_camera`, composed directly from the mission's
`tf_minimal` author calibration. The left-to-right translation expressed in
the left camera frame is `[0.1100771441, -0.0004686583, -0.0012009309]` m.

The white-noise densities are derived from the CPT7 message covariances; bias
random walks remain conservative bootstrap values. No online spatial or
temporal calibration is enabled. This profile must pass a CBS-OFF control run
before any CBS comparison is made.
