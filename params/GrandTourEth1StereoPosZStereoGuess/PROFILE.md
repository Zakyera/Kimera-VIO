# GrandTour ETH-1 front-stereo Kimera native baseline

This validated profile uses `n_gravity = [0, 0, +9.81]`, matching the
stationary AlphaSense accelerometer convention in the downloaded ETH-1 bag.
It uses Kimera's metric stereo relative pose as the backend pose guess,
instead of the IMU-only prediction.

- Mission: `2024-10-01-11-29-55`
- Body/IMU: `/boxi/alphasense/imu` (`alphasense_imu`, about 200 Hz)
- Left: `/boxi/alphasense/front_left/image_raw/compressed`
- Right: `/boxi/alphasense/front_right/image_raw/compressed`
- Cameras: 1440 x 1080, 10 Hz, equidistant distortion
- Baseline: 0.11008469252975077 m
- CBS: disabled for the native-estimator baseline

`T_BS` is `T_alphasense_imu_camera`, composed from the mission's
`tf_minimal` bag. The left-to-right translation expressed in the left camera
frame is `[0.1100771441, -0.0004686583, -0.0012009309]` m.

This profile retains the Newer College Alphasense IMU noise values only as a
bootstrap. It must pass a native 30-second smoke test and full-sequence
reference evaluation before any CBS comparison is made.
