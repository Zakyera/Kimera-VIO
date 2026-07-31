import numpy as np

# Original transformation matrix (from camera to IMU)
Tic = np.array([
    [ 0.00671209, -0.00971447,  0.99993029,  0.14118513],
    [-0.99993865, -0.00887635,  0.00662591,  0.1552482],
    [ 0.00881136, -0.99991342, -0.00977345,  0.01917531],
    [ 0.0,         0.0,         0.0,         1.0]
])

# Extract rotation and translation
R = Tic[:3, :3]
t = Tic[:3, 3]

# Compute inverse
R_inv = R.T
t_inv = -R_inv @ t

# Assemble inverse transformation
Tci = np.eye(4)
Tci[:3, :3] = R_inv
Tci[:3, 3] = t_inv

# Print the result
np.set_printoptions(precision=8, suppress=True)
print("Inverse Transformation (Tci):")
print(Tci)
