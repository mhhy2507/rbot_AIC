#ques 5: 
#a
import numpy as np
from scipy.linalg import expm
from spatialmath.base import angvec2tr, r2q, q2r

# Dữ liệu
a = np.array([2, 3, 4])
a = a / np.linalg.norm(a)  # Chuẩn hóa trục
theta = 0.5

# 1. Rodrigues' formula
K = np.array([
    [0, -a[2], a[1]],
    [a[2], 0, -a[0]],
    [-a[1], a[0], 0]
])
R_rodrigues = np.eye(3) + np.sin(theta)*K + (1-np.cos(theta))*(K @ K)

# 2. Matrix exponential
R_expm = expm(theta * K)

# 3. Toolbox function
R_toolbox = angvec2tr(theta, a)[:3, :3]

print("Rodrigues:\n", R_rodrigues)
print("Exponential:\n", R_expm)
print("Toolbox angvec2tr:\n", R_toolbox)

#b
print("Rodrigues vs Exponential:", np.allclose(R_rodrigues, R_expm))
print("Rodrigues vs Toolbox:", np.allclose(R_rodrigues, R_toolbox))
print("Exponential vs Toolbox:", np.allclose(R_expm, R_toolbox))
#c
# Tìm quaternion từ rotation matrix
q = r2q(R_rodrigues)
print("Quaternion:", q)

# Chuyển lại rotation matrix từ quaternion
R_from_q = q2r(q)
print("Rotation from quaternion:\n", R_from_q)
print("Rodrigues vs Quaternion matrix:", np.allclose(R_rodrigues, R_from_q))