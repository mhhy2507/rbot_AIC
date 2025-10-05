import numpy as np
from spatialmath.base import qrand, q2r, tr2eul, tr2rpy
from scipy.spatial.transform import Slerp, Rotation
import matplotlib.pyplot as plt

# Sinh quaternion đơn vị ngẫu nhiên
q1 = qrand()
q2 = qrand()
print("Random unit quaternion 1:", q1)
print("Random unit quaternion 2:", q2)

# Chuyển sang rotation matrix
R1 = q2r(q1)
R2 = q2r(q2)
print("Rotation matrix 1:\n", R1)
print("Rotation matrix 2:\n", R2)


# Chuyển sang Euler angles (ZYZ convention, mặc định)
eul_zyz_1 = tr2eul(R1)
eul_zyz_2 = tr2eul(R2)
print("Euler angles (ZYZ) 1:", eul_zyz_1)
print("Euler angles (ZYZ) 2:", eul_zyz_2)

# Chuyển sang RPY angles (ZYX convention, mặc định)
rpy_zyx_1 = tr2rpy(R1)
rpy_zyx_2 = tr2rpy(R2)
print("RPY angles (ZYX) 1:", rpy_zyx_1)
print("RPY angles (ZYX) 2:", rpy_zyx_2)

# Nội suy quaternion bằng scipy Slerp
key_rots = Rotation.from_quat([q1, q2])
key_times = [0, 1]
slerp = Slerp(key_times, key_rots)
ts = np.linspace(0, 1, 100)
interp_rots = slerp(ts)
interp_quats = interp_rots.as_quat()

# Nội suy Euler angles (ZYZ)
eul_traj = np.array([eul_zyz_1 + (eul_zyz_2 - eul_zyz_1) * t for t in ts])
# Nội suy RPY angles (ZYX)
rpy_traj = np.array([rpy_zyx_1 + (rpy_zyx_2 - rpy_zyx_1) * t for t in ts])

# Vẽ quỹ đạo các góc
plt.figure(figsize=(12, 6))
plt.subplot(1, 3, 1)
plt.plot(ts, interp_quats)
plt.title('Quaternion Slerp')
plt.xlabel('t')
plt.legend(['q0', 'q1', 'q2', 'q3'])

plt.subplot(1, 3, 2)
plt.plot(ts, eul_traj)
plt.title('Euler (ZYZ) Interpolation')
plt.xlabel('t')
plt.legend(['phi', 'theta', 'psi'])

plt.subplot(1, 3, 3)
plt.plot(ts, rpy_traj)
plt.title('RPY (ZYX) Interpolation')
plt.xlabel('t')
plt.legend(['roll', 'pitch', 'yaw'])

plt.tight_layout()
plt.show()