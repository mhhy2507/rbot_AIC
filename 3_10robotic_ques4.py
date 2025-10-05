#Ques 4:
#Nội suy Euler angles (góc Euler):
#Nội suy quaternion (Slerp):
#Nội suy twist exponential:

import numpy as np
from spatialmath.base import rpy2r, tr2rpy, q2r, r2q
from scipy.spatial.transform import Slerp, Rotation
from scipy.linalg import logm, expm
import matplotlib.pyplot as plt

# Tạo hai orientation ngẫu nhiên
R1 = rpy2r(0.1, 0.2, 0.3)
R2 = rpy2r(1.0, 1.2, 1.3)

ts = np.linspace(0, 1, 100)

# 1. Nội suy Euler angles
rpy1 = tr2rpy(R1)
rpy2 = tr2rpy(R2)
rpy_traj = np.array([rpy1 + (rpy2 - rpy1) * t for t in ts])
R_euler = np.array([rpy2r(*rpy) for rpy in rpy_traj])

# 2. Nội suy quaternion (Slerp)
q1 = r2q(R1)
q2 = r2q(R2)
key_rots = Rotation.from_quat([q1, q2])
slerp = Slerp([0, 1], key_rots)
R_quat = slerp(ts).as_matrix()

# 3. Nội suy twist exponential
S = logm(np.dot(R1.T, R2))
R_twist = np.array([np.dot(R1, expm(t * S)) for t in ts])

# Vẽ quỹ đạo của trục z sau khi biến đổi
fig = plt.figure(figsize=(10, 6))
for i, (R_traj, label) in enumerate(zip([R_euler, R_quat, R_twist], ['Euler', 'Quaternion', 'Twist exponential'])):
    zs = [R[:, 2] for R in R_traj]
    zs = np.array(zs)
    plt.plot(zs[:, 0], zs[:, 1], label=label)
plt.xlabel('z_x')
plt.ylabel('z_y')
plt.title('Quỹ đạo trục z của end-effector')
plt.legend()
plt.axis('equal')
plt.show()

#b) Đánh giá sự khác biệt
#Euler: Có thể không mượt, dễ bị nhảy góc hoặc gimbal lock.
#Quaternion (Slerp): Mượt mà, không singularity, lý tưởng cho robot.
#Twist exponential: Mượt, ý nghĩa vật lý rõ ràng (chuyển động quanh trục cố định).
#c) Liên hệ hiện tượng mất ổn định khi robot hàn
#Khi robot hàn di chuyển qua nhiều góc Euler khác nhau, có thể gặp singularity (gimbal lock), dẫn đến chuyển động không mượt, robot có thể quay đột ngột hoặc không kiểm soát được hướng.
#Sử dụng quaternion hoặc twist exponential giúp tránh hiện tượng này, đảm bảo chuyển động ổn định và mượt mà hơn cho đầu hàn.