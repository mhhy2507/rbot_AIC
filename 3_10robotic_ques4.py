#Ques 4:
#Nội suy Euler angles (góc Euler):
#Nội suy quaternion (Slerp):
#Nội suy twist exponential:

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Slerp, Rotation
from scipy.linalg import logm, expm

# (TUỲ CHỌN) Nếu bạn có spatialmath, dùng cho RPY:
try:
    from spatialmath.base import rpy2r, tr2rpy, r2q, q2r  # chỉ dùng rpy2r, tr2rpy
    HAS_SPATIALMATH = True
except Exception:
    HAS_SPATIALMATH = False

# =========================
# 1) Tạo hai orientation
# =========================
# Dùng RPY (roll, pitch, yaw) để tạo R1, R2 cho dễ so sánh
rpy1 = np.array([0.1, 0.2, 0.3])
rpy2 = np.array([1.0, 1.2, 1.3])

if HAS_SPATIALMATH:
    R1 = rpy2r(*rpy1)   # spatialmath: thứ tự r,p,y quanh x,y,z
    R2 = rpy2r(*rpy2)
else:
    # scipy: 'xyz' nghĩa là quay tuần tự quanh x, rồi y, rồi z
    R1 = Rotation.from_euler('xyz', rpy1).as_matrix()
    R2 = Rotation.from_euler('xyz', rpy2).as_matrix()

ts = np.linspace(0.0, 1.0, 200)

# =========================
# 2) Nội suy Euler (linear angles)
# =========================
if HAS_SPATIALMATH:
    # tr2rpy để chắc chắn đúng quy ước đang dùng
    rpy1_exact = rpy1  # vì R1 được tạo từ rpy1
    rpy2_exact = rpy2
    rpy_traj = np.array([rpy1_exact + (rpy2_exact - rpy1_exact) * t for t in ts])
    R_euler = np.array([rpy2r(*rpy) for rpy in rpy_traj])
else:
    # nội suy thẳng trên góc Euler 'xyz' của scipy
    rpy_traj = np.array([rpy1 + (rpy2 - rpy1) * t for t in ts])
    R_euler = Rotation.from_euler('xyz', rpy_traj).as_matrix()

# =========================
# 3) Nội suy Quaternion (SLERP) - CÁCH SỬA CHUẨN
# =========================
# Tránh lỗi thứ tự phần tử quaternion: tạo Rotation trực tiếp từ ma trận
key_rots = Rotation.from_matrix([R1, R2])
slerp = Slerp([0.0, 1.0], key_rots)
R_quat = slerp(ts).as_matrix()

# =========================
# 4) Nội suy Twist exponential (geodesic trên SO(3))
#     R(t) = R1 * exp( t * log(R1^T R2) )
# =========================
S = logm(R1.T @ R2)
R_twist = np.array([R1 @ expm(t * S) for t in ts])

# =========================
# 5) Hàm tiện ích đo “độ mượt”
# =========================
def geodesic_angle(Ra, Rb):
    # góc giữa Ra và Rb: theta = acos((trace(Ra^T Rb)-1)/2)
    Rrel = Ra.T @ Rb
    val = (np.trace(Rrel) - 1.0) / 2.0
    val = np.clip(val, -1.0, 1.0)
    return np.arccos(val)

def theta_curve(R_traj, R_ref):
    return np.array([geodesic_angle(R_ref, R) for R in R_traj])

def omega_curve(R_traj, ts):
    # omega(t_k) ≈ angle( R_k^T R_{k+1} ) / dt
    omegas = np.zeros(len(ts))
    for k in range(len(ts) - 1):
        dt = ts[k+1] - ts[k]
        omegas[k] = geodesic_angle(R_traj[k], R_traj[k+1]) / dt
    omegas[-1] = omegas[-2]
    return omegas

# Tính theta(t) và omega(t) cho 3 phương pháp
theta_e = theta_curve(R_euler, R1)
theta_q = theta_curve(R_quat,  R1)
theta_t = theta_curve(R_twist, R1)

omega_e = omega_curve(R_euler, ts)
omega_q = omega_curve(R_quat,  ts)
omega_t = omega_curve(R_twist, ts)

# =========================
# 6) VẼ: Quỹ đạo trục z
# =========================
plt.figure(figsize=(9, 6))
for R_traj, label in [(R_euler, 'Euler'),
                      (R_quat,  'Quaternion (SLERP)'),
                      (R_twist, 'Twist exponential')]:
    zs = np.array([R[:, 2] for R in R_traj])      # cột thứ 3 là trục z của end-effector
    plt.plot(zs[:, 0], zs[:, 1], label=label)
plt.xlabel('z_x')
plt.ylabel('z_y')
plt.title('Quỹ đạo trục z của end-effector')
plt.axis('equal')
plt.legend()
plt.tight_layout()

# =========================
# 7) VẼ: Góc geodesic theta(t)
# =========================
plt.figure(figsize=(9, 5))
plt.plot(ts, theta_e, label='Euler')
plt.plot(ts, theta_q, label='Quaternion (SLERP)')
plt.plot(ts, theta_t, label='Twist exponential')
plt.xlabel('t')
plt.ylabel(r'$\theta(t)$ [rad]')
plt.title('Khoảng cách quay từ R1 đến R(t)')
plt.legend()
plt.tight_layout()

# =========================
# 8) VẼ: Tốc độ góc omega(t)
# =========================
plt.figure(figsize=(9, 5))
plt.plot(ts, omega_e, label='Euler')
plt.plot(ts, omega_q, label='Quaternion (SLERP)')
plt.plot(ts, omega_t, label='Twist exponential')
plt.xlabel('t')
plt.ylabel(r'$\omega(t)$ [rad/s] (tính theo tham số t)')
plt.title('Tốc độ góc tức thời dọc quỹ đạo')
plt.legend()
plt.tight_layout()

plt.show()


#b) Đánh giá sự khác biệt
#Euler: Có thể không mượt, dễ bị nhảy góc hoặc gimbal lock.
#Quaternion (Slerp): Mượt mà, không singularity, lý tưởng cho robot.
#Twist exponential: Mượt, ý nghĩa vật lý rõ ràng (chuyển động quanh trục cố định).
#c) Liên hệ hiện tượng mất ổn định khi robot hàn
#Khi robot hàn di chuyển qua nhiều góc Euler khác nhau, có thể gặp singularity (gimbal lock), dẫn đến chuyển động không mượt, robot có thể quay đột ngột hoặc không kiểm soát được hướng.
#Sử dụng quaternion hoặc twist exponential giúp tránh hiện tượng này, đảm bảo chuyển động ổn định và mượt mà hơn cho đầu hàn.