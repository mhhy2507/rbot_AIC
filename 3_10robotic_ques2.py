#Ques 2: Calculate the rotation matrix from eular angles and find out the angles using tr2eul

import numpy as np
from spatialmath.base import eul2tr, tr2eul

# Euler angles (phi, theta, psi)
eul = [np.pi/4, np.pi/2, np.pi/3]
R = eul2tr(*eul)[:3, :3]
print("Rotation matrix:\n", R)

# Lấy lại các góc Euler từ ma trận quay
eul_back = tr2eul(R)
print("Recovered Euler angles:", eul_back)
print("\n")

#Ques 2: Khi θ = 0, hai trục quay đầu và cuối trùng nhau, mất một bậc tự do
import numpy as np
from spatialmath.base import eul2tr, tr2eul
eul_lock = [np.pi/4, 0, np.pi/3]
R_lock = eul2tr(*eul_lock)[:3, :3]
print("Rotation matrix (gimbal lock):\n", R_lock)
eul_lock_back = tr2eul(R_lock)
print("Recovered Euler angles (gimbal lock):", eul_lock_back)