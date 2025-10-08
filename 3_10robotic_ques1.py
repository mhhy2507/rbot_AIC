#Ques 1a:



#Ques 1b: Generate 100 random rotation matrices and verify their properties
#Key characteristics include: it's always square, transforms vectors without 
#changing their magnitude or shape, preserves angles between vectors, and its 
# columns (and rows) are orthonormal unit vectors


import numpy as np
from spatialmath.base import angvec2tr

def random_axis_angle():
    axis = np.random.randn(3)
    axis /= np.linalg.norm(axis)
    angle = np.random.uniform(0, 2*np.pi)
    return angle, axis

rotations = []
for _ in range(100):
    angle, axis = random_axis_angle()
    R = angvec2tr(angle, axis)[:3, :3]  # Extract rotation part
    rotations.append(R)
    print(f'Rotation Matrix:\n{R}')
    det = np.linalg.det(R)
    ortho = np.allclose(R @ R.T, np.eye(3), atol=1e-8)
    print(f"Determinant: {det:.6f}, Orthogonal: {ortho}")



#Ques 1c: Calculate the distance between two orientations by using logarithm of rotation matrix
from scipy.linalg import logm

# Sinh hai ma trận quay ngẫu nhiên
angle1, axis1 = random_axis_angle()
angle2, axis2 = random_axis_angle()
R1 = angvec2tr(angle1, axis1)[:3, :3]
R2 = angvec2tr(angle2, axis2)[:3, :3]
print (f"Rotation Matrix 1:\n{R1}")
print (f"Rotation Matrix 2:\n{R2}")
# Tính khoảng cách
R_rel = R1.T @ R2
log_R = logm(R_rel)
distance = np.linalg.norm(log_R, 'fro')
print(f"Distance between orientations: {distance}") 