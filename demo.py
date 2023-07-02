import numpy as np
from matplotlib import pyplot as plt

# get data
data = np.load("h3.npy", allow_pickle=True).item()
verts3d = np.array(data['verts'])
vcolors = np.array(data['vertex_colors'])
faces = np.array(data['face_indices'])
c_org = np.array(data['cam_eye'])
c_up = np.array(data['cam_up'])
c_lookat = np.array(data['cam_lookat'])
ka = np.array(data['ka'])
kd = np.array(data['kd'])
ks = np.array(data['ks'])
phong = np.array(data['n'])
light_pos = np.array(data['light_positions'])
light_val = np.array(data['light_intensities'])
Ia = np.array(data['Ia'])
M = np.array(data['M'])
N = np.array(data['N'])
W = np.array(data['W'])
bg_color = np.array(data['bg_color'])
focal = data['focal']