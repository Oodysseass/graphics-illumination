import numpy as np
from matplotlib import pyplot as plt
from functions import render_object
from classes import PhongMaterial, PointLight

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
H = np.array(data['H'])
bg_color = np.array(data['bg_color'])
focal = data['focal']

# init scene
mat = PhongMaterial(ka, kd, ks, phong)
lights = []
for i in range(len(light_pos)):
    lights.append(PointLight(light_pos[i], light_val[i]))

# shoooooooot
img = render_object("gouraud", focal, c_org, c_lookat, c_up, bg_color, M, N, \
                    H, W, verts3d, vcolors, faces, mat, lights, Ia)
plt.imsave('0.jpg', img)

img = render_object("phong", focal, c_org, c_lookat, c_up, bg_color, M, N, \
                    H, W, verts3d, vcolors, faces, mat, lights, Ia)
plt.imsave('1.jpg', img)
