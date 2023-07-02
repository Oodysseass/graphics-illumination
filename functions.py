import numpy as np
from classes import edge, PhongMaterial, PointLight

def light(point, normal, vcolor, cam_pos, mat, lights):
    ## init
    I = np.zeros(3, 1)
    normal = normal / np.linalg.norm(normal)

    ## ambient light
    I = I + mat.ka * lights.intensity.T
    
    ## diffuse reflection
    # point to source vector
    L = (lights.pos - point) / np.linalg.norm(lights.pos - point)

    # inner product
    inner = np.dot(normal.flatten(), L.flatten())

    I = I + mat.kd * inner * lights.intensity.T

    ## specular reflection
    # point to camera vector
    V = (cam_pos - point) / np.linalg.norm(cam_pos - point)

    # calculate inner product
    temp = 2 * inner * normal - L
    inner = np.dot(temp.flatten(), V.flatten())

    I = I + mat.ks * (inner ** mat.nphong) * lights.intensity.T


    return I