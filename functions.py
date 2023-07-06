import numpy as np
from classes import edge, PhongMaterial, PointLight

def light(point, normal, vcolor, cam_pos, mat, lights, Ia):
    ## init
    I = np.zeros(3, 1)
    normal = normal / np.linalg.norm(normal)

    ## ambient light
    I = I + mat.ka * Ia
    
    ## diffuse reflection
    # point to source vector
    for light in lights:
        L = (light.pos - point) / np.linalg.norm(light.pos - point)

        # inner product
        inner = np.dot(normal.flatten(), L.flatten())

        I = I + mat.kd * inner * light.intensity.T * vcolor

    ## specular reflection
    # point to camera vector
    for light in lights:
        V = (cam_pos - point) / np.linalg.norm(cam_pos - point)

        # calculate inner product
        inner = np.dot(normal.flatten(), L.flatten())
        temp = 2 * inner * normal - L
        inner = np.dot(temp.flatten(), V.flatten())

        I = I + mat.ks * (inner ** mat.nphong) * light.intensity.T * vcolor

    return np.clip(I, 0, 1)