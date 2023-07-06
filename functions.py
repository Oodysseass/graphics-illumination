import numpy as np
from classes import edge, PhongMaterial, PointLight

# calculates light of given point
def light(point, normal, vcolor, cam_pos, mat, lights, Ia):
    ## init
    I = np.zeros(3, 1)
    normal = normal / np.linalg.norm(normal)

    ## ambient light
    I += mat.ka * Ia
    
    ## diffuse reflection
    # point to source vector
    for light in lights:
        L = (light.pos - point) / np.linalg.norm(light.pos - point)

        # inner product
        inner = np.dot(normal.flatten(), L.flatten())

        I += mat.kd * inner * light.intensity.T * vcolor

    ## specular reflection
    # point to camera vector
    for light in lights:
        V = (cam_pos - point) / np.linalg.norm(cam_pos - point)

        # calculate inner product
        inner = np.dot(normal.flatten(), L.flatten())
        temp = 2 * inner * normal - L
        inner = np.dot(temp.flatten(), V.flatten())

        I += mat.ks * (inner ** mat.nphong) * light.intensity.T * vcolor

    return np.clip(I, 0, 1)

# calculates normal vectors of
def calculate_normals(verts, faces):
    normals = np.zeros(verts.shape)

    # for each triangle
    for face in faces.T:
        triangle = verts[:, face]

        # calculate normal of the triangle as the cross product of two edges
        AB = triangle[:, 1] - triangle[:, 0]
        AC = triangle[:, 2] - triangle[:, 0]
        normal = np.cross(AB, AC)

        normals[:, face] += normal

    # normalize (normalize normals ha ha ha)
    norms = np.linalg.norm(normals, axis=0)
    normals /= norms

    return normals
