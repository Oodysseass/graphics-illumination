import numpy as np
from classes import PhongMaterial, PointLight
from projection import projection
from shaders import shade_gouraud, shade_phong

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

# renders 3d object to 2d photo
def render_object(shader, focal, eye, lookat, up, bg_color, M, N, H, W, \
    verts, vert_colors, faces, mat, n, lights, light_amb):
    # calculate normal vectors
    normals = calculate_normals(verts, faces)

    # get pixel coordinates of points
    n2d, depth = projection(verts, H, W, M, N, focal, eye, lookat, up)

    # find the points outside the photo
    outside_points = np.logical_or.reduce((n2d[0] < 0, n2d[0] > N - 1, \
                                           n2d[1] < 0, n2d[1] > M - 1))

    # get the indices of those points
    to_remove = np.where(outside_points)[0]

    # remove the points and corresponding triangles
    n2d = np.delete(n2d, to_remove, axis=1)
    faces = faces[:, ~np.isin(faces, to_remove).any(axis=0)]

    # calculate depth of each triagle
    triangles_depth = np.array(np.mean(depth[faces.T], axis = 1))

    # sort faces triangles depth
    indices = np.flip(np.argsort(triangles_depth))
    triangles_depth = triangles_depth[indices]
    faces = faces[:, indices]

    # initialize image to background color
    img = np.ones((M, N, 3)) * bg_color

    # shade each triangle accordingly
    if shader == "gouraud":
        for face in faces.T:
            img = shade_gouraud(n2d[:, face], normals[:, face], \
                                vert_colors[:, 3], \
                                np.mean(n2d[:, face], axis=1), \
                                eye, mat, lights, light_amb, img)
    elif shader == "phong":
        for face in faces.T:
            img = shade_phong(n2d[:, face], normals[:, face], \
                                vert_colors[:, 3], \
                                np.mean(n2d[:, face], axis=1), \
                                eye, mat, lights, light_amb, img)

    return img
