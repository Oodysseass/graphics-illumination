import numpy as np
import math
from classes import Edge
from projection import projection

# calculates light of given point
def light(point, normal, vcolor, cam_pos, mat, lights, Ia):
    ## init
    I = np.zeros((1, 3))

    ## ambient light
    I += mat.ka * Ia

    ## diffuse and specular reflection
    # point to camera vector
    V = (cam_pos - point) / np.linalg.norm(cam_pos - point)
    for light in lights:
        # point to source vector
        L = (light.pos - point) / np.linalg.norm(light.pos - point)

        # inner products
        inner1 = np.dot(normal, L)
        temp = 2 * inner1 * normal - L
        inner2 = np.dot(temp, V)

        I += mat.kd * inner1 * light.intensity * vcolor
        I += mat.ks * (inner2 ** mat.nphong) * light.intensity * vcolor

    return np.clip(I, 0, 1)

# calculates normal vectors of vertices
def calculate_normals(verts, faces):
    normals = np.zeros(verts.shape)

    # for each triangle
    for face in faces.T:
        triangle = verts[:, face]

        # calculate normal of the triangle as the cross product of two edges
        AB = triangle[:, 1] - triangle[:, 0]
        AC = triangle[:, 2] - triangle[:, 0]
        normal = np.cross(AB, AC)

        normals[:, face[0]] += normal
        normals[:, face[1]] += normal
        normals[:, face[2]] += normal

    # normalize (normalize normals ha ha ha)
    norms = np.linalg.norm(normals, axis=0)
    normals /= norms

    return normals

def interpolate_vectors(p1, p2, V1, V2, xy, dim):
    if p1[dim - 1] == p2[dim - 1]:
        return V1
    l = (xy - p1[dim - 1]) / (p2[dim - 1] - p1[dim -1 ])

    V = (1 - l) * V1 + l * V2

    return V

# implements gouraud shading
def shade_gouraud(verts_p, verts_n, verts_c, bcoords, \
                  cam_pos, mat, lights, light_amb, X):
    Y = X

    # calculate color of each vertex
    vcolors = np.zeros((3, 3))
    for i in range(3):
        vcolors[i] = light(bcoords, verts_n[:, i], verts_c[:, i], cam_pos, \
                           mat, lights, light_amb)

    # algorithm from previous exercises
    vertices = verts_p.T

    if all(vertices[0] == vertices[1]) and all(vertices[1] == vertices[2]):
        Y[vertices[0, 1], vertices[0, 0]] = \
            np.mean(vcolors, axis=0)
        return Y

    edges = [Edge() for _ in range(3)]
    edges[0] = Edge(np.array([vertices[0], \
                                 vertices[1]]))
    edges[1] = Edge(np.array([vertices[1], \
                                 vertices[2]]))
    edges[2] = Edge(np.array([vertices[2], \
                                 vertices[0]]))

    y_min = min([edge.y_min[1] for edge in edges])
    y_max = max([edge.y_max for edge in edges])

    actives = 0
    for edge in edges:
        if edge.y_min[1] == y_min:
            edge.active = True
            actives = actives + 1
    border_points = []

    if actives == 3:
        for edge in edges:
            if edge.m == float('-inf'):
                edge.active = False
                actives = actives - 1

    if actives == 3:
        for i, edge in enumerate(edges):
            if edge.m == 0:
                for x in range(edge.vertices[0][0], edge.vertices[1][0] + 1):
                    Y[y_min, x] = \
                        interpolate_vectors(edge.vertices[0], edge.vertices[1],\
                                            vcolors[i], vcolors[(i + 1) % 3], \
                                            x, 1)
                actives = actives - 1
                edge.active = False
            else:
                border_points.append([edge.y_min[0] + 1 / edge.m, edge.m, i])
        y_min = y_min + 1

    if len(border_points) == 0:
        for i, edge in enumerate(edges):
            if edge.active:
                border_points.append([edge.y_min[0], edge.m, i])

    for y in range(y_min, y_max + 1):
        border_points = sorted(border_points, key=lambda x: x[0])

        color_A = interpolate_vectors(edges[border_points[0][2]].vertices[0], \
                                      edges[border_points[0][2]].vertices[1], \
                                      vcolors[border_points[0][2]], \
                                      vcolors[(border_points[0][2] + 1) % 3], \
                                      y, 2)
        color_B = interpolate_vectors(edges[border_points[1][2]].vertices[0], \
                                      edges[border_points[1][2]].vertices[1], \
                                      vcolors[border_points[1][2]], \
                                      vcolors[(border_points[1][2] + 1) % 3], \
                                      y, 2)

        for x in range(math.floor(border_points[0][0] + 0.5), \
                       math.floor(border_points[1][0] + 0.5) + 1):
            Y[y, x] = interpolate_vectors( \
                np.array([math.floor(border_points[0][0] + 0.5), y]), \
                np.array([math.floor(border_points[1][0] + 0.5) + 1, y]), \
                color_A, color_B, x, 1)

        if y == y_max:
            break

        for point in border_points:
            point[0] = point[0] + 1 / point[1]

        for i, edge in enumerate(edges):
            if edge.y_min[1] == y + 1:
                edge.active = True
                actives = actives + 1
                border_points.append([edge.y_min[0], edge.m, i])

        if actives == 3:
            if border_points[-1][1] == 0:
                del border_points[-1]
                continue
            for i, edge in enumerate(edges):
                if edge.y_max == y + 1:
                    if border_points[0][2] == i:
                        del border_points[0]
                    else:
                        del border_points[1]
                    edge.active = False
                    actives = actives - 1
                    break

    return Y

# implements phong shading
def shade_phong(verts_p, verts_n, verts_c, bcoords, \
                  cam_pos, mat, lights, light_amb, X):
    Y = X

    # algorithm from previous exercises
    # interpolation in normal vectors is added wherever
    # interpolation for color already existed
    vertices = verts_p.T
    vcolors = verts_c.T

    if all(vertices[0] == vertices[1]) and all(vertices[1] == vertices[2]):
        Y[vertices[0, 1], vertices[0, 0]] = \
            np.mean(vcolors, axis=0)
        return Y

    edges = [Edge() for _ in range(3)]
    edges[0] = Edge(np.array([vertices[0], \
                                 vertices[1]]))
    edges[1] = Edge(np.array([vertices[1], \
                                 vertices[2]]))
    edges[2] = Edge(np.array([vertices[2], \
                                 vertices[0]]))

    y_min = min([edge.y_min[1] for edge in edges])
    y_max = max([edge.y_max for edge in edges])

    actives = 0
    for edge in edges:
        if edge.y_min[1] == y_min:
            edge.active = True
            actives = actives + 1
    border_points = []

    if actives == 3:
        for edge in edges:
            if edge.m == float('-inf'):
                edge.active = False
                actives = actives - 1

    if actives == 3:
        for i, edge in enumerate(edges):
            if edge.m == 0:
                for x in range(edge.vertices[0][0], edge.vertices[1][0] + 1):
                    color = interpolate_vectors(edge.vertices[0], \
                                            edge.vertices[1], vcolors[i], \
                                            vcolors[(i + 1) % 3], x, 1)
                    normal = interpolate_vectors(edge.vertices[0], \
                                            edge.vertices[1], verts_n[:, i], \
                                            verts_n[:, (i + 1) % 3], x, 1)
                    Y[y_min, x] = light(bcoords, normal, color, cam_pos, mat,
                                        lights, light_amb)

                actives = actives - 1
                edge.active = False
            else:
                border_points.append([edge.y_min[0] + 1 / edge.m, edge.m, i])
        y_min = y_min + 1

    if len(border_points) == 0:
        for i, edge in enumerate(edges):
            if edge.active:
                border_points.append([edge.y_min[0], edge.m, i])

    for y in range(y_min, y_max + 1):
        border_points = sorted(border_points, key=lambda x: x[0])

        color_A = interpolate_vectors(edges[border_points[0][2]].vertices[0], \
                                      edges[border_points[0][2]].vertices[1], \
                                      vcolors[border_points[0][2]], \
                                      vcolors[(border_points[0][2] + 1) % 3], \
                                      y, 2)
        color_B = interpolate_vectors(edges[border_points[1][2]].vertices[0], \
                                      edges[border_points[1][2]].vertices[1], \
                                      vcolors[border_points[1][2]], \
                                      vcolors[(border_points[1][2] + 1) % 3], \
                                      y, 2)
        # interpolate normal vectors
        normal_A = interpolate_vectors(edges[border_points[0][2]].vertices[0], \
                                       edges[border_points[0][2]].vertices[1], \
                                       verts_n[:, border_points[0][2]], \
                                       verts_n[:, (border_points[0][2] + 1) % 3], \
                                       y, 2)
        normal_B = interpolate_vectors(edges[border_points[1][2]].vertices[0], \
                                       edges[border_points[1][2]].vertices[1], \
                                       verts_n[:, border_points[1][2]], \
                                       verts_n[:, (border_points[1][2] + 1) % 3], \
                                       y, 2)

        for x in range(math.floor(border_points[0][0] + 0.5), \
                       math.floor(border_points[1][0] + 0.5) + 1):
            color = interpolate_vectors( \
                np.array([math.floor(border_points[0][0] + 0.5), y]), \
                np.array([math.floor(border_points[1][0] + 0.5) + 1, y]), \
                color_A, color_B, x, 1)
            normal = interpolate_vectors( \
                np.array([math.floor(border_points[0][0] + 0.5), y]), \
                np.array([math.floor(border_points[1][0] + 0.5) + 1, y]), \
                normal_A, normal_B, x, 1)
            Y[y, x] = light(bcoords, normal, color, cam_pos, mat, \
                            lights, light_amb)

        if y == y_max:
            break

        for point in border_points:
            point[0] = point[0] + 1 / point[1]

        for i, edge in enumerate(edges):
            if edge.y_min[1] == y + 1:
                edge.active = True
                actives = actives + 1
                border_points.append([edge.y_min[0], edge.m, i])

        if actives == 3:
            if border_points[-1][1] == 0:
                del border_points[-1]
                continue
            for i, edge in enumerate(edges):
                if edge.y_max == y + 1:
                    if border_points[0][2] == i:
                        del border_points[0]
                    else:
                        del border_points[1]
                    edge.active = False
                    actives = actives - 1
                    break

    return Y

# renders 3d object to 2d photo
def render_object(shader, focal, eye, lookat, up, bg_color, M, N, H, W, \
    verts, vert_colors, faces, mat, lights, light_amb):
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
                                vert_colors[:, face], \
                                np.mean(verts[:, face], axis=0), \
                                eye, mat, lights, light_amb, img)
    elif shader == "phong":
        for face in faces.T:
            img = shade_phong(n2d[:, face], normals[:, face], \
                                vert_colors[:, face], \
                                np.mean(verts[:, face], axis=0), \
                                eye, mat, lights, light_amb, img)

    return img
