import numpy as np

class Edge:
    def __init__(self, vertices=None):
        if vertices is not None:
            # points of edge
            self.vertices = vertices
            # cordinates of yk_min, yk_max
            if vertices[0, 1] < vertices[1, 1]:
                self.y_min = vertices[0, :]
                self.y_max = vertices[1, 1]
            else:
                self.y_min = vertices[1, :]
                self.y_max = vertices[0, 1]
            # slope of edge
            if vertices[0, 0] != vertices[1, 0]:
                self.m = (vertices[0, 1] - vertices[1, 1]) / \
                    (vertices[0, 0] - vertices[1, 0])
            # edge is essentially a point
            # -inf just for flag
            elif np.all(vertices[0, :] == vertices[1, :]):
                self.m = float('-inf')
            # vertical edge
            else:
                self.m = float('inf')
            self.active = False
        else:
            self.ordinal = None
            self.vertices = None
            self.y_min = None
            self.y_max = None
            self.m = None
            self.active = None
    
    def __str__(self):
        return f"Vertices: {self.vertices}, \
            y_min: {self.y_min}, y_max: {self.y_max}, m: {self.m}, \
                active: {self.active}"


class PhongMaterial:
    def __init__(self, ka: float, kd: float, ks: float, nphong: int):
        self.ka = ka
        self.kd = kd
        self.ks = ks
        self.nphong = nphong


class PointLight:
    def __init__(self, pos: np.ndarray, intensity: np.ndarray):
        assert pos.shape == (1, 3)
        assert intensity.shape == (1, 3)
        
        self.pos = pos
        self.intensity = intensity

