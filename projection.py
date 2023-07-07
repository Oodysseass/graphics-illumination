import numpy as np
import math

# calculates rotation matrix around an axis
def rotmat(theta, u):
    # make sure its unit vector
    u = u / np.linalg.norm(u)

    # rotation matrix
    R = np.array([[(1 - math.cos(theta)) * (u[0] ** 2) + math.cos(theta),
                   (1 - math.cos(theta)) * u[0] * u[1] - math.sin(theta) * u[2],
                   (1 - math.cos(theta)) * u[0] * u[2] + math.sin(theta) * u[1]],
                  [(1 - math.cos(theta)) * u[1] * u[0] + math.sin(theta) * u[2],
                   (1 - math.cos(theta)) * (u[1] ** 2) + math.cos(theta),
                   (1 - math.cos(theta)) * u[1] * u[2] - math.sin(theta) * u[0]],
                  [(1 - math.cos(theta)) * u[2] * u[0] - math.sin(theta) * u[1],
                   (1 - math.cos(theta)) * u[2] * u[1] + math.sin(theta) * u[0],
                   (1 - math.cos(theta)) * (u[2] ** 2) + math.cos(theta)]])

    return R

# rotates and moves point c_p
def RotateTranslate(c_p, theta, u, A, t):
    # get matrix
    R = rotmat(theta, u)

    # move to A
    c_p = c_p.T - A

    # rotate
    c_p = np.dot(R, c_p.T)

    # move back to starting system
    c_q = c_p.T + A

    # displace by t
    c_q = c_q + t

    return c_q.T

# calculates new coordinates of c_p
# when we move and rotate start point of system
def ChangeCoordinateSystem(c_p, R, c_0):
    # L^(-1)
    R_inv = R.T

    # new coordinates
    # transposes are primarily used cause of python
    c_temp = c_p.T - c_0
    d_p = np.dot(R_inv, c_temp.T)

    return d_p

# projects point in wcs to ccs using pinhole model
def PinHole(f, cv, cx, cy, cz, p3d):
    # rotation matrix
    R = np.array([cx, cy, cz]).T

    # change coordinate system to ccs
    p3d_ccs = ChangeCoordinateSystem(p3d, R, cv)

    # extract z coordinate
    depth = p3d_ccs[2, :].T.flatten()

    # project
    x_proj = f * p3d_ccs[0, :] / depth
    y_proj = f * p3d_ccs[1, :] / depth
    p2d = np.array([x_proj[0, :], y_proj[0,:]])

    return p2d, depth

# projects the same without knowing ccs unit vectors
def CameraLookingAt(f, cv, cK, cup, p3d):
    # calculate unit vectors
    cz = cK / np.linalg.norm(cK)

    t = cup - np.dot(cup.T,  cz) * cz
    cy = t / np.linalg.norm(t)

    cy = cy.T
    cz = cz.T

    cx = np.cross(cy, cz)

    # use pinhole
    p2d, depth = PinHole(f, cv.T, cx, cy, cz, p3d)

    return p2d, depth

# maps coords of projected points to pixels
def rasterize(p2d, Rows, Columns, H, W):
    # scale to new range
    # + (H/w)/2 because camera coords are in range [-H/2, H/2]
    n2d = np.array([(p2d[0, :] + H / 2) * (Rows - 1) / H, \
                    (p2d[1, :] + W / 2) * (Columns - 1) / W])

    # round
    n2d = np.round(n2d).astype(int)

    return n2d

# projects 3d points using pinhole model and calculates pixel coords
def projection(p3d, H, W, Rows, Columns, f, cv, cK, cup):
    # project points
    p2d, depth = CameraLookingAt(f, cv, cK - cv, cup, p3d)
    # get pixel coords
    n2d = rasterize(p2d, Rows, Columns, H, W)

    return n2d

