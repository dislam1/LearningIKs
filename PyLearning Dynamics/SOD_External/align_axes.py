import numpy as np

def align_axes(axes_handle):
    # Set the x and y axis labels of the current axes to be aligned to the orientation of the axes.
    # (c) M. Arthington; modified by M. Zhong

    az, el = axes_handle.viewLim
    Raz = np.array([[np.cos(np.deg2rad(az)), np.sin(np.deg2rad(az)), 0],
                    [-np.sin(np.deg2rad(az)), np.cos(np.deg2rad(az)), 0],
                    [0, 0, 1]])
    Rel = np.array([[1, 0, 0],
                    [0, np.cos(np.deg2rad(el)), -np.sin(np.deg2rad(el))],
                    [0, np.sin(np.deg2rad(el)), np.cos(np.deg2rad(el))]])
    u = axes_handle.get_proj().get_matrix().T[2]

    if not np.allclose(u, [0, 0, 1]):
        p = axes_handle.get_position()
        t = axes_handle.get_position()
        v = (p - t)
        v = v / np.linalg.norm(v)  # View vector from camera to target
        u = u / np.linalg.norm(u)  # Camera up vector
        q = np.cross(v, u)
        q = q / np.linalg.norm(q)

        # Get the x axis's projection into the view plane and then find its angle wrt the up vector
        xH = np.cross([1, 0, 0], v)
        xH = xH / np.linalg.norm(xH)
        xH = np.cross(xH, v)
        thetax = -np.rad2deg(np.arccos(np.dot(xH, u))) + 90

        # Check which way the label needs to be rotated
        if np.dot(q, xH) > 0:
            thetax = -thetax

        # Get the y axis's projection into the view plane and then find its angle wrt the up vector
        yH = np.cross([0, 1, 0], v)
        yH = yH / np.linalg.norm(yH)
        yH = np.cross(yH, v)
        thetay = -np.rad2deg(np.arccos(np.dot(yH, u))) + 90
        if np.dot(q, yH) > 0:
            thetay = -thetay

        # Get the z axis's projection into the view plane and then find its angle wrt the up vector
        zH = np.cross([0, 0, 1], v)
        zH = zH / np.linalg.norm(zH)
        zH = np.cross(zH, v)
        thetaz = -np.rad2deg(np.arccos(np.dot(zH, u))) + 90

        if np.dot(q, zH) > 0:
            thetaz = -thetaz
    else:
        # When rotate3d has been used, the up vector isn't set by MATLAB correctly
        # Calculate current orientation of x and y axes in view coordinates
        xax = np.dot(np.dot(Rel, Raz), np.array([1, 0, 0]))
        yax = np.dot(np.dot(Rel, Raz), np.array([0, 1, 0]))

        # Project x and y into current viewing plane
        n1 = np.cross(xax, [0, 1, 0])
        x = np.cross([0, 1, 0], n1)

        n1 = np.cross(yax, [0, 1, 0])
        y = np.cross([0, 1, 0], n1)

        thetax = np.degrees(np.arctan2(x[2], x[0]))
        thetay = np.degrees(np.arctan2(y[2], y[0]))
        if not any(x):
            thetax = 0
        if not any(y):
            thetay = 0
        thetaz = 90

    # Orientate these labels to be aligned with the axis directions.
    axes_handle.set_xlabel(axes_handle.get_xlabel(), rotation=thetax)
    axes_handle.set_ylabel(axes_handle.get_ylabel(), rotation=thetay)
    axes_handle.set_zlabel(axes_handle.get_zlabel(), rotation=thetaz)
