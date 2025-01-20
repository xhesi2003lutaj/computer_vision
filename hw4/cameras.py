from typing import Tuple

import numpy as np


def camera_from_world_transform(d: float = 1.0) -> np.ndarray:
    """Define a transformation matrix in homogeneous coordinates that
    transforms coordinates from world space to camera space, according
    to the coordinate systems in Question 1.


    Args:
        d (float, optional): Total distance of displacement between world and camera
            origins. Will always be greater than or equal to zero. Defaults to 1.0.

    Returns:
        T (np.ndarray): Left-hand transformation matrix, such that c = Tw
            for world coordinate w and camera coordinate c as column vectors.
            Shape = (4,4) where 4 means 3D+1 for homogeneous.
    """

    R = np.array([
        [1/np.sqrt(2), 0, -1/np.sqrt(2)],
        [0, 1, 0],
        [-1/np.sqrt(2), 0, -1/np.sqrt(2)]
    ])

    t = np.array([0, 0, 1])

    
    T = np.eye(4)  
    T[:3, :3] = R  
    T[:3, 3] = t   

    assert T.shape == (4, 4)
    return T


def apply_transform(T: np.ndarray, points: np.ndarray) -> Tuple[np.ndarray]:
    """Apply a transformation matrix to a set of points.

    Hint: You'll want to first convert all of the points to homogeneous coordinates.
    Each point in the (3,N) shape edges is a length 3 vector for x, y, and z, so
    appending a 1 after z to each point will make this homogeneous coordinates.

    You shouldn't need any loops for this function.

    Args:
        T (np.ndarray):
            Left-hand transformation matrix, such that c = Tw
                for world coordinate w and camera coordinate c as column vectors.
            Shape = (4,4) where 4 means 3D+1 for homogeneous.
        points (np.ndarray):
            Shape = (3,N) where 3 means 3D and N is the number of points to transform.

    Returns:
        points_transformed (np.ndarray):
            Transformed points.
            Shape = (3,N) where 3 means 3D and N is the number of points.
    """
    N = points.shape[1]
    assert points.shape == (3, N)

    # You'll replace this!
    points_transformed = np.zeros((3, N))

    # YOUR CODE HERE
    points_homogeneous = np.vstack([points, np.ones(points.shape[1])])  
    
    transformed_homogeneous = T @ points_homogeneous 
    
    points_transformed = transformed_homogeneous[:3] 

    assert points_transformed.shape == (3, N)
    return points_transformed


def intersection_from_lines(
    a_0: np.ndarray, a_1: np.ndarray, b_0: np.ndarray, b_1: np.ndarray
) -> np.ndarray:
    """Find the intersection of two lines (infinite length), each defined by a
    pair of points.

    Args:
        a_0 (np.ndarray): First point of first line; shape `(2,)`.
        a_1 (np.ndarray): Second point of first line; shape `(2,)`.
        b_0 (np.ndarray): First point of second line; shape `(2,)`.
        b_1 (np.ndarray): Second point of second line; shape `(2,)`.

    Returns:
        np.ndarray: the intersection of the two lines definied by (a0, a1)
                    and (b0, b1).
    """
    # Validate inputs
    assert a_0.shape == a_1.shape == b_0.shape == b_1.shape == (2,)
    assert a_0.dtype == a_1.dtype == b_0.dtype == b_1.dtype == float

 
    A1 = a_1[1] - a_0[1]  
    B1 = a_0[0] - a_1[0]  
    C1 = A1 * a_0[0] + B1 * a_0[1]

    A2 = b_1[1] - b_0[1]  
    B2 = b_0[0] - b_1[0]  
    C2 = A2 * b_0[0] + B2 * b_0[1]

    det = A1 * B2 - A2 * B1

    if abs(det) < 1e-10:
        raise ValueError("The lines are parallel and do not intersect.")

    x = (B2 * C1 - B1 * C2) / det
    y = (A1 * C2 - A2 * C1) / det
    intersection = np.array([x, y])

    assert intersection.shape == (2,)
    assert intersection.dtype == np.float64
    print("intersection ",intersection)
    return intersection
 


def optical_center_from_vanishing_points(
    v0: np.ndarray, v1: np.ndarray, v2: np.ndarray
) -> np.ndarray:
    """Compute the optical center of our camera intrinsics from three vanishing
    points corresponding to mutually orthogonal directions.

    Hints:
    - Your `intersection_from_lines()` implementation might be helpful here.
    - It might be worth reviewing vector projection with dot products.

    Args:
        v0 (np.ndarray): Vanishing point in image space; shape `(2,)`.
        v1 (np.ndarray): Vanishing point in image space; shape `(2,)`.
        v2 (np.ndarray): Vanishing point in image space; shape `(2,)`.

    Returns:
        np.ndarray: Optical center; shape `(2,)`.
    """
    assert v0.shape == v1.shape == v2.shape == (2,), "Vanishing points must be 2D vectors."
    

    v0_h = np.append(v0, 1)  
    v1_h = np.append(v1, 1)
    v2_h = np.append(v2, 1)

    line_v0_v1 = np.cross(v0_h, v1_h)
    line_v1_v2 = np.cross(v1_h, v2_h)
    line_v2_v0 = np.cross(v2_h, v0_h)

    optical_center_h = np.cross(line_v0_v1, line_v1_v2)
    optical_center_h = optical_center_h / optical_center_h[2]  

    optical_center = optical_center_h[:2]

    assert optical_center.shape == (2,)
    return optical_center


def focal_length_from_two_vanishing_points(
    v0: np.ndarray, v1: np.ndarray, optical_center: np.ndarray
) -> np.ndarray:
    """Compute focal length of camera, from two vanishing points and the
    calibrated optical center.

    Args:
        v0 (np.ndarray): Vanishing point in image space; shape `(2,)`.
        v1 (np.ndarray): Vanishing point in image space; shape `(2,)`.
        optical_center (np.ndarray): Calibrated optical center; shape `(2,)`.

    Returns:
        float: Calibrated focal length.
    """

    assert v0.shape == v1.shape == optical_center.shape == (2,), "Wrong shape!"

    c_x, c_y = optical_center
    v0_x, v0_y = v0
    v1_x, v1_y = v1

    dx0, dy0 = v0_x - c_x, v0_y - c_y
    dx1, dy1 = v1_x - c_x, v1_y - c_y

    lhs = dx0 * dx1 + dy0 * dy1

    denominator = dx0 * dy1 - dy0 * dx1

    if denominator == 0:
        raise ValueError("Vanishing points are not orthogonal or too close.")

    f_squared = lhs / denominator

    if f_squared < 0:
        raise ValueError("Computed focal length is negative, check your vanishing points.")

    f = np.sqrt(f_squared)

    return f


def physical_focal_length_from_calibration(
    f: float, sensor_diagonal_mm: float, image_diagonal_pixels: float
) -> float:
    """Compute the physical focal length of our camera, in millimeters.

    Args:
        f (float): Calibrated focal length, using pixel units.
        sensor_diagonal_mm (float): Length across the diagonal of our camera
            sensor, in millimeters.
        image_diagonal_pixels (float): Length across the diagonal of the
            calibration image, in pixels.

    Returns:
        float: Calibrated focal length, in millimeters.
    """

    print(f"focal length in pixels: {f}")
    print(f"sensor diagonal (mm): {sensor_diagonal_mm}")
    print(f"image diagonal (pixels): {image_diagonal_pixels}")
    
    if image_diagonal_pixels == 0:
        raise ValueError("Image diagonal in pixels cannot be zero.")
    
    f_mm = f * (sensor_diagonal_mm / image_diagonal_pixels)
    
    print(f"Computed physical focal length (mm): {f_mm}")

    return f

