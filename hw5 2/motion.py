import numpy as np
from skimage.transform import pyramid_gaussian
from skimage.filters import sobel_h, sobel_v, gaussian
from skimage.feature import corner_harris, corner_peaks

def lucas_kanade(img1, img2, keypoints, window_size=5):
    """ Estimate flow vector at each keypoint using Lucas-Kanade method.

    Args:
        img1 - Grayscale image of the current frame. Flow vectors are computed
            with respect to this frame.
        img2 - Grayscale image of the next frame.
        keypoints - Keypoints to track. Numpy array of shape (N, 2).
        window_size - Window size to determine the neighborhood of each keypoint.
            A window is centered around the current keypoint location.
            You may assume that window_size is always an odd number.
    Returns:
        flow_vectors - Estimated flow vectors for keypoints. flow_vectors[i] is
            the flow vector for keypoint[i]. Numpy array of shape (N, 2).

    Hints:
        - You may use np.linalg.inv to compute inverse matrix
    """
    assert window_size % 2 == 1, "window_size must be an odd number"

    flow_vectors = []
    w = window_size // 2

    # Compute partial derivatives
    Iy, Ix = np.gradient(img1)
    It = img2 - img1

    # For each [y, x] in keypoints, estimate flow vector [vy, vx]
    # using Lucas-Kanade method and append it to flow_vectors.
    for y, x in keypoints:
        # Keypoints can be loacated between integer pixels (subpixel locations).
        # For simplicity, we round the keypoint coordinates to nearest integer.
        # In order to achieve more accurate results, image brightness at subpixel
        # locations can be computed using bilinear interpolation.
        y, x = int(round(y)), int(round(x))
        
        # a window around the keypoint
        x_min = max(x - w, 0)
        x_max = min(x + w + 1, img1.shape[1])
        y_min = max(y - w, 0)
        y_max = min(y + w + 1, img1.shape[0])

        # the gradient values within the window
        Ix_window = Ix[y_min:y_max, x_min:x_max]
        Iy_window = Iy[y_min:y_max, x_min:x_max]
        It_window = It[y_min:y_max, x_min:x_max]

        # Stack the gradients to form the A matrix
        A = np.vstack((Ix_window.flatten(), Iy_window.flatten())).T
        b = -It_window.flatten()

        try:
            v, resids, rank, s = np.linalg.lstsq(A, b, rcond=None)
            flow_vectors.append(v)
        except np.linalg.LinAlgError:
            flow_vectors.append([0, 0])

    return np.array(flow_vectors)

def iterative_lucas_kanade(img1, img2, keypoints, window_size=9, num_iters=7, g=None):
    assert window_size % 2 == 1, "window_size must be an odd number"

    if g is None:
        g = np.zeros(keypoints.shape)

    flow_vectors = []
    w = window_size // 2

    Iy, Ix = np.gradient(img1)

    for (y, x), (gy, gx) in zip(keypoints, g):
        v = np.zeros(2)
        y1 = int(round(y))
        x1 = int(round(x))

        if not (w <= y1 < img1.shape[0] - w and w <= x1 < img1.shape[1] - w):
            flow_vectors.append(np.zeros(2))  
            continue

        G = np.zeros((2, 2))
        for i in range(-w, w + 1):
            for j in range(-w, w + 1):
                if 0 <= y1 + i < Ix.shape[0] and 0 <= x1 + j < Ix.shape[1]:
                    ix = Ix[y1 + i, x1 + j]
                    iy = Iy[y1 + i, x1 + j]
                    G[0, 0] += ix * ix
                    G[0, 1] += ix * iy
                    G[1, 0] += iy * ix
                    G[1, 1] += iy * iy

        if np.linalg.cond(G) > 1e10:  
            flow_vectors.append(np.zeros(2))
            continue

        G_inv = np.linalg.pinv(G)  

        for _ in range(num_iters):
            vx, vy = v
            y2 = int(round(y + gy + vy))
            x2 = int(round(x + gx + vx))

            if not (0 <= y2 < img1.shape[0] and 0 <= x2 < img1.shape[1]):
                break

            b_k = np.zeros(2)
            for i in range(-w, w + 1):
                for j in range(-w, w + 1):
                    if 0 <= y1 + i < Ix.shape[0] and 0 <= x1 + j < Ix.shape[1]:
                        ix = Ix[y1 + i, x1 + j]
                        iy = Iy[y1 + i, x1 + j]
                        delta_I = img1[y1 + i, x1 + j] - img2[y1 + i, x1 + j]
                        b_k[0] += delta_I * ix
                        b_k[1] += delta_I * iy

            v_k = np.dot(G_inv, b_k)
            v += v_k

        flow_vectors.append([v[1], v[0]])

    return np.array(flow_vectors)
        

def pyramid_lucas_kanade(img1, img2, keypoints,
                         window_size=9, num_iters=7,
                         level=2, scale=2):

    """ Pyramidal Lucas Kanade method

    Args:
        img1 - same as lucas_kanade
        img2 - same as lucas_kanade
        keypoints - same as lucas_kanade
        window_size - same as lucas_kanade
        num_iters - number of iterations to run iterative LK method
        level - Max level in image pyramid. Original image is at level 0 of
            the pyramid.
        scale - scaling factor of image pyramid.

    Returns:
        d - final flow vectors
    """

    # Build image pyramids of img1 and img2
    pyramid1 = tuple(pyramid_gaussian(img1, max_layer=level, downscale=scale))
    pyramid2 = tuple(pyramid_gaussian(img2, max_layer=level, downscale=scale))

    # Initialize pyramidal guess
    g = np.zeros(keypoints.shape)

    for L in range(level, -1, -1):
        keypoints_scaled = keypoints / (scale ** L)

        d_L = iterative_lucas_kanade(pyramid1[L], pyramid2[L],
                                     keypoints_scaled + g, 
                                     window_size=window_size, 
                                     num_iters=num_iters)
        
        # Update guess for next level
        g = scale * (g + d_L)

        d = g
        ### YOUR CODE HERE
        pass
        ### END YOUR CODE

    return d

def compute_error(patch1, patch2):
    """ Compute MSE between patch1 and patch2

        - Normalize patch1 and patch2
        - Compute mean square error between patch1 and patch2

    Args:
        patch1 - Grayscale image patch of shape (patch_size, patch_size)
        patch2 - Grayscale image patch of shape (patch_size, patch_size)
    Returns:
        error - Number representing mismatch between patch1 and patch2
    """
    assert patch1.shape == patch2.shape, 'Differnt patch shapes'
    error = 0
    patch1 = (patch1 - np.mean(patch1)) / np.std(patch1)
    
    patch2 = (patch2 - np.mean(patch2)) / np.std(patch2)
    
    error = np.mean((patch1 - patch2) ** 2)
    return error

def track_features(frames, keypoints,
                   error_thresh=1.5,
                   optflow_fn=pyramid_lucas_kanade,
                   exclude_border=5,
                   **kwargs):

    """ Track keypoints over multiple frames

    Args:
        frames - List of grayscale images with the same shape.
        keypoints - Keypoints in frames[0] to start tracking. Numpy array of
            shape (N, 2).
        error_thresh - Threshold to determine lost tracks.
        optflow_fn(img1, img2, keypoints, **kwargs) - Optical flow function.
        kwargs - keyword arguments for optflow_fn.

    Returns:
        trajs - A list containing tracked keypoints in each frame. trajs[i]
            is a numpy array of keypoints in frames[i]. The shape of trajs[i]
            is (Ni, 2), where Ni is number of tracked points in frames[i].
    """

    kp_curr = keypoints
    trajs = [kp_curr]
    patch_size = 3 
    w = patch_size // 2 

    for i in range(len(frames) - 1):
        I = frames[i]
        J = frames[i+1]
        flow_vectors = optflow_fn(I, J, kp_curr, **kwargs)
        kp_next = kp_curr + flow_vectors

        new_keypoints = []
        for yi, xi, yj, xj in np.hstack((kp_curr, kp_next)):

            yi = int(round(yi)); xi = int(round(xi))
            yj = int(round(yj)); xj = int(round(xj))
            # point falls outside the image
            if yj > J.shape[0]-exclude_border-1 or yj < exclude_border or\
               xj > J.shape[1]-exclude_border-1 or xj < exclude_border:
                continue

            patchI = I[yi-w:yi+w+1, xi-w:xi+w+1]
            patchJ = J[yj-w:yj+w+1, xj-w:xj+w+1]
            error = compute_error(patchI, patchJ)
            if error > error_thresh:
                continue

            new_keypoints.append([yj, xj])

        kp_curr = np.array(new_keypoints)
        trajs.append(kp_curr)

    return trajs


def IoU(bbox1, bbox2):
    """ Compute IoU of two bounding boxes

    Args:
        bbox1 - 4-tuple (x, y, w, h) where (x, y) is the top left corner of
            the bounding box, and (w, h) are width and height of the box.
        bbox2 - 4-tuple (x, y, w, h) where (x, y) is the top left corner of
            the bounding box, and (w, h) are width and height of the box.
    Returns:
        score - IoU score
    """
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2
    score = 0
    x_left = max(x1, x2)
    y_top = max(y1, y2)
    x_right = min(x1 + w1, x2 + w2)
    y_bottom = min(y1 + h1, y2 + h2)

    if x_right <= x_left or y_bottom <= y_top:
        return 0  

    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    bbox1_area = w1 * h1
    bbox2_area = w2 * h2

    union_area = bbox1_area + bbox2_area - intersection_area

    # Compute the IoU score
    score = intersection_area / union_area

    ### YOUR CODE HERE
    pass
    ### END YOUR CODE

    return score


