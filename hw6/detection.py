import numpy as np
import skimage
from skimage import feature, data, color, exposure, io
from skimage.transform import rescale, resize, downscale_local_mean
from skimage.filters import gaussian
from scipy import signal
from scipy.ndimage import interpolation
import math
from skimage.feature import hog

def hog_feature(image, pixel_per_cell=8):
    """
    Compute hog feature for a given image.

    Important:
    - Use the hog function provided by skimage to generate both the
      feature vector and the visualization image.
    - For the block normalization parameter, use L1!

    Args:
        image: an image with object that we want to detect.
        pixel_per_cell: number of pixels in each cell, an argument for hog descriptor.

    Returns:
        hog_feature: a vector of hog representation.
        hog_image: an image representation of hog provided by skimage.
    """
    ### YOUR CODE HERE
    hog_feature, hog_image = hog(
        image, 
        pixels_per_cell=(pixel_per_cell, pixel_per_cell),
        block_norm='L1',
        visualize=True
    )
    print("hog feature ",hog_feature)
    print("hog image ",hog_image)

    return (hog_feature, hog_image)

def sliding_window(image, template_feature, step_size, window_size, pixel_per_cell=8,
                   return_unresized_response=False):
    """
    A sliding window that checks each different location in the image
    and finds which location has the highest HOG score.

    Args:
        image: np array of size (H, W). The image to apply sliding window.
        template_feature: an array of size (m,).
            The HOG representation of the object you want to find.
        step_size: int of the step size to move the window.
        window_size: pair of ints (height, width) of the window.
        return_unresized_response: bool to indicate whether to return unresized response map.

    Returns:
        max_score: float of the highest HOG score.
        maxr: int of the row where max_score is found (top-left of window).
        maxc: int of the column where max_score is found (top-left of window).
        response_map: np array of size (ceil H / step_size, ceil W / step_size).
        response_map_resized: np array of size (H, W).
    """
    winH, winW = window_size
    H, W = image.shape

    pad_image = np.pad(
        image,
        ((winH // 2, winH - winH // 2),
         (winW // 2, winW - winW // 2)),
        mode='constant'
    )

    max_score = -float('inf')  #negative infinity for comparison
    maxr, maxc = 0, 0
    response_map = np.zeros((math.ceil(H / step_size), math.ceil(W / step_size)))

    for r in range(0, H, step_size):
        for c in range(0, W, step_size):
            window = pad_image[r:r + winH, c:c + winW]

            hog_feature, _ = hog(
                window,
                pixels_per_cell=(pixel_per_cell, pixel_per_cell),
                block_norm='L1',
                visualize=True
            )
            # print("mbrena")

            score = np.dot(hog_feature, template_feature)

            response_map[r // step_size, c // step_size] = score

            if score > max_score:
                max_score = score
                maxr, maxc = r, c

    # resizing map to the same size as the input image
    response_map_resized = resize(response_map, (H, W), mode='constant', anti_aliasing=True)

    if return_unresized_response:
        return max_score, maxr, maxc, response_map_resized, response_map
    else:
        return max_score, maxr, maxc, response_map_resized


def pyramid(image, scale=0.9, min_size=(200, 100)):
    """
    Generate image pyramid using the given image and scale.
    Reducing the size of the image until either the height or
    width is below the minimum limit. In the ith iteration,
    the image is resized to scale^i of the original image.

    This function is mostly completed for you -- only a termination
    condition is needed.

    Args:
        image: np array of (h,w), an image to scale.
        scale: float of how much to rescale the image each time.
        min_size: pair of ints showing the minimum height and width.

    Returns:
        images: list containing pair of
            (the current scale of the image, resized image).
    """
    images = []

    # Yield the original image
    current_scale = 1.0
    images.append((current_scale, image))

    while True:
        # Use "break" to exit this loop when termination conditions are met.

        # Compute the new dimensions of the image and resize it
        current_scale *= scale
        # image = rescale(image, scale, mode='constant')
        image = rescale(image, scale, mode='constant', anti_aliasing=True, channel_axis=None)
        if image.shape[0] < min_size[0] or image.shape[1] < min_size[1]:
            break
        # Yield the next image in the pyramid
        images.append((current_scale, image))

    return images


def pyramid_score(image, template_feature, shape, step_size=20,
                  scale=0.9, pixel_per_cell=8):
    """
    Calculate the maximum score found in the image pyramid using sliding window.

    Args:
        image: np array of (h,w).
        template_feature: the hog representation of the object you want to detect.
        shape: shape of window you want to use for the sliding_window.

    Returns:
        max_score: float of the highest hog score.
        maxr: int of row where the max_score is found.
        maxc: int of column where the max_score is found.
        max_scale: float of scale when the max_score is found.
        max_response_map: np array of the response map when max_score is found.
    """
    max_score = 0
    maxr = 0
    maxc = 0
    max_scale = 1.0
    max_response_map = np.zeros(image.shape)

    images = pyramid(image, scale)
    for s, scaled_image in images:

        score, r, c, response_map = sliding_window(
            scaled_image, template_feature, step_size, shape, pixel_per_cell
        )

        if score > max_score:
            max_score = score
            maxr = r
            maxc = c
            max_scale = s
            max_response_map = response_map

    return max_score, maxr, maxc, max_scale, max_response_map


def compute_displacement(part_centers, face_shape):
    """
    Calculate the mu and sigma for each part. d is the array where each row
    is the main center (face center) minus the part center. Since in our
    dataset, the face is the full image, face center could be computed by
    finding the center of the image. Vector mu is computed by taking an average
    from the rows of d. And sigma is the standard deviation among the rows.

    Hint: d is the array of distances between part centers and the face center,
    such that d[i,0] is the face's center-x coordinate minus the ith part's x
    coordinate, and similarly for y coordinates.

    Hint: you should leave mu as floats instead of rounding to integers,
    because our next step of applying the shift in shift_heatmap will
    interpolate the shift, which is valid for float shifts.

    Args:
        part_centers: np array of (n,2) containing centers
            of one part in each image.
        face_shape: np array of (h,w) that indicates the shape of a face.
    Returns:
        mu: (2,) vector.
        sigma: (2,) vector.

    """
    d = np.zeros((part_centers.shape[0], 2))

    d = np.array([face_shape[0] / 2, face_shape[1] / 2]) - part_centers

    mu = d.mean(axis=0)
    sigma = d.std(axis=0)

    return mu, sigma


def shift_heatmap(heatmap, mu):
    """
    First normalize the heatmap to make sure that all the values
    are not larger than 1. This kind of normalization can be achieved via
    dividing by the maximum value of the heatmap.

    Then shift the heatmap based on the vector mu.

    Hint: use the interpolation.shift function provided by scipy.ndimage.

    Note: the arguments are copied to ensure your code doesn't alter them.
    The copies are the same type and shape as the originals.

    Args:
        heatmap: np array of (h,w).
        mu: vector array of (1,2).
    Returns:
        new_heatmap: np array of (h,w).
    """
    heatmap = np.copy(heatmap)
    heatmap = heatmap / np.max(heatmap)
    new_heatmap = interpolation.shift(heatmap, mu)

    return new_heatmap


def gaussian_heatmap(heatmap_face, heatmaps, sigmas):
    """
    Apply gaussian filter with the given sigmas to the corresponding heatmaps.
    Then add the filtered heatmaps together with the face heatmap.
    Find the index where the maximum value in the heatmap is found.

    Hint: use gaussian function provided by skimage.

    Note: the arguments are copied to ensure your code doesn't alter them.
    The copies are the same type and shape as the originals.

    Args:
        heatmap_face: np array of (h,w), corresponding to the face heatmap.
        heatmaps: list of [np array of (h,w)], corresponding to the parts heatmaps.
        sigmas: list of [np array of (2,)], corresponding to the parts sigmas.
    Return:
        heatmap: np array of (h,w), corresponding to sum of gaussian-filtered heatmaps.
        maxr: int of row where the heatmap maximum is found.
        maxc: int of column where the heatmap maximum is found.
    """
    heatmap_face = np.copy(heatmap_face)
    heatmaps = list(np.copy(heatmaps))
    sigmas = list(np.copy(sigmas))

    #initializing final heatmap with the face heatmap
    heatmap = heatmap_face

    for h, sigma in zip(heatmaps, sigmas):
        filtered_heatmap = gaussian(h, sigma=sigma, mode='constant')
        heatmap += filtered_heatmap

    maxr, maxc = np.unravel_index(np.argmax(heatmap), heatmap.shape)

    return heatmap, maxr, maxc


def detect_multiple(image, response_map):
    """
    Extra credit
    """
    detected_faces = None
    ### YOUR CODE HERE
    pass
    ### END YOUR CODE
    return detected_faces

