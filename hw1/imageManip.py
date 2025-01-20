import math


from PIL import Image
import numpy as np
from PIL import Image
from skimage import color, io
import skimage.io

def load(image_path):
    """Loads an image from a file path.

    HINT: Look up `skimage.io.imread()` function.

    Args:
        image_path: file path to the image.

    Returns:
        out: numpy array of shape(image_height, image_width, 3).
    """
    out = None

    ### YOUR CODE HERE
    out=skimage.io.imread(image_path)

    # Let's convert the image to be between the correct range.
    out = out.astype(np.float64) / 255
    return out


def crop_image(image, start_row, start_col, num_rows, num_cols):
    """Crop an image based on the specified bounds.

    Args:
        image: numpy array of shape(image_height, image_width, 3).
        start_row (int): The starting row index we want to include in our cropped image.
        start_col (int): The starting column index we want to include in our cropped image.
        num_rows (int): Number of rows in our desired cropped image.
        num_cols (int): Number of columns in our desired cropped image.

    Returns:
        out: numpy array of shape(num_rows, num_cols, 3).
    """

    out = None

    out = image[start_row:start_row+num_rows, start_col:start_col+num_cols, :]

    return out


def dim_image(image):
    """Change the value of every pixel by following

                        x_n = 0.5*x_p^2

    where x_n is the new value and x_p is the original value.

    Args:
        image: numpy array of shape(image_height, image_width, 3).

    Returns:
        out: numpy array of shape(image_height, image_width, 3).
    """

    # image = image.astype(np.float32) / 255.0
    
    out = 0.5 * (np.power(image,2))
    
    out = np.clip(out, 0, 1)
    # # Convert back to 8-bit format (values between 0 and 255)
    out = (out * 255).astype(np.uint8)
    
    return out


def resize_image(input_image, output_rows, output_cols):
    """Resize an image using the nearest neighbor method.

    Args:
        input_image (np.ndarray): RGB image stored as an array, with shape
            `(input_rows, input_cols, 3)`.
        output_rows (int): Number of rows in our desired output image.
        output_cols (int): Number of columns in our desired output image.

    Returns:
        np.ndarray: Resized image, with shape `(output_rows, output_cols, 3)`.
    """
    input_rows, input_cols, channels = input_image.shape
    assert channels == 3

    # 1. Create the resized output image
    output_image = np.zeros(shape=(output_rows, output_cols, 3))

    # 2. Populate the `output_image` array using values from `input_image`
    #    > This should require two nested for loops!
    row_scale_factor = input_rows / output_rows
    col_scale_factor = input_cols / output_cols

    for i in range(output_rows):
        for j in range(output_cols):
            input_i = int(i * row_scale_factor)
            input_j = int(j * col_scale_factor)

            output_image[i, j, :] = input_image[input_i, input_j, :]

    return output_image


def rotate2d(point, theta):
    """Rotate a 2D coordinate by some angle theta.

    Args:
        point (np.ndarray): A 1D NumPy array containing two values: an x and y coordinate.
        theta (float): An theta to rotate by, in radians.

    Returns:
        np.ndarray: A 1D NumPy array containing your rotated x and y values.
    """
    assert point.shape == (2,)
    assert isinstance(theta, float)

    x, y = point
    
    # Apply the 2D rotation formula
    x_prime = x * np.cos(theta) - y * np.sin(theta)
    y_prime = x * np.sin(theta) + y * np.cos(theta)
    
    # Return the rotated coordinates as a NumPy array
    return np.array([x_prime, y_prime])


def rotate_image(input_image, theta):
    """Rotate an image by some angle theta.

    Args:
        input_image (np.ndarray): RGB image stored as an array, with shape
            `(input_rows, input_cols, 3)`.
        theta (float): Angle to rotate our image by, in radians.

    Returns:
        (np.ndarray): Rotated image, with the same shape as the input.
    """
    input_rows, input_cols, channels = input_image.shape
    assert channels == 3

    output_image = np.zeros_like(input_image)
    center_i, center_j = input_rows // 2, input_cols // 2

    for i in range(input_rows):
        for j in range(input_cols):

            i_shifted = i - center_i
            j_shifted = j - center_j

            input_i = i_shifted * np.cos(theta) - j_shifted * np.sin(theta)
            input_j = i_shifted * np.sin(theta) + j_shifted * np.cos(theta)

            input_i = int(input_i + center_i)
            input_j = int(input_j + center_j)

            if 0 <= input_i < input_rows and 0 <= input_j < input_cols:
                output_image[i, j, :] = input_image[input_i, input_j, :]

    return output_image
