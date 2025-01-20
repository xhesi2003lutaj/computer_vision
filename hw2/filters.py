import numpy as np


def conv_nested(image, kernel):

    Hi, Wi = image.shape

    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))


    image_padded = np.zeros((Hi + Hk -1, Wi + Wk -1))
    image_padded[:-Hk//2,:-Wk//2] = image
    for m in range(Hi):
        for n in range(Wi):
            for i in range(Hk):
                for j in range(Wk):
                    out[m,n] += image_padded[m - i +1,n - j +1] * kernel[i, j]

    return out

def zero_pad(image, pad_height, pad_width):

    H, W = image.shape
    out = None

    out = np.pad(image, ((pad_height,pad_height),(pad_width,pad_width)),'constant', constant_values = 0)

    return out


def conv_fast(image, kernel):

    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))

    h = kernel
    h1 = np.flipud(h)
    h2 = np.fliplr(h1)
    Hk, Wk = np.shape(kernel) 
    Hi, Wi = np.shape(image)
    y = Hk//2
    x = Wk//2
    f = image
    f = zero_pad(f, y, x)
    out = np.copy(image)
    for i in range(y, Hi + y):
        # print("brenda")
        for j in range(x, Wi + x):
            sum1 = f[i-y:i+y+1,j-x:j+x+1]
            out[i-y,j-x] = (sum1 * h2).sum()

    return out

def conv_helper(image, kernel):

    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    h = np.delete(kernel,0,axis=0)
    x = Hk//2
    y = Wk//2
    out = np.copy(image)
    f = zero_pad(image, x, y)

    for i in range(x, Hi + x):
        for j in range(y, Wi + y):
            # print("ktu")
            sum1 = f[i-x:i+x-1,j-y:j+y+1]
            out[i-x,j-y] = (sum1 * h).sum()

    return out

def cross_correlation(f, g):

    out = None

    out = conv_helper(f,g)

    return out

def zero_mean_cross_correlation(f, g):

    out = None

    mean = np.mean(g)
    out = conv_helper(f, g - mean)
    # print("out")

    return out

def normalized_cross_correlation(f, g):
    """ Normalized cross-correlation of f and g.

    Normalize the subimage of f and the template g at each step
    before computing the weighted sum of the two.

    Hint: you should look up useful numpy functions online for calculating 
          the mean and standard deviation.

    Args:
        f: numpy array of shape (Hf, Wf).
        g: numpy array of shape (Hg, Wg).

    Returns:
        out: numpy array of shape (Hf, Wf).
    """

    out = None

    g = np.delete(g,0,axis=0)
    Hi, Wi = f.shape
    Hk, Wk = g.shape
    x = Hk//2
    y = Wk//2
    out = np.copy(f)
    temp = zero_pad(f, x, y)
    g = 1/np.std(g) * (g - np.mean(g))
    for i in range(x, Hi + x):
        for j in range(y, Wi + y):
            sum1 = temp[i-x:i+x+1,j-y:j+y+1]
            sum1 = 1/np.std(sum1) * (sum1 - np.mean(sum1))
            # print(sum1)
            out[i-x,j-y] = (sum1 * g).sum()

    return out
