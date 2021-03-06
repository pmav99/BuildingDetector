import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.ndimage import filters
import skimage.color
import skimage.feature


def calc_color_hist(image, bins=16, norm=True):
    """
    This function calculates and returns the color histogram feature
    vector for a given image.
    :param image: The image to calculate the color histogram for
    :param bins: The number of bins per color channel
    :param norm: Boolean, normalizes counts to sum to 1 (within channel)
    :returns: The color histogram feature vector (length 3*bins)
    """
    hist = np.zeros((3*bins,1))
    # Fill feature vector by channel
    color = ('b', 'g', 'r')
    for i,col in enumerate(color):
        channelHist = cv2.calcHist(images=[image], channels=[i], mask=None,\
                                   histSize=[bins], ranges=[0,256])
        # Normalize histogram
        if norm:
            channelHist = channelHist / channelHist.sum()
        hist[bins*i:bins*(i+1)] = channelHist
    return hist


def plot_color_hist(image, bins=16, norm=True):
    """
    This function computes the color histogram feature vector and plots
    the color histogram with each channel (BGR).
    :param image: The image to plot the color histogram for
    :param bins: The number of bins per color channel
    :param norm: Normalizes counts to sum to 1
    """
    # Compute histogram feature vector
    hist = calc_color_hist(image, bins, norm)
    # Plotting histogram
    color = ('b', 'g', 'r')
    for i,col in enumerate(color):
        plt.bar(range(bins*i,bins*(i+1)), hist[bins*i:bins*(i+1)],\
                color=col)
    plt.xlim([0,3*bins])
    frame = plt.gca()
    frame.axes.get_xaxis().set_visible(False)
    plt.title('Normalized histograms for each color channel (BGR)')
    plt.show()


def compute_hog(image_patch):
    """
    This function takes an image patch as input and returns a 576-element HOG
    feature vector.
    :param image_patch: Input image patch
    :returns: HOG feature vector
    """
    gray_patch = skimage.color.rgb2gray(image_patch)
    hog = skimage.feature.hog(gray_patch, orientations=9, pixels_per_cell=(20,20),
        cells_per_block=(2,2), visualise=False)
    return hog


def plot_hist(hist, bin_edges, title='Histogram'):
    """
    This function plots a histogram.
    :param hist: The histogram vector
    :param bin_edges: The histogram bin edge values
    """
    plt.bar(bin_edges[:-1], hist, width=(bin_edges[1] - bin_edges[0]))
    plt.title(title)
    plt.xlabel('Bins')
    plt.ylabel('Frequency')
    plt.show()


def compute_features(patch):
    """
    This function computes features for an image patch.
    :param patch: 80x80 image patch
    :returns: Feature vector
    """
    color = calc_color_hist(patch)
    color = color.flatten()
    hog = compute_hog(patch)
    features = np.concatenate((color, hog), axis=0)
    return features





'''
DEPRECATED CODE

def compute_hog(image_patch, normalize=True):
    """
    This function takes an image patch as input and returns a 9-element
    vector containing the histogram of oriented gradients and a 16-element
    vector containing the histogram of gradient magnitudes.
    :param image_patch: The image patch on which to compute the HOG
    :returns: The HOG feature vector, the edges of the bins, and the max magnitude array
    """
    # Converting to signed integer
    image_patch = image_patch.astype(np.int_)
    height = image_patch.shape[0]
    width = image_patch.shape[1]
    
    # Compute gradients for each channel
    if len(image_patch.shape) > 2:
        channels = image_patch.shape[2]
        magnitude = np.zeros((height, width, channels))
        direction = np.zeros((height, width, channels))
        for i in range(channels):
            magnitude[:,:,i], direction[:,:,i] = compute_gradient(image_patch[:,:,i])
        # Select maximum gradient from all channels at each pixel
        max_index = np.argmax(magnitude, axis=2)
        max_magnitude = np.zeros((height, width))
        max_direction = np.zeros((height, width))
        for i in range(height):
            for j in range(width):
                max_magnitude[i,j] = magnitude[i,j,max_index[i,j]]
                max_direction[i,j] = direction[i,j,max_index[i,j]]
    
    # If only one channel
    else:
        max_magnitude, max_direction = compute_gradient(image_patch)
    
    # Computing simple histogram of gradients
    hog, hog_bins = np.histogram(max_direction, bins=9, range=(0,180),\
                                     weights=max_magnitude, density=False)
    if normalize:
        hog = hog.astype(float) / sum(hog)

    # Computing histogram of gradient magnitudes
    magnitude_hist, magnitude_bins = np.histogram(max_magnitude, bins=16, range=(0,256))
    if normalize:
        magnitude_hist = magnitude_hist.astype(float) / sum(magnitude_hist)

    # Replace nan values with 0
    hog = np.nan_to_num(hog)
    magnitude_hist = np.nan_to_num(magnitude_hist)

    return (hog, hog_bins, magnitude_hist, magnitude_bins, max_magnitude)


def compute_advanced_hog(image_patch, normalize=True, divisions=4):
    """
    This function takes an image patch as input and returns full-patch
    and partial-patch HOG and gradient magnitude vectors.
    :param image_patch: The image patch on which to compute the features
    :param normalize: Normalizes the histograms before returning them
    :param divisions: The number of divisions to make in each dimension
    :returns: Full HOG and gradient magnitude vectors
    """
    # Getting size of image patch
    height = image_patch.shape[0]
    width = image_patch.shape[1]

    # Compute full-patch HOG and gradient magnitudes
    (hog, hog_bins, magnitude_hist, magnitude_bins, max_magnitude) = \
                    compute_hog(image_patch)

    # Split patch into equal size patches, add features from each
    h_patch = height / divisions
    w_patch = width / divisions
    for i in range(divisions):  
        for j in range(divisions):
            patch_ij = image_patch[i*h_patch:(i+1)*h_patch,
                                   j*w_patch:(j+1)*w_patch,:]
            # Compute sub-patch features
            (hog_ij, hog_bins_ij, magnitude_hist_ij, magnitude_bins_ij,
             max_magnitude_ij) = compute_hog(patch_ij)
            # Concatenate with full-patch features
            hog = np.concatenate((hog, hog_ij), axis=1)
            magnitude_hist = np.concatenate((magnitude_hist,
                                            magnitude_hist_ij), axis=1)

    # Replace nan values with 0
    hog = np.nan_to_num(hog)
    magnitude_hist = np.nan_to_num(magnitude_hist)
    
    return (hog, magnitude_hist)


def cart2pol(x, y):
    """
    This function converts from cartesian to polar coordinates.
    :param x: Cartesian x coordinate
    :param y: Cartesian y coordinate
    :returns: Magnitude and direction as a 2-tuple
    """
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return (rho, phi)


def compute_gradient(channel):
    """
    This function computes the gradient magnitudes and directions for one
    channel of an image patch.
    :param channel: One channel of an image patch
    :returns: Magnitude and direction arrays of the same dimensions as the
    input channel of the image patch
    """
    # Convolution kernels, directions will start from 0 at x-axis going counterclockwise (standard)
    horiz_kernel = np.array([[1, 0, -1]])
    vert_kernel = np.array([[-1], [0], [1]])
    
    # Compute horizontal and vertical components of gradient
    horiz = filters.convolve(channel, horiz_kernel, mode='nearest')
    vert = filters.convolve(channel, vert_kernel, mode='nearest')
    
    # Create arrays for magnitude and direction
    magnitude = np.zeros(channel.shape)
    direction = np.zeros(channel.shape)
    
    # Compute magnitude and direction for gradient at each pixel
    for i in range(channel.shape[0]):
        for j in range(channel.shape[1]):
            magnitude[i,j], direction[i,j] = cart2pol(horiz[i,j], vert[i,j])
            # Converting angles to degrees and taking modulo 180
            direction[i,j] = np.rad2deg(direction[i,j]) % 180
    
    # Return computed gradient
    return (magnitude, direction)
'''