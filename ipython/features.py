import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.ndimage import filters

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
    # Convolution kernels
    horiz_kernel = np.array([[1, 0, -1]])
    vert_kernel = np.array([[1], [0], [-1]])
    
    # Compute horizontal and vertical components of gradient
    horiz = filters.convolve(channel, horiz_kernel, mode='reflect')
    vert = filters.convolve(channel, vert_kernel, mode='reflect')
    
    # Create arrays for magnitude and direction
    magnitude = np.zeros(channel.shape)
    direction = np.zeros(channel.shape)
    
    # Compute magnitude and direction for gradient at each pixel
    for i in range(channel.shape[0]):
        for j in range(channel.shape[1]):
            magnitude[i,j], direction[i,j] = cart2pol(horiz[i,j], vert[i,j])
            # Converting angles to degrees and taking absolute value
            direction[i,j] = abs(np.rad2deg(direction[i,j]))
    
    # Return computed gradient
    return (magnitude, direction)


def compute_hog(image_patch, normalize=True):
    """
    This function takes an image patch as input and returns a 9-element
    vector containing the histogram of oriented gradients.
    :param image_patch: The image patch on which to compute the HOG
    :returns: The HOG feature vector and the edges of the bins
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
    hist, bin_edges = np.histogram(max_direction, bins=9, range=(0,180),\
                                     weights=max_magnitude, density=False)
    if normalize:
        hist = hist / sum(hist)
    return (hist, bin_edges)


def plot_hog(hist, bin_edges):
    """
    This function plots the Histogram of Oriented Gradients.
    :param hist: The HOG vector
    :param bin_edges: The HOG bin edge values
    """
    plt.bar(bin_edges[:-1], hist, width=20)
    plt.title('Simple HOG')
    plt.xlabel('Direction bins')
    plt.ylabel('Frequency')
    plt.show()
