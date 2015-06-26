import numpy as np
import cv2
import matplotlib.pyplot as plt

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
