import numpy as np
import cv2
from sklearn import neighbors
from PIL import Image
from skimage.measure import label
import glob
import time
import os
from utils import *
from features import *

def find_markers(image_fn, template_fn, thresh=0.05):
    """
    This function performs template matching on an image with a template
    and thresholds the result, returning the labeled markers and the
    total number of markers.
    :param image_fn: Path of input image
    :param template_fn: Path of template to match
    :param thresh: Threshold value for template matching
    :returns: Labeled markers and number of markers
    """
    # Match template and threshold result
    matched = match_template(image_fn, template_fn)
    threshed = np.copy(matched)
    threshed[np.where(matched > thresh)] = 0
    threshed[np.where(matched <= thresh)] = 1
    
    # Marker labeling
    markers, num_markers = label(threshed, return_num=True)
    return (markers, num_markers)


def find_centroids(markers, num_markers):
    """
    This function finds the centroid for each roof candidate.
    :param markers: Array of pixels corresponding to labeled candidates
    :param num_markers: Number of candidates (+1 for background)
    :returns: Candidate centroids
    """
    centroids = np.zeros((num_markers-1, 2))
    for i in range(num_markers-1):
        location = np.where(markers==(i + 1))
        centroids[i,:] = np.array([location[0].mean(),
                                   location[1].mean()]).round()
    return centroids


def get_training_data(image_data_fn, use_color=True, use_hog=True,
                      use_mag=True):
    """
    This function gets the right training data for the classifier that
    we want to use.
    :param image_data_fn: Path of csv containing training data
    :param use_color: Use color histogram for classification
    :param use_hog: Use histogram of oriented gradients
    :param use_mag: Use gradient magnitudes
    :returns: Training data, training labels, and a label encoder
    """
    # Load training data
    (features, colors, hogs, mags, train_labels, label_encoder) = \
                import_image_data(image_data_fn)
    
    # Getting correct training data
    if use_color and use_hog and use_mag:
        train_data = features
    elif use_color and use_hog:
        train_data = np.concatenate((colors, hogs), axis=1)
    elif use_color and use_mag:
        train_data = np.concatenate((colors, mags), axis=1)
    elif use_hog and use_mag:
        train_data = np.concatenate((hogs, mags), axis=1)
    elif use_color:
        train_data = colors
    elif use_hog:
        train_data = hogs
    elif use_mag:
        train_data = mags
    else:
        print 'Error: No features were specified.'
    
    return (train_data, train_labels, label_encoder)


def get_color_vector(image, centroid, color_rad):
    """
    This functions gets the color histogram vector for a candidate in an
    image.
    :param image: Input image
    :param centroid: Centroid of candidate
    :param color_rad: Radius for computing color histogram
    :returns: Color histogram vector (1x48)
    """
    row = centroid[0]
    col = centroid[1]
    # Get candidate patch
    x_min = max(0, (row - color_rad))
    x_max = min((image.shape[0] - 1), (row + color_rad + 1))
    y_min = max(0, (col - color_rad))
    y_max = min((image.shape[1] - 1), (col + color_rad + 1))
    patch = image[x_min:x_max, y_min:y_max, :]
    # Return color histogram vector
    return calc_color_hist(patch).flatten()


def get_hog_vector(image, centroid, hog_rad):
    """
    This functions gets the color histogram vector for a candidate in an
    image.
    :param image: Input image
    :param centroid: Centroid of candidate
    :param hog_rad: Radius for computing hog features
    :returns: HOG vector (1x9) and gradient magnitude vector (1x16)
    """
    row = centroid[0]
    col = centroid[1]
    # Get candidate patch
    x_min = max(0, (row - hog_rad))
    x_max = min((image.shape[0] - 1), (row + hog_rad + 1))
    y_min = max(0, (col - hog_rad))
    y_max = min((image.shape[1] - 1), (col + hog_rad + 1))
    patch = image[x_min:x_max, y_min:y_max, :]
    # Return color histogram vector
    (hog, hog_bins, magnitude_hist, magnitude_bins, max_magnitude) = \
                compute_hog(patch)
    return (hog, magnitude_hist)


def get_test_data(image_fn, centroids, use_color=True, use_hog=True,
                  use_mag=True, color_rad=15, hog_rad=30):
    """
    This function gets the right candidate features for the classifier
    that we want to use.
    :param image_fn: Path of input image
    :param centroids: Candidate centroids
    :param use_color: Use color histogram for classification
    :param use_hog: Use histogram of oriented gradients
    :param use_mag: Use gradient magnitudes
    :param color_rad: Radius for computing color histogram
    :param hog_rad: Radius for computing hog features
    :returns: Test data (candidate feature vectors)
    """
    # Load input image
    image = cv2.imread(image_fn)
    
    # Number of candidates
    n = centroids.shape[0]
    
    # Feature vector length
    bins_color = 48
    bins_hog = 9
    bins_mag = 16
    
    # Getting correct training data
    if use_color and use_hog and use_mag:
        test_data = np.zeros((n, bins_color + bins_hog + bins_mag))
        for i in range(n):
            color_data = get_color_vector(image, centroids[i,:],
                                        color_rad)
            hog_data, mag_data = get_hog_vector(image, centroids[i,:],
                                               hog_rad)
            test_data[i,:] = np.concatenate((color_data, hog_data,
                                            mag_data), axis=1)
    elif use_color and use_hog:
        test_data = np.zeros((n, bins_color + bins_hog))
        for i in range(n):
            color_data = get_color_vector(image, centroids[i,:],
                                        color_rad)
            hog_data, mag_data = get_hog_vector(image, centroids[i,:],
                                               hog_rad)
            test_data[i,:] = np.concatenate((color_data, hog_data),
                                            axis=1)
    elif use_color and use_mag:
        test_data = np.zeros((n, bins_color + bins_mag))
        for i in range(n):
            color_data = get_color_vector(image, centroids[i,:],
                                        color_rad)
            hog_data, mag_data = get_hog_vector(image, centroids[i,:],
                                               hog_rad)
            test_data[i,:] = np.concatenate((color_data, mag_data),
                                            axis=1)
    elif use_hog and use_mag:
        test_data = np.zeros((n, bins_hog + bins_mag))
        for i in range(n):
            hog_data, mag_data = get_hog_vector(image, centroids[i,:],
                                               hog_rad)
            test_data[i,:] = np.concatenate((hog_data, mag_data),
                                            axis=1)
    elif use_color:
        test_data = np.zeros((n, bins_color))
        for i in range(n):
            color_data = get_color_vector(image, centroids[i,:],
                                        color_rad)
            test_data[i,:] = color_data
    elif use_hog:
        test_data = np.zeros((n, bins_hog))
        for i in range(n):
            hog_data, mag_data = get_hog_vector(image, centroids[i,:],
                                               hog_rad)
            test_data[i,:] = hog_data
    elif use_mag:
        test_data = np.zeros((n, bins_mag))
        for i in range(n):
            hog_data, mag_data = get_hog_vector(image, centroids[i,:],
                                               hog_rad)
            test_data[i,:] = mag_data
    else:
        print 'Error: No features were specified.'
    
    return test_data


def classify_candidates(train_data, train_labels, test_data,
                        label_encoder, k=3):
    """
    This function classifies all candidate roofs within an image and
    returns the predicted labels and classification probabilities.
    :param train_data: Training data
    :param train_labels: Training labels
    :param test_data: Test data
    :param label_encoder: Encoder to convert class predictions to labels
    :param neighbors: Number of neighbors for knn classifier
    :returns: Predicted labels and classification probabilities
    """
    # Create and fit a nearest neighbors classifier
    knn = neighbors.KNeighborsClassifier(n_neighbors=k)
    # Train on training observations
    knn.fit(train_data, train_labels)
    # Predict labels for test set
    predictions = knn.predict(test_data)
    labels = label_encoder.inverse_transform(predictions)
    probs = knn.predict_proba(test_data)
    
    return (predictions, labels, probs)


def annotate_results(image_fn, centroids, predictions, labels):
    """
    This function annotates the original image with the classification
    results and displays the annotated image.
    :param image_fn: Path of input image
    :param centroids: Candidate centroids
    :param predictions: Candidate class predictions
    :param labels: Candidate class labels
    :returns: Annotated image
    """
    # Load satellite image for annotation
    image = cv2.imread(image_fn)
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_color = [(255,0,255), (0,0,255), (0,255,0), (255,0,0)]
    
    # Annotate image for writing to output
    for i in range(centroids.shape[0]):
        cv2.putText(image, labels[i], (int(centroids[i,1]),
                                       int(centroids[i,0])),
                    font, 0.6, text_color[predictions[i]], thickness=2,
                    lineType=cv2.CV_AA)
        cv2.circle(image, (int(centroids[i,1]), int(centroids[i,0])),
                   radius=4, color=text_color[predictions[i]],
                   thickness=-1, lineType=8)
    
    # Return annotated image
    return image


def classify_image(image_fn, template_fn, image_data_fn, out_fn,
                   use_color=True, use_hog=True, use_mag=True,
                   thresh=0.05, color_rad=15, hog_rad=30, k=3,
                   display=False):
    """
    This function classifies all candidate roofs within an image and
    saves a labeled version of the original image. It returns the
    candidate centroid locations, predicted labels, and 
    classification probabilities.
    :param image_fn: Path of input image
    :param template_fn: Path of template used to find candidates
    :param image_data_fn: Path of csv containing training data
    :param out_fn: Path of output image
    :param use_color: Use color histogram for classification
    :param use_hog: Use histogram of oriented gradients
    :param use_mag: Use gradient magnitudes
    :param thresh: Threshold value for template matching
    :param color_rad: Radius for computing color histogram
    :param hog_rad: Radius for computing hog features
    :param k: Number of neighbors for knn classifier
    :param display: Choose True to display plots
    :returns: Candidate centroids, predicted label, and classification
    probabilities
    """
    # Template match, threshold, and segment
    markers, num_markers = find_markers(image_fn, template_fn)
    if display:
        plt.figure()
        plt.imshow(markers)
        plt.show()

    # If at least one candidate is found
    if num_markers > 1:
        
        # Find centroids
        centroids = find_centroids(markers, num_markers)
        
        # Get training data
        (train_data, train_labels, label_encoder) = \
                    get_training_data(image_data_fn, use_color,
                                      use_hog, use_mag)
        
        # Get test data
        test_data = get_test_data(image_fn, centroids, use_color,
                                  use_hog, use_mag)
        
        # Classify candidates
        (predictions, labels, probs) = classify_candidates(train_data,
                                             train_labels, test_data,
                                             label_encoder, k)
        
        # Annotate and display candidate classification results
        image = annotate_results(image_fn, centroids, predictions,
                                 labels)
        
        # Display classification results
        if display:
            # Load image for display
            converted_image = Image.open(image_fn).convert('P')
            # Colors and weights for each class
            colors = ['m', 'r', 'g', 'b']
            weights = ['normal', 'bold', 'normal', 'normal']
            # Annotate plot for display
            plt.figure()
            plt.imshow(converted_image)
            for i in range(centroids.shape[0]):
                plt.plot(centroids[i,1], centroids[i,0], 'o',
                         color=colors[predictions[i]])
                plt.text(centroids[i,1], centroids[i,0], labels[i],
                         color=colors[predictions[i]], fontsize=15,
                         fontweight=weights[predictions[i]])
            plt.xlim(0,400)
            plt.ylim(0,400)
            plt.gca().invert_yaxis()
            plt.show()
        
    # If no candidates are found
    else:
    	centroids = None
        labels = None
        probs = None
        if display:
            plt.figure()
            plt.imshow(converted_image)
            plt.show()
        # Load input image
        image = cv2.imread(image_fn)
    
    # Write output image
    cv2.imwrite(out_fn, image)
    
    return (centroids, labels, probs)


def batch_classify(in_dir, out_dir, template_fn, image_data_fn,
                   use_color=True, use_hog=True, use_mag=True,
                   thresh=0.05, color_rad=15, hog_rad=30, k=3):
    """
    This function classifies all images in the input directory and
    saves annotated output images in the output directory.
    :param in_dir: Input directory
    :param out_dir: Output directory
    :param template_fn: Path of template used to find candidates
    :param image_data_fn: Path of csv containing training data
    :param use_color: Use color histogram for classification
    :param use_hog: Use histogram of oriented gradients
    :param use_mag: Use gradient magnitudes
    :param thresh: Threshold value for template matching
    :param color_rad: Radius for computing color histogram
    :param hog_rad: Radius for computing hog features
    :param k: Number of neighbors for knn classifier
    """
    t_start = time.time()
    count = 0
    
    # Determine correct suffix
    if use_color and use_hog and use_mag:
        suffix = '_all'
    elif use_color and use_hog:
        suffix = '_color_hog'
    elif use_color and use_mag:
        suffix = '_color_mag'
    elif use_hog and use_mag:
        suffix = '_hog_mag'
    elif use_color:
        suffix = '_color'
    elif use_hog:
        suffix = '_hog'
    elif use_mag:
        suffix = '_mag'
    else:
        print 'Error: No features were specified.'
    
    # Process all images in input directory
    for image_fn in glob.glob(in_dir + '*'):
        out_fn = out_dir + str(count) + suffix + '.png'
        classify_image(image_fn, template_fn, image_data_fn, out_fn,
                       use_color, use_hog, use_mag, thresh, color_rad,
                       hog_rad, k)
        count += 1
    
    t_end = time.time()
    print 'Classified {} images in {} seconds.'.format(count,
                                                      (t_end - t_start))

