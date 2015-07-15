import cv2
import numpy as np
import glob
import os
import time
import pandas as pd
from sklearn import preprocessing

def match_template(imageFn, templateFn, method=cv2.TM_SQDIFF_NORMED):
    """
    This function does template matching with the specified image and template and returns the result of the template matching.
    :param imageFn: Filename of the image
    :param templateFn: Filename of the template
    :param method: The template matching method to use
    :returns: Template-matching result as a numpy array
    """
    # Import both image and template as grayscale
    image = cv2.imread(imageFn, 0)
    template = cv2.imread(templateFn, 0)
    # Matching the template without padding input
    result = cv2.matchTemplate(image, template, method=method)
    # Return the template matching result
    return result


def threshold_template(imageFn, templateFn, threshold=0.1, method=cv2.TM_SQDIFF_NORMED):
    """
    This function does template matching, then thresholds the results and returns an array that is 1 where the result is below the threshold.
    :param imageFn: Filename of image
    :param templateFn: Filename of template
    :param threshold: Threshold to use
    :param method: The template matching method to use
    :returns: An array with the locations where the results meet the threshold
    """
    # Do the template matching
    result = match_template(imageFn, templateFn, method)
    # Threshold the result
    locations = np.where(result <= threshold)
    # Return the thresholded locations
    return locations


def write_thresholded(imageFn, templateFn, outFn, threshold=0.1, method=cv2.TM_SQDIFF_NORMED, color=(0,255,0)):
    """
    This function writes the resulting thresholded image to an output file, adding color to the areas of the image that meet the threshold.
    :param imageFn: Filename of image
    :param templateFn: Filename of template
    :param outFn: Filename of output image
    :param threshold: Threshold to use
    :param method: The template matching method to use
    :param color: The color to add on thresholded regions
    """
    # Get the locations where threshold is satisfied
    locations = threshold_template(imageFn, templateFn, threshold, method)
    # Load original image and grayscale template
    image = cv2.imread(imageFn)
    template = cv2.imread(templateFn, 0)
    # Get template pixel dimensions
    height, width = template.shape
    # Set all pixels meeting threshold to desired color
    for pixel in zip(*locations[::1]):
        image[pixel[0] + width/2, pixel[1] + height/2] = color
    # Save image to output file location
    cv2.imwrite(outFn, image)


def batch_write(imageDir, templateFn, outDir, suffix=None, threshold=0.1, method=cv2.TM_SQDIFF_NORMED, color=(0,255,0)):
    """
    This function writes the resulting thresholded images for all images in the image directory.
    :param imageDir: Input image directory
    :param templateFn: Filename of template
    :param outDir: Output image directory
    :param suffix: String to add to output image names; default is template name
    :param threshold: Threshold to use
    :param method: The template matching method to use
    :param color: The color to add on thresholded regions
    """
    # Time the batch operation
    startTime = time.time()
    count = 0
    # Get template filename
    template = os.path.basename(templateFn)
    # Process all images in image directory
    for imageFn in glob.glob(imageDir + '*'):
        fn = os.path.basename(imageFn)[:-4]
        if suffix==None:
            outFn = outDir + fn + template
        else:
            outFn = outDir + fn + suffix + '.png'
        write_thresholded(imageFn, templateFn, outFn, threshold, method, color)
        count += 1
    # Display stats for batch processing
    endTime = time.time()
    elapsed = endTime - startTime
    print 'Template-matched {} images in {} seconds.'.format(count, elapsed)


def append_image_data(class_name, image_dir, csv_in, csv_out=None):
    """
    This function looks in the specified image directory and adds image
    data for all images within the directory to the database loaded from
    the input csv file, then writes the output database to the output
    csv.
    :param class_name: Class of images contained in directory
    :param image_dir: Directory containing images
    :param csv_in: Path to the input csv
    :param csv_out: Path of output csv
    """
    # If no output csv specified, write to input csv
    if csv_out == None:
        csv_out = csv_in
    # Loading image data from csv
    imageData = pd.read_csv(csv_in)
    # Need to convert imageIDs to strings (some get converted to ints)
    imageData.imageID = imageData.imageID.astype(str)
    # Drop first column
    imageData.drop('Unnamed: 0', axis=1, inplace=True)
    # Iterate through images in directory and add image data to DataFrame
    count = 0
    for imagePath in glob.iglob(image_dir + '*.png'):
        count += 1
        #print 'Processing image {}'.format(count)
        # Get image basename with extension
        imageFn = os.path.basename(imagePath)
        # Get image ID
        imageID = imageFn[:-4]
        # Read in image and get dimensions
        image = cv2.imread(imagePath)
        h, w = image.shape[:2]
        # Calculate color histogram
        hist = calc_color_hist(image, bins=16)
        hist = np.transpose(hist)
        # Create temporary histogram DataFrame
        histdf = pd.DataFrame(hist)
        colNames = list(histdf.columns.values)
        tempNames = [str(i+1) for i in colNames]
        newNames = ['hist' + i for i in tempNames]
        histdf.columns = newNames
        # Add to DataFrame
        img_data = pd.DataFrame({'class': [class_name], 'imageID': [imageID],\
                                 'filename': [imageFn], 'width': [w],\
                                 'height': [h]})
        # Add histogram DataFrame
        img_data = pd.concat([img_data, histdf], axis=1)
        # Continues indexing
        imageData = imageData.append(img_data, ignore_index=True)
    print 'Processed {} images in total'.format(count)
    # Drop any rows with duplicated information
    imageData.drop_duplicates(cols=['class', 'imageID'], inplace=True)
    # Reorganize columns
    imageData = imageData[['class', 'imageID', 'filename',\
                           'width', 'height', 'hist1', 'hist2',\
                           'hist3', 'hist4', 'hist5', 'hist6',\
                           'hist7', 'hist8', 'hist9', 'hist10',\
                           'hist11', 'hist12', 'hist13', 'hist14',\
                           'hist15', 'hist16', 'hist17', 'hist18',\
                           'hist19', 'hist20', 'hist21', 'hist22',\
                           'hist23', 'hist24', 'hist25', 'hist26',\
                           'hist27', 'hist28', 'hist29', 'hist30',\
                           'hist31', 'hist32', 'hist33', 'hist34',\
                           'hist35', 'hist36', 'hist37', 'hist38',\
                           'hist39', 'hist40', 'hist41', 'hist42',\
                           'hist43', 'hist44', 'hist45', 'hist46',\
                           'hist47', 'hist48']]
    # Reset index in case duplicate rows were dropped
    imageData.reset_index(drop=True)
    # Writing image data to output csv
    imageData.to_csv(csv_out)
    print 'Wrote data for {} images to csv.'.format(imageData.shape[0])


def import_image_data(csv_in):
    """
    This function reads image data from a csv file and returns Numpy
    arrays containing the image features and the image labels so that
    the data is ready for use. It also returns a label encoder so that
    we can go back and forth between class string labels and integer
    encodings.
    :param csv_in: Path of csv file containing image data
    :returns: Image feature vectors, image labels, and label encoder
    """
    # Loading image data from csv
    imageData = pd.read_csv(csv_in)
    # Need to convert imageIDs to strings (some get converted to ints)
    imageData.imageID = imageData.imageID.astype(str)
    # Drop first column
    imageData.drop('Unnamed: 0', axis=1, inplace=True)
    # Get class labels
    labels = imageData.as_matrix(columns=['class'])
    # Making 2D array into 1D array
    labels = labels.flatten()
    # Encode labels with value between 0 and n_classes-1
    le = preprocessing.LabelEncoder()
    # roof = 0, water = 2, vegetation = 1
    le.fit(labels)
    print le.classes_ # displays the set of classes
    labels = le.transform(labels)
    print 'Got class labels for {} training data points.'.format\
          (labels.shape[0])
    # Get color histograms
    bins = ['hist' + str(i+1) for i in range(48)]
    hists = imageData.as_matrix(columns=bins)
    print 'Got feature vector for {} training data points.'.format\
          (hists.shape[0])
    return (hists, labels, le)