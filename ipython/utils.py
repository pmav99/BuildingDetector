import cv2
import numpy as np
import glob
import os
import time
import pandas as pd
from sklearn import preprocessing
from PIL import Image
import matplotlib.pyplot as plt
from features import *
import urllib


def sample_image(out_fn, lat, lon, height=400, width=400, zoom=19):
    """
    This function uses the Google Static Maps API to download and save
    one satellite image.
    :param out_fn: Output filename for saved image
    :param lat: Latitude of image center
    :param lon: Longitude of image center
    :param height: Height of image in pixels
    :param width: Width of image in pixels
    :param zoom: Zoom level of image
    :return: True if valid image saved, False if no image saved
    """
    # Google Static Maps API key
    api_key = 'AIzaSyAejgapvGncLMRiMlUoqZ2h6yRF-lwNYMM'
    
    # Save extra tall satellite image
    height_buffer = 100
    url_pattern = 'https://maps.googleapis.com/maps/api/staticmap?center=%0.6f,%0.6f&zoom=%s&size=%sx%s&maptype=satellite&key=%s'
    url = url_pattern % (lat, lon, zoom, width, height + height_buffer, api_key)
    urllib.urlretrieve(url, out_fn)

    # Cut out text at the bottom of the image
    image = cv2.imread(out_fn)
    image = image[(height_buffer/2):(height+height_buffer/2),:,:]
    cv2.imwrite(out_fn, image)

    # Check file size and delete invalid images < 10kb
    fs = os.stat(out_fn).st_size
    if fs < 10000:
        os.remove(out_fn)
        return False
    else:
        return True


def sample_dhs(image_data, cell_id, cell_lat, cell_lon, samples,
               out_dir):
    """
    This function samples multiple images at random for a DHS location
    and saves them in the output directory.
    :param image_data: DataFrame containing image metadata
    :param cell_id: Cell ID of DHS location
    :param cell_lat: Latitude of DHS location
    :param cell_lon: Longitude of DHS location
    :param samples: Number of samples to get
    :param out_dir: Directory for sampled images
    :returns: DataFrame containing updated image metadata
    """
    t_start = time.time()
    
    # Sample images
    count = 0
    while count < samples:
        # Randomly sample within half-degree cell
        lat = cell_lat + np.random.uniform(-0.25, 0.25)
        lon = cell_lon + np.random.uniform(-0.25, 0.25)
        # Determine output filename
        fn = str(cell_id) + '_' + str(count) + '.png'
        out_fn = out_dir + fn
        # Save image
        check_image = sample_image(out_fn, lat, lon)
        if check_image:
            # Update image metadata
            temp_data = pd.DataFrame({'image': [fn], 'cellid': [cell_id],
                                     'cell_lat': [cell_lat], 'cell_lon': 
                                     [cell_lon], 'lat': [lat], 'lon': 
                                     [lon]})
            image_data = pd.concat([image_data, temp_data])
            count += 1
    
    # Return updated image metadata
    t_end = time.time()
    print 'Sampled {} images from DHS cell {} in {} seconds.'.format(
                                                    samples, cell_id,
                                                    (t_end - t_start))
    return image_data


def display_image(imageFn):
    """
    This function loads an image and displays it in true color.
    :param imageFn: Filename of the image
    """
    image = Image.open(imageFn)
    convertedImage = image.convert('P')
    plt.figure()
    plt.imshow(convertedImage)
    plt.show()


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


def process_image(image_path, class_name, image_data, hog_type='advanced'):
    """
    This function process one image and adds its data to the dataframe
    containing all of the training image data.
    :param image_path: Path to image to process
    :param class_name: Class of image sample
    :param image_data: Dataframe containing training data
    :param hog_type: Advanced computes sub-patch features as well
    :returns: Dataframe containing new image data
    """
    # Get image basename with extension
    image_fn = os.path.basename(image_path)
    # Get image ID
    imageID = image_fn[:-4]
    
    # Read in image and get dimensions
    image = cv2.imread(image_path)
    h, w = image.shape[:2]
    
    # Compute features
    features = compute_features(image)
    features = features.reshape(1, features.size)
    # Create features dataframe
    features_df = pd.DataFrame(features)
    feature_names = ['color' + str(i + 1) for i in range(48)]
    feature_names = feature_names + ['hog' + str(i + 1) for i in range(324)]
    features_df.columns = feature_names

    # Create dataframe for image
    img_data = pd.DataFrame({'class': [class_name], 'imageID': [imageID],
        'filename': [image_fn], 'width': [w], 'height': [h]})
    img_data = pd.concat([img_data, features_df], axis=1)
    
    # Add new image data to image dataframe and continue indexing
    return image_data.append(img_data, ignore_index=True)


def store_image_data(image_dir, csv_out):
    """
    This function goes through each of the class sample directories
    within the samples directory and computes color histogram and HOG
    features for each image within, then writes the data to a csv file
    for later use.
    :param image_dir: Path to the directory containing image samples
    :param csv_out: Path of output csv file containing data
    """
    start_time = time.time()
    
    # Setting column names of dataframe
    columns = ['class', 'imageID', 'filename', 'width', 'height']
    columns = columns + ['color' + str(i + 1) for i in range(48)]
    columns = columns + ['hog' + str(i + 1) for i in range(324)]
    
    # Create data frame for image data
    image_data = pd.DataFrame(columns=columns)
    
    # Keep track of total images processed
    count_total = 0
    
    # Search sample directory for all sample class subdirectories
    for class_dir in glob.iglob(image_dir + '*'):
        # Get class name
        class_name = os.path.basename(class_dir)
        # Add data for each image in class subdirectory
        count_class = 0
        check0_time = time.time()
        for image_path in glob.iglob(class_dir + '/*.png'):
            # Add data for each image
            image_data = process_image(image_path, class_name, image_data)
            count_total += 1
            count_class += 1
            # Print update for every 100 images
            if count_class % 100 == 0:
                print 'Processed {} images for {} class.'.format(count_class,
                    class_name)
        # Print update for class
        check1_time = time.time()
        print 'Processed {} images for {} class in {} seconds.'.format(
            count_class,class_name,(check1_time-check0_time))
        check0_time = check1_time
    
    # Drop rows with duplicated information
    image_data.drop_duplicates(['class', 'filename'], inplace=True)
    # Reorganize columns
    image_data = image_data[columns]
    # Writing image data to csv
    image_data.to_csv(csv_out)
    end_time = time.time()
    print 'Processed {} images total in {} seconds.'.format(count_total,
        (end_time - start_time))


def import_image_data(csv_in, display=False):
    """
    This function reads image data from a csv file and returns Numpy
    arrays containing the image features and the image labels so that
    the data is ready for use. It also returns a label encoder so that
    we can go back and forth between class string labels and integer
    encodings.
    :param csv_in: Path of csv file containing image data
    :param display: Print out updates
    :returns: All features vectors, color histogram vectors, HOG vectors,
    image labels, and label encoder
    """
    # Loading image data from csv
    image_data = pd.read_csv(csv_in)
    # Need to convert imageIDs to strings (some get converted to ints)
    image_data.imageID = image_data.imageID.astype(str)
    # Drop first column
    image_data.drop('Unnamed: 0', axis=1, inplace=True)
    # Get class labels
    labels = image_data.as_matrix(columns=['class'])
    # Making 2D array into 1D array
    labels = labels.flatten()
    # Encode labels with value between 0 and n_classes-1
    label_encoder = preprocessing.LabelEncoder()
    # dirt = 0, roof = 1, vegetation = 2, water = 3
    label_encoder.fit(labels)
    if display:
        print label_encoder.classes_ # displays the set of classes
    labels = label_encoder.transform(labels)
    if display:
        print 'Got class labels for {} training data points.'.format\
              (labels.shape[0])
    # Get color histograms
    color_columns = ['color' + str(i+1) for i in range(48)]
    colors = image_data.as_matrix(columns=color_columns)
    # Get hog vectors
    hog_columns = ['hog' + str(i+1) for i in range(324)]
    hogs = image_data.as_matrix(columns=hog_columns)
    # Join into all features vectors
    all_columns = color_columns + hog_columns
    features = image_data.as_matrix(columns=all_columns)
    if display:
        print 'Got feature vectors for {} training data points.'.format\
              (features.shape[0])
    return (features, colors, hogs, labels, label_encoder)






'''
def import_image_data(csv_in, display=False):
    """
    This function reads image data from a csv file and returns Numpy
    arrays containing the image features and the image labels so that
    the data is ready for use. It also returns a label encoder so that
    we can go back and forth between class string labels and integer
    encodings.
    :param csv_in: Path of csv file containing image data
    :param display: Print out updates
    :returns: All features vectors, color histogram vectors, HOG vectors,
    gradient magnitude vectors, image labels, and label encoder
    """
    # Loading image data from csv
    image_data = pd.read_csv(csv_in)
    # Need to convert imageIDs to strings (some get converted to ints)
    image_data.imageID = image_data.imageID.astype(str)
    # Drop first column
    image_data.drop('Unnamed: 0', axis=1, inplace=True)
    # Get class labels
    labels = image_data.as_matrix(columns=['class'])
    # Making 2D array into 1D array
    labels = labels.flatten()
    # Encode labels with value between 0 and n_classes-1
    label_encoder = preprocessing.LabelEncoder()
    # dirt = 0, roof = 1, vegetation = 2, water = 3
    label_encoder.fit(labels)
    if display:
        print label_encoder.classes_ # displays the set of classes
    labels = label_encoder.transform(labels)
    if display:
        print 'Got class labels for {} training data points.'.format\
              (labels.shape[0])
    # Get color histograms
    color_columns = ['color' + str(i+1) for i in range(48)]
    colors = image_data.as_matrix(columns=color_columns)
    # Get hog vectors
    hog_columns = ['hog' + str(i+1) for i in range(9*17)]
    hogs = image_data.as_matrix(columns=hog_columns)
    # Get gradient magnitude vectors
    mag_columns = ['mag' + str(i+1) for i in range(16*17)]
    mags = image_data.as_matrix(columns=mag_columns)
    # Join into all features vectors
    all_columns = color_columns + hog_columns + mag_columns
    features = image_data.as_matrix(columns=all_columns)
    if display:
        print 'Got feature vectors for {} training data points.'.format\
              (features.shape[0])
    return (features, colors, hogs, mags, labels, label_encoder)


def process_image(image_path, class_name, image_data, hog_type='advanced'):
    """
    This function process one image and adds its data to the dataframe
    containing all of the training image data.
    :param image_path: Path to image to process
    :param class_name: Class of image sample
    :param image_data: Dataframe containing training data
    :param hog_type: Advanced computes sub-patch features as well
    :returns: Dataframe containing new image data
    """
    # Get image basename with extension
    image_fn = os.path.basename(image_path)
    # Get image ID
    imageID = image_fn[:-4]
    
    # Read in image and get dimensions
    image = cv2.imread(image_path)
    h, w = image.shape[:2]
    
    # Calculate color histogram
    color = calc_color_hist(image)
    color = np.transpose(color)
    # Create color histogram dataframe
    color_df = pd.DataFrame(color)
    col_names = list(color_df.columns.values)
    new_names = ['color' + str(i + 1) for i in col_names]
    color_df.columns = new_names
    
    # Calculate hog and gradient magnitude features
    if hog_type == 'advanced':
        (hog, mag) = compute_advanced_hog(image)
    else:
        (hog, hog_bins, mag, mag_bins, max_mag) = compute_hog(image)
    hog = hog.reshape(1, hog.size)
    mag = mag.reshape(1, mag.size)
    # Create hog dataframe
    hog_df = pd.DataFrame(hog)
    col_names = list(hog_df.columns.values)
    new_names = ['hog' + str(i + 1) for i in col_names]
    hog_df.columns = new_names
    
    # Create gradient magnitude dataframe
    mag_df = pd.DataFrame(mag)
    col_names = list(mag_df.columns.values)
    new_names = ['mag' + str(i + 1) for i in col_names]
    mag_df.columns = new_names
    
    # Create dataframe for image
    img_data = pd.DataFrame({'class': [class_name], 'imageID':\
                             [imageID], 'filename': [image_fn],\
                             'width': [w], 'height': [h]})
    img_data = pd.concat([img_data, color_df, hog_df, mag_df],\
                       axis=1)
    
    # Add new image data to image dataframe and continue indexing
    return image_data.append(img_data, ignore_index=True)


def store_image_data(image_dir, csv_out):
    """
    This function goes through each of the class sample directories
    within the samples directory and computes color histogram and HOG
    features for each image within, then writes the data to a csv file
    for later use.
    :param image_dir: Path to the directory containing image samples
    :param csv_out: Path of output csv file containing data
    """
    start_time = time.time()
    
    # Setting column names of dataframe
    columns = ['class', 'imageID', 'filename', 'width', 'height']
    columns = columns + ['color' + str(i + 1) for i in range(48)]
    columns = columns + ['hog' + str(i + 1) for i in range(9*17)]
    columns = columns + ['mag' + str(i + 1) for i in range(16*17)]
    
    # Create data frame for image data
    image_data = pd.DataFrame(columns=columns)
    
    # Keep track of total images processed
    count_total = 0
    
    # Search sample directory for all sample class subdirectories
    for class_dir in glob.iglob(image_dir + '*'):
        # Get class name
        class_name = os.path.basename(class_dir)
        # Add data for each image in class subdirectory
        count_class = 0
        for image_path in glob.iglob(class_dir + '/*.png'):
            # Add data for each image
            image_data = process_image(image_path, class_name,\
                                       image_data)
            count_total += 1
            count_class += 1
            # Print update for every 100 images
            if count_class % 100 == 0:
                print 'Processed {} images for {} class.'.format(count_class,\
                                                         class_name)
        # Print update for class
        print 'Processed {} images for {} class.'.format(count_class,\
                                                         class_name)
    
    # Drop rows with duplicated information
    image_data.drop_duplicates(['class', 'filename'], inplace=True)
    # Reorganize columns
    image_data = image_data[columns]
    # Writing image data to csv
    image_data.to_csv(csv_out)
    end_time = time.time()
    print 'Processed {} images total in {} seconds.'.format(count_total,\
                                                (end_time - start_time))
'''