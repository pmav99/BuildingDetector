import cv2
import numpy as np
import glob
import os
import time


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
