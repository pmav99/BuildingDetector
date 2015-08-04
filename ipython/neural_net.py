import numpy as np
import cv2
import glob
import time
import os
import classify_roof

net = classify_roof.init_net('/afs/cs.stanford.edu/u/nealjean/scratch/GitHub/poverty/nightlights_cnn/configs/VGG_S_roof_deploy.prototxt',\
         '/afs/cs.stanford.edu/u/nealjean/scratch/VGG_S_roof_iter_7500.caffemodel')
mean_arr = classify_roof.get_mean_arr('/afs/cs.stanford.edu/u/nealjean/scratch/VGG_mean.binaryproto')

# Classify each image patch
width = 80
height = 80
image_count = 0
font = cv2.FONT_HERSHEY_TRIPLEX
annotation_color = (0,255,0) # green
t0 = time.time()
in_dir = '../images/forest/classify/input/'
annotated_dir = '../images/forest/classify/annotated/'
for image_fn in glob.glob(in_dir + '*'):
    # Load in image and one to annotate
    image = cv2.imread(image_fn)
    image_out = cv2.imread(image_fn)
    image_count += 1
    print 'Classifying patches in image {}.'.format(image_count)
    # Get dimensions of input image
    max_x, max_y = image.shape[:2]
    # Pull image patches as long as they fit in the image
    count = 0
    top_left_x = 0
    # Lists to store locations and probabilities for detected roofs
    locations = []
    roof_probabilities = []
    patches = []
    while (top_left_x + (height - 1) < max_x):
        top_left_y = 0
        while (top_left_y + (width - 1) < max_y):
            # Get feature vector from image patch
            patch = image[top_left_x:top_left_x+height,
                          top_left_y:top_left_y+width, :]
            patches.append(patch)
            locations.append((top_left_x, top_left_y))
            """
            #color = calc_color_hist(patch)
            #color = color.flatten()
            #(hog, hog_bins, magnitude_hist, magnitude_bins,
            # max_magnitude) = compute_hog(patch)
            #(hog, magnitude_hist) = compute_advanced_hog(patch)
            #feature = np.concatenate((color, hog, magnitude_hist),
                                    axis=0)
            # Classify image patch
            #predict = clf.predict(feature)[0] # 0 = nonroof, 1 = roof
            #probs = clf.predict_proba(feature)[0]
            #nonroof_prob = probs[0]
            #roof_prob = probs[1]
            # If classified as a roof, store location and probability
            if predict == 1:
                locations.append((top_left_x, top_left_y))
                roof_probabilities.append(roof_prob)
            # Decide where to save image patch
            if predict == 1:
                out_fn = (roof_dir + str(roof_prob) + '_' +
                          os.path.basename(image_fn)[:-4] +
                          '_' + str(count) +  '.png')
            else:
                out_fn = (nonroof_dir + str(nonroof_prob) + '_' +
                          os.path.basename(image_fn)[:-4] +
                          '_' + str(count) + '.png')
            # Save image patch
            #save_patch(image, out_fn, top_left_x, top_left_y,
            #               width, height)
            """
            count += 1
            top_left_y += (width/2)
        top_left_x += (height/2)

    # Classify all patches
    probs = classify_roof.get_prob(net, mean_arr, patches)
    roof_probabilities = np.asarray(probs)[:,1]
    locations = [(x[0], x[1]) for x, y in zip(locations, roof_probabilities) if y > 0.5]
    roof_probabilities = roof_probabilities[roof_probabilities > 0.5].tolist()
    """
    locations = np.asarray(locations)
    locations = locations[roof_probabilities > 0.5]
    if len(locations.shape) < 2:
        locations = [locations.tolist()]
    locations = [(x[0], x[1]) for ]
    print locations
    """
    # Annotate image with non-maximum suppression
    annotate_locs = []
    annotate_probs = []
    while locations:
        max_prob = max(roof_probabilities)
        max_ind = roof_probabilities.index(max_prob)
        current_prob = roof_probabilities.pop(max_ind)
        current_loc = locations.pop(max_ind)
        # Check to see if overlapping
        overlap = False
        for i in range(len(annotate_locs)):
            if ((abs(current_loc[0] - annotate_locs[i][0]) < width) and
                (abs(current_loc[1] - annotate_locs[i][1]) < height)):
                overlap = True
        # If not overlapping, annotate and add to annotated list
        if not overlap:
            cv2.putText(image_out, str(current_prob),
                        (current_loc[1] + height/2, current_loc[0] + width/2),
                        font, 0.5, annotation_color, thickness=1,
                        lineType=cv2.CV_AA)
            cv2.rectangle(image_out, (current_loc[1], current_loc[0]),
                         (current_loc[1]+height-1, current_loc[0]+width-1),
                         annotation_color)
            annotate_locs.append(current_loc)
            annotate_probs.append(current_prob)
    # Save annotated image
    image_out_fn = annotated_dir + os.path.basename(image_fn)
    cv2.imwrite(image_out_fn, image_out)
t1 = time.time()
print 'Classified image patches for {} images in {} seconds.'.format(
                                        image_count, (t1-t0))