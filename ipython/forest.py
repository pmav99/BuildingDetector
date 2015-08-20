import numpy as np
import cv2
from features import *
from utils import *
from sklearn import ensemble


def train_forest_classifier(csv_in, boost=False):
	"""
	This function trains a random forest classifier on the stored roof/nonroof
	training examples.
	:param csv_in: CSV file containing training data
	:param boost: If true, use AdaBoost
	:returns: Random forest roof classifier
	"""
	(features, colors, hogs, labels, label_encoder) = \
		import_image_data(csv_in, display=True)
    # Initialize classifier
	if boost:
		clf = ensemble.RandomForestClassifier()
		classifier = ensemble.AdaBoostClassifier(base_estimator=clf,
			n_estimators=50, random_state=0)
	else:
		classifier = ensemble.RandomForestClassifier(n_estimators=50,
			random_state=0, class_weight='auto')
	# Train classifier
	classifier.fit(features, labels)

	return classifier


def classify_patch(patch, classifier, thresh=0.5):
	"""
	This function takes in an 80x80 image patch and classifies it as roof or
	nonroof. It returns the roof prediction and roof probability.
	:param patch: Image patch
	:param classifier: Classifies roof/nonroof image patches
	:returns: Roof prediction and probability
	"""
	# Compute features
	features = compute_features(patch)
	# Classify patch
	probs = classifier.predict_proba(features)[0]
	prob = probs[1]
	if prob > 0.5:
		predict = 1
	else:
		predict = 0
	return (predict, prob)


def max_roofs(locs, probs, width=80, height=80):
	"""
	This function takes in the locations and probabilities of roofs found in
	a 400x400 pixel image and returns the locations and probabilities after
	non-maximum suppression.
	:param locs: Locations of candidate roofs
	:param probs: Probabilities of candidate roofs
	:param width: Width of image patches
	:param height: Height of image patches
	:returns: Locations and probabilities of roofs after non-maximum suppression
	"""
	max_locs = []
	max_probs = []
	while locs:
		max_prob = max(probs)
		max_ind = probs.index(max_prob)
		current_prob = probs.pop(max_ind)
		current_loc = locs.pop(max_ind)
		# Check for overlap
		overlap = False
		for i in range(len(max_locs)):
			if ((abs(current_loc[0] - max_locs[i][0]) < width) and
				(abs(current_loc[1] - max_locs[i][1]) < height)):
				overlap = True
		# If no overlap, add to max locations
		if not overlap:
			max_locs.append(current_loc)
			max_probs.append(current_prob)

	return (max_locs, max_probs)


def find_roofs(image, classifier):
	"""
	This function takes in a 400x400 pixel satellite image and finds all roofs. It
	returns the location and probability of roofs found.
	:param image: Input satellite image
	:param classifier: Classifies roof/nonroof image patches
	:returns: Locations and probabilities of roofs
	"""
	# Image patch dimensions
	width = 80
	height = 80
	# Dimensions of input image
	max_x, max_y = image.shape[:2]
	# Classify each image patch
	locs = []
	probs = []
	tl_x = 0
	while (tl_x + (height - 1) < max_x):
		tl_y = 0
		while (tl_y + (width - 1) < max_y):
			patch = image[tl_x:tl_x+height, tl_y:tl_y+width,:]
			# Classify patch
			(predict, prob) = classify_patch(patch, classifier)
			# Store roof locations and probabilities
			if predict:
				locs.append((tl_x, tl_y))
				probs.append(prob)
			tl_y += (width/2)
		tl_x += (height/2)

	# Non-maximum suppression
	(locs, probs) = max_roofs(locs, probs, width, height)

	return (locs, probs)


def count_roofs(image, classifier):
	"""
	This function takes in a 400x400 pixel satellite image and classifies each image
	patch as roof or nonroof. It uses non-maximum suppression to determine the total
	number of roofs.
	:param image: Input satellite image
	:param classifier: Classifies roof/nonroof image patches
	:returns: Number of roofs found
	"""
	# Find all patches with roofs
	(locs, probs) = find_roofs(image, classifier)
	# Count the number of roofs found
	count = len(locs)
	return count


def batch_count_roofs(in_dir, csv_in, csv_out):
	"""
	This function takes in a folder of 400x400 pixel satellite images and counts the
	roofs in each image. It stores the image name and roof count in a csv file.
	:param in_dir: Input directory containing images
	:param csv_in: Input csv containing training data
	:param csv_out: Output csv containing image roof counts
	"""
	# Training classifier
	classifier = train_forest_classifier(csv_in, boost=False)
	# Create dataframe to store roof counts
	image_data = pd.DataFrame(columns=['filename', 'roof_count'])
	# Process images
	t0 = time.time()
	count = 0
	for image_fn in glob.glob(in_dir + '*'):
		image = cv2.imread(image_fn)
		roof_count = count_roofs(image, classifier)
		image_data.loc[count] = [image_fn, roof_count]
		count += 1
		if count % 100 == 0:
			t1 = time.time()
			print 'Processed {} images so far in {} seconds.'.format(count, (t1-t0))
	# Write data to csv
	image_data.to_csv(csv_out)
	tf = time.time()
	print 'Stored roof counts for {} images in {} seconds.'.format(count, (tf-t0))