import os
import numpy as np
from tqdm import tqdm

from skimage.io import imread
from skimage.transform import resize

from keras.preprocessing.image import ImageDataGenerator

TRAIN_PATH = 'data/stage1_train/'
TEST_PATH = 'data/stage1_test/'

def preprocess_train(input_size):
	#data/
	#    stage1_train/
	#         image_id/
	#             images/
	#             masks/
	#         image_id/
	# ...
	print('Creating training generator...')

	# TODO: Pass in the save path via arg
	if os.path.isfile('./data/train_X.npy') is True and os.path.isfile('./data/train_Y.npy') is True:
		print('Loading previously processed training data...')
		# try to load the formatted training data
		train_X = np.load('./data/train_X.npy')
		train_Y = np.load('./data/train_Y.npy')
	else:
		print('Processing training data...')
		train_ids = next(os.walk(TRAIN_PATH))[1] # getting the name of the images

		train_X = np.zeros((len(train_ids), input_size[0], input_size[1], input_size[2]), dtype=np.uint8) # width, height, channels
		train_Y = np.zeros((len(train_ids), input_size[0], input_size[1], 1), dtype=np.bool)

		for index_, image_id in tqdm(enumerate(train_ids), total=len(train_ids)):
		    # base path
		    path = TRAIN_PATH + image_id
		    # getting the images
		    image_path = path + '/images/' + image_id + '.png'
		    image = imread(image_path)
		    resized = resize(image, (input_size[0], input_size[1], input_size[2]), mode='constant', preserve_range=True)
		    train_X[index_] = resized
		    # getting the masks
		    complete_mask = np.zeros((input_size[0], input_size[1], 1), dtype=np.bool)
		    for mask_id in next(os.walk(path + '/masks/'))[2]:
		        mask_path = path + '/masks/' + mask_id
		        mask = imread(mask_path)
		        resized_mask = resize(mask, (input_size[0], input_size[1], 1), mode='constant', preserve_range=True)
		        # creating one mask for all the masks for this image
		        complete_mask = np.maximum(resized_mask, complete_mask)
		    train_Y[index_] = complete_mask

	    # save the formatted training data
		np.save('./data/train_X', train_X)
		np.save('./data/train_Y', train_Y)

	return (train_X, train_Y)


def preprocess_test(input_size):
	test_ids = next(os.walk(TEST_PATH))[1]

	test_image_sizes = [] # we are going to resize the predicted test images back to original size

	for index_, test_id in tqdm(enumerate(test_ids), total=len(test_ids)):
	    image_path = TEST_PATH + test_id + '/images/' + test_id + '.png'
	    image = imread(image_path)
	    test_image_sizes.append((image.shape[0], image.shape[1]))
	    resized = resize(image, (input_size[0], input_size[1], input_size[2]), mode='constant', preserve_range=True)
	    test_X[index_] = resized

	return (test_X, test_image_sizes)


def create_train_generator():
	return ImageDataGenerator(rotation_range=45, width_shift_range=0.25, height_shift_range=0.25, horizontal_flip=True, vertical_flip=True)
