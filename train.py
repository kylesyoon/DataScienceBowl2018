import argparse
import os
import random
import numpy as np
import pandas as pd
from tqdm import tqdm

from skimage.morphology import label

from keras.models import Model, load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import Adam

from keras_unet.unet import unet
from preprocessing.generator import preprocess_train, preprocess_test, create_train_generator


IMG_WIDTH = IMG_HEIGHT = 256
IMG_CHANNELS = 3

NUM_EPOCHS = 30
STEPS_PER_EPOCH = 600 # there are 670 images


def create_model():
	print('Creating model...')

	model = unet(input=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
	optimizer = Adam(lr=1e-5)

	model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=[]) # TODO: Metrics

	model.summary()

	return model


def create_callbacks():
	print('Creating callbacks...')
	early = EarlyStopping(patience=3, verbose=1)
	checkpoint = ModelCheckpoint('./snapshots/keras_unet.h', verbose=1, save_best_only=True)


def rle_encoding(x):
    '''
    x: numpy array of shape (height, width), 1 - mask, 0 - background
    Returns run length as list
    '''
    dots = np.where(x.T.flatten() == 1)[0] # .T sets Fortran order down-then-right
    run_lengths = []
    prev = -2
    for b in dots:
        if (b > prev + 1): run_lengths.extend((b + 1, 0))
        run_lengths[-1] += 1
        prev = b
        
    return run_lengths


def prob_to_rles(x, cutoff=0.5):
    lab_img = label(x > cutoff)
    for i in range(1, lab_img.max() + 1):
        yield rle_encoding(lab_img == i)


def main():
	# preprocess data 
	train_X, train_Y = preprocess_train(input_size=(IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS))

	test_X, test_image_sizes = preprocess_test(input_size=(IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS))

	# create generators
	train_generator = create_train_generator()
	train_generator.fit(train_X)

	# create the model
	model = create_model()

	# create the callbacks
	callbacks = create_callbacks()

	# train
	print('Starting training...')
	model.fit_generator(train_generator.flow(train_X, train_Y), epochs=NUM_EPOCHS, steps_per_epoch=STEPS_PER_EPOCH, callbacks=callbacks, verbose=1)
	print('Finished training...')

	print('Starting submission...')
	predictions = model.predict(test_X, verbose=1)

	preds = np.squeeze(predictions)

	# resizing the predictions to original size
	preds_resized = []
	for index_, pred in enumerate(preds):
	    image = resize(pred, test_image_sizes[index_], mode='constant', preserve_range=True)
	    preds_resized.append(image)

	new_test_ids = []
	rles = []
	for n, id_ in enumerate(test_ids):
	    rle = list(prob_to_rles(preds_resized[n]))
	    rles.extend(rle)
	    new_test_ids.extend([id_] * len(rle))

	submission = pd.DataFrame()
	submission['ImageId'] = new_test_ids
	submission['EncodedPixels'] = pd.Series(rles).apply(lambda x: ' '.join(str(y) for y in x))
	submission.to_csv('keras_unet.csv', index=False)


if __name__ == '__main__':
	main()