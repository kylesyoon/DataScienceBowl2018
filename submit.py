import argparse
import numpy as np
import pandas as pd

from skimage.transform import resize
from skimage.morphology import label

from keras.models import Model, load_model

from preprocessing.generator import preprocess_test


IMG_WIDTH = IMG_HEIGHT = 256
IMG_CHANNELS = 3


def rle_encoding(x):
    '''
    x: numpy array of shape (height, width), 1 - mask, 0 - background
    Returns run length as list
    '''
    dots = np.where(x.T.flatten() == 1)[
        0]  # .T sets Fortran order down-then-right
    run_lengths = []
    prev = -2
    for b in dots:
        if (b > prev + 1):
            run_lengths.extend((b + 1, 0))
        run_lengths[-1] += 1
        prev = b

    return run_lengths


def prob_to_rles(x, cutoff=0.5):
    lab_img = label(x > cutoff)
    for i in range(1, lab_img.max() + 1):
        yield rle_encoding(lab_img == i)


def main(args=None):
	parser = argparse.ArgumentParser()
	parser.add_argument('--path', help='File path to the model to use for submission')

	args = parser.parse_args()
	model_path = args.path

	if model_path is None:
		print('--path is required')
		return

	model = load_model(model_path)
	test_X, test_ids, test_image_sizes = preprocess_test(input_size=(IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS))

	preds = model.predict(test_X, verbose=1)
	preds = np.squeeze(preds)

	# resizing the predictions to original size
	preds_resized = []
	for index, pred in enumerate(preds):
	    image = resize(pred, test_image_sizes[index], mode='constant', preserve_range=True)
	    preds_resized.append(image)

	pred_ids = []
	rles = []
	for index, id_ in enumerate(test_ids):
	    rle = list(prob_to_rles(preds_resized[index]))
	    rles.extend(rle)
	    pred_ids.extend([id_] * len(rle))

	submission = pd.DataFrame()
	submission['ImageId'] = pred_ids
	submission['EncodedPixels'] = pd.Series(rles).apply(lambda x: ' '.join(str(y) for y in x))
	submission.to_csv('submission.csv', index=False)


if __name__ == '__main__':
    main()
