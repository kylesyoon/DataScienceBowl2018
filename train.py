import argparse
import os
import random
import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm

from keras.models import Model, load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import Adam

from keras_unet.unet import unet
from preprocessing.generator import preprocess_train, create_train_generator, create_val_generator


IMG_WIDTH = IMG_HEIGHT = 256
IMG_CHANNELS = 3

BATCH_SIZE = 32
NUM_EPOCHS = 50
STEPS_PER_EPOCH = 600  # there are 670 images


def create_model():
    print('Creating model...')

    model = unet(input=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
    optimizer = Adam(lr=1e-3)

    model.compile(optimizer=optimizer,
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    model.summary()

    return model


def create_callbacks():
    print('Creating callbacks...')
    checkpoint = ModelCheckpoint('./snapshots/keras_unet_{epoch:02d}.h5', verbose=1)

    return [checkpoint]


def create_generators():
    return create_train_generator(), create_val_generator()


def main():
    # preprocess data
    train_X, train_Y, val_X, val_Y = preprocess_train(
        input_size=(IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS))

    # create generators
    train_generator, val_generator = create_generators()
    train_generator.fit(train_X)
    val_generator.fit(val_X)

    # create the model
    model = create_model()

    # create the callbacks
    callbacks = create_callbacks()

    if os.path.isfile('./snapshots/keras_unet.h5') is True:
        print('loading model...')
        model = load_model('./snapshots/keras_unet.h5')
    else:
        # train
        print('Starting training...')
        history = model.fit_generator(train_generator.flow(train_X, train_Y, batch_size=BATCH_SIZE),
                                      epochs=NUM_EPOCHS,
                                      steps_per_epoch=STEPS_PER_EPOCH,
                                      callbacks=callbacks,
                                      verbose=1,
                                      validation_data=val_generator.flow(
                                          val_X, val_Y, batch_size=BATCH_SIZE),
                                      validation_steps=STEPS_PER_EPOCH * 0.2)

        print('Finished training...')
        # save training history
        with open('history.pkl', 'wb') as history_file:
            pickle.dump(history.history, history_file)


if __name__ == '__main__':
    main()
