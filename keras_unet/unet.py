from keras.layers import Input, Conv2D, Lambda, MaxPooling2D, Conv2DTranspose, concatenate, Dropout
from keras.models import Model

def unet(input):
	inputs = Input((input[0], input[1], input[2])) # height, width, channels

	conv_1 = Conv2D(16, (3, 3), activation='relu', padding='same')(inputs)
	conv_1 = Conv2D(16, (3, 3), activation='relu', padding='same')(conv_1)
	pool_1 = MaxPooling2D((2, 2))(conv_1)

	conv_2 = Conv2D(32, (3, 3), activation='relu', padding='same')(pool_1)
	conv_2 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv_2)
	pool_2 = MaxPooling2D((2, 2))(conv_2)

	conv_3 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool_2)
	conv_3 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv_3)
	pool_3 = MaxPooling2D((2, 2))(conv_3)

	conv_4 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool_3)
	conv_4 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv_4)
	pool_4 = MaxPooling2D((2, 2))(conv_4)

	conv_5 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool_4)
	conv_5 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv_5)

	dropout = Dropout(0.2)(conv_5)

	up_6 = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(dropout)
	up_6 = concatenate([up_6, conv_4], axis=3)
	conv_6 = Conv2D(128, (3, 3), activation='relu', padding='same')(up_6)
	conv_6 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv_6)

	up_7 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv_6)
	up_7 = concatenate([up_7, conv_3], axis=3)
	conv_7 = Conv2D(64, (3, 3), activation='relu', padding='same')(up_7)
	conv_7 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv_7)

	up_8 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv_7)
	up_8 = concatenate([up_8, conv_2], axis=3)
	conv_8 = Conv2D(32, (3, 3), activation='relu', padding='same')(up_8)
	conv_8 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv_8)

	up_9 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv_8)
	up_9 = concatenate([up_9, conv_1], axis=3)
	conv_9 = Conv2D(16, (3, 3), activation='relu', padding='same')(up_9)
	conv_9 = Conv2D(16, (3, 3), activation='relu', padding='same')(conv_9)

	# TODO: Drop-out layers at the end of the contracting path perform further implicit data augmentation.

	conv_10 = Conv2D(1, (1, 1), activation='sigmoid')(conv_9)

	return Model(inputs=[inputs], outputs=[conv_10])
	