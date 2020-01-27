import cv2
import numpy as np
import tensorflow as tf

def preprocess(image):
	image = (image)/127.5 - 1
	image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
	image = image.reshape(-1, IMG_SIZE, IMG_SIZE, 3)
	return image

def prepare_model(IMG_SHAPE):
	base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE, include_top=False, weights='imagenet')
	base_model.trainable = False
	global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
	prediction_layer = tf.keras.layers.Dense(1)
	model = tf.keras.Sequential([
		base_model,
		global_average_layer,
		prediction_layer
		])
	base_learning_rate = 0.0001
	model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=base_learning_rate), loss='binary_crossentropy',	metrics=['accuracy'])
	return model

LABEL_NAMES = ['dog', 'cat']

IMG_SIZE = 150

IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)

img = input('Enter the path to image: ')

if img:
	model = prepare_model(IMG_SHAPE)

	model.load_weights('models/cats_dogs.h5')

	image = cv2.imread(img, cv2.IMREAD_COLOR)

	orig = image

	image = preprocess(image)

	prediction = model.predict(image)

	if(np.max(prediction)>0):
		
		label = LABEL_NAMES[0]

	else:

		label = LABEL_NAMES[1]	

	new_score = np.max(prediction)

	text = '{} | confidence score = {:.2f}'.format(label, new_score)
	print('\n',text, '\n')
	cv2.putText(orig, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
	cv2.imshow("Cats vs Dogs", orig)

	cv2.waitKey(0)

	cv2.destroyAllWindows()