import cv2
import numpy as np
import tensorflow as tf

LABEL_NAMES = ['t_shirt', 'trouser', 'pullover', 'dress', 'coat', 'sandal', 'shirt', 'sneaker', 'bag', 'ankle_boots']

img = input('Enter the path to image: ')

def init_CNN_model(image):
  model = tf.keras.models.Sequential()
  model.add(tf.keras.layers.BatchNormalization(input_shape=image.shape[1:]))
  model.add(tf.keras.layers.Conv2D(64, (5, 5), padding='same', activation='elu'))
  model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2,2)))

  model.add(tf.keras.layers.BatchNormalization(input_shape=image.shape[1:]))
  model.add(tf.keras.layers.Conv2D(128, (5, 5), padding='same', activation='elu'))
  model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

  model.add(tf.keras.layers.BatchNormalization(input_shape=image.shape[1:]))
  model.add(tf.keras.layers.Conv2D(256, (5, 5), padding='same', activation='elu'))
  model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2,2)))

  model.add(tf.keras.layers.Flatten())
  model.add(tf.keras.layers.Dense(256))
  model.add(tf.keras.layers.Activation('elu'))
  # model.add(tf.keras.layers.Dropout(0.5))
  model.add(tf.keras.layers.Dense(10))
  model.add(tf.keras.layers.Activation('softmax'))
  return model

if img:

	image = cv2.imread(img, cv2.IMREAD_GRAYSCALE)

	orig = cv2.imread(img, cv2.IMREAD_COLOR)

	image = cv2.resize(image, (28, 28))

	image = image.reshape(-1, 28, 28, 1)

	model = init_CNN_model(image)

	model.load_weights('models/fashion.h5')	

	prediction = model.predict(image)

	label = LABEL_NAMES[np.argmax(prediction)]

	new_score = np.max(prediction)

	text = '{} | confidence score = {:.2f}'.format(label, new_score)
	print('\n',text, '\n')
	cv2.putText(orig, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
	cv2.imshow("Fashion MNIST Classifier", orig)

	cv2.waitKey(0)

	cv2.destroyAllWindows()