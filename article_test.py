import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Input, SimpleRNN, LSTM, Bidirectional, Dense, GRU, Embedding, Dropout
from tensorflow.keras.models import Sequential 

def preprocess(text):
	MAX_VOCAB = 30
	tokenizer = Tokenizer(num_words=MAX_VOCAB, split=' ')
	tokenizer.fit_on_texts(text)
	X = tokenizer.texts_to_sequences(text)
	X = pad_sequences(X)
	return X

def GRU_model(X):
	model = Sequential()
	model.add(Embedding(2500,128,input_length=X.shape[1]))
	model.add(GRU((64), return_sequences=True))
	model.add(GRU((32)))
	# model.add(Dropout(0.2))
	model.add(Dense(8,activation='relu'))
	model.add(Dense(5,activation='softmax'))
	model.compile(loss=tf.keras.losses.categorical_crossentropy,optimizer='adam',metrics=['accuracy'])
	return model

LABEL_NAMES = ['wired', 'github', 'google source', 'medium', 'nytimes']

text = input('\nEnter the article\'s title: ')

if text:

	X = preprocess(text)

	model = GRU_model(X)

	model.load_weights('models/article.h5')

	predictions = model.predict(X)

	label = LABEL_NAMES[int(np.min(predictions))]

	score = np.max(predictions)

	print('\nHighest Probable Publisher: {} | Confidence Score: {:.2f}'.format(label, score))