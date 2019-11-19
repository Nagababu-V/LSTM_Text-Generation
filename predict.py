import numpy
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
from keras.models import load_model
import sys


#Loading the data

filename = "data.txt"
raw_text = open(filename, 'r', encoding='utf-8').read()
raw_text = raw_text.lower()

chars = sorted(list(set(raw_text)))
char_to_int = dict((c, i) for i, c in enumerate(chars))
int_to_char =dict((i,c) for i ,c in enumerate(chars))

num_chars = len(raw_text)
num_vocab = len(chars)

seq_length = 100
dataX = []
dataY = []
for i in range(0, num_chars - seq_length, 1):
	seq_in = raw_text[i:i + seq_length]
	seq_out = raw_text[i + seq_length]
	dataX.append([char_to_int[char] for char in seq_in])
	dataY.append(char_to_int[seq_out])
num_patterns = len(dataX)


X = numpy.reshape(dataX, (num_patterns, seq_length, 1))
y = np_utils.to_categorical(dataY)

print(X[0].shape)
model = Sequential()
model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2])))
model.add(Dropout(0.2))
model.add(Dense(y.shape[1], activation='softmax'))
model.load_weights('Trained_text_MOdel.h5')
model.compile(loss='categorical_crossentropy', optimizer='adam')

start = numpy.random.randint(0, len(dataX)-1)
pattern = dataX[start]

# generate characters
for i in range(1000):
	x = numpy.reshape(pattern, (1, len(pattern), 1))
	x = x / float(num_vocab)
	prediction = model.predict(x, verbose=0)
	index = numpy.argmax(prediction)
	result = int_to_char[index]
	seq_in = [int_to_char[value] for value in pattern]
	sys.stdout.write(result)
	pattern.append(index)
	pattern = pattern[1:len(pattern)]
#
