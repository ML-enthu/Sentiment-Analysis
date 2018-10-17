# CNN for the IMDB problem
import numpy
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten, GRU
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.callbacks import EarlyStopping, ModelCheckpoint

import csv

import pandas as pd
# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)


from keras.preprocessing.text import Tokenizer

max_words = 1000
batch_size = 32
epochs = 5
model_file_name = 'model.h5'
# load the dataset but only keep the top n words, zero the rest
top_words = 5000
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=top_words)
# pad dataset to a maximum review length in words
max_words = 500

X_train = sequence.pad_sequences(X_train, maxlen=max_words)
X_test = sequence.pad_sequences(X_test, maxlen=max_words)

model = Sequential()
model.add(Embedding(top_words, 32, input_length=max_words))
model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(250, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

earlyStopping = EarlyStopping(monitor='val_loss', patience=2, verbose=0, mode='min')
mcp_save = ModelCheckpoint('model.h5', save_best_only=True, verbose=1, monitor='val_loss', mode='min')
callbacks = [earlyStopping,mcp_save]


# Fit the model
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=2, batch_size=128, callbacks = callbacks, verbose=1)
# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=1)
print("Accuracy: %.2f%%" % (scores[1]*100))

predictions = model.predict_classes(X_test, verbose=0)
df = pd.DataFrame()  
df['id'] = range(len(X_test))
df['sentiment'] = predictions
#submissions=pd.DataFrame({"Id": list(range(1,len(predictions)+1)),"Label": predictions})
df.to_csv("submissions.csv", index=False, header=True)
