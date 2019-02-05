
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from keras.callbacks import TensorBoard
from keras.layers import Dense, Embedding, LSTM
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split

from Embedding_Utilities import Utilities

util = Utilities()
data = pd.read_csv('../../Sentiment Analysis Data/input/Processed_Sentiment Analysis Dataset 2.csv')


tokenizer = Tokenizer(num_words=util.MAX_NUM_WORDS)
tokenizer.fit_on_texts(data['SentimentText'].values)
X = tokenizer.texts_to_sequences(data['SentimentText'].values)
X = pad_sequences(X,maxlen=util.TWEET_LENGTH)

lstm_out = 196

word_index = tokenizer.word_index
embedding_matrix,nb_words = util.load_glove_model(word_index)

model = Sequential()
model.add(Embedding(nb_words, util.EMBEDDING_DIM,input_length = util.TWEET_LENGTH,
                    dropout=0.2,weights=[embedding_matrix],trainable=False))
model.add(LSTM(lstm_out, dropout_U=0.2, dropout_W=0.2, activation='relu'))
model.add(Dense(2,activation='softmax'))
model.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])
print(model.summary())

Y = pd.get_dummies(data['Sentiment']).values
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.20, random_state = 42)
print(X_train.shape,Y_train.shape)
print(X_test.shape,Y_test.shape)

batch_size = 256
tb = TensorBoard(log_dir='./lstm_glove', histogram_freq=0, batch_size=batch_size, write_graph=True, write_grads=False, write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)
hist = model.fit(X_train, Y_train, epochs = 8, batch_size=batch_size, verbose = 1, callbacks=[tb])
model.save('./model_LSTM_GloVe.h5')