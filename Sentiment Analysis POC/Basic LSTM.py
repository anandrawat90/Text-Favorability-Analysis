import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from keras.callbacks import TensorBoard, CSVLogger, ModelCheckpoint
from keras.layers import Dense, Embedding, LSTM
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
from Embedding_Utilities import load_data

# data = pd.read_csv('../Sentiment Analysis Data/Labeled Dataset/Processed_Sentiment Analysis Dataset 2.csv')
# data = data[['SentimentText','Sentiment']]

model_name = 'basic_GloVe_lstm_model'
# max_features = 20000
# tokenizer = Tokenizer(num_words=max_features)
# tokenizer.fit_on_texts(data['SentimentText'].values)
# X = tokenizer.texts_to_sequences(data['SentimentText'].values)
# X = pad_sequences(X,maxlen=140)
# Y = pd.get_dummies(data['Sentiment']).values
util = load_data(model_name)
X = util.X
Y = pd.get_dummies(util.Y).values
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.20, random_state = util.RANDOM_SEED)
validation_size = 1500

X_validate = X_test[-validation_size:]
Y_validate = Y_test[-validation_size:]
X_test = X_test[:-validation_size]
Y_test = Y_test[:-validation_size]
batch_size = 256


embed_dim = 200
lstm_out = 196

model = Sequential()
model.add(Embedding(util.NB_WORDS, util.EMBEDDING_DIM,input_length = util.TWEET_LENGTH, dropout=0.2))
model.add(LSTM(lstm_out, dropout_U=0.2, dropout_W=0.2))
model.add(Dense(2,activation='softmax'))
model.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])
print(model.summary())

csv_logger = CSVLogger('logs/'+model_name+'.csv', append=False)
checkpoint_name = 'models/'+model_name+ '/{epoch:02d}-{loss:.2f}-{val_loss:.2f}.hdf5'
checkpoint = ModelCheckpoint(checkpoint_name, monitor='val_loss',
                             verbose=0, save_best_only=True,
                             save_weights_only=False, mode='auto', period=1)
tb = TensorBoard(log_dir='summary/'+model_name, histogram_freq=1, batch_size=batch_size, write_graph=True,
                 write_grads=False, write_images=False, embeddings_freq=0, embeddings_layer_names=None,
                 embeddings_metadata=None)
model.fit(X_train, Y_train, batch_size=batch_size,
          epochs=8, verbose=1, callbacks=[csv_logger, tb, checkpoint], validation_data=(X_validate, Y_validate))
#
#
# score,acc = model.evaluate(X_test, Y_test, verbose = 0, batch_size = batch_size)
# print("score: %.2f" % (score))
# print("acc: %.2f" % (acc))
#
# pos_cnt, neg_cnt, pos_correct, neg_correct = 0, 0, 0, 0
# for x in range(len(X_validate)):
#
#     result = model.predict(X_validate[x].reshape(1, X_test.shape[1]), batch_size=1, verbose=2)[0]
#
#     if np.argmax(result) == np.argmax(Y_validate[x]):
#         if np.argmax(Y_validate[x]) == 0:
#             neg_correct += 1
#         else:
#             pos_correct += 1
#
#     if np.argmax(Y_validate[x]) == 0:
#         neg_cnt += 1
#     else:
#         pos_cnt += 1
#
# print("pos_acc", pos_correct / pos_cnt * 100, "%")
# print("neg_acc", neg_correct / neg_cnt * 100, "%")
#
