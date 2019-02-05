# from keras.wrappers.scikit_learn import KerasRegressor
from keras.callbacks import ModelCheckpoint, TensorBoard, CSVLogger
from keras.layers import Embedding, Dense, CuDNNLSTM,CuDNNGRU, Bidirectional
from keras.models import Sequential
# import numpy as np
from sklearn.model_selection import train_test_split

from Embedding_Utilities import load_data


# def lstm_model(util):
#     lstm_out = 196
#     model = Sequential()
#     model.add(Embedding(util.NB_WORDS, util.EMBEDDING_DIM, input_length=util.TWEET_LENGTH,
#                         weights=[util.embedding_matrix], trainable=False))
#     model.add(CuDNNLSTM(lstm_out, return_sequences=True))
#     model.add(CuDNNLSTM(lstm_out, go_backwards=True))
#     model.add(Dense(1))
#     model.compile(loss='mean_squared_error', optimizer='nadam', metrics=['mae'])
#     return model

def gru_model(util):
    gru_out = 196
    model = Sequential()
    model.add(Embedding(util.NB_WORDS, util.EMBEDDING_DIM, input_length=util.TWEET_LENGTH,
                        weights=[util.embedding_matrix], trainable=False))
    model.add(Bidirectional(CuDNNGRU(gru_out, return_sequences=True)))
    model.add(Bidirectional(CuDNNGRU(gru_out, go_backwards=True)))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='nadam', metrics=['mae'])
    return model

# X_validate = X_test[-validation_size:]
# Y_validate = Y_test[-validation_size:]
# X_test = X_test[:-validation_size]
# Y_test = Y_test[:-validation_size]

# model_name = 'LR_TL_cuDnnLSTM_N196_GloVe'
#
# util = load_data(model_name, data_source='../Sentiment Analysis Data/Collected Dataset/'
#                                          'WorldCupDraw Preprocessed Dataset wo Dups.csv', data_label='text',
#                  target_label='positive_percentage')
# X = util.X
# Y = util.Y
#
# model = lstm_model(util)
# # model = load_model('models/LR_TL_BI_cuDnnLSTM_N196_GloVe_e20/10-59.01-134.28.hdf5')
# csv_logger = CSVLogger(util.LOG_DIR, append=True)
# checkpoint_name = util.MODEL_DIR + '/{epoch:02d}-{loss:.2f}-{val_loss:.2f}.hdf5'
# checkpoint = ModelCheckpoint(checkpoint_name, monitor='val_loss',
#                              verbose=0, save_best_only=True,
#                              save_weights_only=False, mode='auto', period=1)
# tb = TensorBoard(log_dir=util.SUMMARY_DIR, histogram_freq=0.2, batch_size=batch_size, write_graph=True,
#                  write_grads=False, write_images=False, embeddings_freq=0, embeddings_layer_names=None,
#                  embeddings_metadata=None)
# model.fit(X_train, Y_train, batch_size=batch_size,
#           epochs=100, verbose=0, callbacks=[csv_logger, tb, checkpoint], validation_data=(X_test, Y_test))
# print(X_test,Y_test)

# score = model.evaluate(X_test, Y_test, verbose=1, batch_size=batch_size)

# print(score)

model_name = 'LR_TL_BI_cuDnnGRU_N196_GloVe'

util = load_data(model_name, data_source='../Sentiment Analysis Data/Collected Dataset/'
                                         'WorldCupDraw Preprocessed Dataset wo Dups.csv', data_label='text',
                 target_label='positive_percentage')
X = util.X
Y = util.Y
batch_size = 256
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, random_state=42)
validation_size = int(len(X_test) * 0.20)

model = gru_model(util)
# model = load_model('models/LR_TL_BI_cuDnnLSTM_N196_GloVe_e20/10-59.01-134.28.hdf5')
csv_logger = CSVLogger(util.LOG_DIR, append=True)
checkpoint_name = util.MODEL_DIR + '/{epoch:02d}-{loss:.2f}-{val_loss:.2f}.hdf5'
checkpoint = ModelCheckpoint(checkpoint_name, monitor='val_loss',
                             verbose=0, save_best_only=True,
                             save_weights_only=False, mode='auto', period=1)
tb = TensorBoard(log_dir=util.SUMMARY_DIR, histogram_freq=0.2, batch_size=batch_size, write_graph=True,
                 write_grads=False, write_images=False, embeddings_freq=0, embeddings_layer_names=None,
                 embeddings_metadata=None)
model.fit(X_train, Y_train, batch_size=batch_size,
          epochs=100, verbose=0, callbacks=[csv_logger, tb, checkpoint], validation_data=(X_test, Y_test))
# print(X_test,Y_test)

# score = model.evaluate(X_test, Y_test, verbose=1, batch_size=batch_size)

# print(score)
