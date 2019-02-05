
import numpy as np  # linear algebra
from keras.callbacks import TensorBoard, CSVLogger, ModelCheckpoint
from keras.layers import Dense, Embedding, CuDNNLSTM, Bidirectional
from keras.models import Sequential
from sklearn.model_selection import StratifiedKFold
from pandas import get_dummies

from Embedding_Utilities import load_data

util = load_data('TL_BI_cuDNNLSTM_L2_N196_GloVe_3f_final')


lstm_out = 196

model = Sequential()
model.add(Embedding(util.NB_WORDS, util.EMBEDDING_DIM,input_length = util.TWEET_LENGTH,
                    dropout=0.2,weights=[util.embedding_matrix], trainable=False))
model.add(Bidirectional(CuDNNLSTM(lstm_out, return_sequences=True)))
model.add(Bidirectional(CuDNNLSTM(lstm_out, go_backwards=True)))
model.add(Dense(util.NUM_CLASSES,activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='nadam',metrics = ['accuracy'])
print(model.summary())
kfold = StratifiedKFold(n_splits=3,shuffle=True,random_state=util.RANDOM_SEED)
cvscores = []
X = util.X
Y = util.Y
Data_X = X
Data_Y = Y
batch_size = 256
i=0
for train,test in kfold.split(X,Y):
    fold = str(i+1)
    csv_logger = CSVLogger(util.LOG_DIR, append=True)
    foldY_train = get_dummies(Y[train]).values
    foldY_test = get_dummies(Y[test]).values
    checkpoint_name = util.MODEL_DIR + '/' + fold + '.{epoch:02d}-{loss:.2f}-{val_loss:.2f}.hdf5'
    checkpoint = ModelCheckpoint(checkpoint_name, monitor='val_loss',
                                 verbose=0, save_best_only=False,
                                 save_weights_only=False, mode='auto', period=1)
    tb = TensorBoard(log_dir=util.SUMMARY_DIR+fold, histogram_freq=1, batch_size=batch_size, write_graph=True,
                     write_grads=False,write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)
    hist = model.fit(X[train],foldY_train,batch_size=batch_size,
                     epochs=2,verbose=1,callbacks=[csv_logger,tb,checkpoint],validation_data=(X[test],foldY_test))
    i += 1
Data_Y = get_dummies(Data_Y).values
score,acc = model.evaluate(Data_X, Data_Y, verbose = 1, batch_size = batch_size)
print("score: %.2f" % (score))
print("acc: %.2f" % (acc))
