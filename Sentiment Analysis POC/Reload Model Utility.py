import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from keras.callbacks import TensorBoard, CSVLogger
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split

from Embedding_Utilities import Utilities

util = Utilities()
data = pd.read_csv('../../Sentiment Analysis Data/input/Processed_Sentiment Analysis Dataset 2.csv')
data_labels = {1:"Postive",0:"Negative"}

tokenizer = Tokenizer(num_words=util.MAX_NUM_WORDS)
tokenizer.fit_on_texts(data['SentimentText'].values)
X = tokenizer.texts_to_sequences(data['SentimentText'].values)
X = pad_sequences(X,maxlen=util.TWEET_LENGTH)
Y = pd.get_dummies(data['Sentiment']).values
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.20, random_state = 42)
print(X_train.shape,Y_train.shape)
print(X_test.shape,Y_test.shape)
validation_size = 1500

X_validate = X_test[-validation_size:]
Y_validate = Y_test[-validation_size:]
X_test = X_test[:-validation_size]
Y_test = Y_test[:-validation_size]

batch_size = 256
lstm_out = 196

word_index = tokenizer.word_index
embedding_matrix,nb_words = util.load_glove_model(word_index)

model = load_model('./models/model_TL_GRU_GloVe.h5')
csv_logger = CSVLogger('./logs/model_TL_GRU_GloVe_16.csv')
tb = TensorBoard(log_dir='./summary/TL_cuDnnGRU_glove_e10', histogram_freq=1, batch_size=batch_size,
                 write_graph=True, write_grads=False, write_images=False, embeddings_freq=0,
                 embeddings_layer_names=None, embeddings_metadata=None)
hist = model.fit(X_train, Y_train, epochs = 8, batch_size=batch_size, verbose = 1,
                 callbacks=[tb,csv_logger],validation_data=(X_validate,Y_validate))

model.save('./models/model_TL_GRU_GloVe_e16.h5')

score,acc = model.evaluate(X_test, Y_test, verbose = 1, batch_size = batch_size)
print("score: %.2f" % (score))
print("acc: %.2f" % (acc))

pos_cnt, neg_cnt, pos_correct, neg_correct = 0, 0, 0, 0
for x in range(len(X_validate)):

    result = model.predict(X_validate[x].reshape(1, X_test.shape[1]), batch_size=1, verbose=2)[0]

    if np.argmax(result) == np.argmax(Y_validate[x]):
        if np.argmax(Y_validate[x]) == 1:
            neg_correct += 1
        else:
            pos_correct += 1

    if np.argmax(Y_validate[x]) == 0:
        neg_cnt += 1
    else:
        pos_cnt += 1

print("pos_acc", pos_correct / pos_cnt * 100, "%")
print("neg_acc", neg_correct / neg_cnt * 100, "%")