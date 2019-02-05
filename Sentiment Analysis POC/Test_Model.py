from keras import models
from Embedding_Utilities import load_data
from os.path import join
from pandas import get_dummies
from numpy import asarray
from numpy import argmax

# model_name = "TL_BI_cuDNNLSTM_L2_N196_GloVe_3f_final/3.02-0.30-0.32.hdf5"
# model_path = join('models/',model_name)
# util = load_data(model_name)
#
# X = util.X
# Y = get_dummies(util.Y).values
# print(X[:5])
# print(Y[:5])
# model = models.load_model(model_path)
#
# batch_size = 256
#
# score,acc = model.evaluate(X, Y, verbose = 1, batch_size = batch_size)
# print("score: %.2f" % (score))
# print("acc: %.2f" % (acc))
#
# pos_cnt, neg_cnt, pos_correct, neg_correct = 0, 0, 0, 0
# for x in range(len(X)):
#
#     result = model.predict(X[x].reshape(1, X.shape[1]), batch_size=1, verbose=2)[0]
#
#     if argmax(result) == argmax(Y[x]):
#         if argmax(Y[x]) == 0:
#             neg_correct += 1
#         else:
#             pos_correct += 1
#
#     if argmax(Y[x]) == 0:
#         neg_cnt += 1
#     else:
#         pos_cnt += 1
#
# print("pos_acc", pos_correct / pos_cnt * 100, "%")
# print("neg_acc", neg_correct / neg_cnt * 100, "%")

model_to_load = "models/LR_TL_BI_cuDnnLSTM_N196_GloVe_e100/69-25.80-124.41.hdf5"
util = load_data('LR_TL_BI_cuDnnLSTM_N196_GloVe_e100', data_source='../Sentiment Analysis Data/Collected Dataset/'
                                         'WorldCupDraw Preprocessed Dataset wo Dups.csv', data_label='text',
                 target_label='positive_percentage')

X = asarray(['Germany vs. Mexico will be fun!','The moment England were drawn with Belgium...  #bbcfootball'])
X = util.convert_data(X)
model = models.load_model(model_to_load)
print(model.predict(X))