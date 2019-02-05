from keras import models
from Embedding_Utilities import load_data
from os.path import join
import pandas as pd
from numpy import argmax

model_name = "TL_BI_cuDNNLSTM_L2_N196_GloVe_5f_final/5.02-0.24-0.27.hdf5"
model_path = join('models/',model_name)
util = load_data(model_name)

replies_data = pd.read_csv('../Sentiment Analysis Data/Collected Dataset/Replies_WorldCupDraw_Cleansed.csv')

replies_tweet = util.convert_data(replies_data.text.values)

model = models.load_model(model_path)

batch_size = 256

result = model.predict(replies_tweet, batch_size=batch_size, verbose=1)

prediction = []

total_count = replies_data.text.size
neg_cnt, pos_cnt = 0,0
for tweet_number in range(replies_data.text.size):
    prediction.append(argmax(result[tweet_number]))
    if argmax(result[tweet_number]) == 0:
        neg_cnt += 1
    else:
        pos_cnt += 1

pos_per = (pos_cnt/total_count)
neg_per = (neg_cnt/total_count)
print("#Replies %d, #Positive Replies: %d, #Negative Replies: %d\n"
      "Per Positive Replies: %.2f , Per Negative Replies: %.2f"%(total_count,pos_cnt,neg_cnt,pos_per,neg_per))

replies_data['Prediction'] = prediction

replies_data.to_csv(path_or_buf='Replies_WorldCupDraw_Cleansed_Finalized_2.csv',index=False,encoding='utf-8')