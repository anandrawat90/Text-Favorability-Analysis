from os.path import join, exists, abspath
from os import makedirs
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import pandas as pd
import pickle


class Utilities:
    MODE = 'DEBUG'
    def __init__(self,data_dir=None,data_label='SentimentText',target_label='Sentiment'):
        # Constants for running the network
        self.MAX_NUM_WORDS = 20000  # only consider top 20000
        self.EMBEDDING_DIM = 200
        self.TWEET_LENGTH = 140  # max length of sequence
        self.VALIDATION_SPLIT = 0.20  # validation to split, 80% testing 20% validation
        self.MODEL_NAME = None
        self.MASTER_DIR = './'
        self.MODELS_DIR_BASE = join(self.MASTER_DIR, 'models/')
        self.LOG_DIR = join(self.MASTER_DIR, 'logs/')
        self.SUMMARY_DIR = join(self.MASTER_DIR, 'summary/')
        self.PROJECT_DIR = '../'
        data_source = '../Sentiment Analysis Data/Labeled Dataset/Processed_Sentiment Analysis Dataset 2.csv'
        if data_dir:
            data_source = data_dir
        self.DATA_SOURCE = data_source
        self.DATA_LABEL = data_label
        self.TARGET_LABEL = target_label
        self.MODEL_DIR = None
        self.WORD_INDEX = None
        self.X = None
        self.Y = None
        self.NUM_CLASSES = 2
        self.NB_WORDS = 0
        self.RANDOM_SEED = 42
        self.embedding_matrix = None
        self.word_dict = None
        self.load_data()

    def load_glove_model(self):
        if self.WORD_INDEX is None:
            print('Initialize Data First')
            exit(-1)
        if self.embedding_matrix is None:
            embedding_index = {}
            with open(join(self.PROJECT_DIR, 'Word Embedding/glove.twitter.27B.200d.txt'),
                      encoding='utf-8') as glove_embedding:
                for line in glove_embedding:
                    emb_data = line.split()
                    word = emb_data[0]
                    coefs = np.asarray(emb_data[1:], dtype='float32')
                    embedding_index[word] = coefs

            self.NB_WORDS = min(self.MAX_NUM_WORDS, len(self.WORD_INDEX))
            # use embedding_index and word_index to compute embedding_matrix
            self.embedding_matrix = np.zeros((self.NB_WORDS, self.EMBEDDING_DIM))
            word_not_in_embedding = []
            for word, i in self.WORD_INDEX.items():
                if i >= self.NB_WORDS:
                    continue
                embedding_vector = embedding_index.get(word)
                if embedding_vector is not None:
                    # words not found in embedding_index will be all zeros
                    self.embedding_matrix[i] = embedding_vector
                else:
                    # create a list of words not in embedding to statistical analysis
                    word_not_in_embedding.append((word, i, self.word_dict[word]))
            print('found %d number of words, %d numbers of words missing from embedding'%(len(self.embedding_matrix),len(word_not_in_embedding)))
            with open('words not found.csv','w') as dummy:
                dummy.write('word,index,rank\n')
                for word in word_not_in_embedding:
                    dummy.write(word[0]+","+str(word[1])+","+str(word[2])+'\n')

    def set_model_name(self, model_name):
        self.MODEL_NAME = model_name
        self.LOG_DIR = join(self.LOG_DIR, self.MODEL_NAME) + '.csv'
        self.SUMMARY_DIR = join(self.SUMMARY_DIR, self.MODEL_NAME)
        self.MODEL_DIR = join(self.MODELS_DIR_BASE, self.MODEL_NAME)
        if self.MODE != 'DEBUG':
            makedirs(self.MODEL_DIR)
            makedirs(self.SUMMARY_DIR)

    def load_data(self):
        data = pd.read_csv(self.DATA_SOURCE)
        tokenizer = Tokenizer(num_words=self.MAX_NUM_WORDS)
        tokenizer.fit_on_texts(data[self.DATA_LABEL].values)
        X = tokenizer.texts_to_sequences(data[self.DATA_LABEL].values)
        self.X = pad_sequences(X, maxlen=self.TWEET_LENGTH)
        self.Y = data[self.TARGET_LABEL]
        self.WORD_INDEX = tokenizer.word_index
        self.word_dict = tokenizer.word_counts
        self.load_glove_model()

    def convert_data(self,input_queries):
        tokenizer = Tokenizer(num_words=self.MAX_NUM_WORDS)
        tokenizer.fit_on_texts(input_queries)
        formatted_input = tokenizer.texts_to_sequences(input_queries)
        formatted_input = pad_sequences(formatted_input, maxlen=self.TWEET_LENGTH)
        return formatted_input


def load_data(model_name,data_source=None,data_label = '',target_label = ''):
    util = None
    preprocessed_object = 'preProcessedData.pkl'
    if data_source:
        preprocessed_object = data_source +"."+preprocessed_object
    if exists(preprocessed_object):
        with open(preprocessed_object, 'rb') as output:
            print('Using utility file:', abspath(preprocessed_object))
            util = pickle.load(output)
    else:
        if data_source:
            util = Utilities(data_source,data_label,target_label)
        else:
            util = Utilities()
        with open(preprocessed_object, 'wb') as output:
            pickle.dump(util, output, pickle.HIGHEST_PROTOCOL)
            print('Created new utility file:',abspath(preprocessed_object))
    util.set_model_name(model_name)
    return util

if __name__ == '__main__':
    util = load_data('testing',data_source='../Sentiment Analysis Data/Collected Dataset/WorldCupDraw Preprocessed Dataset.csv',data_label='text',target_label='positive_percentage')
    # util = load_data('testing')
    print(util.MODEL_DIR, util.LOG_DIR, util.SUMMARY_DIR, util.MODEL_NAME)
    print(len(util.WORD_INDEX), len(util.embedding_matrix))
    print(len(util.X),len(util.Y))
