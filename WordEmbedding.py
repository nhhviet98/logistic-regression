from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from gensim.models import Word2Vec
from tqdm import tqdm
import time
import numpy as np


class WordEmbedding():
    def __init__(self):
        self.stop_words = stopwords.words('english')
        self.word2vec = None
        self.all_words = None
        self.tokenizer = None
        self.mean = None
        self.std = None

    @staticmethod
    def _stemming(data: list) -> list:
        '''
        Stemming for training set
        :param data: list
            list of sentences in training set
        :return: list
            all words after stemming
        '''
        poster_stemmer = PorterStemmer()
        all_words = [[poster_stemmer.stem(word) for word in sentence] for sentence in data]
        return all_words

    def norm_data(self, x: list):
        '''
        Normalize data with mean and std
        :param x: list
            Input data
        :return: np.ndarray
            Data after normalize
        '''
        self.mean = np.mean(x)
        self.std = np.std(x)
        return (x - self.mean) / self.std

    def norm_data_test(self, x_test: list) -> np.ndarray:
        '''
        Normalize data of test set using mean and std of training set
        :param x_test: list
            Input data
        :return: np.ndarray
            Test set after normalize
        '''
        return (x_test - self.mean) / self.std

    def _vectorized(self, data_list: list) -> list:
        '''
        tokenized for sentences in data set
        :param data_list: list
            list of data in training set
        :return: list
            all words after tokenized
        '''
        tokenizer = RegexpTokenizer(r'[a-zA-Z0-9]+')
        poster_stemmer = PorterStemmer()
        self.all_words = []
        for data in tqdm(data_list):
            data_tokenized = data.lower()
            data_tokenized = tokenizer.tokenize(data_tokenized)
            data_tokenized = [poster_stemmer.stem(word) for word in data_tokenized
                              if word not in self.stop_words and len(word) > 2]
            self.all_words.append(data_tokenized)
        return self.all_words

    def _word2vec(self, data_list: list):
        '''
        Word to vec for training set using gensim library
        :param data_list: list
            list of sentences in training set
        :return: None
        '''
        self.all_words = self._vectorized(data_list)
        t1 = time.time()
        self.word2vec = Word2Vec(self.all_words, min_count=5, size=300,
                                 window=30, workers=10, iter=30)
        t2 = time.time()
        print("time word2vec", t2 - t1)
        for i, data in tqdm(enumerate(self.all_words)):
            data_tokenized = [word for word in data if word in self.word2vec.wv.index2word]
            self.all_words[i] = data_tokenized
        print('yeah yeah')

    def doc2vec(self, data_list: list) -> np.ndarray:
        '''
        transform sentences to vector using Word2Vec
        :param data_list: list
            List of sentences
        :return: np.ndarray
            Vector of all sentences in training set
        '''
        self._word2vec(data_list)
        param_vec = []
        for param in self.all_words:
            param_vec.append(np.sum(self.word2vec.wv[param], axis=0)/len(param))
        param_vec = self.norm_data(param_vec)
        return param_vec

    def fit_transform(self, x_test):
        x_test_embedded = []
        tokenizer = RegexpTokenizer(r'[a-zA-Z0-9]+')
        poster_stemmer = PorterStemmer()
        for data in tqdm(x_test):
            data_tokenized = data.lower()
            data_tokenized = tokenizer.tokenize(data_tokenized)
            data_tokenized = [poster_stemmer.stem(word) for word in data_tokenized
                              if word not in self.stop_words and len(word) > 2]
            data_tokenized = [word for word in data_tokenized if word in self.word2vec.wv.index2word]
            x_test_embedded.append(np.sum(self.word2vec.wv[data_tokenized], axis=0)/len(data_tokenized))
        return x_test_embedded
