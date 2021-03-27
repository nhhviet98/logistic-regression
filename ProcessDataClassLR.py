import pandas as pd


class ProcessData:
    def __init__(self):
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None

    def read_train_csv(self, train_file_path: str):
        '''
        read data from file path
        :param train_file_path: str
            path of training file
        :return: list
            Data of training set
        '''
        df_train = pd.read_csv(train_file_path)
        self.x_train = df_train['message'].to_list()[:]
        self.y_train = df_train['topic'].to_list()[:]
        return self.x_train, self.y_train

    def read_test_csv(self, test_file_path: str, labels_test_file_path: str):
        '''
        read test data from file path
        :param test_file_path: str
            path of input testing path
        :param labels_test_file_path: str
            path of output testing path
        :return: list
            Data of testing set
        '''
        df_test = pd.read_csv(test_file_path)
        df_label_test = pd.read_csv(labels_test_file_path)
        self.x_test = df_test['message'].to_list()[:]
        self.y_test = df_label_test['topic'].to_list()[:]
        return self.x_test, self.y_test

