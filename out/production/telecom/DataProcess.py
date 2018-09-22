import pandas as pd
from sklearn.model_selection import GroupKFold


class DataProcess:

    def __init__(self, train_file_name, test_file_name, df=None, dft=None):
        self.train_file_name = train_file_name
        self.test_file_name = test_file_name
        self.df = df
        self.dft = dft

    def data_input(self):
        self.df = pd.read_csv(self.train_file_name)
        self.dft = pd.read_csv(self.test_file_name)
        print 'before processing train data shape:', self.df.shape
        print 'before processing test data shape:', self.dft.shape
        return self.fill_nan()

    def fill_nan(self):

        self.df = self.df[~self.df.isin(['\N'])].fillna(0)
        self.df = self.df.reset_index(drop=True)
        self.dft = self.dft[~self.dft.isin(['\N'])].fillna(0)
        self.dft = self.dft.reset_index(drop=True)

        print 'after processing train data shape:', self.df.shape
        print 'after processing test data shape:', self.dft.shape
        return self.df, self.dft

    @staticmethod
    def get_split_data(df_bin, dft_bin):
        user_id = [user[0] for user in df_bin[['user_id']].values]
        group = GroupKFold(n_splits=5)

        for train_idx, valid_idx in group.split(df_bin.drop(['label'], axis=1), df_bin[['label']], groups=user_id):
            train = df_bin.iloc[train_idx]
            valid = df_bin.iloc[valid_idx]
            X_train = train.drop(['label'], axis=1)
            y_train = train[['label']]
            X_valid = valid.drop(['label'], axis=1)
            y_valid = valid[['label']]
            break

        X_test = dft_bin

        return X_train, y_train, X_valid, y_valid, X_test
