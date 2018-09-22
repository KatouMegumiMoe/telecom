import pandas as pd
from sklearn.model_selection import GroupKFold


class DataProcess:

    def __init__(self, file_name, df=None):
        self.file_name = file_name
        self.df = df

    def data_input(self):
        self.df = pd.read_csv(self.file_name)
        print 'before processing data shape:', self.df.shape
        return self.fill_nan()

    def fill_nan(self):
        self.df = self.df[~self.df.isin(['\N'])].dropna(axis=0, how='any')
        self.df = self.df.reset_index(drop=True)
        print 'after processing data shape:', self.df.shape
        return self.df

    @staticmethod
    def get_valid_data(df_bin):
        user_id = df_bin[['user_id']].values.tolist()
        group = GroupKFold(n_splits=5)

        for train_idx, test_idx in group.split(df_bin.drop(['label'], axis=1), df_bin[['label']], groups=user_id):
            train = df_bin.iloc[train_idx]
            test = df_bin.iloc[test_idx]
            X_train = train.drop(['label'], axis=1)
            y_train = train[['label']]
            X_test = test.drop(['label'], axis=1)
            y_test = test[['label']]
            break

        return X_train, y_train, X_test, y_test

    @staticmethod
    def get_test_data(train, test):
        X_train = train.drop(['label'], axis=1)
        y_train = train[['label']]
        X_test = test.drop(['label'], axis=1)
        y_test = test[['label']]

        return X_train, y_train, X_test, y_test

