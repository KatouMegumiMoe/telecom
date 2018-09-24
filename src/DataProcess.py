import pandas as pd
from sklearn.model_selection import GroupKFold
from Constant import Const


class DataProcess:

    def __init__(self):
        self.df = None
        self.dft = None
        self.services = list()

    def data_input(self, train_file_name, test_file_name, mode):
        self.df = pd.read_csv(train_file_name)
        self.df = self.fill_data(self.df)
        self.df = self.label_process(self.df)

        if not mode:
            self.dft = pd.read_csv(test_file_name)
            self.dft = self.fill_data(self.dft)
        else:
            self.dft = None

        return self.df, self.dft

    @staticmethod
    def fill_data(data_frame):
        data_frame = data_frame[~data_frame.isin(['\N'])].fillna(0)
        data_frame = data_frame.reset_index(drop=True)
        # data_frame['contract_time'] = data_frame['contract_time'].replace(-1, 0)

        return data_frame

    def label_process(self, data_frame):
        self.services = data_frame.groupby(['current_service']).count().index.values
        for label_id in range(Const.CATEGORY_NUM):
            service = self.services[label_id]
            data_frame['current_service'] = data_frame['current_service'].replace(service, label_id)

        return data_frame

    @staticmethod
    def get_split_data(df_bin, dft_bin, mode):
        user_id = [user[0] for user in df_bin[['user_id']].values]
        group = GroupKFold(n_splits=10)

        for train_idx, valid_idx in group.split(df_bin.drop(['current_service'], axis=1),
                                                df_bin[['current_service']],
                                                groups=user_id):
            train = df_bin.iloc[train_idx]
            valid = df_bin.iloc[valid_idx]
            X_train = train.drop(['current_service'], axis=1)
            y_train = train[['current_service']]
            X_valid = valid.drop(['current_service'], axis=1)
            y_valid = valid[['current_service']]
            break

        if not mode:
            X_train = df_bin.drop(['current_service'], axis=1)
            y_train = df_bin[['current_service']]

        X_test = dft_bin

        return X_train, y_train, X_valid, y_valid, X_test

    def transform_index(self, result):
        for label_id in range(Const.CATEGORY_NUM):
            service = self.services[label_id]
            result['predict'] = result['predict'].replace(label_id, int(service))
        result = result[['user_id', 'predict']]
        result.to_csv(Const.SUBMISSION_FILE_NAME, index=False)
