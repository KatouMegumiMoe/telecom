import pandas as pd
from sklearn.model_selection import GroupKFold
from Constant import Const


class DataProcess:

    def __init__(self, train_file_name, test_file_name, df=None, dft=None):
        self.train_file_name = train_file_name
        self.test_file_name = test_file_name
        self.df = df
        self.dft = dft

    def data_input(self):
        self.df = pd.read_csv(self.train_file_name)
        self.dft = pd.read_csv(self.test_file_name)
        return self.fill_nan()

    def fill_nan(self):

        self.df = self.df[~self.df.isin(['\N'])].fillna(0)
        self.df = self.df.reset_index(drop=True)
        self.dft = self.dft[~self.dft.isin(['\N'])].fillna(0)
        self.dft = self.dft.reset_index(drop=True)

        return self.df, self.dft

    @staticmethod
    def transform_to_binary(df, dft, df_binary=pd.DataFrame(), dft_binary=pd.DataFrame()):
        all_service = df.groupby(['current_service']).count().index.values
        service_list = list()
        user_list = list()
        label_list = list()

        for row in range(df.shape[0]):
            current_service = int(df.at[row, 'current_service'])
            user_id = df.at[row, 'user_id']

            for service_id in all_service:
                service_list.append(service_id)
                user_list.append(user_id)
                label_list.append(1) if service_id == current_service else label_list.append(0)

        df_binary['service_id'] = service_list
        df_binary['user_id'] = user_list
        df_binary['label'] = label_list

        df_train = pd.merge(df, df_binary, how='left', on='user_id')

        service_list = list()
        user_list = list()

        for row in range(dft.shape[0]):
            user_id = str(dft.at[row, 'user_id'])

            for service_id in all_service:
                service_list.append(service_id)
                user_list.append(user_id)

        dft_binary['service_id'] = service_list
        dft_binary['user_id'] = user_list

        df_test = pd.merge(dft, dft_binary, how='left', on='user_id')

        return df_train, df_test

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

    @staticmethod
    def sort_index(dft, df):
        index = [item for item in range(dft.shape[0])]
        dft['sort'] = index
        result = pd.merge(df, dft, how='left', on='user_id')
        result = result[['user_id', 'predict', 'sort']]
        result = result.sort_values(['sort'])
        result = result[['user_id', 'predict']]
        result.to_csv(Const.SUBMISSION_FILE_NAME, index=False)
