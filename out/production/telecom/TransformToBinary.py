import pandas as pd


class TransformToBinary:

    def __init__(self, df, dft):
        self.df = df
        self.dft = dft

    def transform_to_binary(self):
        df_binary = pd.DataFrame()
        all_service = self.df.groupby(['current_service']).count().index.values
        service_list = list()
        user_list = list()
        label_list = list()

        for row in range(self.df.shape[0]):
            current_service = int(self.df.at[row, 'current_service'])
            user_id = self.df.at[row, 'user_id']

            for service_id in all_service:
                service_list.append(service_id)
                user_list.append(user_id)
                label_list.append(1) if service_id == current_service else label_list.append(0)

        df_binary['service_id'] = service_list
        df_binary['user_id'] = user_list
        df_binary['label'] = label_list

        df_train = pd.merge(self.df, df_binary, how='left', on='user_id')

        dft_binary = pd.DataFrame()
        service_list = list()
        user_list = list()

        for row in range(self.dft.shape[0]):
            user_id = str(self.dft.at[row, 'user_id'])

            for service_id in all_service:
                service_list.append(service_id)
                user_list.append(user_id)

        dft_binary['service_id'] = service_list
        dft_binary['user_id'] = user_list

        df_test = pd.merge(self.dft, dft_binary, how='left', on='user_id')

        return df_train, df_test




