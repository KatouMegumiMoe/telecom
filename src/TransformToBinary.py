import pandas as pd


class TransformToBinary:

    def __init__(self, df, df_binary=pd.DataFrame()):
        self.df = df
        self.df_binary = df_binary

    def transform_to_binary(self):
        all_service = self.df.groupby(['current_service']).count().index.values
        service_list = list()
        user_list = list()
        label_list = list()

        for row in range(self.df.shape[0]):
            current_service = int(self.df.at[row, 'current_service'])
            user_id = str(self.df.at[row, 'user_id'])

            for service_id in all_service:
                service_list.append(service_id)
                user_list.append(user_id)
                label_list.append(1) if service_id == current_service else label_list.append(0)

        self.df_binary['service_id'] = service_list
        self.df_binary['user_id'] = user_list
        self.df_binary['label'] = label_list

        return pd.merge(self.df, self.df_binary, how='left', on='user_id')


