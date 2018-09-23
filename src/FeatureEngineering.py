

class FeatureEngineering:

    def __init__(self):
        self.df = None
        self.dft = None

    def feature_process(self, df, dft):
        self.df = FeatureEngineering.change_by_row(df)
        self.dft = FeatureEngineering.change_by_row(dft)

        self.df = FeatureEngineering.remove_feature(self.df)
        self.dft = FeatureEngineering.remove_feature(self.dft)

        return self.df, self.dft

    @staticmethod
    def change_by_row(df):
        min_fee = list()
        max_fee = list()
        avg_fee = list()

        for row in range(df.shape[0]):
            min_fee.append(min(df.at[row, '1_total_fee'],
                               df.at[row, '2_total_fee'],
                               df.at[row, '3_total_fee'],
                               df.at[row, '4_total_fee']))
            max_fee.append(max(df.at[row, '1_total_fee'],
                               df.at[row, '2_total_fee'],
                               df.at[row, '3_total_fee'],
                               df.at[row, '4_total_fee']))
            avg_fee.append((df.at[row, '1_total_fee'] +
                            df.at[row, '2_total_fee'] +
                            df.at[row, '3_total_fee'] +
                            df.at[row, '4_total_fee'])/4.0)

        df['min_fee'] = min_fee
        df['max_fee'] = max_fee
        df['avg_fee'] = avg_fee

        return df

    @staticmethod
    def remove_feature(df):
        df = df.drop(['complaint_level',
                      'former_complaint_num',
                      'former_complaint_fee',
                      'is_promise_low_consume',
                      'net_service'], axis=1)
        return df
