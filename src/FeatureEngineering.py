

class FeatureEngineering:

    def __init__(self):
        self.df = None
        self.dft = None

    def feature_process(self, df, dft, mode):
        self.df = FeatureEngineering.change_by_row(df)
        self.df = FeatureEngineering.remove_feature(self.df)

        if not mode:
            self.dft = FeatureEngineering.change_by_row(dft)
            self.dft = FeatureEngineering.remove_feature(self.dft)
        else:
            self.dft = dft

        return self.df, self.dft

    @staticmethod
    def change_by_row(df):
        min_fee = list()
        # avg_pay = list()

        for row in range(df.shape[0]):
            min_fee.append(min(df.at[row, '1_total_fee'],
                               df.at[row, '2_total_fee'],
                               df.at[row, '3_total_fee'],
                               df.at[row, '4_total_fee']))
            # avg_pay.append(float(df.at[row, 'pay_num'])/float(df.at[row, 'pay_times']))
        df['min_fee'] = min_fee
        # df['avg_pay'] = avg_pay
        return df

    @staticmethod
    def remove_feature(df):
        return df
