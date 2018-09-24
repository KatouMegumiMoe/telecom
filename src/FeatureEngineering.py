import pandas as pd


class FeatureEngineering:

    def __init__(self, mode):
        self.df = None
        self.dft = None
        self.mode = mode

    def feature_process(self, df, dft):
        self.df = FeatureEngineering.change_by_row(df)
        self.df = FeatureEngineering.remove_feature(self.df)

        if not self.mode:
            self.dft = FeatureEngineering.change_by_row(dft)
            self.dft = FeatureEngineering.remove_feature(self.dft)
        else:
            self.dft = dft

        return self.df, self.dft

    @staticmethod
    def change_by_row(df):
        min_fee = list()
        max_fee = list()
        approx_fee_min = list()
        approx_fee_max = list()
        max_traffic = list()
        sum_traffic = list()
        max_call_time = list()
        max_local_and_call = list()

        for row in range(df.shape[0]):
            min_fee_item = min(df.at[row, '1_total_fee'],
                               df.at[row, '2_total_fee'],
                               df.at[row, '3_total_fee'],
                               df.at[row, '4_total_fee'])
            max_fee_item = max(df.at[row, '1_total_fee'],
                               df.at[row, '2_total_fee'],
                               df.at[row, '3_total_fee'],
                               df.at[row, '4_total_fee'])
            min_fee.append(min_fee_item)
            max_fee.append(max_fee_item)
            sum_traffic.append(df.at[row, 'local_trafffic_month'] + df.at[row, 'last_month_traffic'])

            if df.at[row, 'local_trafffic_month'] > df.at[row, 'last_month_traffic']:
                max_traffic.append(df.at[row, 'local_trafffic_month'])
            else:
                max_traffic.append(df.at[row, 'last_month_traffic'])

            if df.at[row, 'service1_caller_time'] > df.at[row, 'service2_caller_time']:
                max_call_time.append(df.at[row, 'service1_caller_time'])
                max_local_and_call.append(df.at[row, 'service1_caller_time'] + df.at[row, 'local_caller_time'])
            else:
                max_call_time.append(df.at[row, 'service2_caller_time'])
                max_local_and_call.append(df.at[row, 'service2_caller_time'] + df.at[row, 'local_caller_time'])

            if df.at[row, 'many_over_bill'] == 1:
                approx_fee_min.append(0)
                approx_fee_max.append(min_fee_item)
            else:
                approx_fee_min.append(max_fee_item)
                approx_fee_max.append(9999)

        df['min_fee'] = min_fee
        df['max_fee'] = max_fee
        df['approx_fee_min'] = approx_fee_min
        df['approx_fee_max'] = approx_fee_max
        df['max_traffic'] = max_traffic
        df['sum_traffic'] = sum_traffic
        df['max_call_time'] = max_call_time
        df['max_local_and_call'] = max_local_and_call

        return df

    @staticmethod
    def remove_feature(df):

        df = df.drop(['former_complaint_num',
                      'pay_times',
                      'complaint_level',
                      'former_complaint_fee'], axis=1)

        return df
