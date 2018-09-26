import pandas as pd
import numpy as np
from Constant import Const


def process_null(df):
    df[['gender']] = pd.to_numeric(df.gender, errors='coerce')
    df[['age']] = pd.to_numeric(df.age, errors='coerce')
    df[['fee_1_month']] = pd.to_numeric(df.fee_1_month, errors='coerce')
    df[['fee_2_month']] = pd.to_numeric(df.fee_2_month, errors='coerce')
    df[['fee_3_month']] = pd.to_numeric(df.fee_3_month, errors='coerce')
    df[['fee_4_month']] = pd.to_numeric(df.fee_4_month, errors='coerce')

    df = df.replace(np.nan, Const.MISSING_NUM)
    return df


def save_data(df, file_name):
    df.to_csv(file_name, index=False)


if __name__ == '__main__':
    train = pd.read_csv(Const.TRAIN_FILE_NAME_ORIGIN)
    test = pd.read_csv(Const.TEST_FILE_NAME_ORIGIN)

    train.columns = ['service_type',
                     'is_mix_service',
                     'online_time',
                     'fee_1_month',
                     'fee_2_month',
                     'fee_3_month',
                     'fee_4_month',
                     'traffic_0_month',
                     'is_over_fee',
                     'contract_type',
                     'contract_time',
                     'is_promise_low_consume',
                     'net_service',
                     'pay_times',
                     'pay_num',
                     'traffic_1_month',
                     'traffic_local_0_month',
                     'call_local',
                     'call_service_1_month',
                     'call_service_2_month',
                     'gender',
                     'age',
                     'complaint_level',
                     'complaint_former_num',
                     'complaint_former_fee',
                     'current_service',
                     'user_id']
    test.columns = ['service_type',
                    'is_mix_service',
                    'online_time',
                    'fee_1_month',
                    'fee_2_month',
                    'fee_3_month',
                    'fee_4_month',
                    'traffic_0_month',
                    'is_over_fee',
                    'contract_type',
                    'contract_time',
                    'is_promise_low_consume',
                    'net_service',
                    'pay_times',
                    'pay_num',
                    'traffic_1_month',
                    'traffic_local_0_month',
                    'call_local',
                    'call_service_1_month',
                    'call_service_2_month',
                    'gender',
                    'age',
                    'complaint_level',
                    'complaint_former_num',
                    'complaint_former_fee',
                    'user_id']

    train = process_null(train)
    save_data(train, Const.TRAIN_FILE_NAME)
    test = process_null(test)
    save_data(test, Const.TEST_FILE_NAME)


