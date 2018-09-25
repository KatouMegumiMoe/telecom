import pandas as pd
from Constant import Const


def process_gender(df):
    df[['gender']] = df[['gender']].replace('1', 1)
    df[['gender']] = df[['gender']].replace('2', 2)
    df[['gender']] = df[['gender']].replace('01', 1)
    df[['gender']] = df[['gender']].replace('02', 2)
    df[['gender']] = df[['gender']].replace('00', 0)
    df[['gender']] = df[['gender']].replace('0', 0)

    print df[['gender']].drop_duplicates(keep='first').values.tolist()

    return df


def process_test_null(df):
    df[['2_total_fee']] = df[['2_total_fee']].replace('\N', Const.MISSING_NUM)
    df[['3_total_fee']] = df[['3_total_fee']].replace('\N', Const.MISSING_NUM)
    df[['gender']] = df[['gender']].replace('\N', Const.MISSING_NUM)
    df[['age']] = df[['age']].replace('\N', Const.MISSING_NUM)

    return df


def process_train_null(df):
    df = df[~(df['2_total_fee'].isin(['\N']) | df['3_total_fee'].isin(['\N']) | df['gender'].isin(['\N']) | df['age'].isin(['\N']))]
    df[['1_total_fee']] = df[['1_total_fee']].astype('float64')
    df[['2_total_fee']] = df[['2_total_fee']].astype('float64')
    df[['3_total_fee']] = df[['3_total_fee']].astype('float64')
    df[['4_total_fee']] = df[['4_total_fee']].astype('float64')
    return df


def save_data(df, file_name):
    df.to_csv(file_name, index=False)


if __name__ == '__main__':
    train = pd.read_csv(Const.TRAIN_FILE_NAME_ORIGIN)
    test = pd.read_csv(Const.TEST_FILE_NAME_ORIGIN)
    print train.shape
    train = process_train_null(train)
    print train.shape
    train = process_gender(train)
    save_data(train, Const.TRAIN_FILE_NAME)

    test = process_test_null(test)
    test = process_gender(test)
    save_data(test, Const.TEST_FILE_NAME)


