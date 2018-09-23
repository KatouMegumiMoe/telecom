from DataProcess import *
from XGBoostModel import *


if __name__ == '__main__':
    dp = DataProcess(Const.TRAIN_FILE_NAME, Const.TEST_FILE_NAME)
    df, dft = dp.data_input()
    df_bin, dft_bin = dp.transform_to_binary(df, dft)
    X_train, y_train, X_valid, y_valid, X_test = dp.get_split_data(df_bin, dft_bin)

    xgb = XGBoostModel(X_train, y_train, X_valid, y_valid, X_test)
    # result = xgb.train_model()
    # dp.sort_index(dft, result)
    xgb.load_model()
