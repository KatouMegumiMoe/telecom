from DataProcess import *
from XGBoostModel import *
from FeatureEngineering import *
from TimeCost import *


if __name__ == '__main__':
    tc = TimeCost()

    dp = DataProcess()
    df, dft = dp.data_input(Const.TRAIN_FILE_NAME, Const.TEST_FILE_NAME)
    tc.print_event()

    df_bin, dft_bin = dp.transform_to_binary(df, dft)
    tc.print_event()

    fe = FeatureEngineering()
    df_bin, dft_bin = fe.feature_process(df_bin, dft_bin)
    tc.print_event()

    X_train, y_train, X_valid, y_valid, X_test = dp.get_split_data(df_bin, dft_bin)
    tc.print_event()

    xgb = XGBoostModel(X_train, y_train, X_valid, y_valid, X_test)
    result = xgb.train_model()
    tc.print_event()

    dp.sort_index(dft, result)
    # xgb.load_model()
