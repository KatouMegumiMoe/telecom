from DataProcess import *
from XGBoostModel import *
from FeatureEngineering import *
from TimeCost import *


if __name__ == '__main__':

    mode = Const.VALID_MODE
    tc = TimeCost()

    dp = DataProcess()
    df, dft = dp.data_input(Const.TRAIN_FILE_NAME, Const.TEST_FILE_NAME, mode)
    tc.print_event()

    fe = FeatureEngineering()
    df, dft = fe.feature_process(df, dft, mode)
    tc.print_event()

    X_train, y_train, X_valid, y_valid, X_test = dp.get_split_data(df, dft, mode)
    tc.print_event()

    xgb = XGBoostModel(X_train, y_train, X_valid, y_valid, X_test, mode)
    result = xgb.train_model()
    tc.print_event()

    if not mode:
        dp.transform_index(result)

    # xgb.load_model()
