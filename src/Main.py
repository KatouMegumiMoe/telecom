from DataProcess import *
from XGBoostModel import *
from FeatureEngineering import *
from TimeCost import *


if __name__ == '__main__':

    mode = Const.VALID_MODE
    mode = Const.PREDICT_MODE
    tc = TimeCost()

    dp = DataProcess(mode)
    df, dft = dp.data_input(Const.TRAIN_FILE_NAME, Const.TEST_FILE_NAME)
    tc.print_event()

    fe = FeatureEngineering(mode)
    df, dft = fe.feature_process(df, dft)
    tc.print_event()

    X_train, y_train, X_valid, y_valid, X_test = dp.get_split_data(df, dft)
    tc.print_event()

    xgb = XGBoostModel(mode)
    result = xgb.train_model(X_train, y_train, X_valid, y_valid, X_test)
    tc.print_event()

    if not mode:
        dp.transform_index(result)

    # xgb.load_model()
