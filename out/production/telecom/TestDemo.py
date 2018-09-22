from DataProcess import *
from TransformToBinary import *
from XGBoostModel import *


dp = DataProcess(Const.TRAIN_FILE_NAME, Const.TEST_FILE_NAME)
df, dft = dp.data_input()

ttb = TransformToBinary(df, dft)
df_bin, dft_bin = ttb.transform_to_binary()
X_train, y_train, X_valid, y_valid, X_test = dp.get_split_data(df_bin, dft_bin)

xgb = XGBoostModel(X_train, y_train, X_valid, y_valid, X_test)
xgb.train()
