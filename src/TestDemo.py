from DataProcess import *
from TransformToBinary import *
from XGBoostModel import *


dp = DataProcess(Const.TRAIN_FILE_NAME)
df = dp.data_input()

ttb = TransformToBinary(df)
df_bin = ttb.transform_to_binary()
print df_bin[['user_id', 'service_id', 'label']]

X_train, y_train, X_test, y_test = dp.get_valid_data(df_bin)
xgb = XGBoostModel(X_train, y_train, X_test, y_test)
predict = xgb.train()
print 'the total f-score:', xgb.evaluation(predict, X_test, y_test)
