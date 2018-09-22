from DataProcess import *
from TransformToBinary import *
from XGBoostModel import *


dp = DataProcess(Const.TRAIN_FILE_NAME)
df = dp.data_input()

ttb = TransformToBinary(df)
df_bin = ttb.transform_to_binary()

'''
X_train, y_train, X_test, y_test = dp.get_valid_data(df_bin)
xgb = XGBoostModel(X_train, y_train, X_test, y_test)
predict = xgb.train()
print 'the total f-score:'
print xgb.evaluation(predict, X_test, y_test)
'''

dp = DataProcess(Const.TEST_FILE_NAME)
dft = dp.data_input()
dft_bin = ttb.transform_to_binary()

X_train, y_train, X_test, y_test = dp.get_test_data(df_bin, dft_bin)
xgb = XGBoostModel(X_train, y_train, X_test, y_test)
predict = xgb.train()
result = xgb.result_merge(predict, X_test, y_test)
submission = result[['user_id', 'service_id']]
submission.columns = ['user_id', 'predict']
submission.to_csv(Const.SUBMISSION_FILE_NAME)
result.to_csv(Const.RESULT_FILE_NAME)
