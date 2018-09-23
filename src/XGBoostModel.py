import xgboost as xgb
from Constant import Const
import operator
import pandas as pd


class XGBoostModel:

    def __init__(self, X_train, y_train, X_valid, y_valid, X_test, num_round=Const.NUM_ROUND, model=None):
        self.params = {'colsample_bytree': 0.8,
                       'silent': 1,
                       'eval_metric': 'error',
                       'eta': 0.05,
                       'learning_rate': 0.1,
                       'njob': 8,
                       'min_child_weight': 1,
                       'subsample': 0.8,
                       'seed': 0,
                       'objective': 'binary:logistic',
                       'max_depth': 5,
                       'gamma': 0.0,
                       'booster': 'gbtree'}

        self.X_train = X_train
        self.y_train = y_train
        self.X_valid = X_valid
        self.y_valid = y_valid
        self.X_test = X_test

        self.dtrain = xgb.DMatrix(X_train.drop(['current_service', 'user_id'], axis=1), y_train)
        self.dvalid = xgb.DMatrix(X_valid.drop(['current_service', 'user_id'], axis=1), y_valid)
        self.dtest = xgb.DMatrix(X_test.drop(['user_id'], axis=1))

        self.num_round = num_round
        self.model = model

    def train_model(self):
        watchlist = [(self.dtrain, 'train'), (self.dvalid, 'valid')]
        self.model = xgb.train(self.params, self.dtrain, evals=watchlist, num_boost_round=self.num_round)
        self.model.save_model(Const.MODEL_FILE_NAME)

        self.valid_model()
        result = self.predict_model()
        return result

    def load_model(self):
        self.model = xgb.Booster({'nthread':4})
        self.model.load_model(Const.MODEL_FILE_NAME)
        self.valid_model()

        importance = self.model.get_fscore()
        importance = sorted(importance.items(), key=operator.itemgetter(1))
        print pd.DataFrame(importance, columns=['feature', 'score'])

    def valid_model(self):
        prediction = self.model.predict(self.dvalid)
        err = XGBoostModel.evaluation(prediction, self.X_valid, self.y_valid)
        print 'the total f-score:', err

    def predict_model(self):
        prediction = self.model.predict(self.dtest)
        result = XGBoostModel.test_result_merge(prediction, self.X_test)
        XGBoostModel.save_data(result)
        return result

    @staticmethod
    def test_result_merge(predict_result, test):
        result = test
        result['predict'] = predict_result
        result = result.reset_index(drop=True)
        result['predict'] = result['predict'].astype('float64')
        return result.iloc[result.groupby(['user_id']).apply(lambda x: x['predict'].idxmax())]

    @staticmethod
    def valid_result_merge(predict_result, test, label):
        result = test
        result['predict'] = predict_result
        result['label'] = label
        result = result.reset_index(drop=True)
        result['predict'] = result['predict'].astype('float64')
        return result.iloc[result.groupby(['user_id']).apply(lambda x: x['predict'].idxmax())]

    @staticmethod
    def evaluation(predict_result, test, label):
        result = XGBoostModel.valid_result_merge(predict_result, test, label)
        score = 0.0
        service_list = result.groupby(['current_service']).count().index.values

        for service in service_list:
            tp = result[(result['service_id'] == result['current_service']) & (result['current_service'] == service)].shape[0]
            fp = result[(result['service_id'] != result['current_service']) & (result['service_id'] == service)].shape[0]
            fn = result[(result['service_id'] != result['current_service']) & (result['current_service'] == service)].shape[0]

            try:
                precision = float(tp)/(tp+fp)
                recall = float(tp)/(tp+fn)
                score += 2*precision*recall/(precision+recall)
            except ZeroDivisionError:
                print 'zero error in service:', service
                continue

        return (score/len(service_list))**2

    @staticmethod
    def save_data(result):
        submission = result[['user_id', 'service_id']]
        submission.columns = ['user_id', 'predict']
        submission.to_csv(Const.SUBMISSION_FILE_NAME, index=False)
        result.to_csv(Const.RESULT_FILE_NAME)
