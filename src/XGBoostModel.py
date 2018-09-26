import xgboost as xgb
from Constant import Const
import operator
import pandas as pd


class XGBoostModel:

    def __init__(self, mode):
        self.params = {'colsample_bytree': 0.8,
                       'silent': 1,
                       'eval_metric': 'mlogloss',
                       'eta': 0.1,
                       'learning_rate': 0.1,
                       'njob': 8,
                       'min_child_weight': 1,
                       # 'subsample': 0.8,
                       'seed': 0,
                       'objective': 'multi:softmax',
                       'max_depth': 5,
                       'gamma': 0.0,
                       'booster': 'gbtree',
                       'num_class': Const.CATEGORY_NUM}

        self.dtrain = None
        self.dvalid = None
        self.dtest = None

        self.model = None
        self.mode = mode

    def train_model(self, X_train, y_train, X_valid, y_valid, X_test, result=None):

        self.dtrain = xgb.DMatrix(X_train.drop(['user_id'], axis=1), y_train, missing=Const.MISSING_NUM)
        if self.mode:
            self.dvalid = xgb.DMatrix(X_valid.drop(['user_id'], axis=1), y_valid, missing=Const.MISSING_NUM)
            watchlist = [(self.dtrain, 'train'), (self.dvalid, 'valid')]
        else:
            self.dtest = xgb.DMatrix(X_test.drop(['user_id'], axis=1), missing=Const.MISSING_NUM)
            watchlist = [(self.dtrain, 'train')]

        self.model = xgb.train(self.params,
                               self.dtrain,
                               evals=watchlist,
                               num_boost_round=Const.NUM_ROUND)
        self.model.save_model(Const.MODEL_FILE_NAME)

        if self.mode:
            self.valid_model(X_valid, y_valid)
        else:
            result = self.predict_model(X_test)

        importance = self.model.get_fscore()
        importance = sorted(importance.items(), key=operator.itemgetter(1))
        print pd.DataFrame(importance, columns=['feature', 'score'])

        return result

    def load_model(self):
        self.model = xgb.Booster({'nthread':4})
        self.model.load_model(Const.MODEL_FILE_NAME)
        self.valid_model()

    def valid_model(self, X_valid, y_valid):
        prediction = self.model.predict(self.dvalid)
        err = XGBoostModel.evaluation(prediction, X_valid, y_valid)
        print 'the total f-score:', err

    def predict_model(self, X_test):
        prediction = self.model.predict(self.dtest)
        result = XGBoostModel.test_result_merge(prediction, X_test)
        print 'finished model training'
        XGBoostModel.save_data(result)
        return result

    @staticmethod
    def test_result_merge(predict_result, test):
        result = test
        result['predict'] = predict_result
        result = result.reset_index(drop=True)
        result['predict'] = result['predict'].astype('float64')
        return result

    @staticmethod
    def valid_result_merge(predict_result, test, label):
        result = test
        result['predict'] = predict_result
        result['label'] = label
        result = result.reset_index(drop=True)
        result['predict'] = result['predict'].astype('float64')
        return result

    @staticmethod
    def evaluation(predict_result, test, label):
        result = XGBoostModel.valid_result_merge(predict_result, test, label)
        score = 0.0
        service_list = result.groupby(['label']).count().index.values

        for service in service_list:
            tp = result[(result['label'] == result['predict']) & (result['label'] == service)].shape[0]
            fp = result[(result['label'] != result['predict']) & (result['predict'] == service)].shape[0]
            fn = result[(result['label'] != result['predict']) & (result['label'] == service)].shape[0]

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
        submission = result[['user_id', 'predict']]
        submission.columns = ['user_id', 'predict']
        submission.to_csv(Const.SUBMISSION_FILE_NAME, index=False)
        result.to_csv(Const.RESULT_FILE_NAME)
