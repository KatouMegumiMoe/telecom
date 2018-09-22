import xgboost as xgb
from Constant import Const


class XGBoostModel:

    def __init__(self, X_train, y_train, X_test, y_test, num_round=Const.NUM_ROUND, model=None):
        self.params = {'colsample_bytree': 0.8,
                       'silent': 1,
                       'eval_metric': 'auc',
                       'eta': 0.1,
                       'learning_rate': 0.1,
                       'njob': 8,
                       'min_child_weight': 1,
                       'subsample': 0.8,
                       'seed': 0,
                       'objective': 'reg:linear',
                       'max_depth': 4,
                       'gamma': 0.0,
                       'booster': 'gbtree'}
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.dtrain = xgb.DMatrix(X_train.drop(['current_service', 'user_id'], axis=1), y_train)
        self.dtest = xgb.DMatrix(X_test.drop(['current_service', 'user_id'], axis=1), y_test)
        self.num_round = num_round
        self.model = model

    def train(self):
        watchlist = [(self.dtrain, 'train'), (self.dtest, 'test')]
        self.model = xgb.train(self.params, self.dtrain, evals=watchlist, num_boost_round=self.num_round)
        self.predict()

    def predict(self):
        return self.model.predict(self.dtest)

    @staticmethod
    def evaluation(predict, test, label):
        res = test
        res['predict'] = predict
        res['label'] = label
        res.reset_index(drop=True)
        result = res.iloc[res.groupby(['user_id']).apply(lambda x: x['predict'].idxmax())]

        score = 0.0
        service_list = result.groupby(['current_service']).count().index.values
        for id in service_list:
            tp = result[(result['service_id'] == result['current_service']) & (result['current_service'] == id)].shape[0]
            fp = result[(result['service_id'] != result['current_service']) & (result['service_id'] == id)].shape[0]
            fn = result[(result['service_id'] != result['current_service']) & (result['current_service'] == id)].shape[0]

            precision = float(tp)/(tp+fp)
            recall = float(tp)/(tp+fn)
            score += 2*precision*recall/(precision+recall)

        return (score/len(service_list))**2
