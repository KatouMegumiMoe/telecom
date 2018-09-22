from DataProcess import *
from Constant import Const
from TransformToBinary import *


dp = DataProcess(Const.TRAIN_FILE_NAME)
df = dp.data_input()

ttb = TransformToBinary(df)
df_bin = ttb.transform_to_binary()
print df_bin[['user_id', 'service_id', 'label_id']]

