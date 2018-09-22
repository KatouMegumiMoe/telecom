import pandas as pd
from Constant import Const


dft = pd.read_csv(Const.TEST_FILE_NAME)
df = pd.read_csv(Const.SUBMISSION_FILE_NAME)

index = [item for item in range(dft.shape[0])]
dft['sort'] = index
result = pd.merge(df, dft, how='left', on='user_id')
result 