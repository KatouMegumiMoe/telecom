import pandas as pd


class DataProcess:

    def __init__(self, file_name, df=None):
        self.file_name = file_name
        self.df = df

    def data_input(self):
        self.df = pd.read_csv(self.file_name)
        print 'before processing data shape:', self.df.shape
        return self.fill_nan()

    def fill_nan(self):
        self.df = self.df[~self.df.isin(['\N'])].dropna(axis=0, how='any')
        self.df = self.df.reset_index(drop=True)
        print 'after processing data shape:', self.df.shape
        return self.df
