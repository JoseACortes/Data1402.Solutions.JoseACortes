import DataPrepBase
import LaptopPrep
from sklearn import preprocessing
import pandas as pd

def scale(dataset):
    min_max_scaler = preprocessing.MinMaxScaler()
    return pd.DataFrame(min_max_scaler.fit_transform(dataset.values))