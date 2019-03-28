import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

def ReadTrainData(fileName='./data/OXO_dataset.csv', asNumpy=True, sample_size=None):
    df = pd.read_csv(fileName, sep=',', header=0)
    df.dropna(axis='index', how='any')

    if(sample_size is not None):
        df = df.sample(n=sample_size)

    df_features = df.drop(['Move'], axis=1)
    df_targets = df.loc[:, ['Move']]

    if(asNumpy):
        return df_features.values, np.transpose(df_targets.values)[0]
    else:
        return df_features, df_targets

def EncodeMatrix(le, examples):
    rows, cols = examples.shape

    examples = np.reshape(examples, rows*cols)
    examples = le.transform(examples)
    examples = np.reshape(examples, (rows, cols))

    return examples

def DecodeMatrix(le, examples):
    rows, cols = examples.shape
    examples = np.reshape(examples, rows*cols)
    examples = le.inverse_transform(examples)
    examples = np.reshape(examples, (rows, cols))

    return examples