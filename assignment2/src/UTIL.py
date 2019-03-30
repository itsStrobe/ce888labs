import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

# One Hot Encoding
labels = np.array([['X','X','X','X','X','X','X','X','X'],
                   ['Y','Y','Y','Y','Y','Y','Y','Y','Y'],
                   ['_','_','_','_','_','_','_','_','_']])
enc = OneHotEncoder(handle_unknown='ignore', dtype=np.int, sparse=False)
enc.fit(labels)

def EncodeMatrix(examples):
    examples = enc.transform(examples)

    return examples

def DecodeMatrix(examples):
    examples = enc.inverse_transform(examples)

    return examples

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

def GetInvalidMoves(states, moves):
    invStates = []
    invMoves  = []
    for state, move in zip(states, moves):
        if(state[move] != '_'):
            print(state[move])
            invStates.append(state)
            invMoves.append(move)

    return np.array(invStates), np.array(invMoves)

def SaveExamplesAsDataFrame(X, y, fileName, dirName='./'):
    if not os.path.exists(dirName): os.makedirs(dirName)

    MovesDict = {"[0:0]": X[:, 0].tolist(), 
                 "[0:1]": X[:, 1].tolist(),
                 "[0:2]": X[:, 2].tolist(),
                 "[1:0]": X[:, 3].tolist(),
                 "[1:1]": X[:, 4].tolist(),
                 "[1:2]": X[:, 5].tolist(),
                 "[2:0]": X[:, 6].tolist(),
                 "[2:1]": X[:, 7].tolist(),
                 "[2:2]": X[:, 8].tolist(),
                 "Move" : y.tolist()}

    MovesPd = pd.DataFrame(MovesDict)
    MovesPd.to_csv(dirName + fileName)

    return MovesPd
