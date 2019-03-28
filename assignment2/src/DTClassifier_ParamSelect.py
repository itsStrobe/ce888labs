# Author: Jose Juan Zavala Iglesias
# Created as part of requirements for CE888 Assignment at The University of Essex (2019)
# This file will evaluate between Gini Impurity and Information Gain metrics
# to come up with the best Decision Tree classifier for OXO.
# It then evaluates that the DT comes up with appropiate moves for a sample of the dataset.

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd
import multiprocessing
import os
from UTIL import ReadTrainData
from UTIL import EncodeMatrix

def GetInvalidMoves(states, moves):
    invStates = []
    invMoves  = []
    for state, move in zip(states, moves):
        if(state[move] != '_'):
            print(state[move])
            invStates.append(state)
            invMoves.append(move)

    return np.array(invStates), np.array(invMoves)

# CONSTANTS
RES_DIR = "./Results/"

# Required Operations
cores = multiprocessing.cpu_count()
if not os.path.exists(RES_DIR): os.makedirs(RES_DIR)

# Read whole dataset
X, y = ReadTrainData(fileName="./data/OXO_NewDataset.csv")

# Label Encoding
le = LabelEncoder()
le.fit(["_", "X", "Y"])

# Set DT to test between Gini Impurity and Information Gain metrics and different max levels (avoid overfitting but check for invalid moves)
dt = DecisionTreeClassifier(splitter="best", max_depth=11, min_impurity_decrease=0)
params = {'criterion':['gini', 'entropy'], 'max_depth':[8, 9, 10, 11]}


# Grid Search using 10-Fold Cross Validation
clf = GridSearchCV(dt, params, cv=10, n_jobs=cores, verbose=3)
clf.fit(EncodeMatrix(le, X), y)

resultsPd = pd.DataFrame(clf.cv_results_)
resultsPd.head
resultsPd.to_csv(RES_DIR + "Results.csv")


# Test for invalid moves
X_sample, _ = ReadTrainData(fileName="./data/OXO_NewDataset.csv", sample_size=10000)
y_pred = clf.predict(EncodeMatrix(le, X_sample))

PredMovesDict = {"[0:0]": X_sample[:, 0].tolist(), 
                 "[0:1]": X_sample[:, 1].tolist(),
                 "[0:2]": X_sample[:, 2].tolist(),
                 "[1:0]": X_sample[:, 3].tolist(),
                 "[1:1]": X_sample[:, 4].tolist(),
                 "[1:2]": X_sample[:, 5].tolist(),
                 "[2:0]": X_sample[:, 6].tolist(),
                 "[2:1]": X_sample[:, 7].tolist(),
                 "[2:2]": X_sample[:, 8].tolist(),
                 "Move" : y_pred.tolist()}

PredMovesPd = pd.DataFrame(PredMovesDict)
PredMovesPd.head
PredMovesPd.to_csv(RES_DIR + "PredictedMoves.csv")

X_inv, y_inv = GetInvalidMoves(X_sample, y_pred)

if (X_inv.size > 0):
    InvMovesDict = {"[0:0]": X_inv[:, 0].tolist(), 
                    "[0:1]": X_inv[:, 1].tolist(),
                    "[0:2]": X_inv[:, 2].tolist(),
                    "[1:0]": X_inv[:, 3].tolist(),
                    "[1:1]": X_inv[:, 4].tolist(),
                    "[1:2]": X_inv[:, 5].tolist(),
                    "[2:0]": X_inv[:, 6].tolist(),
                    "[2:1]": X_inv[:, 7].tolist(),
                    "[2:2]": X_inv[:, 8].tolist(),
                    "Move" : y_inv.tolist()}

    InvMovesPd = pd.DataFrame(InvMovesDict)
    InvMovesPd.head
    InvMovesPd.to_csv(RES_DIR + "InvalidMoves.csv")
    print("ERROR: INVALID MOVES FOUND. CHECK " + RES_DIR + "InvalidMoves.csv FOR MORE INFO.")
else:
    print("SUCCESS: NO INVALID MOVES FOUND")
