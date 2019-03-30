# Author: Jose Juan Zavala Iglesias
# Created as part of requirements for CE888 Assignment at The University of Essex (2019)
# This file will evaluate between Gini Impurity and Information Gain metrics
# to come up with the best Decision Tree classifier for OXO.
# It then evaluates that the DT comes up with appropiate moves for a sample of the dataset.

import os
import multiprocessing
import numpy as np
import pandas as pd
from UTIL import EncodeMatrix
from UTIL import ReadTrainData
from UTIL import GetInvalidMoves
from UTIL import SaveExamplesAsDataFrame
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV

# CONSTANTS
RES_DIR = "./Results/"

# Required Operations
cores = multiprocessing.cpu_count()
if not os.path.exists(RES_DIR): os.makedirs(RES_DIR)

# Read whole dataset
X, y = ReadTrainData(fileName="./data/OXO_NewDataset.csv")

# Set DT to test between Gini Impurity and Information Gain metrics and different max levels (avoid overfitting but check for invalid moves)
dt = DecisionTreeClassifier(splitter="best", min_impurity_decrease=0.)
params = {'criterion':['gini', 'entropy'], 'max_depth':[11, 13, 15, 17, 19, 21, 23, 25, 27, 29]}


# Grid Search using 10-Fold Cross Validation
clf = GridSearchCV(dt, params, cv=10, n_jobs=cores, return_train_score=True, verbose=3)
clf.fit(EncodeMatrix(X), y)

resultsPd = pd.DataFrame(clf.cv_results_)
resultsPd.head
resultsPd.to_csv(RES_DIR + "Results.csv")


# Test for invalid moves
X_sample, _ = ReadTrainData(fileName="./data/OXO_NewDataset.csv", sample_size=10000)
y_pred = clf.predict(EncodeMatrix(X_sample))

PredMovesPd = SaveExamplesAsDataFrame(X_sample, y_pred, "PredictedMoves.csv", dirName=RES_DIR)
print("Predicted Moves")
print(PredMovesPd.head)

X_inv, y_inv = GetInvalidMoves(X_sample, y_pred)

if (X_inv.size > 0):
    InvMovesPd = SaveExamplesAsDataFrame(X_inv, y_inv, "InvalidMoves.csv", dirName=RES_DIR)
    print("Predicted Invalid Moves")
    print(InvMovesPd.head)
    print("ERROR: INVALID MOVES FOUND. CHECK " + RES_DIR + "InvalidMoves.csv FOR MORE INFO.")
else:
    print("SUCCESS: NO INVALID MOVES FOUND")
