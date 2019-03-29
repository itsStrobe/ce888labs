# Author: Jose Juan Zavala Iglesias
# Created as part of requirements for CE888 Assignment at The University of Essex (2019)
# This file will provide a Agent object with the capabilities of being trained, loading 
# an existing agent, and saving its own state.

# It defaults to the best parameters determined with the DTClassifier_ParamSelect.py script,
# under the assumption that the parameters extracted from the inital dataset will work for
# later datasets extracted from self-play

import os
import multiprocessing
import numpy as np
import pandas as pd
from UTIL import EncodeMatrix
from UTIL import ReadTrainData
from UTIL import GetInvalidMoves
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.externals import joblib

# CONSTANTS
CLF_DIR   = "./Models/"
MDL_EXT   = ".joblib"
DOT_EXT   = ".dot"
CRITERION = "entropy"
MAX_DEPTH = None

class AgentMoveSelector:
    def __init__(self, criterion=CRITERION, max_depth=MAX_DEPTH):
        self.model = DecisionTreeClassifier(splitter="best", min_impurity_decrease=0., criterion=criterion, max_depth=max_depth)

    def SaveModel(self, name, clf_dir=CLF_DIR):
        if not os.path.exists(clf_dir): os.makedirs(clf_dir)
        joblib.dump(self.model, clf_dir + name + MDL_EXT)
        export_graphviz(self.model, out_file= clf_dir + name + DOT_EXT)    

    def LoadModel(self, name, clf_dir=CLF_DIR):
        self.model = joblib.load(clf_dir + name + MDL_EXT)

    def TrainModel(self, X, y):
        self.model.fit(EncodeMatrix(X), y)

    def MakeMove(self, state, verbose = False):
        if(len(state.shape) != 2):
            move = self.model.predict(EncodeMatrix(state.reshape(1, state.size)))[0]
        else:
            move = self.model.predict(EncodeMatrix(state))[0]
            state = state.reshape(state.size,)

        if(state[move] == '_'):
            return move
        else:
            if(verbose): print("WARNING: DT returned invalid move. Making first valid move.")
            return np.where(state == '_')[0][0]

    def Predict(self, X):
        return self.model.predict(EncodeMatrix(X))
