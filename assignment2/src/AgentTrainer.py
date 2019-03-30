import sys
import numpy as np
import pandas as pd
from DTMoveSelector import AgentMoveSelector
from UTIL import GetInvalidMoves
from UTIL import ReadTrainData
from UTIL import SaveExamplesAsDataFrame

# CONSTANTS
RES_DIR = "./AgentTrainingResults/"

# Parameters
modelName = str(sys.argv[1])
dataDir   = str(sys.argv[2])

# Read training data and move validation data
X, y = ReadTrainData(fileName = dataDir)
X_val, _ = ReadTrainData(fileName = dataDir, sample_size=1000)

# Initialize and Train AgentMoveSelector
agent = AgentMoveSelector()
agent.TrainModel(X, y)

# Save Model
agent.SaveModel(modelName)

# Check for invalid moves
y_pred = agent.Predict(X_val)
X_inv, y_inv = GetInvalidMoves(X_val, y_pred)
if (X_inv.size > 0):
    InvMovesPd = SaveExamplesAsDataFrame(X_inv, y_inv, "InvalidMoves_" + modelName + ".csv", dirName=RES_DIR)
    print("Predicted Invalid Moves")
    print(InvMovesPd.head)
    print("ERROR: INVALID MOVES FOUND. CHECK " + RES_DIR + "InvalidMoves.csv FOR MORE INFO.")
else:
    print("SUCCESS: NO INVALID MOVES FOUND")
