from UTIL import ReadTrainData
from GameMaster import Play
from DTMoveSelector import AgentMoveSelector

# CONSTANTS
NUM_VS_MCTS= 1000
NUM_VS_PREV= 10
DATA_ROOT  = './data/'
OG_DATA    = './data/OXO_dataset.csv'
DATA_PREFIX= 'dataset_'
MCTS_NAME  = 'Random'
MDL_PREFIX = 'Model_'
IT         = 0

def CreateNewPlayer(dataset, playerName):
    X, y = ReadTrainData(fileName=dataset)
    Player = AgentMoveSelector()
    Player.TrainModel(X, y)
    Player.SaveModel(playerName)

# First model with original dataset (MCTS vs MCTS)
X_, y_ = ReadTrainData(fileName=OG_DATA)
Player = AgentMoveSelector()
Player.TrainModel(X_, y_)
Player.SaveModel(MDL_PREFIX + str(IT))

Play(MDL_PREFIX + str(IT), MCTS_NAME, NUM_VS_MCTS, DATA_PREFIX + str(IT))
IT += 1

for i in range(10):
    for j in range(10):
        newPlayer = MDL_PREFIX + str(IT)
        CreateNewPlayer(DATA_ROOT + DATA_PREFIX + str(IT-1) + '.csv', newPlayer)
        Play(newPlayer, MCTS_NAME, NUM_VS_MCTS, DATA_PREFIX + str(IT))
        IT += 1
    
    newModel = MDL_PREFIX + str(IT-1)
    oldModel = MDL_PREFIX + str(IT-11)
    Play(newModel, oldModel, NUM_VS_PREV, newModel + 'vs' + oldModel, savePerformance=True)
