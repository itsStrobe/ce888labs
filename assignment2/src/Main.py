import time
from UTIL import ReadTrainData
from GameMaster import Play
from DTMoveSelector import AgentMoveSelector

# CONSTANTS
MCTS_TESTS   = 10
PREV_TESTS   = 1
MATCHES_MCTS = 100
MATCHES_PREV = 10
DATA_ROOT    = './data/'
OG_DATA      = './data/OXO_dataset.csv'
DATA_PREFIX  = 'dataset_'
MCTS_NAME    = 'Random'
MDL_PREFIX   = 'Model_'
IT           = 0

def CreateNewPlayer(dataset, playerName):
    X, y = ReadTrainData(fileName=dataset)
    Player = AgentMoveSelector()
    Player.TrainModel(X, y)
    Player.SaveModel(playerName)

# First model with original dataset (MCTS vs MCTS)
X_, y_ = ReadTrainData(fileName=OG_DATA, sample_size=100)
Player = AgentMoveSelector()
Player.TrainModel(X_, y_)
Player.SaveModel(MDL_PREFIX + str(IT))

Play(MDL_PREFIX + str(IT), MCTS_NAME, MATCHES_MCTS, DATA_PREFIX + str(IT))
IT += 1

start_time = time.time()
for i in range(PREV_TESTS):
    for j in range(MCTS_TESTS):
        newPlayer = MDL_PREFIX + str(IT)
        CreateNewPlayer(DATA_ROOT + DATA_PREFIX + str(IT-1) + '.csv', newPlayer)
        Play(newPlayer, MCTS_NAME, MATCHES_MCTS, DATA_PREFIX + str(IT))
        IT += 1
    
    newModel = MDL_PREFIX + str(IT-1)
    oldModel = MDL_PREFIX + str(IT-11)
    Play(newModel, oldModel, MATCHES_PREV, newModel + 'vs' + oldModel, savePerformance=True)
    
print("Elapsed Time:", time.time() - start_time)
