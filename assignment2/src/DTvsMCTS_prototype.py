# This is a very simple implementation of the UCT Monte Carlo Tree Search algorithm in Python 2.7.
# The function UCT(rootstate, itermax, verbose = False) is towards the bottom of the code.
# It aims to have the clearest and simplest possible code, and for the sake of clarity, the code
# is orders of magnitude less efficient than it could be made, particularly by using a 
# state.GetRandomMove() or state.DoRandomRollout() function.
# 
# Example GameState classes for Nim, OXO and Othello are included to give some idea of how you
# can write your own GameState use UCT in your 2-player game. Change the game to be played in 
# the UCTPlayGame() function at the bottom of the code.
# 
# Written by Peter Cowling, Ed Powley, Daniel Whitehouse (University of York, UK) September 2012.
# 
# Licence is granted to freely use and distribute for any sensible/legal purpose so long as this comment
# remains in any distributed code.
# 
# For more information about Monte Carlo Tree Search check out our web site at www.mcts.ai

# ATTENTION:
# This is a modification of the file described above.
# This file is a first, UNFINISHED, version of a DT vs MCTS program for the game of OXO (Tic-Tac-Toe)
# It will load a pre-trained DT model to play against the implemented MCTS agent
# Modified by Jose Juan Zavala Iglesias for CE888 Assignment at The University of Essex (2019)

from math import *
import sys
import random
import numpy as np
from DTAgent import AgentMoveSelector
from sklearn.tree import DecisionTreeClassifier

# ----- UTIL ----- #

def transform_OXO_state(turn, state):
    state_to_string = {
        0: ["_", "_"],
        1: ["X", "Y"],
        2: ["Y", "X"]
    }
    
    new_state = [state_to_string[elem][turn - 1] for elem in state]
    
    return new_state
    
def GetWinnerMoves(winner, match_moves):
    # TODO
    
    return winning_moves
    
# ----- MCTS Implementation ----- #

class GameState:
    """ A state of the game, i.e. the game board. These are the only functions which are
        absolutely necessary to implement UCT in any 2-player complete information deterministic 
        zero-sum game, although they can be enhanced and made quicker, for example by using a 
        GetRandomMove() function to generate a random move during rollout.
        By convention the players are numbered 1 and 2.
    """
    def __init__(self):
            self.playerJustMoved = 2 # At the root pretend the player just moved is player 2 - player 1 has the first move
        
    def Clone(self):
        """ Create a deep clone of this game state.
        """
        st = GameState()
        st.playerJustMoved = self.playerJustMoved
        return st

    def DoMove(self, move):
        """ Update a state by carrying out the given move.
            Must update playerJustMoved.
        """
        self.playerJustMoved = 3 - self.playerJustMoved
        
    def GetMoves(self):
        """ Get all possible moves from this state.
        """
    
    def GetResult(self, playerjm):
        """ Get the game result from the viewpoint of playerjm. 
        """

    def __repr__(self):
        """ Don't need this - but good style.
        """
        pass

class OXOState:
    """ A state of the game, i.e. the game board.
        Squares in the board are in this arrangement
        012
        345
        678
        where 0 = empty, 1 = player 1 (X), 2 = player 2 (O)
    """
    def __init__(self):
        self.playerJustMoved = 2 # At the root pretend the player just moved is p2 - p1 has the first move
        self.board = [0,0,0,0,0,0,0,0,0] # 0 = empty, 1 = player 1, 2 = player 2
        
    def Clone(self):
        """ Create a deep clone of this game state.
        """
        st = OXOState()
        st.playerJustMoved = self.playerJustMoved
        st.board = self.board[:]
        return st

    def DoMove(self, move):
        """ Update a state by carrying out the given move.
            Must update playerToMove.
        """
        assert move >= 0 and move <= 8 and move == int(move) and self.board[move] == 0
        self.playerJustMoved = 3 - self.playerJustMoved
        self.board[move] = self.playerJustMoved
        
    def GetMoves(self):
        """ Get all possible moves from this state.
        """
        return [i for i in range(9) if self.board[i] == 0]
    
    def GetResult(self, playerjm):
        """ Get the game result from the viewpoint of playerjm. 
        """
        for (x,y,z) in [(0,1,2),(3,4,5),(6,7,8),(0,3,6),(1,4,7),(2,5,8),(0,4,8),(2,4,6)]:
            if self.board[x] == self.board[y] == self.board[z]:
                if self.board[x] == playerjm:
                    return 1.0
                else:
                    return 0.0
        if self.GetMoves() == []: return 0.5 # draw
        assert False # Should not be possible to get here

    def __repr__(self):
        s= ""
        for i in range(9): 
            s += ".XO"[self.board[i]]
            if i % 3 == 2: s += "\n"
        return s

class Node:
    """ A node in the game tree. Note wins is always from the viewpoint of playerJustMoved.
        Crashes if state not specified.
    """
    def __init__(self, move = None, parent = None, state = None):
        self.move = move # the move that got us to this node - "None" for the root node
        self.parentNode = parent # "None" for the root node
        self.childNodes = []
        self.wins = 0
        self.visits = 0
        self.untriedMoves = state.GetMoves() # future child nodes
        self.playerJustMoved = state.playerJustMoved # the only part of the state that the Node needs later
        
    def UCTSelectChild(self):
        """ Use the UCB1 formula to select a child node. Often a constant UCTK is applied so we have
            lambda c: c.wins/c.visits + UCTK * sqrt(2*log(self.visits)/c.visits to vary the amount of
            exploration versus exploitation.
        """
        s = sorted(self.childNodes, key = lambda c: c.wins/c.visits + sqrt(2*log(self.visits)/c.visits))[-1]
        return s
    
    def AddChild(self, m, s):
        """ Remove m from untriedMoves and add a new child node for this move.
            Return the added child node
        """
        n = Node(move = m, parent = self, state = s)
        self.untriedMoves.remove(m)
        self.childNodes.append(n)
        return n
    
    def Update(self, result):
        """ Update this node - one additional visit and result additional wins. result must be from the viewpoint of playerJustmoved.
        """
        self.visits += 1
        self.wins += result

    def __repr__(self):
        return "[M:" + str(self.move) + " W/V:" + str(self.wins) + "/" + str(self.visits) + " U:" + str(self.untriedMoves) + "]"

    def TreeToString(self, indent):
        s = self.IndentString(indent) + str(self)
        for c in self.childNodes:
             s += c.TreeToString(indent+1)
        return s

    def IndentString(self,indent):
        s = "\n"
        for i in range (1,indent+1):
            s += "| "
        return s

    def ChildrenToString(self):
        s = ""
        for c in self.childNodes:
             s += str(c) + "\n"
        return s


def UCT(rootstate, itermax, verbose = False):
    """ Conduct a UCT search for itermax iterations starting from rootstate.
        Return the best move from the rootstate.
        Assumes 2 alternating players (player 1 starts), with game results in the range [0.0, 1.0]."""

    rootnode = Node(state = rootstate)

    for i in range(itermax):
        node = rootnode
        state = rootstate.Clone()

        # Select
        while node.untriedMoves == [] and node.childNodes != []: # node is fully expanded and non-terminal
            node = node.UCTSelectChild()
            state.DoMove(node.move)

        # Expand
        if node.untriedMoves != []: # if we can expand (i.e. state/node is non-terminal)
            m = random.choice(node.untriedMoves) 
            state.DoMove(m)
            node = node.AddChild(m,state) # add child and descend tree

        # Rollout - this can often be made orders of magnitude quicker using a state.GetRandomMove() function
        while state.GetMoves() != []: # while state is non-terminal
            state.DoMove(random.choice(state.GetMoves()))

        # Backpropagate
        while node != None: # backpropagate from the expanded node and work back to the root node
            node.Update(state.GetResult(node.playerJustMoved)) # state is terminal. Update node with result from POV of node.playerJustMoved
            node = node.parentNode

    # Output some information about the tree - can be omitted
    if (verbose): print(rootnode.TreeToString(0))
    else: print(rootnode.ChildrenToString())

    return sorted(rootnode.childNodes, key = lambda c: c.visits)[-1].move # return the move that was most visited
                
def UCTPlayGame(NewPlayer, save_moves = False):
    """ Play a sample game between two UCT players where each player gets a different number 
        of UCT iterations (= simulations = tree nodes).
    """
    state = OXOState() # uncomment to play OXO
    moves_performed = np.array([], dtype=str).reshape(0, 10)
        
    while (state.GetMoves() != []):
        print(str(state))
        if state.playerJustMoved == 2:
            # Player 1 - X
            DT_state = transform_OXO_state(1, state.board)
            m = NewPlayer.MakeMove(np.array(DT_state))
        else:
            # Player 2 - O
            m = UCT(rootstate = state, itermax = 1000, verbose = False)
        print("Best Move: " + str(m) + "\n")
        
        if save_moves:
            current_state = transform_OXO_state(3 - state.playerJustMoved, state.board)
            current_state.append(m)
            moves_performed = np.append(moves_performed, [current_state], axis=0)

        state.DoMove(m)
    
    # Winner:
    # 0 -> Draw
    # 1 -> X (DT)
    # 2 -> Y (MCTS)
    winner = 0
    
    if state.GetResult(state.playerJustMoved) == 1.0:
        print("Player " + str(state.playerJustMoved) + " wins!")
        winner = state.playerJustMoved
    elif state.GetResult(state.playerJustMoved) == 0.0:
        print("Player " + str(3 - state.playerJustMoved) + " wins!")
        winner = 3 - state.playerJustMoved
    else: print("Nobody wins!")
    
    if save_moves:
        return winner, moves_performed
        
    return winner

if __name__ == "__main__":
    """ Play a given number of games to the end
        DT vs MCTS
    """
    modelFileName = str(sys.argv[1])
    num_games = int(sys.argv[2])
    save_file = str(sys.argv[3])
    
    wins = 0
    draw = 0
    loss = 0
    
    # DecisionTreeClassifier
    NewPlayer = AgentMoveSelector()
    NewPlayer.LoadModel(modelFileName)
    
    moves = np.array(["[0:0]", "[0:1]", "[0:2]", "[1:0]", "[1:1]", "[1:2]", "[2:0]", "[2:1]", "[2:2]", "Move"], dtype=str).reshape(1, 10)
    
    for i in range(num_games):
        winner, match_moves = UCTPlayGame(NewPlayer, save_moves = True)
        # winning_moves = GetWinnerMoves(winner, match_moves)
        # moves = np.append(moves, winning_moves, axis=0)
        
        if(winner == 0):
            draw += 1
        
        if(winner == 1):
            wins += 1
            
        if(winner == 2):
            loss += 1
    
    np.savetxt("./data/" + save_file + ".csv", moves, delimiter=",", fmt="%s")
    
    print("Matches played:", num_games)
    print("Wins:", wins)
    print("Draws:", draw)
    print("Losses:", loss)
    print("-------------")
    print("DT Win Rate:", wins / num_games)
    print("DT No-Loss Rate:", (wins + draw) / num_games)
        