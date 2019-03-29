import os
import sys
import numpy as np
from GameAgent import Agent_Random, Agent_DT, OXOState, transform_OXO_state

def GetWinnerMoves(winner, match_moves):
    if(winner == 0): return np.empty((1, 10))
    winning_moves = match_moves[winner - 1].reshape(1, match_moves.shape[1])
    it = winner + 1

    while(it < match_moves.shape[0]):
        winning_moves = np.append(winning_moves, match_moves[it].reshape(1, match_moves.shape[1]), axis=0)
        it += 2
    
    return winning_moves

def UCTPlayGame(Player1, Player2, save_moves = False):
    """ Play a sample game between two UCT players where each player gets a different number 
        of UCT iterations (= simulations = tree nodes).
    """
    state = OXOState() # Play OXO
    moves_performed = np.array([], dtype=str).reshape(0, 10)
        
    while (state.GetResult(state.playerJustMoved) == -1.0):
        print(str(state))
        if state.playerJustMoved == 2:
            # Player 1 - X
            m = Player1.UCT(state, 100, verbose = False)
        else:
            # Player 2 - O
            m = Player2.UCT(state, 1000, verbose = False)
        print("Best Move for Player " + str(3 - state.playerJustMoved) + ": " + str(m) + "\n")
        
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

def Play(player1_ModelName, player2_ModelName, num_games, save_file, saveOnlyWins=False, savePerformance=False):
    wins = 0
    draw = 0
    loss = 0

    # Set up Player 1
    if(player1_ModelName != "Random"):
        Player1 = Agent_DT(player1_ModelName)
    else:
        Player1 = Agent_Random()
    
    # Set up Player 2
    if(player2_ModelName != "Random"):
        Player2 = Agent_DT(player2_ModelName)
    else:
        Player2 = Agent_Random()
    
    moves = np.array(["[0:0]", "[0:1]", "[0:2]", "[1:0]", "[1:1]", "[1:2]", "[2:0]", "[2:1]", "[2:2]", "Move"], dtype=str).reshape(1, 10)
    
    for i in range(num_games):
        winner, match_moves = UCTPlayGame(Player1, Player2, save_moves = True)
        if(saveOnlyWins):
            if(winner > 0):
                winning_moves = GetWinnerMoves(winner, match_moves)
                moves = np.append(moves, winning_moves, axis=0)
        else:
            moves = np.append(moves, match_moves, axis=0)
        
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

    if(savePerformance):
        if not os.path.exists('./performance/'): os.makedirs('./performance/')
        with open('./performance/' + player1_ModelName + '_performance.txt', 'w') as file:  # Use file to refer to the file object
            file.write("Matches played: "  + str(num_games))
            file.write("Wins: "            + str(wins))
            file.write("Draws: "           + str(draw))
            file.write("Losses: "          + str(loss))
            file.write("-----------------------------")
            file.write("DT Win Rate: "     + str(wins / num_games))
            file.write("DT No-Loss Rate: " + str((wins + draw) / num_games))

if __name__ == "__main__":
    """ Play a given number of games to the end
        DT vs MCTS
    """
    player1_ModelName = str(sys.argv[1])
    player2_ModelName = str(sys.argv[2])
    num_games         = int(sys.argv[3])
    save_file         = str(sys.argv[4])

    Play(player1_ModelName, player2_ModelName, num_games, save_file)
