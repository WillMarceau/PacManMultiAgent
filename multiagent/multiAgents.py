# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        newGhostPositions = [ghostState.getPosition() for ghostState in newGhostStates]
        prevPos = currentGameState.getPacmanPosition()

            
        ghostDistance = float('inf')
        foodCount = 0
        foodDistance = float('inf')
        scared = min(newScaredTimes)
        closeGhostDistance = float('inf')

        score = successorGameState.getScore()

        # pen for remaining food
        score -= 1.5 * foodCount

        # reward for eating food
        if currentGameState.getNumFood() > successorGameState.getNumFood():
            score += 50

        for element in newGhostPositions:
            ghostDistance = manhattanDistance(newPos, element)
            # big pen for being close to ghost
            if ghostDistance < 2:
                score -= 50
            else:
            # reward for being far in general
                score -= (1 / 1 + ghostDistance) * 1.5

        for i in range(newFood.width):
            for j in range(newFood.height):
                if newFood[i][j]:
                    foodCount += 1
                    if manhattanDistance(newPos, (i, j)) < foodDistance:
                        foodDistance = manhattanDistance(newPos, (i, j))

        
        # pen for remaining food
        score -= 5 * foodCount

        # no food left
        if foodDistance == float('inf'):
            score += 10
        elif foodDistance == 0:
            score += 10
        # reward for being close to food
        else:
            score -= 2 * foodDistance

        # reward for chasing scared ghosts
        for ghost in newGhostStates:
            if ghost.scaredTimer > 0:
                scaredDistance = manhattanDistance(newPos, ghost.getPosition())
                score += 25 / (scaredDistance + 1)
        



        "*** YOUR CODE HERE ***"
        return score

def scoreEvaluationFunction(currentGameState: GameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        bestMove = None
        bestScore = float('-inf')
        moves = gameState.getLegalActions(0)
        for move in moves:
            nextState = gameState.generateSuccessor(0, move)
            v = self.value(nextState, self.depth, 1)

            if v > bestScore:
                bestScore = v
                bestMove = move
        return bestMove

    def value(self, gameState: GameState, depth: int, index: int):

        if gameState.isWin():
            return self.evaluationFunction(gameState)
        elif gameState.isLose():
            return self.evaluationFunction(gameState)
        elif depth == 0:
            return self.evaluationFunction(gameState)
        else:
            # pacman turn
            if index == 0:
                return self.maxValue(gameState, depth, index)
            # ghost turn
            elif index < gameState.getNumAgents():
                return self.minValue(gameState, depth, index)
            # new depth reached, pacman turn again
            else:
               return self.value(gameState, depth - 1, index = 0)
    
    def maxValue(self, gameState: GameState, depth: int, index: int):
            
            v = float('-inf')
            # get moves
            moves = gameState.getLegalActions(index)

            # increment index
            #self.index += 1

            # for valid moves
            for move in moves:

                # generate next state
                nextState = gameState.generateSuccessor(index, move)

                # take max of successor and v in get action func
                v = max(v, self.value(nextState, depth, index + 1))
            

            # return v
            return v

    def minValue(self, gameState: GameState, depth: int, index: int):
        v = float('inf')

        moves = gameState.getLegalActions(index)

        # increment index
        #self.index += 1

        # for valid moves
        for move in moves:

            # generate next state
            nextState = gameState.generateSuccessor(index, move)

            # take max of successor and v in get action func
            v = min(v, self.value(nextState, depth, index + 1))

        # return v
        return v

        


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        bestMove = None
        bestScore = float('-inf')
        alpha = float('-inf')
        beta = float('inf')
        moves = gameState.getLegalActions(0)
        for move in moves:
            nextState = gameState.generateSuccessor(0, move)
            v = self.value(nextState, self.depth, 1, alpha, beta)

            if v > bestScore:
                bestScore = v
                bestMove = move
            alpha = max(alpha, bestScore)
        return bestMove
    
    def value(self, gameState: GameState, depth: int, index: int, alpha: int, beta: int):

        if gameState.isWin():
            return scoreEvaluationFunction(gameState)
        elif gameState.isLose():
            return scoreEvaluationFunction(gameState)
        elif depth == 0:
            return scoreEvaluationFunction(gameState)
        else:
            # pacman turn
            if index == 0:
                return self.maxValue(gameState, depth, index, alpha, beta)
            # ghost turn
            elif index < gameState.getNumAgents():
                return self.minValue(gameState, depth, index, alpha, beta)
            # new depth reached, pacman turn again
            else:
               return self.value(gameState, depth - 1, 0, alpha, beta)
    
    def maxValue(self, gameState: GameState, depth: int, index: int, alpha: int, beta: int):
            
            v = float('-inf')
            # get moves
            moves = gameState.getLegalActions(index)

            # increment index
            #self.index += 1
            # for valid moves
            for move in moves:

                # generate next state
                nextState = gameState.generateSuccessor(index, move)

                # take max of successor and v in get action func
                v = max(v, self.value(nextState, depth, (index + 1), alpha, beta))
                alpha = max(alpha, v)
                if v > beta:
                    return v
                #alpha = max(alpha, v)
            
            # return v
            return v

    def minValue(self, gameState: GameState, depth: int, index: int, alpha: int, beta: int):
        v = float('inf')

        moves = gameState.getLegalActions(index)

        # for valid moves
        for move in moves:

            # generate next state
            nextState = gameState.generateSuccessor(index, move)

            # take max of successor and v in get action func
            v = min(v, self.value(nextState, depth, (index + 1), alpha, beta))
            beta = min(beta, v)
            if v < alpha:
                return v

        # return v
        return v


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
       # print("RUN THIS")
        bestMove = None
        bestScore = float('-inf')
        moves = gameState.getLegalActions(0)
        for move in moves:
            nextState = gameState.generateSuccessor(0, move)
            v = self.value(nextState, self.depth, 1)

            if v > bestScore:
                bestScore = v
                bestMove = move
        return bestMove
    
    def value(self, gameState: GameState, depth: int, index: int):

        if gameState.isWin():
            return self.evaluationFunction(gameState)
        elif gameState.isLose():
            return self.evaluationFunction(gameState)
        elif depth == 0:
            return self.evaluationFunction(gameState)
        else:
            # pacman turn
            if index == 0:
                return self.maxValue(gameState, depth, index)
            # ghost turn
            elif index < gameState.getNumAgents():
                return self.expValue(gameState, depth, index)
            # new depth reached, pacman turn again
            else:
               return self.value(gameState, depth - 1, index = 0)
    
    def maxValue(self, gameState: GameState, depth: int, index: int):
            
            v = float('-inf')
            # get moves
            moves = gameState.getLegalActions(index)

            # increment index
            #self.index += 1

            # for valid moves
            for move in moves:

                # generate next state
                nextState = gameState.generateSuccessor(index, move)

                # take max of successor and v in get action func
                v = max(v, self.value(nextState, depth, index + 1))
            

            # return v
            return v

    def expValue(self, gameState: GameState, depth: int, index: int):
        v = 0

        # get moves
        moves = gameState.getLegalActions(index)

        # for valid moves
        for move in moves:

            # generate successor
            nextState = gameState.generateSuccessor(index, move)
            prob = 1 / len(moves)

            v += prob * self.value(nextState, depth, index + 1)
        
        return v


def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    # inits
    pos = currentGameState.getPacmanPosition()
    food = currentGameState.getFood()
    ghostStates = currentGameState.getGhostStates()
    ghostPositions = [ghostState.getPosition() for ghostState in ghostStates]
    capsules = currentGameState.getCapsules()

    ghostDistance = float('inf')
    foodCount = 0
    foodDistance = float('inf')
    secondFoodDistance = float('inf')
    

    # base score
    score = currentGameState.getScore()
    #score = 0

    #ghosts
    for element in ghostPositions:
        ghostDistance = manhattanDistance(pos, element)
        if ghostDistance < 1:
            score -= 50
        else:
            score += 0

    capDistance = float('inf')
    for cap in capsules:
        if capDistance > manhattanDistance(pos, cap):
            capDistance = manhattanDistance(pos, cap)

    if capDistance == float('inf'):
        capDistance = 0

    if capDistance == 0:
        score 
    # 3 
    if (ghostDistance < 3):
        if capDistance == 0:
            score -= capDistance * 25
        else:
            score -= capDistance * 2
    else:
        score += 0
    # food
    foodDistanceArray = []
    tmpDistance = ('inf')
    for i in range(food.width):
        for j in range(food.height):
            if food[i][j]:
                foodCount += 1
                tmpDistance = manhattanDistance(pos, (i, j))
                foodDistanceArray.append(tmpDistance)
    
    #print(foodDistanceArray)
    if len(foodDistanceArray) >= 2:
        foodDistance = foodDistanceArray[0]
        secondFoodDistance = foodDistanceArray[1]
    elif len(foodDistanceArray) >= 1:
        foodDistance = foodDistanceArray[0]
        secondFoodDistance = -40
    else:
        foodDistance = -40
        secondFoodDistance = -40
    # pen for remaining food
    score -= 200 * foodCount
    # no food left
    if foodDistance == float('inf'):
        score += 10
    elif foodDistance == 0:
        score += 20
    else:
        score -= 4 * foodDistance

    if secondFoodDistance == float('inf'):
        score = 0
    else:
        score -= 4.5 * secondFoodDistance
    
    #print(foodDistance, secondFoodDistance, currentGameState.getScore())

    
    # reward for chasing scared ghosts
    for ghost in ghostStates:
        if ghost.scaredTimer > 0:
            scaredDistance = manhattanDistance(pos, ghost.getPosition())
            score += 25 / (scaredDistance + 1)
    #print("score", score)
    return score
    

# Abbreviation
better = betterEvaluationFunction
