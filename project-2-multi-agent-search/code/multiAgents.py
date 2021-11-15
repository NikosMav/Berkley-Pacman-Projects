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

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
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

    def evaluationFunction(self, currentGameState, action):
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
        newFood = currentGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()

        #print newGhostStates[0]

        "Taking Food into consideration"
        foodList = newFood.asList()
        closestFood = float("inf")
        for food in foodList:
            closestFood = min(closestFood, manhattanDistance(newPos, food))

        "Taking Ghosts into consideration"
        for ghostState in successorGameState.getGhostPositions():
            distance = manhattanDistance(newPos, ghostState)
            if distance < 2:     #Very close to ghost (returns inf because the best choice is to move)
                return - float("inf")

        return -closestFood

def scoreEvaluationFunction(currentGameState):
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

    def getAction(self, gameState):
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
        """
        "*** YOUR CODE HERE ***"

        output = self.minMaxDecision(gameState, 0, 0)
        return output[0]

    def minMaxDecision(self, gameState, deepness, agentIndex):
        if agentIndex == gameState.getNumAgents():    #Pacman and ghosts completed their Actions.
            agentIndex = 0               #Next player is Pacman.
            deepness += 1                   #Depth increases by one after everyone played their turn.
        if (deepness==self.depth or gameState.isWin() or gameState.isLose()):
            return [None, self.evaluationFunction(gameState)]
        elif (agentIndex == 0):
            return self.maxValue(gameState, deepness, agentIndex)
        else:
            return self.minValue(gameState, deepness, agentIndex)

    def maxValue(self, gameState, deepness, agentIndex):
        bestAction = ["Stop", -float("inf")]

        for legalAction in gameState.getLegalActions(agentIndex):
            succValue = self.minMaxDecision(gameState.generateSuccessor(agentIndex, legalAction), deepness, agentIndex + 1)
            testVal = succValue[1]
            if testVal > bestAction[1]:
                bestAction = [legalAction, testVal]
        return bestAction

    def minValue(self, gameState, deepness, agentIndex):
        bestAction = ["Stop", float("inf")]

        for legalAction in gameState.getLegalActions(agentIndex):
            succValue = self.minMaxDecision(gameState.generateSuccessor(agentIndex, legalAction), deepness, agentIndex + 1)
            testVal = succValue[1]
            if testVal < bestAction[1]:
                bestAction = [legalAction, testVal]
        return bestAction


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        output = self.minMaxDecision(gameState, 0, 0, -float("inf"), float("inf"))
        return output[0]

    def minMaxDecision(self, gameState, deepness, agentIndex, a, b):
        if agentIndex == gameState.getNumAgents():
            agentIndex = 0
            deepness += 1
        if (deepness==self.depth or gameState.isWin() or gameState.isLose()):
            return [None, self.evaluationFunction(gameState)]
        elif (agentIndex == 0):
            return self.maxValue(gameState, deepness, agentIndex, a, b)
        else:
            return self.minValue(gameState, deepness, agentIndex, a, b)

    def maxValue(self, gameState, deepness, agentIndex, a, b):
        bestAction = ["Stop", -float("inf")]

        for legalAction in gameState.getLegalActions(agentIndex):
            succValue = self.minMaxDecision(gameState.generateSuccessor(agentIndex, legalAction), deepness, agentIndex + 1, a, b)
            testVal = succValue[1]
            if testVal > bestAction[1]:
                bestAction = [legalAction, testVal]
            if testVal > b:
                return [legalAction, testVal]
            a = max(a, testVal)
        return bestAction

    def minValue(self, gameState, deepness, agentIndex, a, b):
        bestAction = ["Stop", float("inf")]

        for legalAction in gameState.getLegalActions(agentIndex):
            succValue = self.minMaxDecision(gameState.generateSuccessor(agentIndex, legalAction), deepness, agentIndex + 1, a, b)
            testVal = succValue[1]
            if testVal < bestAction[1]:
                bestAction = [legalAction, testVal]
            if testVal < a:
                return [legalAction, testVal]
            b = min(b, testVal)
        return bestAction

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        outputList = self.expectiDecision(gameState, 0, 0)
        return outputList[0]

    def expectiDecision(self, gameState, deepness, agentIndex):
        if agentIndex == gameState.getNumAgents():
            agentIndex = 0
            deepness += 1
        if (deepness==self.depth or gameState.isWin() or gameState.isLose()):
            return [None, self.evaluationFunction(gameState)]
        elif (agentIndex == 0):
            return self.maxValue(gameState, deepness, agentIndex)
        else:
            return self.expectiValue(gameState, deepness, agentIndex)

    def maxValue(self, gameState, deepness, agentIndex):
        bestAction = ["Stop", -float("inf")]

        for legalAction in gameState.getLegalActions(agentIndex):
            succValue = self.expectiDecision(gameState.generateSuccessor(agentIndex, legalAction), deepness, agentIndex + 1)
            testVal = succValue[1]
            if testVal > bestAction[1]:
                bestAction = [legalAction, testVal]
        return bestAction

    def expectiValue(self, gameState, deepness, agentIndex):
        expectedAction = ["Stop", 0]

        ghostActions = gameState.getLegalActions(agentIndex)
        probability = 1.0/len(ghostActions)       #Choosing uniformly, probability for each action is equal for all actions.

        for legalAction in ghostActions:
            succValue = self.expectiDecision(gameState.generateSuccessor(agentIndex, legalAction), deepness, agentIndex + 1)
            testVal = succValue[1]
            expectedAction[0]= legalAction
            expectedAction[1] += testVal * probability

        return expectedAction

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """

    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()

    "Taking Food into consideration"
    foodList = newFood.asList()
    closestFood = float("inf")
    for food in foodList:
        closestFood = min(closestFood, manhattanDistance(newPos, food))

    "Taking Ghosts into consideration"
    for ghostState in currentGameState.getGhostPositions():
        distance = manhattanDistance(newPos, ghostState)
        if distance < 2:     #Very close to ghost (returns inf because the best choice is to move)
            return - float("inf")

    capsules = currentGameState.getCapsules()
    capsulesNum = len(capsules)

    return currentGameState.getScore() - capsulesNum + 1.0/closestFood

# Abbreviation
better = betterEvaluationFunction
