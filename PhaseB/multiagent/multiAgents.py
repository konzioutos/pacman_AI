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
import random
import util

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
        scores = [self.evaluationFunction(
            gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(
            len(scores)) if scores[index] == bestScore]
        # Pick randomly among the best
        chosenIndex = random.choice(bestIndices)

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
        newFood = successorGameState.getFood()

        "*** YOUR CODE HERE ***"
        score = successorGameState.getScore()

        minFoodDist = float("+inf")
        for fpos in newFood.asList():
            if(util.manhattanDistance(fpos, newPos) < minFoodDist):
                minFoodDist = util.manhattanDistance(
                    fpos, newPos)  # apostasi apo to fagito
        # print("FOOD", minFoodDist)

        minGhostDis = float("+inf")
        for gpos in successorGameState.getGhostPositions():
            if(util.manhattanDistance(newPos, gpos) < minGhostDis):
                minGhostDis = util.manhattanDistance(
                    newPos, gpos)  # apostasi apo to fantasma
            # otan plisiazei to fantasma para poli

            if util.manhattanDistance(newPos, gpos) < 5:
                score -= 100
        # print("GHOST", minGhostDis)

        score += minGhostDis/minFoodDist  # ypologismos tou score

        if action == Directions.STOP:  # otan stamataei na kineitai
            score -= 100

        return score


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

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        self.index = 0  # Pacman is always agent index 0
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
        # util.raiseNotDefined()

        def minimax(gameState, depth, agentIndex):
            if agentIndex == 0:  # ftiaxnw to epomeno depth
                new_depth = depth-1
            else:
                new_depth = depth
            # ftiaxnw ton epomeno agent
            new_index = (agentIndex + 1) % gameState.getNumAgents()
            if new_depth == 0 or gameState.isWin() or gameState.isLose():
                return self.evaluationFunction(gameState), None

            if agentIndex == 0:
                #max_list = {}
                max_value = float("-inf")
                max_action = None
                for action in gameState.getLegalActions(agentIndex):
                    succ = gameState.generateSuccessor(
                        agentIndex, action)  # diadoxous gia pacman
                    value = minimax(succ, new_depth, new_index)[0]
                    if value > max_value:
                        max_value = value
                        max_action = action
                return max_value, max_action  # epistrefw to value kai action

            else:
                #min_list = {}
                min_value = float("+inf")
                min_action = None
                for action in gameState.getLegalActions(agentIndex):
                    succ = gameState.generateSuccessor(
                        agentIndex, action)  # diadoxous gia fantasmata
                    value = minimax(succ, new_depth, new_index)[0]
                    if value < min_value:
                        min_value = value
                        min_action = action
                return min_value, min_action  # epistrefw to value kai action

        return minimax(gameState, self.depth+1, self.index)[1]


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        # util.raiseNotDefined()

        def AlphaBeta(gameState):
            max_val = float("-inf")
            alpha = float("-inf")
            beta = float("inf")
            next_action = None
            for action in gameState.getLegalActions(0):
                succ = gameState.generateSuccessor(
                    0, action)  # pairnw diadoxous gia pacman
                value = minValue(succ, 0, 1, alpha, beta)
                if(value > max_val):  # vriskw th kaliteri epomeni energeia
                    max_val = value
                    next_action = action
                if value > beta:
                    return next_action
                alpha = max(alpha, max_val)
            return next_action

        def maxValue(gameState, depth, alpha, beta):
            depth += 1
            if gameState.isWin() == True or gameState.isLose() == True or depth == self.depth:
                return self.evaluationFunction(gameState)
            max_val = float("-inf")
            for action in gameState.getLegalActions(0):
                succ = gameState.generateSuccessor(
                    0, action)  # pairnw diadoxous gia pacman
                max_val = max(max_val, minValue(succ, depth, 1, alpha, beta))
                if max_val > beta:  # an symvei auto epistrefw
                    return max_val
                alpha = max(alpha, max_val)
            return max_val

        def minValue(gameState, depth, agentIndex, alpha, beta):
            if gameState.isWin() == True or gameState.isLose() == True or depth == self.depth:
                return self.evaluationFunction(gameState)
            min_value = float('inf')
            for action in gameState.getLegalActions(agentIndex):
                succ = gameState.generateSuccessor(
                    agentIndex, action)  # pairnw diadoxous gia ta fantasmata
                if agentIndex == gameState.getNumAgents() - 1:  # teleutaio fantasma
                    min_value = min(min_value, maxValue(
                        succ, depth, alpha, beta))
                    if alpha > min_value:
                        return min_value
                    beta = min(beta, min_value)
                else:

                    min_value = min(min_value, minValue(succ, depth,
                                                        agentIndex + 1, alpha, beta))  # kalw anadromi gia ta ypoloipa fantasmata
                if min_value < alpha:  # an symvei ayto epistrefw
                    return min_value
                beta = min(beta, min_value)
            return min_value

        return AlphaBeta(gameState)


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction
          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
          The expectimax function returns a tuple of (actions,
        """
        "*** YOUR CODE HERE ***"
        # util.raiseNotDefined()
        def expectimax(gameState, depth, agentIndex):
            if gameState.isWin() or gameState.isLose() or depth == 0:
                return self.evaluationFunction(gameState), None
            if agentIndex == gameState.getNumAgents():  # ftiaxnw to epomeno vathos kai to index tou epomenou agent
                new_depth = depth-1
                new_index = 1
            else:
                new_depth = depth
                new_index = agentIndex+1

            if agentIndex == 1:  # max, an eimaste edw epistrefei to megisto value kai action apo ton successor
                max_value = float("-inf")
                max_action = None
                for action in gameState.getLegalActions(0):
                    succ = gameState.generateSuccessor(
                        0, action)  # pairnw diadoxous gia pacman
                    value = expectimax(succ, depth, 2)[0]
                    if value > max_value:  # vriskw to value kai action
                        max_value = value
                        max_action = action
                return max_value, max_action

            else:  # chance node,an eimaste edw epistrefw to average value apo tous diadoxous
                min_value = float("+inf")
                min_action = None
                min_list = []
                for action in gameState.getLegalActions(agentIndex-1):
                    succ = gameState.generateSuccessor(
                        agentIndex-1, action)  # pairnw diadoxous gia fantasmata
                    if agentIndex == gameState.getNumAgents():
                        value = expectimax(succ, new_depth, new_index)[0]
                    else:
                        value = expectimax(succ, new_depth, new_index)[0]
                    min_list.append(value)
                    if value < min_value:  # vriskw to value kai action
                        min_value = value
                        min_action = action
                # return min_value, min_action
                return float(sum(min_list)/len(min_list)), min_action

        return expectimax(gameState, self.depth, 1)[1]


def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).
      DESCRIPTION: <write something here so we know what you did>
      Evaluate state by  :
            * closest food
            * food left
            * capsules left
            * distance to ghost
    """
    "*** YOUR CODE HERE ***"
    score = currentGameState.getScore()
    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newGhostStates = currentGameState.getGhostStates()

    newScaredTimes = []
    foodDist = []
    ghostDist = []

    for ghostState in newGhostStates:
        # gia na dw an fovatai to fantasma
        newScaredTimes.append(ghostState.scaredTimer)

    foodsLeft = len(newFood.asList())  # posa fagita emeinan

    sum_scared = sum(newScaredTimes)

    for fpos in newFood.asList():
        foodDist.append(util.manhattanDistance(newPos, fpos))

    if (newFood.asList()):  # kontinotero kai makritero fagito
        closestFood = min(foodDist)
        furthestFood = max(foodDist)
    else:
        return score

    for ghost in newGhostStates:
        ghostDist.append(manhattanDistance(newPos, ghost.getPosition()))

    minGhostDist = min(ghostDist)  # to pio kontino fantasma sto pacman

    if (sum_scared > 0):  # an einai fovismeno to fantasma piginaino pros ekei
        score += -minGhostDist - closestFood

    if (foodsLeft == 1):  # an einai ena fagito mono tote exoume mono closestFood giati einai to idio
        score += minGhostDist - closestFood

    else:
        score += minGhostDist - (furthestFood + closestFood)

    return score
    # util.raiseNotDefined()


# Abbreviation
better = betterEvaluationFunction
