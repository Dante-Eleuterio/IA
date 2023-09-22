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
        legalMoves = gameState.getLegalActions()

        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) 

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
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood().asList()
        newGhostStates = successorGameState.getGhostStates()

        score = successorGameState.getScore()

        food_distances = [manhattanDistance(newPos, food) for food in newFood]
        if len(food_distances) > 0:
            min_food_distance = min(food_distances)
            score += 1.0 / min_food_distance

        for ghostState in newGhostStates:
            ghost_pos = ghostState.getPosition()
            ghost_distance = manhattanDistance(newPos, ghost_pos)
            if ghost_distance < 2:
                score -= 1000
            else:
                score -= 1.0 / ghost_distance

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
        self.index = 0 
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
        best_action = None
        best_value = float("-inf")

        legal_actions = gameState.getLegalActions(0)

        for action in legal_actions:
            successor = gameState.generateSuccessor(0, action)
            value = self.min_value(successor, 1, 0)  
            if value > best_value:
                best_value = value
                best_action = action

        return best_action
    
    def max_value(self, gameState, depth):
        if gameState.isWin() or gameState.isLose() or depth == self.depth:
            return self.evaluationFunction(gameState)

        value = float("-inf")
        
        legal_actions = gameState.getLegalActions(0)

        for action in legal_actions:
            successor = gameState.generateSuccessor(0, action)
            value = max(value, self.min_value(successor, 1, depth))

        return value

    def min_value(self, gameState, agent_index, depth):
        if gameState.isWin() or gameState.isLose() or depth == self.depth:
            return self.evaluationFunction(gameState)

        value = float("inf")

        legal_actions = gameState.getLegalActions(agent_index)

        if agent_index == gameState.getNumAgents() - 1:
            for action in legal_actions:
                successor = gameState.generateSuccessor(agent_index, action)
                value = min(value, self.max_value(successor, depth + 1))
        else:
            for action in legal_actions:
                successor = gameState.generateSuccessor(agent_index, action)
                value = min(value, self.min_value(successor, agent_index + 1, depth))

        return value

class AlphaBetaAgent(MultiAgentSearchAgent):
    def getAction(self, gameState):
        def max_value(gameState, depth, alpha, beta):
            current_depth = depth + 1
            if gameState.isWin() or gameState.isLose() or current_depth == self.depth:
                return self.evaluate(gameState)
            max_val = float("-inf")
            possible_actions = gameState.getLegalActions(0)
            alpha1 = alpha
            for action in possible_actions:
                successor = gameState.generateSuccessor(0, action)
                max_val = max(max_val, min_value(successor, current_depth, 1, alpha1, beta))
                if max_val > beta:
                    return max_val
                alpha1 = max(alpha1, max_val)
            return max_val

        def min_value(gameState, depth, agentIndex, alpha, beta):
            min_val = float("inf")
            if gameState.isWin() or gameState.isLose():
                return self.evaluate(gameState)
            possible_actions = gameState.getLegalActions(agentIndex)
            beta1 = beta
            for action in possible_actions:
                successor = gameState.generateSuccessor(agentIndex, action)
                if agentIndex == (gameState.getNumAgents() - 1):
                    min_val = min(min_val, max_value(successor, depth, alpha, beta1))
                    if min_val < alpha:
                        return min_val
                    beta1 = min(beta1, min_val)
                else:
                    min_val = min(min_val, min_value(successor, depth, agentIndex + 1, alpha, beta1))
                    if min_val < alpha:
                        return min_val
                    beta1 = min(beta1, min_val)
            return min_val

        possible_actions = gameState.getLegalActions(0)
        best_score = float("-inf")
        chosen_action = ''
        alpha = float("-inf")
        beta = float("inf")
        for action in possible_actions:
            next_state = gameState.generateSuccessor(0, action)
            score = min_value(next_state, 0, 1, alpha, beta)
            if score > best_score:
                chosen_action = action
                best_score = score
            if score > beta:
                return chosen_action
            alpha = max(alpha, score)
        return chosen_action
    def evaluate(self, gameState):
        return gameState.getScore()
class ExpectimaxAgent(MultiAgentSearchAgent):
    def getAction(self, gameState):
        def max_value(gameState, depth):
            current_depth = depth + 1
            if gameState.isWin() or gameState.isLose() or current_depth == self.depth:
                return self.evaluate(gameState)
            max_val = float("-inf")
            possible_actions = gameState.getLegalActions(0)
            for action in possible_actions:
                successor = gameState.generateSuccessor(0, action)
                max_val = max(max_val, exp_value(successor, current_depth, 1))
            return max_val

        def exp_value(gameState, depth, agentIndex):
            exp_val = 0.0
            if gameState.isWin() or gameState.isLose():
                return self.evaluate(gameState)
            possible_actions = gameState.getLegalActions(agentIndex)
            probability = 1.0 / len(possible_actions)
            for action in possible_actions:
                successor = gameState.generateSuccessor(agentIndex, action)
                if agentIndex == (gameState.getNumAgents() - 1):
                    exp_val += max_value(successor, depth) * probability
                else:
                    exp_val += exp_value(successor, depth, agentIndex + 1) * probability
            return exp_val

        possible_actions = gameState.getLegalActions(0)
        best_score = float("-inf")
        chosen_action = ''
        for action in possible_actions:
            next_state = gameState.generateSuccessor(0, action)
            score = exp_value(next_state, 0, 1)
            if score > best_score:
                chosen_action = action
                best_score = score
        return chosen_action

    def evaluate(self, gameState):
        return gameState.getScore()

def betterEvaluationFunction(currentGameState):
    pacmanPos = currentGameState.getPacmanPosition()

    food = currentGameState.getFood()

    ghostStates = currentGameState.getGhostStates()

    capsules = currentGameState.getCapsules()

    foodDistances = [manhattanDistance(pacmanPos, foodPos) for foodPos in food.asList()]
    minFoodDistance = min(foodDistances, default=1)  # Avoid division by zero
    foodScore = 1.0 / minFoodDistance

    capsuleDistances = [manhattanDistance(pacmanPos, capsule) for capsule in capsules]
    minCapsuleDistance = min(capsuleDistances, default=1)  # Avoid division by zero
    capsuleScore = 1.0 / minCapsuleDistance

    ghostDistances = [manhattanDistance(pacmanPos, ghost.getPosition()) for ghost in ghostStates]
    ghostScore = sum(1.0 / distance for distance in ghostDistances)

    gameScore = currentGameState.getScore()

    evaluation = (2 * gameScore) + (1.5 * foodScore) + (3 * capsuleScore) - (4 * ghostScore)

    return evaluation


# Abbreviation
better = betterEvaluationFunction
