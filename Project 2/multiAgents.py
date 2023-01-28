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
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** CS3568 YOUR CODE HERE ***"
        "Decribe your function:"
        
        distance = []
        foodList = currentGameState.getFood().asList()
        
        # if action is stop it returns negative infinite
        if action == 'Stop':
            return -float("inf")
        
        # Iterate through all possible gostStates and returns negative infinate ...
        # ... if next pacman position is gostposition and scaredTimer for that gost is 0
        for ghostState in newGhostStates:
            if ghostState.getPosition() == newPos and ghostState.scaredTimer  == 0:
                return -float("inf") 
        
        # Iterate through all food list in all the remaining food positions, calculate the manhattenDistance
        # of the food to the new gost positions and appends the negative distance to list(distance)
        for food in foodList:
            dist = util.manhattanDistance(food, newPos)
            distance.append(-1 * dist)
        
        return successorGameState.getScore() + max(distance) # returns the max distance of the new pacman position to remaining food positions.

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

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** CS3568 YOUR CODE HERE ***"
        "PS. It is okay to define your own new functions. For example, value, min_function,max_function"
        
        numGhosts = gameState.getNumAgents() - 1
        return self.max_function(gameState, 1, numGhosts)

    def max_function(self, gameState, depth, numGhosts):
        
        agentIndex = 0
        initialGhostIndex = 1
        
        # validating whether current game state is win state or lost state, ...
        # ... returns the evaluationFunction of game state to get the score
        if(gameState.isWin() or gameState.isLose()):
          return self.evaluationFunction(gameState)
      
        # assigning negative infinity value to "v" variable.
        v = -float("inf")
        
        best_action = Directions.STOP # initializing the best action to STOP
        
        # Iterating through list of legal actions of the agent 
        for action in gameState.getLegalActions(agentIndex):
            
            # get the successor Value from minimum function with sending ...  
            # ... ghost index and depth and successor function to min_function.
            successor = gameState.generateSuccessor(agentIndex, action)  
            successorValue = self.min_function(successor, depth, initialGhostIndex, numGhosts)
            
            if successorValue > v:
                v = successorValue
                best_action = action  # get the action at depth 0 to choose best action for pacman to traverse through optimal path.

        if depth > 1:
            return v
        
        return best_action

    def min_function(self, gameState, depth, agentIndex, numGhosts):
        
        # validating whether current game state is win state or lost state, ...
        # ... returns the evaluationFunction of game state to get the score
        if gameState.isWin() or gameState.isLose():
          return self.evaluationFunction(gameState)
        
        # assigning positive infinity value to "v" variable.
        v = float("inf")
        
        # get the successors of possible legal actions of the agent.
        legalActions = gameState.getLegalActions(agentIndex)
        successors = [gameState.generateSuccessor(agentIndex, action) for action in legalActions]
        
        # validating the agentIndex count is equal to number of ghosts to choose min/max functions.
        if agentIndex == numGhosts:
            
            # validating whether depth reached max depth, to choose 
            # minimum value of max function of successor states or minimum value 
            # of manhattan scores of next states.
            if depth < self.depth:
                for successor in successors:
                    v = min(v, self.max_function(successor, depth + 1, numGhosts))
            else:
                for successor in successors:
                    v = min(v, self.evaluationFunction(successor))
        
        else:
            
            # choosing the minimum value of min function of successor states i.e from remaining ghost states.
            for successor in successors:
                v = min(v, self.min_function(successor, depth, agentIndex + 1, numGhosts))
        
        return v

        util.raiseNotDefined()

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** CS3568 YOUR CODE HERE ***"
        "PS. It is okay to define your own new functions. For example, value, min_function,max_function"
        
        def max_function(state, agentIndex, depth, alpha, beta):

            agentIndex = 0
            legalActions = state.getLegalActions(agentIndex) #initializing legal actions and agent index
            
            # validating whether no legal actions to pacman and current depth is the max depth of the tree.
            if not legalActions  or depth == self.depth:
                return self.evaluationFunction(state)
            
            # initializing the negative infinity to "v" variable
            v = -float("inf")
            
            # Iterate through all legalActions of pacman agent, get the possible successor states,
            # update v variable with the maximum value of variable v and min function value.
            for action in legalActions:    
                successor = state.generateSuccessor(agentIndex, action)
                v = max(v, min_function(successor, agentIndex + 1, depth + 1, alpha, beta))
                
                # compare v, beta, to return v if v is greater.
                if v > beta:
                    return v
                
                # update the alpha with maximum of aplha and v
                alpha = max(alpha, v)
            
            return v
        
        def min_function(state, agentIndex, depth, alpha, beta):
            
            # get the number of ghosts count and possible legal actions of agents 
            numGhosts = gameState.getNumAgents() - 1
            legalActions = state.getLegalActions(agentIndex)
            
            # If no legal actions, return the manhattan distance of gamestate
            if not legalActions:
                return self.evaluationFunction(state)
            
            # initializing the infinate value to v variable.
            v = float("inf")
            
            # validate if agent index is equal to number of ghosts, then choose 
            # minimum of v, max function, else minimum of min function.
            if agentIndex == numGhosts:
                for action in legalActions:
                    
                    # update v with minimum of v, max function
                    successor = state.generateSuccessor(agentIndex, action)
                    v =  min(v, max_function(successor, agentIndex,  depth, alpha, beta))
                    
                    if v < alpha:
                        return v
                    
                    # update beta with minimum of beta, v
                    beta = min(beta, v)

            else:
                for action in legalActions:
                    
                    # update v with minimum of v, min function
                    successor = state.generateSuccessor(agentIndex, action)
                    v =  min(v, min_function(successor, agentIndex + 1, depth, alpha, beta))
                    
                    if v < alpha:
                        return v
                    
                    # update beta with min of beta, v
                    beta = min(beta, v)

            return v
        
        # get the legal actions of pacman agent.
        legalActions = gameState.getLegalActions(0)
        alpha = -float("inf")
        beta = float("inf")
        
        # iterate through legal actions, get the values of min functions at root node,
        # based on the alpha beta pruning, return the max value of that.
        allActions = {}
        for action in legalActions:
            
            successor = gameState.generateSuccessor(0, action)
            value = min_function(successor, 1, 1, alpha, beta)
            allActions[action] = value

            if value > beta:
                return action
            alpha = max(value, alpha)

        return max(allActions, key=allActions.get)
        
        util.raiseNotDefined()

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
        "*** CS3568 YOUR CODE HERE ***"
        "PS. It is okay to define your own new functions. For example, value, min_function,max_function"
        
        def exp_function(state, agentIndex, depth):
            # Initialize number of actions for ghost agent and ghosts at initial state.
            numGhosts = gameState.getNumAgents() - 1
            legalActions = state.getLegalActions(agentIndex)
        	
        	# if no legal actions, return manhattan distance of the state.
            if not legalActions:
                return self.evaluationFunction(state)
        	# initializing the probability and v value
            ev = 0
            probabilty = 1.0 / len(legalActions)
        	
        	# Iterating through all legal actions, for all ghost agents choose exp_function,
        	# else if pacman agent, choose max_function, based on that update v value with probability.
            for action in legalActions:
                
                if agentIndex == numGhosts:
                    exp_value =  max_function(state.generateSuccessor(agentIndex, action), agentIndex,  depth)
                
                else:
                    exp_value = exp_function(state.generateSuccessor(agentIndex, action), agentIndex + 1, depth)
                
                ev += exp_value * probabilty
            return ev
        
        
        
        def max_function(state, agentIndex, depth):
                    
        	# initialize the agentIndex and legal actions of pacman agent.
        	agentIndex = 0
        	legalActions = state.getLegalActions(agentIndex)
        	
        	# validate if no legal actions and depth is max depth, then return manhattan distance. 
        	if not legalActions  or depth == self.depth:
        		return self.evaluationFunction(state)
        	
        	# update v with maximum value of expecitvalue of successor states.
        	v =  max(exp_function(state.generateSuccessor(agentIndex, action), agentIndex + 1, depth + 1) for action in legalActions)
        
        	return v
        
        actions = gameState.getLegalActions(0)
        # Iterate through legal actions of pacman at the initial state and calculate expecti-max
        # of all actions, return the maximum of all actions.
        allActions = {}
        for action in actions:
        	allActions[action] = exp_function(gameState.generateSuccessor(0, action), 1, 1)
        
        return max(allActions, key=allActions.get)

        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** CS3568 YOUR CODE HERE ***"
    
    currentPos = currentGameState.getPacmanPosition()
    currentFood = currentGameState.getFood().asList()
    currentGhostStates = currentGameState.getGhostStates()
    currentScaredTimes = [ghostState.scaredTimer for ghostState in currentGhostStates]
    currentCapsule = currentGameState.getCapsules()

    if currentGameState.isWin():
        return 123456

    for state in currentGhostStates:
        if state.getPosition() == currentPos and state.scaredTimer == 1:
            return -123456

    score = 0

    # list of manhattan distances of food from current position of pacman
    foodDistance = [util.manhattanDistance(currentPos, food) for food in currentFood]
    nearestFood = min(foodDistance) # nearest food distance to pacman
    
    # decrease the current score with respect to length of current food
    score -= len(currentFood)

    if currentCapsule:
        
        # list of manhattan distance distance of capusles from pacman
        capsuleDistance = [util.manhattanDistance(currentPos, capsule) for capsule in currentCapsule]
        nearestCapsule = min(capsuleDistance) # nearest food capsule
        
        # increase the score with respect to distance of nearest capsule
        score += float(1/nearestCapsule)
    
    # list of manhattan distances of ghosts from current position of pacman
    currentGhostDistances = [util.manhattanDistance(currentPos, ghost.getPosition()) for ghost in currentGameState.getGhostStates()]
    nearestCurrentGhost = min(currentGhostDistances) # nearest ghost distance
    scaredTime = sum(currentScaredTimes)   # sum of scared times
    
    # validating nearest ghost distance greater than or equal to 1 and also validating the scared time of ghost.
    if nearestCurrentGhost >= 1:
        if scaredTime < 0:
            score -= 1/nearestCurrentGhost
        else:
            score += 1/nearestCurrentGhost

    return currentGameState.getScore() + score


    util.raiseNotDefined()


better = betterEvaluationFunction
