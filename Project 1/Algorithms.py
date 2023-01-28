import util

class DFS(object):
    def depthFirstSearch(self, problem):
        """
        Search the deepest nodes in the search tree first
        [2nd Edition: p 75, 3rd Edition: p 87]

        Your search algorithm needs to return a list of actions that reaches
        the goal.  Make sure to implement a graph search algorithm
        [2nd Edition: Fig. 3.18, 3rd Edition: Fig 3.7].

        To get started, you might want to try some of these simple commands to
        understand the search problem that is being passed in:

        print "Start:", problem.getStartState()
        print "Is the start a goal?", problem.isGoalState(problem.getStartState())
        print "Start's successors:", problem.getSuccessors(problem.getStartState())
        """
        "*** TTU CS3568 YOUR CODE HERE ***"

        possibleState = util.Stack()          # Reading stack data type from util
        startState = problem.getStartState()  # Assigning start state to a variable
        visitedStates = []                    # Assigning start state as default assigned variable
        possibleState.push(([], startState))  # push the start state to stack variables
        
        
        # looging through possible states until not empty
        while not possibleState.isEmpty():
            
            actions, currentVisitedState = possibleState.pop()  # pop the LIFO variable from stack
            visitedStates.append(currentVisitedState)           # visited nodes are appended here to expand all possible nodes.
            
            if problem.isGoalState(currentVisitedState):      # validate whether current state is goal state.
                return actions
            
            nextStates = problem.getSuccessors(currentVisitedState)  # Get the child states to the current state
            
            for child in reversed(nextStates):                 # loop through child states in reverse order ...
                                                               # ... to push to stack the respective child ...
                if child[0] not in visitedStates:             # ... state and actions if child is not in visited states
                    # visitedStates.append(child[0])
                    action = child[1]
                    possibleState.push((actions + [action], child[0]))
        
        util.raiseNotDefined()

class BFS(object):
    def breadthFirstSearch(self, problem):
        "*** TTU CS5368 YOUR CODE HERE ***"
        
        possibleState = util.Queue()          # Assigning the Queue datatype from util
        startState = problem.getStartState()  # Assigning the starting node to variable
        visitedStates = [startState]          # Added start value to visited states list
        possibleState.push(([], startState))  # Added start state and actions to initial state
        
        # looping through possible states until not empty
        while not possibleState.isEmpty():
            actions, currentVisitedState = possibleState.pop()
            # visitedStates.append(currentVisitedState)
            
            if problem.isGoalState(currentVisitedState):    # validate whether current state is goal state
                return actions
            
            nextStates = problem.getSuccessors(currentVisitedState)  # Get the child states of the current state
            
            for state in nextStates:                        # loop through child states and push the child states ...
                                                            # ... and actions to Queue which are not in visited states
                if state[0] not in visitedStates:
                    visitedStates.append(state[0])          # To avoid expanding nodes twice in a case, visited states are appended here
                    possibleState.push((actions + [state[1]], state[0]))
        
        util.raiseNotDefined()

class UCS(object):
    def uniformCostSearch(self, problem):
        "*** TTU CS3568 YOUR CODE HERE ***"
        
        possibleState = util.PriorityQueue()            # Assigning the Priority Queue datatype from util
        visitedStates = {}                              # Initialize visited states with empty dictionary
        startState = problem.getStartState()            # Assign start state to a variable
        possibleState.push((0, [], startState), 0)      # Push the Initial state item with zero cost to priority queue
        
        while not possibleState.isEmpty():              # Looping until possible states is not empty
            
            cost, actions, currentVisitedState = possibleState.pop()  # Get the node with least cost from possible states
            
            if((currentVisitedState not in visitedStates) or 
                    (cost < visitedStates[currentVisitedState])):     # validate if node is not visited or cost is less than if prev visited cost
                
                visitedStates[currentVisitedState] = cost             # update visited state with current state and its cost
                
                if problem.isGoalState(currentVisitedState):          # return actions if Goal state is reached in the current state
                    return actions
                
                else:
                    nextStatesValues = problem.getSuccessors(         # If goal state not reached, get the successer nodes
                        currentVisitedState
                        )
                    
                    for nextState, nextAction, nextCost in nextStatesValues:    # Iterate through child/successor nodes 
                        
                        cummulativeActions = actions + [nextAction]
                        # cummilativeCost = cost + nextCost
                        cummulativeCost = problem.getCostOfActions(cummulativeActions) # Get cummulative actions and costs
                        
                        possibleState.update(   
                            (cummulativeCost, cummulativeActions, nextState),       # update the possible states with child/successor state items and costs
                            cummulativeCost)

        util.raiseNotDefined()


class aSearch (object):
    def nullHeuristic( state, problem=None):
        """
        A heuristic function estimates the cost from the current state to the nearest goal in the provided SearchProblem.  This heuristic is trivial.
        """
        return 0
    def aStarSearch(self,problem, heuristic=nullHeuristic):
        "Search the node that has the lowest combined cost and heuristic first."
        "*** TTU CS3568 YOUR CODE HERE ***"
        
        possibleState = util.PriorityQueue()          # Assign the priority Queue data type from util 
        visitedStates = []                            # Initialize the visitedStates variable with empty list
        startState = problem.getStartState()          # Assign the start state to a variable
        possibleState.push((0, [], startState), 0)    # Push the Initial state item with zero cost to priority queue
        
        while not possibleState.isEmpty():            # looping while the possible state is not Empty
            
            cost, actions, currentVisitedState = possibleState.pop()  # Get the state item with least cost from  priority queue
            
            visitedStates.append((cost, currentVisitedState))    # Append visited states with cost and current state
            # visitedStates[currentVisitedState] = cost
            
            if problem.isGoalState(currentVisitedState):     # validate if current state is Goal to return actions
                return actions
            
            else:
                nextStates = problem.getSuccessors(currentVisitedState)   # If Goal is not reached, Get the successors/child states
                
                for nextState, nextAction, nextCost in nextStates:        # Iterate through next states
                    
                    cummulativeActions = actions + [nextAction]
                    cummulativeCost = problem.getCostOfActions(cummulativeActions)   # Get cummulative actions and cost
                    # cummulativeCost = cost + nextCost
                    prev_visited = False
                    
                    for visited in visitedStates:                   # Gothrough all visited states to validate if current state ...
                        visitedCost, visitedState = visited         # ... is already visited and cummulative cost is greater.
                        
                        if((nextState == visitedState) and          
                            (cummulativeCost >= visitedCost)):
                            prev_visited = True
                    
                    if not prev_visited:                      # If current state is not visited then update successor/child state items ...
                        possibleState.push(                   # ... with cummulative cost and priority to queue as cummulative cost with heuristic cost.
                            (cummulativeCost, cummulativeActions, nextState),
                            cummulativeCost + heuristic(nextState, problem))
                        visitedStates.append((cummulativeCost, nextState))      
                        # visitedStates[nextState] = cummulativeCost
        
        util.raiseNotDefined()

