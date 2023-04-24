# search.py
# ---------
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util


class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return [s, s, w, s, w, w, s, w]


def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.
    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.
    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:
    print "Start:", problem.getStartState()
    print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    print "Start's successors:", problem.getSuccessors(problem.getStartState())
    """

    "*** YOUR CODE HERE ***"

    s = util.Stack()  # xrisimopoiw stiva
    visited = set()  # ena set gia tous komvous pou exw episkefthei
    # kanw push ton arxiko komvo kai to keno monopati tou
    s.push((problem.getStartState(), []))

    while not s.isEmpty():
        curr = s.pop()  # kanoume pop kai pairnume to komvo kai to path
        curr_node = curr[0]
        curr_path = curr[1]

        if problem.isGoalState(curr_node):  # otan ftanoume to stoxo
            print("DFS DONE")
            return curr_path
        else:
            if curr_node not in visited:  # mpainoume edw an den exei episkeftei o komvos
                visited.add(curr_node)  # ton prostheoume sto set
                successors = problem.getSuccessors(
                    curr_node)  # pairnoume tous diadoxous
                for succ in successors:  # kai gia to kathe ena pairnoume ton komvo kai to path
                    next_node = succ[0]
                    next_path = succ[1]
                    # ypologizoume to path apo thn arxi mexri ton komvo auton
                    path = curr_path+[next_path]
                    # kai ton kanoume push mazi me olokliro to path
                    s.push((next_node, path))
    util.raiseNotDefined()


def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"

    # einai h idia diadikasia me parapanw apla ayth th fora douleoume me oura

    queue = util.Queue()
    visited = set()
    queue.push((problem.getStartState(), []))

    while not queue.isEmpty():
        curr = queue.pop()
        curr_node = curr[0]
        curr_path = curr[1]

        if problem.isGoalState(curr_node):
            print("BFS DONE")
            return curr_path
        else:
            if curr_node not in visited:
                visited.add(curr_node)
                successors = problem.getSuccessors(curr_node)
                for succ in successors:
                    next_node = succ[0]
                    next_path = succ[1]
                    path = curr_path+[next_path]
                    queue.push((next_node, path))

    util.raiseNotDefined()


def uniformCostSearch(problem):
    "Search the node of least total cost first. "
    "*** YOUR CODE HERE ***"

    util.raiseNotDefined()


def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0


def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"

    # paromoia diadikasia alla twra me oura proteraiotitas

    p_q = util.PriorityQueue()
    visited = set()
    p_q.push((problem.getStartState(), []), heuristic(  # kanoume push kai to kostos alla kai thn eurestiki synarthsh
        problem.getStartState(), problem))

    while not p_q.isEmpty():
        curr = p_q.pop()
        curr_node = curr[0]
        curr_path = curr[1]
        if problem.isGoalState(curr_node):
            print("A* DONE")
            return curr_path
        else:
            if curr_node not in visited:
                visited.add(curr_node)
                successors = problem.getSuccessors(curr_node)
                for succ in successors:
                    next_node = succ[0]
                    next_path = succ[1]
                    path = curr_path+[next_path]
                    # ypologizoume to kostos tou monopatiou
                    cost = problem.getCostOfActions(path)

                    # kalipto th periptwsh pou h euretikh synartisi einai none
                    if(heuristic(next_node, problem) is not None):
                        p_q.push((next_node, path), cost +  # an den einai kanw push to komvo to monopati kai to costos auksimeno me to euristiko kostos
                                 heuristic(next_node, problem))
                    else:
                        p_q.push((next_node, path), cost)

    util.raiseNotDefined()


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
