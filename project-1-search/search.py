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
from game import Directions

class Node:
    def __init__ (self, previous, state, cost, previousDir):
        self.parent = previous
        self.cost = cost
        self.state = state
        self.previousDir = previousDir

    def getState(self):
        return self.state
    def getCost(self):
        return self.cost
    def getPreviousDir(self):
        return self.previousDir
    def getParent(self):
        return self.parent
    def __eq__(self, OtherNode):
        return self.getState() == OtherNode.getState()

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
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:"""

    currentNode =  problem.getStartState()
    fringe  = util.Stack()
    visited = set()
    directions = []
    newNode = Node(None, currentNode, 0, None)
    fringe.push(newNode)

    while not fringe.isEmpty():
        currentNode = fringe.pop()

        if problem.isGoalState(currentNode.getState()):
            while not currentNode.getPreviousDir() == None:
                directions.insert(0, currentNode.getPreviousDir())
                currentNode = currentNode.getParent()
            return directions

        else:
            visited.add(currentNode.getState())
            successors = problem.getSuccessors(currentNode.getState())
            #print successors
            for successor in successors:
                if successor[0] not in visited:
                        successorNode = Node(currentNode, successor[0], successor[2], successor[1])
                        fringe.push(successorNode)

    util.raiseNotDefined()

def breadthFirstSearch(problem):

    currentNode =  problem.getStartState()
    fringe  = util.Queue()
    visited = set()
    directions = []
    newNode = Node(None, currentNode, 0, None)
    fringe.push(newNode)

    while not fringe.isEmpty():
        currentNode = fringe.pop()

        if problem.isGoalState(currentNode.getState()):
            while not currentNode.getPreviousDir() == None:
                directions.insert(0, currentNode.getPreviousDir())
                currentNode = currentNode.getParent()
            return directions

        else:
            visited.add(currentNode.getState())
            successors = problem.getSuccessors(currentNode.getState())
            #print successors
            for successor in successors:
                if successor[0] not in visited:
                        successorNode = Node(currentNode, successor[0], successor[2], successor[1])
                        if successorNode not in fringe.list:
                            fringe.push(successorNode)

    util.raiseNotDefined()

def uniformCostSearch(problem):

    currentNode =  problem.getStartState()
    fringe  = util.PriorityQueue()
    visited = set()
    directions = []
    newNode = Node(None, currentNode, 0, None)
    fringe.push(newNode, 0)

    while not fringe.isEmpty():
        currentNode = fringe.pop()

        if problem.isGoalState(currentNode.getState()):
            while not currentNode.getPreviousDir() == None:
                directions.insert(0, currentNode.getPreviousDir())
                currentNode = currentNode.getParent()
            return directions

        else:
            visited.add(currentNode.getState())
            successors = problem.getSuccessors(currentNode.getState())
            #print successors
            for successor in successors:
                if successor[0] not in visited:
                    parentCost = currentNode.getCost()
                    totalCost = successor[2] + parentCost
                    #print totalCost
                    successorNode = Node(currentNode, successor[0], totalCost, successor[1])
                    fringe.update(successorNode, totalCost)

    util.raiseNotDefined()

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):

    currentNode =  problem.getStartState()
    fringe  = util.PriorityQueue()
    visited = set()
    directions = []
    newNode = Node(None, currentNode, 0, None)
    fringe.push(newNode, 0 + heuristic(newNode.getState(), problem))

    while not fringe.isEmpty():
        currentNode = fringe.pop()

        if problem.isGoalState(currentNode.getState()):
            while not currentNode.getPreviousDir() == None:
                directions.insert(0, currentNode.getPreviousDir())
                currentNode = currentNode.getParent()
            return directions

        else:
            visited.add(currentNode.getState())
            successors = problem.getSuccessors(currentNode.getState())
            #print successors
            for successor in successors:
                if successor[0] not in visited:
                    parentCost = currentNode.getCost()
                    totalCost = successor[2] + parentCost
                    successorNode = Node(currentNode, successor[0], totalCost, successor[1])
                    fringe.update(successorNode, totalCost + heuristic(successorNode.getState(), problem))


    util.raiseNotDefined()


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch