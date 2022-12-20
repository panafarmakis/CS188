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
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem: SearchProblem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    start_state = problem.getStartState()  # current state of problem
    start_node = (start_state, None, [])  # node = (curr_state,prev_node,list_of_directions)
    frontier_dfs = util.Stack()  # create Stack for DFS (LIFO)
    reached = set()  # create empty set to keep track of reached states
    frontier_dfs.push(start_node)
    while not frontier_dfs.isEmpty():
        curr_node = frontier_dfs.pop()
        curr_state = curr_node[0]  # state of curr_node
        # prev_node = curr_node[1]  # previous node of curr_node
        if problem.isGoalState(curr_state):
            return curr_node[2]
        if curr_state not in reached:
            reached.add(curr_state)
            for child_state in problem.getSuccessors(curr_state):  # child state = [prev_state/successor,action,
                # stepCost]
                child_node = (child_state[0], curr_node, curr_node[2] + [child_state[1]])
                frontier_dfs.push(child_node)

    return None


def breadthFirstSearch(problem: SearchProblem):
    """Search the shallowest nodes in the search tree first."""
    start_state = problem.getStartState()  # current state of problem
    start_node = (start_state, None, [])  # node = (curr_state,prev_node,list_of_directions)
    frontier_bfs = util.Queue()  # create Queue for BFS
    reached = set()  # create empty set to keep track of reached states
    frontier_bfs.push(start_node)
    while not frontier_bfs.isEmpty():
        curr_node = frontier_bfs.pop()
        curr_state = curr_node[0]  # state of curr_node
        if problem.isGoalState(curr_state):
            return curr_node[2]
        if curr_state not in reached:
            reached.add(curr_state)
            for child_state in problem.getSuccessors(curr_state):  # child state = [prev_state/successor,action,
                # stepCost]
                child_node = (child_state[0], curr_node, curr_node[2] + [child_state[1]])
                frontier_bfs.push(child_node)

    return None

def uniformCostSearch(problem: SearchProblem):
    """Search the node of least total cost first."""

    def priorityFunction(node):
        actual_cost = problem.getCostOfActions(node[2])
        return actual_cost

    start_state = problem.getStartState()  # current state of problem
    start_node = (start_state, None, [])  # node = (curr_state,prev_node,list_of_directions)
    frontier_ucs = util.PriorityQueueWithFunction(priorityFunction)  # create PriorityQ
    # for UCS (with a priority function)
    reached = set()  # create empty set
    frontier_ucs.push(start_node)
    while not frontier_ucs.isEmpty():
        curr_node = frontier_ucs.pop()
        curr_state = curr_node[0]  # state of curr_node
        # prev_node = curr_node[1]  # previous node of curr_node
        if problem.isGoalState(curr_state):
            return curr_node[2]
        if curr_state not in reached:
            reached.add(curr_state)
            for child_state in problem.getSuccessors(curr_state):  # child state = [prev_state/successor,action,
                # stepCost]
                child_node = (child_state[0], curr_node, curr_node[2] + [child_state[1]])
                frontier_ucs.push(child_node)

    return None


def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem: SearchProblem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    def priorityFunction(node):
        actual_cost = problem.getCostOfActions(node[2])
        return actual_cost + heuristic(node[0], problem)  # heuristic(state,problem)

    start_state = problem.getStartState()
    start_node = (start_state, None, [])  # node = (curr_state,prev_node,list_of_directions)
    frontier_astar = util.PriorityQueueWithFunction(priorityFunction)  # create PriorityQ
    # for A* with Priority Function (action cost + Heuristic)
    reached = set()  # create empty set to keep track of reached states
    frontier_astar.push(start_node)
    while not frontier_astar.isEmpty():
        curr_node = frontier_astar.pop()
        curr_state = curr_node[0]  # state of curr_node
        if problem.isGoalState(curr_state):
            return curr_node[2]
        if curr_state not in reached:
            reached.add(curr_state)
            for child_state in problem.getSuccessors(curr_state):  # child state = [prev_state/successor,action,
                # stepCost]
                child_node = (child_state[0], curr_node, curr_node[2] + [child_state[1]])
                frontier_astar.push(child_node)
    return None

# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
