import copy
import csv
import re
import time
import random
from func_timeout import func_timeout
from func_timeout import FunctionTimedOut
from collections import deque
from heapq import heappush, heappop


class Node:
    def __init__(self, board: list):
        # Initialize the board and size attributes
        self.board = board
        self.size = len(board)
        self.back = None

    def __eq__(self, other) -> bool:
        res = True
        # Iterate through all the tiles on the board
        for i in range(self.size):
            for j in range(self.size):
                # If any tiles are different, set the flag to False and break out of the loop
                if self.board[i][j] != other.board[i][j]:
                    res = False
                    break
        return res

    def __lt__(self, other) -> bool:
        # Return whether the size of this node is less than or equal to the size of the other node
        return self.size <= other.size

    def not_in(self, others) -> bool:
        res = True
        # Iterate through the list of other nodes
        for i in range(len(others)):
            # If this node is equal to any of the other nodes, set the flag to False and break out of the loop
            if self.__eq__(others[i]):
                res = False
                break
        return res

    def get_position(self) -> int:
        # Iterate through all the tiles on the board
        for i in range(self.size):
            for j in range(self.size):
                # If the current tile is empty, return its position
                if self.board[i][j] == 0:
                    # Return the position as a single integer, calculated by multiplying the row index by the size of
                    # the board and adding the column index + 1
                    return i * self.size + j + 1


########################################################
# Heuristic 1 : number of displaced tiles
########################################################
def heuristic1(state: Node) -> int:
    # Initialize the score for h
    score1 = state.size * state.size - 1 - state.get_position()
    # Iterate through all the tiles on the board
    for x in range(state.size):
        for y in range(state.size):
            # If the tile is not in the correct position, increase the score for the first heuristic
            if state.board[x][y] != x * state.size + y + 1:
                score1 += 1
    return score1


########################################################
# Heuristic 2 : sum of Manhattan distance
########################################################
def heuristic2(state: Node) -> int:
    # Initialize the score for h
    score2 = 0
    # Iterate through all the tiles on the board
    for x in range(state.size):
        for y in range(state.size):
            # Calculate the correct position of the tile
            x1 = int((state.board[x][y] - 1) / state.size)
            y1 = (state.board[x][y] - 1) % state.size
            # Increase the score for the second heuristic by the Manhattan distance from the tile's current position
            # to its correct position
            score2 = score2 + abs(y1 - y) + abs(x1 - x)
    return score2


def heuristic12(state: Node) -> int:
    # Initialize the scores for both heuristics
    score1 = 0
    score2 = 0
    # Iterate through all the tiles on the board
    for x in range(state.size):
        for y in range(state.size):
            # If the tile is not in the correct position, increase the score for the first heuristic
            if state.board[x][y] != x * state.size + y + 1:
                score1 += 1
            # Calculate the correct position of the tile
            x1 = int((state.board[x][y] - 1) / state.size)
            y1 = (state.board[x][y] - 1) % state.size
            # Increase the score for the second heuristic by the Manhattan distance from the tile's current position
            # to its correct position
            score2 = score2 + abs(y1 - y) + abs(x1 - x)
    # Return the sum of the scores for both heuristics
    return score1 + score2


########################################################
# IDA* Algorithm
########################################################
def id_a_star_search(start: Node, goal: Node) -> list[int]:
    # Initialize the depth limit to 1
    depth_limit = heuristic12(start)
    while True:
        # Perform an ID A* search with the current depth limit
        result = a_star_search(start, goal, 3, depth_limit)
        if result is None:
            depth_limit *= 2
        # If a path was found, return it
        else:
            return result


########################################################
# A* Algorithm
########################################################
def a_star_search(start: Node, goal: Node, heuristic: int, depth_limit: int, ) -> list[int]:
    func_dict = {1: heuristic1, 2: heuristic2, 3: heuristic12}
    # Initialize a stack to store the nodes to be explored
    frontier = []
    # g = start.size * start.size - 1 - start.get_position()
    # Push the starting node onto the stack with the heuristic value as the priority
    heappush(frontier, (func_dict.get(heuristic)(start), start))
    # Initialize a list to store the nodes that have been explored
    explored = []
    while frontier:
        # Get the node with the lowest heuristic value from the frontier
        state = heappop(frontier)[1]
        # Add the node to the list of explored nodes
        explored.append(state)
        # If the current node is the goal, return the path to it
        if state.__eq__(goal):
            path = []
            # Trace the path back to the starting node
            while state.back is not None:
                path.append(state.get_position())
                state = state.back
            path.append(state.get_position())
            path.reverse()
            print('Number of iterations:', len(explored))
            return path
        # If the depth limit has been reached, return None This depth limit is given to the ID A* algorithm to use.
        # If we use the A* algorithm, it is sufficient to give a very large value.
        elif depth_limit - len(explored) <= 0:
            return None
        # Generate the neighbors of the current node
        for neighbor in get_neighbors(state):
            # Create a temporary list to store the nodes in the frontier
            tmp = []
            for f in frontier:
                tmp.append(f[1])
            # If the neighbor has not been explored and is not in the frontier, add it to the frontier
            if neighbor.not_in(explored) and neighbor.not_in(tmp):
                # g = neighbor.size * neighbor.size - 1 - neighbor.get_position()
                heappush(frontier, (func_dict.get(heuristic)(neighbor), neighbor))
    # If no path was found, return None
    return None


########################################################
# BFS Algorithm
########################################################
def bfs_search(start: Node, goal: Node) -> list[int]:
    # Initialize a queue to store the nodes to be explored
    frontier = deque()
    # Add the starting node to the queue
    frontier.append(start)
    # Initialize a list to store the nodes that have been explored
    explored = []
    while frontier:
        # Get the next node in the queue
        state = frontier.popleft()
        # Add the node to the list of explored nodes
        explored.append(state)
        # If the current node is the goal, return the path to it
        if state.__eq__(goal):
            path = []
            # Trace the path back to the starting node
            while state.back is not None:
                path.append(state.get_position())
                state = state.back
            path.append(state.get_position())
            path.reverse()
            print('Number of iterations:', len(explored))
            return path
        # If the depth limit has been reached, return None
        # Generate the neighbors of the current node
        for neighbor in get_neighbors(state):
            # If the neighbor has not been explored and is not in the frontier, add it to the frontier
            if neighbor.not_in(explored) and neighbor.not_in(frontier):
                frontier.append(neighbor)
    # If no path was found, return None
    return None


########################################################
# Bidirectional Search
########################################################
def bidirectional_search(start: Node, end: Node) -> list[int]:
    # Set up two queues, one for each direction
    forward_queue = [start]
    backward_queue = [end]

    # Store the visited nodes in two lists, one for each direction
    forward_visited = []
    backward_visited = []

    # While both queues have elements in them
    while forward_queue and backward_queue:

        # Check the front elements of the two queues
        forward_front = forward_queue.pop(0)
        backward_front = backward_queue.pop(0)

        # If they match, then we have found a path
        if forward_front == backward_front:
            return forward_front

        # Otherwise, add any unvisited neighbors to the respective queues
        for neighbor in get_neighbors(forward_front):
            if neighbor not in forward_visited:
                forward_queue.append(neighbor)
                forward_visited.append(neighbor)

        for neighbor in get_neighbors(backward_front):
            if neighbor not in backward_visited:
                backward_queue.append(neighbor)


def get_neighbors(state: Node) -> list[Node]:
    # Initialize a list to store the neighbors
    neighbors = []
    # Iterate through all the tiles on the board
    for x in range(state.size):
        for y in range(state.size):
            # If the current tile is the empty space, generate its neighbors
            if state.board[x][y] == 0:
                # If the empty space is not in the top row, generate a neighbor by swapping it with the tile above it
                if x > 0:
                    # Create a deep copy of the current state
                    tmp = swap(state, x, y, x - 1, y)
                    # Set the parent of the neighbor to the current state
                    tmp.back = state
                    # Add the neighbor to the list
                    neighbors.append(tmp)
                if x < state.size - 1:
                    tmp = swap(state, x, y, x + 1, y)
                    tmp.back = state
                    neighbors.append(tmp)
                if y > 0:
                    tmp = swap(state, x, y, x, y - 1)
                    tmp.back = state
                    neighbors.append(tmp)
                if y < state.size - 1:
                    tmp = swap(state, x, y, x, y + 1)
                    tmp.back = state
                    neighbors.append(tmp)
    # Return all neighbors
    return neighbors


def swap(state, x1, y1, x2, y2):
    # Create a deep copy of the current state
    state = copy.deepcopy(state)
    # Store the value of the tile at the second position in a temporary variable
    temp = state.board[x2][y2]
    # Swap the values of the two tiles
    state.board[x2][y2] = state.board[x1][y1]
    state.board[x1][y1] = temp
    # Return the modified state
    return state


def read_date():
    datas = csv.reader(open("./data/data.csv", encoding="utf-8-sig"))
    for data in datas:
        print("Number of times the number 0 has been moved:", data[2], " Time limit: 3s")
        print("Start matrix: ", data[0])
        matrix = [[0 for x in range(int(data[1]))] for y in range(int(data[1]))]
        tmp = re.sub("\D", "", data[0])
        ptr = 0
        for i in range(int(data[1])):
            for j in range(int(data[1])):
                matrix[i][j] = int(tmp[ptr])
                ptr += 1
        solve(matrix, goal_generator(int(data[1])), time_limit=10)
        print("----------------------------------------------------------------")


def solve(start_matrix: [[]], goal_matrix: [[]], time_limit: int):
    start = Node(start_matrix)
    goal = Node(goal_matrix)
    path_id_a_star_search = None
    path_a_star_search_1 = None
    path_a_star_search_2 = None

    print("id_a_star_search")
    try:
        time_start = time.time()
        path_id_a_star_search = func_timeout(time_limit, id_a_star_search, args=(start, goal,))
        time_end = time.time()
        print('time_cost', time_end - time_start, 's')
    except FunctionTimedOut as e:
        print("Exceeding the iteration limit")

    print("a_star_search with heuristic1")
    try:
        time_start = time.time()
        path_a_star_search_1 = func_timeout(time_limit, a_star_search, args=(start, goal, 1, 10000,))
        time_end = time.time()
        print('time_cost', time_end - time_start, 's')
    except FunctionTimedOut as e:
        print("Exceeding the iteration limit")

    print("a_star_search with heuristic2")
    try:
        time_start = time.time()
        path_a_star_search_2 = func_timeout(time_limit, a_star_search, args=(start, goal, 2, 10000,))
        time_end = time.time()
        print('time_cost', time_end - time_start, 's')
    except FunctionTimedOut as e:
        print("Exceeding the iteration limit")

    print("a_star_search with heuristic1 and heuristic2")
    try:
        time_start = time.time()
        path_a_star_search_2 = func_timeout(time_limit, a_star_search, args=(start, goal, 3, 10000,))
        time_end = time.time()
        print('time_cost', time_end - time_start, 's')
    except FunctionTimedOut as e:
        print("Exceeding the iteration limit")

    # Exceeding the iteration limit
    print("bidirectional search")
    try:
        time_start = time.time()
        path_bidirectional_search = func_timeout(time_limit, bidirectional_search, args=(start, goal,))
        time_end = time.time()
        print('time_cost', time_end - time_start, 's')
    except FunctionTimedOut as e:
        print("Exceeding the iteration limit")

    print("bfs_search")
    try:
        time_start = time.time()
        path_bfs_search = func_timeout(time_limit, bfs_search, args=(start, goal,))
        time_end = time.time()
        print('time_cost', time_end - time_start, 's')
    except FunctionTimedOut as e:
        print("Exceeding the iteration limit")
    if path_a_star_search_1 is not None:
        print(path_a_star_search_1)
    elif path_a_star_search_2 is not None:
        print(path_a_star_search_2)


def get_valid_actions(row: int, col: int, matrix_size: int) -> []:
    actions = ["up", "down", "left", "right"]
    if row == 0:
        actions.remove("up")
    if row == matrix_size - 1:
        actions.remove("down")
    if col == 0:
        actions.remove("left")
    if col == matrix_size - 1:
        actions.remove("right")
    return actions


def start_generator(n: int, goal_matrix: [[]]) -> [[]]:
    start_matrix = copy.deepcopy(goal_matrix)
    matrix_size = len(start_matrix)
    row = matrix_size - 1
    col = matrix_size - 1
    action_pre = ""
    for i in range(n):
        actions = get_valid_actions(row, col, matrix_size)
        try:
            match action_pre:
                case "up":
                    actions.remove("down")
                case "down":
                    actions.remove("up")
                case "left":
                    actions.remove("right")
                case "right":
                    actions.remove("left")
        except:
            pass

        action_pre = random.choice(actions)
        match action_pre:
            case "up":
                tmp = start_matrix[row][col]
                start_matrix[row][col] = start_matrix[row - 1][col]
                start_matrix[row - 1][col] = tmp
                row -= 1
            case "down":
                tmp = start_matrix[row][col]
                start_matrix[row][col] = start_matrix[row + 1][col]
                start_matrix[row + 1][col] = tmp
                row += 1
            case "left":
                tmp = start_matrix[row][col]
                start_matrix[row][col] = start_matrix[row][col - 1]
                start_matrix[row][col - 1] = tmp
                col -= 1
            case "right":
                tmp = start_matrix[row][col]
                start_matrix[row][col] = start_matrix[row][col + 1]
                start_matrix[row][col + 1] = tmp
                col += 1
    return start_matrix


def goal_generator(length_matrix: int) -> [[]]:
    matrix = [[0 for x in range(length_matrix)] for y in range(length_matrix)]
    for i in range(length_matrix):
        for j in range(length_matrix):
            val = i * length_matrix + j + 1
            if val != length_matrix * length_matrix:
                matrix[i][j] = val
    return matrix


if __name__ == "__main__":
    # If you want to see the results of the test, add the # in front of the line of code below(read_date()).
    # read_date()
    goal = goal_generator(4)
    print(goal)
    start = start_generator(15, goal)
    print(start)
    solve(start, goal, time_limit=10)
