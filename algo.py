import copy
import time
import random
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
                    # Return the position as a single integer, calculated by multiplying the row index by the size of the board and adding the column index + 1
                    return i * self.size + j + 1


def id_a_star_search(start: Node, goal: Node) -> list[int]:
    # Initialize the depth limit to 1
    depth_limit = 1
    while True:
        # Perform an ID A* search with the current depth limit
        result = a_star_search(start, goal, depth_limit)
        if result is None:
            depth_limit *= 2
        # If a path was found, return it
        else:
            return result


def heuristic1(state: Node) -> int:
    # Initialize the score for heuristic
    score1 = 0
    # Iterate through all the tiles on the board
    for x in range(state.size):
        for y in range(state.size):
            # If the tile is not in the correct position, increase the score for the first heuristic
            if state.board[x][y] != x * state.size + y + 1:
                score1 += 1
    return score1


def heuristic2(state: Node) -> int:
    # Initialize the scores for heuristic
    score2 = 0
    # Iterate through all the tiles on the board
    for x in range(state.size):
        for y in range(state.size):
            # Calculate the correct position of the tile
            x1 = int((state.board[x][y] - 1) / state.size)
            y1 = (state.board[x][y] - 1) % state.size
            # Increase the score for the second heuristic by the Manhattan distance from the tile's current position to its correct position
            score2 = score2 + abs(y1 - y) + abs(x1 - x)
    return score2


def heuristic(state: Node) -> int:
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
            # Increase the score for the second heuristic by the Manhattan distance from the tile's current position to its correct position
            score2 = score2 + abs(y1 - y) + abs(x1 - x)
    # Return the sum of the scores for both heuristics
    return score1 + score2


def a_star_search(start: Node, goal: Node, depth_limit: int) -> list[int]:
    # Initialize a stack to store the nodes to be explored
    frontier = []
    # Push the starting node onto the stack with the heuristic value as the priority
    heappush(frontier, (heuristic(start), start))
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
        # If the depth limit has been reached, return None
        # This depth limit is given to the ID A* algorithm to use. If we use the A* algorithm, it is sufficient to give a very large value.
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
                heappush(frontier, (heuristic(neighbor), neighbor))
    # If no path was found, return None
    return None


def bfs_search(start: Node, goal: Node, depth_limit: int) -> list[int]:
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
        # The BFS algorithm often takes a long time to solve this problem, and the purpose of this limit is to prevent this algorithm from taking up too much time and causing lag.
        elif depth_limit - len(explored) <= 0:
            return None
        # Generate the neighbors of the current node
        for neighbor in get_neighbors(state):
            # If the neighbor has not been explored and is not in the frontier, add it to the frontier
            if neighbor.not_in(explored) and neighbor.not_in(frontier):
                frontier.append(neighbor)
    # If no path was found, return None
    return None


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


def solve(start_matrix: [[]], goal_matrix: [[]]) -> list[int]:
    start = Node(start_matrix)
    goal = Node(goal_matrix)

    print("id_a_star_search")
    time_start = time.time()
    path_id_a_star_search = id_a_star_search(start, goal)
    time_end = time.time()
    print('time_cost', time_end - time_start, 's')
    if path_id_a_star_search is None:
        print("Exceeding the iteration limit")

    print("a_star_search")
    time_start = time.time()
    path_a_star_search = a_star_search(start, goal, 1000)
    time_end = time.time()
    print('time_cost', time_end - time_start, 's')
    if id_a_star_search is None:
        print("Exceeding the iteration limit")

    print("bfs_search")
    time_start = time.time()
    path_bfs_search = bfs_search(start, goal, 1000)
    time_end = time.time()
    print('time_cost', time_end - time_start, 's')
    if path_bfs_search is None:
        print("Exceeding the iteration limit")
    return path_a_star_search


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
    goal = goal_generator(6)
    start = start_generator(6, goal)
    print(start)
    solve(start, goal)
