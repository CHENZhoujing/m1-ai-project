import copy
from collections import deque
from heapq import heappush, heappop


class Node:
    def __init__(self, board: list):
        self.board = board
        self.size = len(board)
        self.back = None

    def __eq__(self, other) -> bool:
        res = True
        for i in range(self.size):
            for j in range(self.size):
                if self.board[i][j] != other.board[i][j]:
                    res = False
                    break
        return res

    def __lt__(self, other) -> bool:
        return self.size <= other.size

    def not_in(self, others) -> bool:
        res = True
        for i in range(len(others)):
            if self.__eq__(others[i]):
                res = False
                break
        return res

    def get_position(self) -> int:
        for i in range(self.size):
            for j in range(self.size):
                if self.board[i][j] == 0:
                    return i * self.size + j + 1


def id_a_star_search(start: Node, goal: Node) -> list[int]:
    depth_limit = 1
    while True:
        result = a_star_search(start, goal, depth_limit)
        if result is None:
            depth_limit *= 2
        else:
            return result


def heuristic(state: Node) -> int:
    score1 = 0
    score2 = 0
    for x in range(state.size):
        for y in range(state.size):
            if state.board[x][y] != x * state.size + y + 1:
                score1 += 1
            x1 = int((state.board[x][y] - 1) / state.size)
            y1 = (state.board[x][y] - 1) % state.size
            score2 = score2 + abs(y1 - y) + abs(x1 - x)
    return score1 + score2


def a_star_search(start: Node, goal: Node, depth_limit: int) -> list[int]:
    frontier = []
    heappush(frontier, (heuristic(start), start))
    explored = []
    while frontier:
        state = heappop(frontier)[1]
        explored.append(state)
        if state.__eq__(goal):
            path = []
            while state.back is not None:
                path.append(state.get_position())
                state = state.back
            path.append(state.get_position())
            path.reverse()
            return path
        elif depth_limit - len(explored) <= 0:
            return None
        for neighbor in get_neighbors(state):
            tmp = []
            for f in frontier:
                tmp.append(f[1])
            if neighbor.not_in(explored) and neighbor.not_in(tmp):
                heappush(frontier, (heuristic(neighbor), neighbor))
        print(len(explored))
    return None


def bfs_search(start: Node, goal: Node, depth_limit: int) -> list[int]:
    frontier = deque()
    frontier.append(start)
    explored = []
    while frontier:
        state = frontier.popleft()
        explored.append(state)
        if state.__eq__(goal):
            path = []
            while state.back is not None:
                path.append(state.get_position())
                state = state.back
            path.append(state.get_position())
            path.reverse()
            return path
        elif depth_limit - len(explored) <= 0:
            return None
        for neighbor in get_neighbors(state):
            if neighbor.not_in(explored) and neighbor.not_in(frontier):
                frontier.append(neighbor)
        print(len(explored))
    return None


def get_neighbors(state: Node) -> list[Node]:
    neighbors = []
    for x in range(state.size):
        for y in range(state.size):
            if state.board[x][y] == 0:
                if x > 0:
                    tmp = swap(state, x, y, x - 1, y)
                    tmp.back = state
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
    return neighbors


def swap(state, x1, y1, x2, y2):
    state = copy.deepcopy(state)
    temp = state.board[x2][y2]
    state.board[x2][y2] = state.board[x1][y1]
    state.board[x1][y1] = temp
    return state


def solve(matrix: [[]]) -> list[int]:
    start = Node(matrix)
    goal = Node([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 0]])

    print("id_a_star_search")
    path_id_a_star_search = id_a_star_search(start, goal)
    if path_id_a_star_search is None:
        print("No Solution")
    else:
        print(path_id_a_star_search)

    print("a_star_search")
    path_a_star_search = a_star_search(start, goal, 1000)
    if id_a_star_search is None:
        print("No Solution")
    else:
        print(path_a_star_search)

    print("bfs_search")
    path_bfs_search = bfs_search(start, goal, 1000)
    if path_bfs_search is None:
        print("No Solution")
    else:
        print(path_bfs_search)
    return path_a_star_search


if __name__ == "__main__":
    start = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [0, 13, 14, 15]]
    solve(start)
