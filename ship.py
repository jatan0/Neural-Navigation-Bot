import random
import numpy as np


def generate_ship_layout(D=40):
    grid = np.ones((D, D), dtype=int)
    start_x, start_y = random.randint(1, D - 2), random.randint(1, D - 2)
    grid[start_x, start_y] = 0

    def get_neighbors(x, y):
        neighbors = []
        if x > 0:
            neighbors.append((x - 1, y))
        if x < D - 1:
            neighbors.append((x + 1, y))
        if y > 0:
            neighbors.append((x, y - 1))
        if y < D - 1:
            neighbors.append((x, y + 1))
        return neighbors

    def get_open_candidates():
        candidates = []
        for x in range(D):
            for y in range(D):
                if grid[x, y] == 1:
                    open_neighbors = sum(
                        grid[nx, ny] == 0 for nx, ny in get_neighbors(x, y)
                    )
                    if open_neighbors == 1:
                        candidates.append((x, y))
        return candidates

    candidates = get_open_candidates()
    while candidates:
        x, y = random.choice(candidates)
        grid[x, y] = 0
        candidates = get_open_candidates()

    dead_ends = [
        (x, y)
        for x in range(D)
        for y in range(D)
        if grid[x, y] == 0
        and sum(grid[nx, ny] == 0 for nx, ny in get_neighbors(x, y)) == 1
    ]
    for x, y in random.sample(dead_ends, len(dead_ends) // 2):
        blocked_neighbors = [
            (nx, ny) for nx, ny in get_neighbors(x, y) if grid[nx, ny] == 1
        ]
        if blocked_neighbors:
            nx, ny = random.choice(blocked_neighbors)
            grid[nx, ny] = 0
    return grid


# # test
# if __name__ == "__main__":
#     ship_layout = generate_ship_layout(40)
#     print("ship layout:")
#     print(ship_layout)
