import random


def place_mice(grid, num_mice, mouse_type):
    D = grid.shape[0]
    open_cells = [(x, y) for x in range(D) for y in range(D) if grid[x, y] == 0]
    mice_pos = random.sample(open_cells, num_mice)
    return mice_pos


def move_mice(grid, mice_pos, mouse_type):
    D = grid.shape[0]
    new_positions = []
    for x, y in mice_pos:
        if mouse_type == "stochastic":
            neighbors = [
                (x + dx, y + dy)
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]
                if 0 <= x + dx < D and 0 <= y + dy < D and grid[x + dx, y + dy] == 0
            ]
            if neighbors:
                new_positions.append(random.choice(neighbors))
            else:
                new_positions.append((x, y))
        else:
            new_positions.append((x, y))
    return new_positions


# # test
# if __name__ == "__main__":
#     from ship import generate_ship_layout

#     grid = generate_ship_layout(40)
#     num_mice = 2

#     stat_mice_pos = place_mice(grid, num_mice, "stationary")
#     print("stationary:")
#     print(stat_mice_pos, "\n")

#     stoch_mice_pos = place_mice(grid, num_mice, "stochastic")
#     print("original stochastic:")
#     print(stoch_mice_pos)

#     for _ in range(15):
#         stoch_mice_pos = move_mice(grid, stoch_mice_pos, "stochastic")
#         print("next stochastic:")
#         print(stoch_mice_pos)
