import heapq
from sensor import sensor_reading
from ship import generate_ship_layout
from mice import place_mice


def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def a_star_search(grid, start, goal, avoid=None):
    if avoid is None:
        avoid = set()
    D = grid.shape[0]
    neighbors = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    close_set = set()
    came_from = {}
    gscore = {start: 0}
    fscore = {start: heuristic(start, goal)}
    oheap = []
    heapq.heappush(oheap, (fscore[start], start))
    while oheap:
        current = heapq.heappop(oheap)[1]
        if current == goal:
            data = []
            while current in came_from:
                data.append(current)
                current = came_from[current]
            return data[::-1]
        close_set.add(current)
        for i, j in neighbors:
            neighbor = current[0] + i, current[1] + j
            tentative_g_score = gscore[current] + 1
            if 0 <= neighbor[0] < D and 0 <= neighbor[1] < D:
                if grid[neighbor[0]][neighbor[1]] == 1 or neighbor in avoid:
                    continue
            else:
                continue
            if neighbor in close_set and tentative_g_score >= gscore.get(
                neighbor, float("inf")
            ):
                continue
            if tentative_g_score < gscore.get(
                neighbor, float("inf")
            ) or neighbor not in [i[1] for i in oheap]:
                came_from[neighbor] = current
                gscore[neighbor] = tentative_g_score
                fscore[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                heapq.heappush(oheap, (fscore[neighbor], neighbor))
    return []


def bot_1_strategy(grid, bot_pos, mouse_positions, alpha):
    D = grid.shape[0]
    best_pos = bot_pos
    max_prob = 0
    neighbors = [(0, 1), (0, -1), (1, 0), (-1, 0)]

    for dx, dy in neighbors:
        new_pos = (bot_pos[0] + dx, bot_pos[1] + dy)
        if (
            0 <= new_pos[0] < D
            and 0 <= new_pos[1] < D
            and grid[new_pos[0], new_pos[1]] == 0
        ):
            prob = sum(
                sensor_reading(new_pos, mouse_pos, alpha)
                for mouse_pos in mouse_positions
            ) / len(mouse_positions)
            if prob > max_prob:
                max_prob = prob
                best_pos = new_pos
    return best_pos


def bot_1(grid, bot_pos, mouse_positions, alpha):
    move_count = 0
    sense_count = 0
    while mouse_positions:
        bot_pos = bot_1_strategy(grid, bot_pos, mouse_positions, alpha)
        move_count += 1
        sense_result = any(
            sensor_reading(bot_pos, mouse_pos, alpha) for mouse_pos in mouse_positions
        )
        sense_count += 1
        if bot_pos in mouse_positions:
            mouse_positions.remove(bot_pos)
    total_actions = move_count + sense_count
    return total_actions


def bot_2_strategy(grid, bot_pos, mouse_positions, alpha, step_count):
    D = grid.shape[0]
    max_prob = 0
    best_pos = bot_pos
    if step_count % 2 == 0:
        neighbors = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        for dx, dy in neighbors:
            new_pos = (bot_pos[0] + dx, bot_pos[1] + dy)
            if (
                0 <= new_pos[0] < D
                and 0 <= new_pos[1] < D
                and grid[new_pos[0], new_pos[1]] == 0
            ):
                prob = sum(
                    sensor_reading(new_pos, mouse_pos, alpha)
                    for mouse_pos in mouse_positions
                ) / len(mouse_positions)
                if prob > max_prob:
                    max_prob = prob
                    best_pos = new_pos
    else:
        prob = sum(
            sensor_reading(bot_pos, mouse_pos, alpha) for mouse_pos in mouse_positions
        ) / len(mouse_positions)
        if prob > max_prob:
            max_prob = prob
            best_pos = bot_pos
    return best_pos


def bot_2(grid, bot_pos, mouse_positions, alpha):
    move_count = 0
    sense_count = 0
    step_count = 0

    while mouse_positions:
        bot_pos = bot_2_strategy(grid, bot_pos, mouse_positions, alpha, step_count)
        if step_count % 2 == 0:
            move_count += 1
        else:
            sense_count += 1

        if bot_pos in mouse_positions:
            mouse_positions.remove(bot_pos)

        step_count += 1

    total_actions = move_count + sense_count
    return total_actions


def initialize_sectors(D, sector_size):
    sectors = []
    for i in range(0, D, sector_size):
        for j in range(0, D, sector_size):
            sectors.append((i, j))
    return sectors


def zigzag_pattern(sector, sector_size):
    pattern = []
    for i in range(sector_size):
        for j in range(sector_size):
            if i % 2 == 0:
                pattern.append((sector[0] + i, sector[1] + j))
            else:
                pattern.append((sector[0] + i, sector[1] + (sector_size - j - 1)))
    return pattern


def bot_3_strategy(
    grid,
    bot_pos,
    sector_size,
    current_sector,
    current_pattern,
    sectors,
    D,
    alpha,
    mouse_positions,
):
    if not current_pattern:
        if sectors:
            current_sector = sectors.pop(0)
            current_pattern.extend(zigzag_pattern(current_sector, sector_size))
        else:
            return (
                bot_pos,
                current_sector,
                current_pattern,
            )

    next_pos = current_pattern.pop(0)
    if next_pos[0] >= D or next_pos[1] >= D or grid[next_pos[0], next_pos[1]] == 1:
        return (
            bot_pos,
            current_sector,
            current_pattern,
        )

    sensor_reading_value = any(
        sensor_reading(next_pos, mouse_pos, alpha) for mouse_pos in mouse_positions
    )
    if sensor_reading_value:
        sector_size = max(1, sector_size // 2)
        sectors = initialize_sectors(D, sector_size)

    return next_pos, current_sector, current_pattern


def bot_3(grid, bot_pos, mouse_positions, alpha, sector_size=5):
    move_count = 0
    sense_count = 0
    D = grid.shape[0]
    sectors = initialize_sectors(D, sector_size)
    current_sector = None
    current_pattern = []

    while mouse_positions:
        bot_pos, current_sector, current_pattern = bot_3_strategy(
            grid,
            bot_pos,
            sector_size,
            current_sector,
            current_pattern,
            sectors,
            D,
            alpha,
            mouse_positions,
        )
        move_count += 1
        sense_result = any(
            sensor_reading(bot_pos, mouse_pos, alpha) for mouse_pos in mouse_positions
        )
        sense_count += 1

        if bot_pos in mouse_positions:
            mouse_positions.remove(bot_pos)

    total_actions = move_count + sense_count
    return total_actions


# testing
if __name__ == "__main__":

    grid = generate_ship_layout(40)
    alpha = 0.3
    num_mice = 2

    test_cases = [
        {
            "bot_pos": (20, 20),
            "mouse_type": "stationary",
            "description": "stationary mouse",
        },
        {
            "bot_pos": (10, 10),
            "mouse_type": "stochastic",
            "description": "stochastic mouse",
        },
    ]

    # bot 1 test
    for i, test in enumerate(test_cases):
        bot_pos = test["bot_pos"]
        mouse_type = test["mouse_type"]
        description = test["description"]

        mouse_positions = place_mice(grid, num_mice, mouse_type)
        print(f"test #{i + 1}: {description}")
        print("bot:", bot_pos)
        print("mouse:", mouse_positions)

        total_actions = bot_1(grid, bot_pos, mouse_positions, alpha)
        print(f"# of steps to catch mouse: {total_actions}")
        print()

    # bot 2 test
    for i, test in enumerate(test_cases):
        bot_pos = test["bot_pos"]
        mouse_type = test["mouse_type"]
        description = test["description"]

        mouse_positions = place_mice(grid, num_mice, mouse_type)
        print(f"test #{i + 1}: {description}")
        print("bot:", bot_pos)
        print("mouse:", mouse_positions)

        total_actions = bot_2(grid, bot_pos, mouse_positions, alpha)
        print(f"# of steps to catch mouse: {total_actions}")
        print()

    # bot 3 test
    for i, test in enumerate(test_cases):
        bot_pos = test["bot_pos"]
        mouse_type = test["mouse_type"]
        description = test["description"]

        mouse_positions = place_mice(grid, num_mice, mouse_type)
        print(f"test #{i + 1}: {description}")
        print("bot:", bot_pos)
        print("mouse:", mouse_positions)

        total_actions = bot_3(grid, bot_pos, mouse_positions, alpha)
        print(f"# of steps to catch mouse: {total_actions}")
        print()
