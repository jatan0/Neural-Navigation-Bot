import numpy as np
import random
import math
import pickle
from ship import generate_ship_layout
from mice import place_mice, move_mice
from sensor import sensor_reading
from bot_strategies import bot_3_strategy, initialize_sectors, zigzag_pattern


def collect_data(bot_strategy, num_episodes=100, mouse_type="stochastic"):
    data = []
    D = 40
    alpha = 0.3

    for episode in range(num_episodes):
        grid = generate_ship_layout(D)
        bot_pos = (random.randint(0, D - 1), random.randint(0, D - 1))
        mouse_positions = place_mice(grid, 1, mouse_type)

        initial_sensor_data = get_initial_sensor_data(
            grid, bot_pos, mouse_positions, alpha
        )
        sector_size = 5
        sectors = initialize_sectors(D, sector_size)
        current_sector = None
        current_pattern = []
        episode_actions = []

        while mouse_positions:
            action = bot_strategy(
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

            next_pos, current_sector, current_pattern = action
            move_direction = determine_move_direction(bot_pos, next_pos)
            episode_actions.append(
                (move_direction, "sense" if next_pos == bot_pos else "move")
            )

            bot_pos = next_pos

            if mouse_type == "stochastic":
                mouse_positions = move_mice(grid, mouse_positions, "stochastic")

            if bot_pos in mouse_positions:
                break

        data.append((grid, bot_pos, initial_sensor_data, episode_actions))

    return data


def get_initial_sensor_data(grid, bot_pos, mouse_positions, alpha):
    D = grid.shape[0]
    sensor_data = np.zeros((D, D))
    for mouse_pos in mouse_positions:
        d = abs(bot_pos[0] - mouse_pos[0]) + abs(bot_pos[1] - mouse_pos[1])
        prob_beep = math.exp(-alpha * (d - 1))
        sensor_data[mouse_pos[0], mouse_pos[1]] = prob_beep
    return sensor_data.flatten()


def determine_move_direction(current_pos, next_pos):
    if next_pos[0] < current_pos[0]:
        return "up"
    elif next_pos[0] > current_pos[0]:
        return "down"
    elif next_pos[1] < current_pos[1]:
        return "left"
    elif next_pos[1] > current_pos[1]:
        return "right"
    return "stay"


def save_data(data, mouse_type):
    with open(f"data_{mouse_type}.pkl", "wb") as f:
        pickle.dump(data, f)


data_stochastic = collect_data(
    bot_3_strategy, num_episodes=100, mouse_type="stochastic"
)
save_data(data_stochastic, "stochastic")

data_stationary = collect_data(
    bot_3_strategy, num_episodes=100, mouse_type="stationary"
)
save_data(data_stationary, "stationary")
