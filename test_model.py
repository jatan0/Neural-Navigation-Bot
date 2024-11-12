import torch
import numpy as np
from ship import generate_ship_layout
from mice import place_mice, move_mice
from sensor import sensor_reading
from model import BotModel

model = BotModel(input_size=3202, output_size=5)
model.load_state_dict(torch.load("bot_model.pth"))
model.eval()

action_mapping = {0: "up", 1: "down", 2: "left", 3: "right", 4: "sense"}


def simulate_environment(
    grid, bot_pos, mouse_positions, model, alpha, mouse_type, max_steps=3000
):
    D = grid.shape[0]
    move_count = 0
    sense_count = 0
    steps = 0
    while mouse_positions and steps < max_steps:
        sensor_data = np.zeros((D, D))
        for mouse_pos in mouse_positions:
            d = abs(bot_pos[0] - mouse_pos[0]) + abs(bot_pos[1] - mouse_pos[1])
            prob_beep = np.exp(-alpha * (d - 1))
            sensor_data[mouse_pos[0], mouse_pos[1]] = prob_beep
        state = np.concatenate(
            (grid.flatten(), [bot_pos[0], bot_pos[1]], sensor_data.flatten())
        )
        if state.shape[0] != 3202:
            pad_size = 3202 - state.shape[0]
            state = np.pad(state, (0, pad_size), "constant")
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            output = model(state)
            action = torch.argmax(output).item()
        action_str = action_mapping[action]

        if action_str == "up":
            new_pos = (bot_pos[0] - 1, bot_pos[1])
        elif action_str == "down":
            new_pos = (bot_pos[0] + 1, bot_pos[1])
        elif action_str == "left":
            new_pos = (bot_pos[0], bot_pos[1] - 1)
        elif action_str == "right":
            new_pos = (bot_pos[0], bot_pos[1] + 1)
        else:
            sense_count += 1
            steps += 1
            continue

        if (
            0 <= new_pos[0] < D
            and 0 <= new_pos[1] < D
            and grid[new_pos[0], new_pos[1]] == 0
        ):
            bot_pos = new_pos
            move_count += 1
        if bot_pos in mouse_positions:
            mouse_positions.remove(bot_pos)
        if mouse_type == "stochastic":
            mouse_positions = move_mice(grid, mouse_positions, "stochastic")
        steps += 1
    total_actions = move_count + sense_count
    return total_actions


def evaluate_model(
    model, mouse_type, num_episodes=10, grid_size=40, num_mice=1, alpha=0.3
):
    total_actions_list = []
    for episode in range(num_episodes):
        grid = generate_ship_layout(grid_size)
        bot_pos = (np.random.randint(0, grid_size), np.random.randint(0, grid_size))
        mouse_positions = place_mice(grid, num_mice, mouse_type)
        actions_model = simulate_environment(
            grid, bot_pos, mouse_positions, model, alpha, mouse_type
        )
        total_actions_list.append(actions_model)

        print(f"episode {episode + 1}/{num_episodes}, total: {actions_model}")

    avg_actions = np.mean(total_actions_list)
    print(f"\naverage actions for ({mouse_type}): {avg_actions}")

    return avg_actions


print("catching stationary mice:")
avg_actions_stationary = evaluate_model(model, mouse_type="stationary")

print("\ncatching stochastic mice:")
avg_actions_stochastic = evaluate_model(model, mouse_type="stochastic")
