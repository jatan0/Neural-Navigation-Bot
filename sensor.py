import math
import random


def sensor_reading(bot_pos, mouse_pos, alpha):
    d = abs(bot_pos[0] - mouse_pos[0]) + abs(bot_pos[1] - mouse_pos[1])
    if d == 1:
        return True
    prob_beep = math.exp(-alpha * (d - 1))
    return random.random() < prob_beep


# # test
# if __name__ == "__main__":
#     bot_pos = (5, 5)
#     mouse_pos = (7, 5)
#     alpha = 0.45

#     for i in range(2, 10):
#         print(f"dist from bot: {i}, beep: {sensor_reading(bot_pos, (5 + i, 5), alpha)}")

#     # immediate neighbor
#     print("\n2nd test")
#     print(f"dist from bot: 1, beep: {sensor_reading(bot_pos, (6, 5), alpha)}")
