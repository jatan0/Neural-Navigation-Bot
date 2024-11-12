import pickle

with open("data_stochastic.pkl", "rb") as f:
    data_stochastic = pickle.load(f)

with open("data_stationary.pkl", "rb") as f:
    data_stationary = pickle.load(f)


def inspect_data(data, description):
    first_episode = data[0]
    grid = first_episode[0]
    initial_bot_position = first_episode[1]
    initial_sensor_data = first_episode[2]
    actions = first_episode[3]

    print(f"\n{description}: first episode data:")
    print("\nship:")
    print(grid)

    print("\noriginal bot location:")
    print(initial_bot_position)

    print("\noriginal sensor data:")
    print(initial_sensor_data)


inspect_data(data_stochastic, "Stochastic Mouse")

inspect_data(data_stationary, "Stationary Mouse")
