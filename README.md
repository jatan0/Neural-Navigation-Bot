# Neural-Navigation-Bot
Neural-Navigation-Bot is a simulation project that demonstrates autonomous navigation using neural networks, where a bot learns to efficiently track and capture both stationary and moving targets within maze-like ship environments. The bot processes sensor data to optimize its decision-making and movement strategies, showcasing adaptive behavior in both predictable and stochastic scenarios.

## Features
- Grid-based Environment Simulation: Dynamic ship layout generation with mice placements.
- Deep Learning Model: PyTorch-based neural network to predict bot actions.
- Multiple Mouse Behaviors: Supports stationary and stochastic mouse types.
- Sensor-based Navigation: Uses simulated sensor readings for decision-making.
- Evaluation Metrics: Measures performance in terms of actions taken to catch mice.

## Tech Stack
- Modeling & Framework - PyTorch
- Data Preprocessing - NumPy, Scikit-learn
- Environment Simulation - Python (Custom scripts)
- Data Storage - Pickle (for dataset serialization)
- Visualization - Matplotlib (for loss plot)

<!--
## Project Structure
- **mice.py**: Handles the initialization and movement logic for mice.
- **model.py**: Defines the neural network architecture.
- **preprocess_data.py**: Prepares training data and splits it into training, validation, and test sets.
- **sensor.py**: Contains logic for simulating sensor readings.
- **ship.py**: Generates the grid environment for the bot and mice.
- **test_model.py**: Runs simulations to evaluate the trained model.
- **train_model.py**: Trains the model and provides evaluation results.
- **data_stochastic.pkl & data_stationary.pkl**: Pickle files containing preprocessed datasets.
--!>
