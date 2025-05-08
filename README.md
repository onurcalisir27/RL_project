# Waypoint-Based Reinforcement Learning for Robot Manipulation

This repository implements a waypoint-based reinforcement learning approach for robot manipulation tasks, as described in the paper "Waypoint-Based Reinforcement Learning for Robot Manipulation Tasks" by Mehta et al. The implementation uses TensorFlow and is designed to work with the robosuite simulation environment.

## Prerequisites

- Python 3.8 or higher
- A compatible robosuite environment (e.g., Lift task with Panda robot)
- Optional: GPU support for TensorFlow (recommended for faster training)

## Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. **Create a virtual environment** (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify robosuite installation**:
   Ensure robosuite is properly configured. Test it by running:
   ```bash
   python -c "import robosuite; print(robosuite.__version__)"
   ```
   This should output `1.4.1` or a compatible version.

## Project Structure

- `models.py`: Defines the neural network model (`RNetwork`) used for reward estimation.
- `method.py`: Implements the core algorithm for waypoint optimization and model training.
- `memory.py`: Manages the experience buffer for storing trajectories and rewards.
- `train.py`: Script for training the model on a robosuite task.
- `eval.py`: Script for evaluating the trained model.
- `main.py`: Entry point for running training or evaluation based on configuration.
- `cfg/`: Directory containing configuration files (`config.yaml` and task-specific YAMLs like `Lift.yaml`).
- `models/`: Directory where trained models and evaluation data are saved.

## Usage

### Training

1. **Configure the task**:
   Edit `cfg/config.yaml` to set the desired task and parameters. For example, to train on the Lift task:
   ```yaml
   env_name: Lift
   object: ''
   num_wp: 2
   run_name: 'test'
   n_inits: 5
   render: False
   train: True
   test: False
   ```

2. **Run the training**:
   ```bash
   python main.py
   ```
   This will train the model for the specified number of episodes, saving models and training data to `models/Lift/test/`.

3. **Monitor training**:
   Training progress is logged to TensorBoard. View the logs using:
   ```bash
   tensorboard --logdir runs/
   ```
   Open the provided URL (usually `http://localhost:6006`) in a browser.

### Evaluation

1. **Configure evaluation**:
   Update `cfg/config.yaml` to enable evaluation:
   ```yaml
   train: False
   test: True
   ```

2. **Run the evaluation**:
   ```bash
   python main.py
   ```
   This will evaluate the trained model for the specified number of episodes, saving results to `models/Lift/test/eval_data.pkl`.

### Notes

- **Tasks**: The default configuration is set for the Lift task. To use other tasks (e.g., Stack, Door), create corresponding YAML files in `cfg/task/` based on `Lift.yaml`.
- **Model Saving**: Trained models are saved as TensorFlow checkpoints in `models/{env}/{run_name}/model_{wp_id}_{idx}`.
- **Rendering**: Set `render: True` in `config.yaml` to visualize the robot's actions during training or evaluation (requires a display).
- **Customization**: Adjust hyperparameters in `cfg/task/Lift.yaml` (e.g., `epoch_wp`, `batch_size`) to tune performance.

## Troubleshooting

- **Robosuite Errors**: Ensure robosuite is installed correctly and the environment (e.g., Panda robot) is available. Check the robosuite documentation for setup details.
- **TensorFlow GPU Issues**: Verify TensorFlow GPU support by running `python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"`. Install CUDA and cuDNN if needed.
- **Missing Models**: Ensure trained models exist in the `models/` directory before evaluation.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Acknowledgments

This implementation is based on the paper "Waypoint-Based Reinforcement Learning for Robot Manipulation Tasks" by Shaunak A. Mehta, Soheil Habibian, and Dylan P. Losey.