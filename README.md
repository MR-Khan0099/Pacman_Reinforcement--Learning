# Pacman Reinforcement Learning

This repository contains the implementation of a Pacman game trained using Reinforcement Learning (RL), utilizing OpenAI Gym for the custom environment setup. The agent is trained to play six levels of Pacman with increasing complexity, optimizing its gameplay strategies using Q-learning techniques. The project structure and environment allow for reproducibility and ease of understanding.


## Project Structure
The repository is organized as follows:
```
PACMANOPENAIGYM/
├── Pacman_Reinforcement--Learning/
│   ├── bin/
│   ├── models/
│   │   ├── Level_1/
│   │   ├── Level_2/
│   │   ├── Level_3/
│   │   ├── Level_4/
│   │   ├── Level_5/
│   │   ├── Level_6/
│   └── notebook/
├── src/
│   ├── gym-pacman-environment/
│   │   ├── final_models/
|   |   |--- models
|   |   |-- plots
│   |   |── AStar.py
│   |   ├── Deep-Q-Learning.py
│   |   ├── evaluate.py
│   |   ├── PacmanAgent.py
│   |   ├── utils.py
├── environment.yml
├── requirements.txt
├── README.md
```

### Key Components
1. gym-pacman-environment: Custom Pacman environment built using OpenAI Gym.
2. models: Contains trained models and their respective evaluation plots.
3. src: Source code for training, evaluation, and utilities.
4. requirements.txt: Specifies Python packages needed for the project.
5. environment.yml: Specifies Conda dependencies for environment setup and dependencies.
6. README.md: Documentation and usage instructions.


## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/MR-Khan0099/Pacman_Reinforcement--Learning.git
    ```
2. Navigate to the project directory:
    ```bash
    cd Pacman_Reinforcement_Learning
    ```
3. Install the required dependencies:
    
   1. Option 1: Using Conda
   ```bash
        conda env create -f environment.yml 
    ```
    ```bash
        conda activate PacmanOpenAIGym_FHDortmund 
    ```

    Option 2: Using requirements.txt
    ```bash 
        pip install -r requirements.txt
    ```

## Usage
### Train the Pacman Agent
To train the Pacman agent for a specific level:

1. Update the Level in PacmanAgent.py:
Open src/gym-pacman-environment/PacmanAgent.py and modify the level_name variable to the desired level name. For example:
```python
level_name = "Level_4"  
```
2. Specify Model and Plot Names in Deep-Q-Learning.py:
Open src/gym-pacman-environment//Deep-Q-Learning.py and update the model and plot file names to save the results for the specific level. For example:
```python
model_name = "Pacman_Q_Model_L4_1500.keras"  
plot_name = "Pacman_Q_Rewards_L4_again_1500.png"  
```
code snippet:
```python
# Save the trained model
model_save_dir = "models/"
os.makedirs(model_save_dir, exist_ok=True)
model_file_path = os.path.join(model_save_dir, "Pacman_Q_Model_L4_1500.keras")
```
3. Run the Training Script:
Execute the training script:

```bash
python src/gym-pacman-environment/Deep-Q-Learning.py  
```
During training, you can observe the agent's learning process as it renders in real-time. After training, the reward plots and model will be saved.

### Evaluate the Trained Agent
To evaluate a trained agent:
1. Update the Level in PacmanAgent.py:
Similar to training, update the level_name variable in PacmanAgent.py to match the level you want to evaluate.

2. Specify the Model Name in evaluate.py:
Open src/gym-pacman-environment/evaluate.py and update the model file path to the corresponding trained model for the selected level. For example:
```bash
model_path = "models/Pacman_Q_Model_L6_2500.keras
```

3. Run the Evaluation Script:
Execute the evaluation script:
```bash
python src/gym-pacman-environment/evaluate.py  
```

## Algorithms Implemented

This project implements three major reinforcement learning algorithms for training the Pacman agent:

### 1. Q-Learning
Q-Learning is a value-based RL algorithm where the agent learns a Q-table mapping states to action values. The Q-values are updated iteratively based on the Bellman equation:

\[ Q(s,a) \leftarrow Q(s,a) + \alpha \left( r + \gamma \max_{a} Q(s',a) - Q(s,a) \right) \]

Key Features:
- Uses a discrete state-action mapping table.
- Suitable for environments with a finite number of states and actions.
- Simple to implement and understand.
- Requires a large amount of memory for large state spaces.
- Can be slow to converge for complex environments.
- Relies on exploration strategies to discover optimal policies.
- Sensitive to the choice of learning rate and discount factor.
- Can be combined with function approximation techniques for scalability.
- Often used as a baseline for comparing more advanced RL algorithms.
- Effective for problems with well-defined state transitions and rewards.
- Efficient for small state and action spaces.
### 2. Deep Q-Network (DQN)
DQN uses a neural network to approximate the Q-function instead of a table. This allows it to scale to problems with high-dimensional or continuous state spaces.

#### Key Steps in the Implementation:
1. **Building the Q-Network:**
A neural network with two hidden layers is used:
```python
def build_model():
    model = Sequential([
        tf.keras.Input(shape=(state_size,)),
        Dense(128, activation='relu'),
        Dense(128, activation='relu'),
        Dense(action_space, activation='linear')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate), loss='mse')
    return model
```
- `state_size`: Number of features in the state representation.
- `action_space`: Total number of possible actions (e.g., up, down, left, right).
- Loss function: Mean Squared Error (MSE).

2. **Training the Network:**
The DQ-network is trained using experience replay. A replay buffer stores past experiences for sampling and training:
```python
def train_q_network():
    batch = random.sample(replay_buffer, batch_size)
    states, actions, rewards, next_states, dones = zip(*batch)

    next_q_values = target_network.predict(next_states, verbose=0)
    max_next_q_values = np.max(next_q_values, axis=1)

    targets = rewards + gamma * max_next_q_values * (1 - dones)
    q_values = dq_network.predict(states, verbose=0)
    for i, action in enumerate(actions):
        q_values[i, action] = targets[i]

    dq_network.fit(states, q_values, verbose=0, batch_size=batch_size)
```
**Features:**
- Uses a target network for stability.
- Epsilon-greedy exploration for action selection.
- Periodic updates of the target network.

3. **Reward Calculation:**
Rewards are based on:
- Eating dots (+2 per dot).
- Completing the level (+50 bonus).
- Ghost proximity (-10 penalty).
```python
def calculate_reward(env, last_remaining_dots, done):
    reward = 0
    dots_eaten = last_remaining_dots - env.remainingDots
    reward += dots_eaten * 2
    if env.remainingDots == 0:
        reward += 50
    if done and env.remainingDots > 0:
        reward -= 10
```

4. **Exploration-Exploitation Tradeoff:**
Initial exploration probability (epsilon) decreases exponentially after each episode:
```python
if epsilon > epsilon_min:
    epsilon *= epsilon_decay
```

### 3. Policy Gradient
While this project primarily focuses on Q-Learning and DQN, Policy Gradient methods are included for advanced agent control:
- Instead of learning value functions, policy gradient methods directly optimize the policy.
- Gradient ascent is used to maximize the expected cumulative reward:
\[ \theta \leftarrow \theta + \alpha \nabla_{\theta} E[\text{Reward}] \]
These methods are effective for environments with high-dimensional action spaces or stochastic policies.


### Hyperparameters for Different Levels

The following hyperparameters were used for training the Pacman agent across different levels:

| Parameter            | General Parameters | Level 1 Parameters | Level 2 Parameters | Level 3 Parameters | Level 4 Parameters | Level 5 Parameters | Level 6 Parameters |
|----------------------|--------------------|--------------------|--------------------|--------------------|--------------------|--------------------|--------------------|
| `state_size`         | 49                 | 49                 | 49                 | 49                 | 49                 | 49                 | 49                 |
| `action_space`       | 4                  | 4                  | 4                  | 4                  | 4                  | 4                  | 4                  |
| `gamma`              | 0.95               | 0.95               | 0.95               | 0.95               | 0.95               | 0.95               | 0.95               |
| `epsilon`            | 1.0                | 1.0                | 1.0                | 1.0                | 1.0                | 1.0                | 1.0                |
| `epsilon_min`        | 0.005              | 0.005              | 0.005              | 0.1                | 0.1                | 0.1                | 0.1                |
| `epsilon_decay`      | 0.99               | 0.99               | 0.99               | 0.995              | 0.995              | 0.995              | 0.995              |
| `learning_rate`      | 0.02               | 0.02               | 0.02               | 0.005              | 0.005              | 0.005              | 0.005              |
| `batch_size`         | 64                 | 64                 | 64                 | 64                 | 64                 | 64                 | 64                 |
| `max_replay_size`    | 10000              | 10000              | 10000              | 10000              | 10000              | 10000              | 10000              |
| `episodes`           | 1000/500           | 500                | 1000               | 1000 / 500         | 1500               | 2500               | 2500               |
| `target_update_freq` | 10                 | 10                 | 10                 | 2                  | 2                  | 2                  | 2                  |


#### Penalty for Leaving Dots
```python
pacman_position = env.find_entity(env.render(), "pacman")
hunter_position = env.find_entity(env.render(), "hunter_ghost")
if pacman_position and hunter_position:
    if pacman_position[0] == hunter_position[0] and abs(hunter_position[1] - pacman_position[1]) <= 1:
        reward = -10
    if pacman_position[1] == hunter_position[1] and abs(hunter_position[0] - pacman_position[0]) <= 1:
        reward = -10
```
## Results

The trained agent's performance can be visualized using the provided script above. 

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request.



