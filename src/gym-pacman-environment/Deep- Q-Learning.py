import numpy as np
import random
from collections import deque
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt
import os  # To handle file and directory operations
from gym_pacman_environment.envs import PacmanEnv
from PacmanAgent import PacmanAgent  # Import the PacmanAgent

# Initialize environment
env = PacmanAgent(PacmanEnv())  # Wrap the Pacman environment
state_space = np.prod(env.reset().shape)  # Flatten the state
action_space = env.action_space.n


# # Parameters -level 4 lastest and level5 passed
state_size = 49  # Update according to your state representation
action_space = 4  # Number of possible actions
gamma = 0.95  #discount factor
epsilon = 1.0 #exploration probability
epsilon_min = 0.1 # minimum exploration probability
epsilon_decay = 0.995 # exploration decay
learning_rate = 0.005
batch_size = 64
max_replay_size = 10000
episodes = 1500
target_update_freq = 2


# gamma = 0.99            # Focus on long-term rewards
# epsilon = 0.9           # Slightly lower initial exploration
# epsilon_min = 0.05      # Ensure occasional exploration
# epsilon_decay = 0.99    # Slower exploration decay
# learning_rate = 0.005   # More stable updates
# batch_size = 64         # Faster updates during training
# target_update_freq = 5  # More frequent target updates


## lev 1 and 2
# state_size = 49  # Update according to your state representation
# action_space = 4  # Number of possible actions
# gamma = 0.95
# epsilon = 1.0
# epsilon_min = 0.005
# epsilon_decay = 0.99
# learning_rate = 0.02
# batch_size = 64
# max_replay_size = 10000
# episodes = 100
# target_update_freq = 10

# Replay buffer
replay_buffer = deque(maxlen=max_replay_size)

# Preprocess state
def preprocess_state(state):
    state = np.array(state).flatten()
    if state.shape[0] != state_size:
        raise ValueError(f"State size mismatch! Expected {state_size}, got {state.shape[0]}")
    return state

# Build the Q-network
def build_model():
    model = Sequential([
        tf.keras.Input(shape=(state_size,)),
        Dense(128, activation='relu'),
        Dense(128, activation='relu'),
        Dense(action_space, activation='linear')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate), loss='mse')
    return model

# Initialize Q-network and target network
dq_network = build_model()
target_network = build_model()
target_network.set_weights(dq_network.get_weights())

# Calculate rewards
def calculate_reward(env, last_remaining_dots, done):
    reward = 0
    dots_eaten = last_remaining_dots - env.remainingDots
    reward += dots_eaten * 2  # Stronger incentive for eating dots


    # if last_remaining_dots > env.remainingDots:
    #     reward += 1
    if env.remainingDots == 0:
        reward += 50
    if done and env.remainingDots > 0:
        reward -= 10
    

    # Add a penalty if Pacman is near the hunter ghost
    pacman_position = env.find_entity(env.render(), "pacman")
    hunter_position = env.find_entity(env.render(), "hunter_ghost")
    if pacman_position and hunter_position:
        if pacman_position[0] == hunter_position[0] and abs(hunter_position[1] - pacman_position[1]) <= 1:
            reward = -10
        if pacman_position[1] == hunter_position[1] and abs(hunter_position[0] - pacman_position[0]) <= 1:
            reward = -10

    return reward

# Train the Q-network
def train_q_network():
    if len(replay_buffer) < batch_size:
        return

    batch = random.sample(replay_buffer, batch_size)
    states, actions, rewards, next_states, dones = zip(*batch)

    states = np.array(states)
    next_states = np.array(next_states)
    rewards = np.array(rewards)
    dones = np.array(dones)

    # Predict Q-values for the next states using the target network
    next_q_values = target_network.predict(next_states, verbose=0)
    max_next_q_values = np.max(next_q_values, axis=1)

    # Calculate targets
    targets = rewards + gamma * max_next_q_values * (1 - dones)

    # Update the Q-values for the taken actions
    q_values = dq_network.predict(states, verbose=0)
    for i, action in enumerate(actions):
        q_values[i, action] = targets[i]

    # Train the DQ network
    dq_network.fit(states, q_values, verbose=0, batch_size=batch_size)

# Initialize a list to store rewards for plotting
reward_list = []

# Main training loop
for episode in range(episodes):
    state = preprocess_state(env.reset())
    total_reward = 0
    done = False
    steps = 0
    last_remaining_dots = env.remainingDots

    print(f"Episode {episode + 1} begins")

    while not done:
        if np.random.rand() < epsilon:
            action = np.random.randint(0, action_space)
        else:
            q_values = dq_network.predict(state.reshape(1, -1), verbose=0)
            action = np.argmax(q_values[0])

        next_state, _, done, _ = env.step(action)
        next_state = preprocess_state(next_state)
        reward = calculate_reward(env, last_remaining_dots, done)
        total_reward += reward

        replay_buffer.append((state, action, reward, next_state, done))

        state = next_state
        steps += 1
        last_remaining_dots = env.remainingDots

        train_q_network()

    # Decay epsilon for exploration-exploitation tradeoff
    if epsilon > epsilon_min:
        epsilon *= epsilon_decay

    # Update target network weights periodically
    if (episode + 1) % target_update_freq == 0:
        target_network.set_weights(dq_network.get_weights())

    reward_list.append(total_reward)
    print(f"Episode {episode + 1}: Total Reward: {total_reward}")

print("Training complete!")

# Plot Reward Trends
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(reward_list) + 1), reward_list, label="Total Reward")
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("Reward Trend Over Episodes")
plt.grid()
plt.legend()

# Save the rewards plot
plot_save_dir = "plots/"
os.makedirs(plot_save_dir, exist_ok=True)
# plot_file_path = os.path.join(plot_save_dir, "Pacman_Q_Rewards_L4_t6_gamma_lr_500.png")
plot_file_path = os.path.join(plot_save_dir, "Pacman_Q_Rewards_L6_2000_new.png")

try:
    plt.savefig(plot_file_path)
    print(f"Plot saved successfully at {plot_file_path}")
except Exception as e:
    print(f"Error saving plot: {e}")

# Save the trained model
model_save_dir = "models/"
os.makedirs(model_save_dir, exist_ok=True)
model_file_path = os.path.join(model_save_dir, "Pacman_Q_Model_L6_2000_new.keras")
try:
    dq_network.save(model_file_path)
    print(f"Model saved successfully at {model_file_path}")
except Exception as e:
    print(f"Error saving model: {e}")
