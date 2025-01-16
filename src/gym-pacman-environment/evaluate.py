import numpy as np
import time
import matplotlib.pyplot as plt
from gym_pacman_environment.envs import PacmanEnv
from PacmanAgent import PacmanAgent
import tensorflow as tf
from tensorflow.keras.models import load_model

# Initialize environment and agent with the saved model
model_path = "models/Pacman_Q_Model_L6_newP_20_1000.keras"
model = tf.keras.models.load_model(model_path)
env = PacmanAgent(PacmanEnv())

def preprocess_state(state):
    
    return np.array(state).flatten()
    # Add your preprocessing logic here
    
# Evaluate the model on multiple episodes
def evaluate_model(env, model, episodes=20, render=True):
    reward_per_episode = []
    for episode in range(episodes):
        state = preprocess_state(env.reset())  # Reset environment
        total_reward = 0
        done = False
        steps = 0

        print(f"Starting episode {episode + 1}...")

        while not done:
            if render:
                env.render()
                time.sleep(0.1)

            # Get action using trained model (greedy policy)
            q_values = model.predict(state.reshape(1, -1), verbose=0)
            print(f"Q-values: {q_values}")  # Debugging Q-values
            action = np.argmax(q_values[0])  # Exploit learned policy

            # Step in the environment
            next_state, reward, done, _ = env.step(action)
            next_state = preprocess_state(next_state)
            env.render()
            time.sleep(0.1)
            total_reward += reward
            steps += 1

            # Update state
            state = next_state

            # Debugging step information
            print(f"Step {steps}: Action={action}, Reward={reward}, Total={total_reward}")

        # Store reward for this episode
        reward_per_episode.append(total_reward)
        print(f"Episode {episode + 1} finished with total reward: {total_reward}")

    return reward_per_episode

# Evaluate the agent
episodes_to_evaluate = 20
rewards = evaluate_model(env, model, episodes=episodes_to_evaluate, render=True)

# Analyze results
print("Evaluation complete!")
print(f"Average Reward over {episodes_to_evaluate} episodes: {np.mean(rewards)}")
print(f"Standard Deviation of Rewards: {np.std(rewards)}")

# Plot rewards for evaluation episodes
plt.plot(range(1, len(rewards) + 1), rewards)
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("Evaluation Rewards Over Episodes")
plt.grid()
plt.show()

# plt.savefig("evaluation_rewards.png")

