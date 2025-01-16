import gym # type: ignore
from gym import logger # type: ignore
from gym_pacman_environment.envs import PacmanEnv
from tensorflow.keras.models import load_model # type: ignore
from utils import get_level
import numpy as np # type: ignore

# TODO set the desired number of games to play
episode_count = 100

# Set to False to disable that information about the current state of the game are printed out on the console
# Be aware that the gameworld is printed transposed to the console, to avoid mapping the coordinates and actions
printState = True

# TODO Set this to the desired level
# level_name = "RL01_straight_tunnel"
# level_name = "RL02_square_tunnel_H"
# level_name = "RL03_square_tunnel_R"
# level_name = "RL04_square_tunnel_deadends_H"
# level_name = "RL05_intersecting_tunnels_H_R"
level_name = "RL06_intersecting_tunnels_deadends_H_R"

level = get_level(level_name)

# You can set this to False to change the agent's observation to Box from OpenAIGym - see also PacmanEnv.py
# Otherwise a 2D array of tileTypes will be used
usingSimpleObservations = False

# Defines all possible types of tiles in the game and how they are printed out on the console
# Should not be changed unless you want to change the rules of the game
tileTypes = {
    "empty": " ",
    "wall": "#",
    "dot": "*",
    "pacman": "P",
    "ghost_rnd": "R",
    "ghost_hunter": "H",
}

ENTITIES_MAP = {
    "empty": 0,
    "dot": 2,
    "wall": 1,
    "pacman": 3,
    "random_ghost": 4,
    "hunter_ghost": 5,
}

# Will be automatically set to True by the PacmanAgent if it is used and should not be set manually
usingPythonAgent = False


class PacmanAgent(gym.Wrapper):
    # Set the class attribute
    global usingPythonAgent
    usingPythonAgent = True

    @staticmethod
    def find_entity(state, entity_name):
        ENTITIES_MAP = {
            "pacman": 1,
            "hunter_ghost": 2, 
        }
        entity_value = ENTITIES_MAP.get(entity_name)
        if entity_value is None:
            raise ValueError(f"Entity {entity_name} is not defined in ENTITIES_MAP.")
        
        result = np.where(state == entity_value)
        if result[0].size > 0:
            return (result[0][0], result[1][0])
        return None



    def __init__(self, env_name="gym_pacman_environment:pacman-python-v0",model_path=None):
        """ """

        ##PacmanAgent
        # super(PacmanAgent, self).__init__(gym.make(env_name))
        # self.env_name = env_name
        # self.action_space = self.env.action_space

        ##evaluate and Q-learning
        super(PacmanAgent, self).__init__(env_name)
        self.env_name = PacmanEnv
        self.action_space = self.env.action_space
        self.model = None  # Model will be loaded if a path is provided
        if model_path:
            self.load_model(model_path)
            print("Model loaded successfully.")
            self.model.summary()  # Print the model's architecture

#########Load model####
    def load_model(self, model_path: str):
        """
        Load the pre-trained model for evaluation.
        :param model_path: Path to the saved model.
        """
        try:
            self.model = load_model(model_path)
        except Exception as e:
            raise RuntimeError(f"Failed to load model from {model_path}: {e}")
 #########Load model####

    # def act(self, action: int) -> str:
    #     """
    #     Convert the action from an integer to a string
    #     :param action: The action to be executed as an integer
    #     :return: The action to be executed as a string
    #     """
    #     match action:
    #         case 0:
    #             action = "GO_NORTH"
    #         case 1:
    #             action = "GO_WEST"
    #         case 2:
    #             action = "GO_EAST"
    #         case 3:
    #             action = "GO_SOUTH"
    #         case _:
    #             raise ValueError(f"Invalid action: {action}")
    #     return action
    def act(self ,observation: np.ndarray) -> int:
        """
        Predict the next action using the trained model.
        :param observation: The current observation of the environment
        :return: The predicted action
        """
        # Flatten the observation if it's a 2D grid
        if len(observation.shape) > 1:  # Check if the observation is not already flattened
            observation = observation.flatten()
        
        # Add a batch dimension
        observation = np.expand_dims(observation, axis=0)
        
        # Predict Q-values
        q_values = self.model.predict(observation, verbose=0)
        print("Q-values:", q_values)
        
        # Select action with the highest Q-value
        action = np.argmax(q_values)

        # Convert the action to a string that the environment understands
        # Map the action to the correct string representation
        # action_str = self.int_to_action(action)
        # return action_str
        return action

    def int_to_action(self, action: int) -> str:
        """
        Convert an integer action to a string action.
        :param action: The action to be converted to string.
        :return: The string representation of the action.
        """
        match action:
            case 0:
                action = "GO_NORTH"
            case 1:
                action = "GO_WEST"
            case 2:
                action = "GO_EAST"
            case 3:
                action = "GO_SOUTH"
            case _:
                raise ValueError(f"Invalid action: {action}")
        return action
        # if action == 0:
        #     return "GO_NORTH"
        # elif action == 1:
        #     return "GO_WEST"
        # elif action == 2:
        #     return "GO_EAST"
        # elif action == 3:
        #     return "GO_SOUTH"
        # else:
        #     raise ValueError(f"Invalid action: {action}")

    def step(self, action: int) -> tuple:
        """
        Execute one time step within the environment
        :param action: The action to be executed
        :return: observation, reward, done, info
        """
        # print("Action:", type(action))
        return self.env.step(self.int_to_action(action))

    def reset(self) -> object:
        """
        Resets the state of the environment and returns an initial observation.
        :return: observation (object): the initial observation of the space.
        """
        return self.env.reset()


if __name__ == "__main__":
    # Can also be set to logger.WARN or logger.DEBUG to print out more information during the game
    logger.set_level(logger.DISABLED)

    # Select which gym-environment to run
    env = PacmanAgent()

    # Execute all episodes by resetting the game and play it until it is over
    for i in range(episode_count):
        observation = env.reset()
        reward = 0
        done = False

        while True:
            # Determine the agent's next action based on the current observation and reward and execute it
            env.render()
            # TODO better action selection
            action = env.action_space.sample()
            observation, reward, done, debug = env.step(action)
            if done:
                break

    env.close()







# #     def step(self, action: int) -> tuple:
# #         """
# #         Execute one time step within the environment.
# #         :param action: The action to be executed.
# #         :return: A tuple (observation, reward, done, info).
# #         """
# #         if action not in range(self.action_space.n):
# #             raise ValueError(f"Invalid action {action}. Valid actions are in the range [0, {self.action_space.n - 1}].")

# #         try:
# #             # Perform the step in the wrapped environment
# #             observation, reward, done, info = self.env.step(action)

# #             # Optionally preprocess observation if needed
# #             if isinstance(observation, np.ndarray):
# #                 observation = observation.flatten()  # Example: flatten 2D observations
# #             return observation, reward, done, info

# #         except Exception as e:
# #             raise RuntimeError(f"Error during environment 'step' execution with action {action}.") from e


# #     def reset(self) -> object:
# #         """
# #         Resets the state of the environment and returns an initial observation.
# #         :return: The initial observation of the space.
# #         """
# #         try:
# #             # Perform reset in the wrapped environment
# #             observation = self.env.reset()

# #             # Optionally preprocess observation if needed
# #             if isinstance(observation, np.ndarray):
# #                 observation = observation.flatten()  # Example: flatten 2D observations
# #             return observation

# #         except AttributeError as e:
# #             raise RuntimeError("The environment's 'reset' method failed.") from e


# # if __name__ == "__main__":
# #   

# #     # Execute all episodes by resetting the game and play it until it is over
# #     for i in range(episode_count):
# #         observation = env.reset()
# #         reward = 0
# #         done = False

# #         while not done:
# #             # Determine the agent's next action based on the current observation and reward and execute it
# #             if printState:
# #                 env.render()
# #                 time.sleep(0.1)

# #             # Use the model to decide the action
# #             action = env.act(observation)
# #             print("Action:", action)
# #             observation, reward, done, debug = env.step(action)
            
# #             if done:
# #                 print(f"Episode {i + 1} finished with reward: {reward}")
# #                 break

# #     env.close()



#     def act(self, observation: np.ndarray) -> int:
#         """
#         Predict the next action using the trained model.
#         :param observation: The current observation of the environment.
#         :return: The predicted action.
#         """
#         if self.model is None:
#             raise ValueError("Model not loaded. Cannot predict actions.")

#         # Flatten the observation if it's a 2D grid
#         if len(observation.shape) > 1:
#             observation = observation.flatten()

#         # Add a batch dimension
#         observation = np.expand_dims(observation, axis=0)

#         # Predict Q-values
#         q_values = self.model.predict(observation, verbose=0)
#         action = np.argmax(q_values)
#         return action

#     def step(self, action: int) -> tuple:
#         """
#         Execute one time step within the environment.
#         :param action: The action to be executed.
#         :return: A tuple (observation, reward, done, info).
#         """
#         if action not in range(self.action_space.n):
#             raise ValueError(f"Invalid action {action}. Valid actions are in the range [0, {self.action_space.n - 1}].")

#         try:
#             observation, reward, done, info = self.env.step(action)

#             if isinstance(observation, np.ndarray):
#                 observation = observation.flatten()  # Ensure observation is flattened
#             return observation, reward, done, info

#         except Exception as e:
#             raise RuntimeError(f"Error during environment 'step' execution with action {action}.") from e

#     def reset(self) -> object:
#         """
#         Resets the state of the environment and returns an initial observation.
#         :return: The initial observation of the space.
#         """
#         try:
#             observation = self.env.reset()
#             if isinstance(observation, np.ndarray):
#                 observation = observation.flatten()
#             return observation

#         except AttributeError as e:
#             raise RuntimeError("The environment's 'reset' method failed.") from e



  
