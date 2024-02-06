import numpy as np


class CVRPEnv:
    def __init__(self, num_customers, num_depots, vehicle_capacity):
        self.num_customers = num_customers
        self.num_depots = num_depots
        self.vehicle_capacity = vehicle_capacity
        self.customers = None
        self.depots = None
        self.current_step = 0
        self.total_reward = 0
        self.done = False

    def reset(self):
        self.customers = np.random.rand(self.num_customers, 2) * 100  # generate random customer locations
        self.depots = np.random.rand(self.num_depots, 2) * 100  # generate random depot locations
        self.current_step = 0
        self.total_reward = 0
        self.done = False
        return self._get_observation()

    def step(self, action):
        # Perform the selected action and return the new state, reward, and whether the episode has ended
        # In this example, the action is just a random permutation of the customers
        np.random.shuffle(self.customers)
        state = self._get_observation()
        reward = self._get_reward()
        self.total_reward += reward
        self.current_step += 1
        if self.current_step == self.num_customers:
            self.done = True
        return state, reward, self.done, {}

    def render(self):
        # Visualize the environment (optional)
        pass

    def _get_observation(self):
        # Return the current state of the environment as an observation
        state = np.concatenate([self.depots.reshape(-1), self.customers.reshape(-1)])
        return state

    def _get_reward(self):
        # Calculate the reward for the current state and action
        # In this example, the reward is just the negative distance from the depot to the closest customer
        distances = np.linalg.norm(self.customers - self.depots[:, np.newaxis], axis=2)
        min_distances = np.min(distances, axis=0)
        exceeded_capacities = np.sum(self.customers, axis=1) > self.vehicle_capacity
        unsatisfied_customers = np.sum(min_distances == np.inf)
        return -np.sum(min_distances) - np.sum(exceeded_capacities) - np.sum(unsatisfied_customers)
