import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class VRPModel(nn.Module):
    def __init__(self, num_inputs, num_hidden, num_outputs):
        super(VRPModel, self).__init__()
        self.fc1 = nn.Linear(num_inputs, num_hidden)
        self.fc2 = nn.Linear(num_hidden, num_outputs)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class VRPEnvironment:
    def __init__(self, num_customers, num_vehicles, vehicle_capacity):
        self.num_customers = num_customers
        self.num_vehicles = num_vehicles
        self.vehicle_capacity = vehicle_capacity
        self.reset()

    def reset(self):
        self.demands = np.random.randint(1, 10, self.num_customers)
        self.demands[0] = 0
        self.locations = np.random.rand(self.num_customers, 2)
        self.locations[0] = np.array([0, 0])
        self.routes = [[] for _ in range(self.num_vehicles)]
        self.route_demands = [0 for _ in range(self.num_vehicles)]
        self.current_vehicle = 0
        self.current_location = np.array([0, 0])

    def step(self, action):
        self.routes[self.current_vehicle].append(action)
        self.route_demands[self.current_vehicle] += self.demands[action]
        self.current_location = self.locations[action]
        if self.route_demands[self.current_vehicle] > self.vehicle_capacity:
            self.current_vehicle += 1
            self.route_demands[self.current_vehicle] = self.demands[action]
        reward = 0
        done = False
        if len(self.routes[self.current_vehicle]) == self.num_customers:
            done = True
            for route in self.routes:
                route.append(0)
            reward = -np.sum([np.linalg.norm(np.diff(self.locations[route], axis=0)) for route in self.routes])
        return self.locations[action], reward, done

class VRPReinforcementLearning:
    def __init__(self, num_inputs, num_hidden, num_outputs, num_customers, num_vehicles, vehicle_capacity, learning_rate, gamma):
        self.model = VRPModel(num_inputs, num_hidden, num_outputs)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.gamma = gamma
        self.env = VRPEnvironment(num_customers, num_vehicles, vehicle_capacity)

    def select_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        action_values = self.model(state)
        return torch.argmax(action_values
