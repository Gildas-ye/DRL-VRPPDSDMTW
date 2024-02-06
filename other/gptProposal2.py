import numpy as np
import random


# Définir l'environnement du CVRP
class CVRP:
    def __init__(self, num_customers, capacity):
        self.num_customers = num_customers
        self.capacity = capacity
        self.customers = np.random.randint(1, 10, size=(num_customers, 2))  # coordonnées des clients
        self.distance_matrix = np.zeros((num_customers + 1, num_customers + 1))  # matrice des distances
        for i in range(num_customers + 1):
            for j in range(num_customers + 1):
                self.distance_matrix[i][j] = np.linalg.norm(self.customers[i] - self.customers[j])

    # Fonction qui calcule la récompense pour une action donnée
    def get_reward(self, action):
        # action est une liste des clients visités par un véhicule
        # la récompense est la distance totale parcourue
        dist = 0
        current_capacity = 0
        current_loc = 0  # l'emplacement actuel du véhicule est 0
        for next_loc in action:
            dist += self.distance_matrix[current_loc][next_loc]
            current_capacity += self.customers[next_loc][1]
            if current_capacity > self.capacity:
                return -1  # si la capacité dépasse la limite, la récompense est négative
            current_loc = next_loc
        # Ajouter la distance de retour au dépôt
        dist += self.distance_matrix[current_loc][0]
        return 1.0 / dist


# Définir l'algorithme Q-Learning
class QLearning:
    def __init__(self, env, num_episodes, alpha, gamma, epsilon):
        self.env = env
        self.num_episodes = num_episodes
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.Q = np.zeros((env.num_customers + 1, env.num_customers + 1))  # table Q

    # Fonction pour choisir l'action
    def choose_action(self, state):
        # exploration-exploitation trade-off
        if random.uniform(0, 1) < self.epsilon:
            return random.sample(range(1, self.env.num_customers + 1),
                                 random.randint(1, self.env.num_customers))  # action aléatoire
        else:
            return np.argmax(self.Q[state], axis=1)  # action avec la plus grande valeur Q

    # Fonction pour mettre à jour la table Q
    def update_Q(self, state, action, reward, next_state):
        self.Q[state][action] += self.alpha * (reward + self.gamma * np.max(self.Q[next_state]) - self.Q[state][action])

    # Fonction pour exécuter l'algorithme Q-Learning
    def run(self):
        for episode in range(self.num_episodes):
            state = 0  # l'état initial est le dépôt
            done = False
            while not done:

