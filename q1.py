import numpy as np
from scipy.stats import norm, poisson

# Define the problem
questions = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]
probabilities = [0.99, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
rewards = [100, 500, 1000, 5000, 10000,
           50000, 100000, 500000, 1000000, 5000000]

# Define the state space
states = [(q, p, r) for q, p, r in zip(questions, probabilities, rewards)]

# Define the action space
actions = ['True', 'False']

# Define the reward function


def get_reward(state, action):
    if action == 'True':
        if state[1] >= np.random.uniform():
            return state[2]
        else:
            return 0
    else:
        if state[1] <= np.random.uniform():
            return state[2]
        else:
            return 0


# Initialize the Q-value function
Q = {}
for state in states:
    for action in actions:
        Q[(state, action)] = 0

# Set hyperparameters
epsilon = 0.1
gamma = 1.0
num_episodes = 1000

# Train the AI
for i in range(num_episodes):
    # Choose an initial state
    state = states[np.random.randint(0, len(states))]

    # Play the game until termination
    episode = []
    while True:
        # Choose an action using the exploration-exploitation strategy
        if np.random.uniform() < epsilon:
            action = actions[np.random.randint(0, len(actions))]
        else:
            action = max(actions, key=lambda x: Q[(state, x)])

        # Take the action and observe the new state and reward
        reward = get_reward(state, action)
        next_state = states[np.random.randint(0, len(states))]
        episode.append((state, action, reward))

        # Move to the next state
        state = next_state

        # Terminate if all questions have been answered
        if state[0] == questions[-1]:
            break

    # Update the Q-value function using Monte Carlo updates
    G = 0
    for t in reversed(range(len(episode))):
        state, action, reward = episode[t]
        G = gamma * G + reward
        if (state, action) in Q:
            Q[(state, action)] += (1/num_episodes) * (G - Q[(state, action)])

    # Print the cumulative reward obtained in each episode
    print("Episode {}: {}".format(
        i, np.sum([reward for _, _, reward in episode])))

# Choose the optimal state
optimal_state_action = max(Q, key=Q.get)
optimal_state = optimal_state_action[0]
optimal_action = optimal_state_action[1]

# Print the optimal state and the corresponding action
print("Optimal state: {}, Optimal action: {}".format(
    optimal_state, optimal_action))
