import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from typing import List
import matplotlib.pyplot as plt
import numpy as np


### Some Hyperparameters

MOVE_DICT = {
    0: "nowhere",
    1: "up",
    2: "right",
    3: "down",
    4: "left"
}

REWARD_DICT = {
    1: 0,
    2: 2,
    3: 1,
}

SIZE = 5
COOPERATIVE_DIMS = 2

BATCH_SIZE = 50
LSTM_HIDDEN_DIM = 8
MIN_EPS = 0.01
REPLAY_BUFFER_SIZE = 1000
COMMS_DIM = 5




### The Agents

class Agent:
    def __init__(self, id):
        self.id = id
        self.x = random.randint(0, SIZE - 1)
        self.y = random.randint(0, SIZE - 1)
        self.hidden_state = []

    def __str__(self):
        return str(self.id)
    
    def select_random_move(self):
        move = random.randint(0, 4)
        return move
    
    def randomize_location(self):
        self.x = random.randint(0, SIZE - 1)
        self.y = random.randint(0, SIZE - 1)
    

class BaseDQNLSTM(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=LSTM_HIDDEN_DIM):
        super(BaseDQNLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc1 = nn.Linear(hidden_dim, 32)
        self.fc2 = nn.Linear(32, output_dim)
        self.bn1 = nn.BatchNorm1d(32)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, hx):
        if hx[0].shape[0] == BATCH_SIZE: 
            x = x.unsqueeze(1)
            hx = (hx[0].permute(1, 0, 2), hx[0].permute(1, 0, 2))
        out, hx = self.lstm(x, hx)
        x = self.fc1(out[:, -1, :])
        x = self.bn1(x)
        x = self.dropout(x)
        q_values = self.fc2(x)
        return q_values, hx

class CoopDQNLSTM(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=LSTM_HIDDEN_DIM):
        super(CoopDQNLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim + COOPERATIVE_DIMS, hidden_dim, batch_first=True)
        self.fc1 = nn.Linear(hidden_dim, 32)
        self.fc2 = nn.Linear(32, output_dim)
        self.bn1 = nn.BatchNorm1d(32)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, hx):
        if hx[0].shape[0] == BATCH_SIZE: 
            x = x.unsqueeze(1)
            hx = (hx[0].permute(1, 0, 2), hx[0].permute(1, 0, 2))
        out, hx = self.lstm(x, hx)
        x = self.fc1(out[:, -1, :])
        x = self.bn1(x)
        x = self.dropout(x)
        q_values = self.fc2(x)
        return q_values, hx

# class CoopDQNLSTM(nn.Module):
#     def __init__(self, input_dim, output_dim, hidden_dim=LSTM_HIDDEN_DIM):
#         super(CoopDQNLSTM, self).__init__()
#         self.lstm = nn.LSTM(input_dim + COOPERATIVE_DIMS, hidden_dim, batch_first=True)
#         self.fc1 = nn.Linear(hidden_dim, 32)
#         self.fc2 = nn.Linear(32, output_dim)
#         self.fc3 = nn.Linear(32, COMMS_DIM)
#         self.bn1 = nn.BatchNorm1d(32)
#         self.dropout = nn.Dropout(0.1)

#     def forward(self, x, hx):
#         if hx[0].shape[0] == BATCH_SIZE: 
#             x = x.unsqueeze(1)
#             hx = (hx[0].permute(1, 0, 2), hx[0].permute(1, 0, 2))
#         out, hx = self.lstm(x, hx)
#         x = self.fc1(out[:, -1, :])
#         x = self.bn1(x)
#         x = self.dropout(x)
#         q_values = self.fc2(x)
#         comms = self.fc3(x)
#         return q_values, hx, comms

AGENT_TYPES = {
    "base": BaseDQNLSTM,
    "cooperative": CoopDQNLSTM,
}

class DQNAgent(Agent):
    def __init__(self, id, input_dim, output_dim, agent_type):
        super(DQNAgent, self).__init__(id)
        self.agent_type = agent_type
        self.q_network = AGENT_TYPES[agent_type](input_dim, output_dim)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=0.001)
        self.hidden_state = (torch.zeros(1, 1, LSTM_HIDDEN_DIM), torch.zeros(1, 1, LSTM_HIDDEN_DIM))
        self.memory = deque(maxlen=REPLAY_BUFFER_SIZE)
        self.epsilon = 1

    def select_action(self, observation, hidden_state):
        self.q_network.eval()
        with torch.no_grad():
            observation = torch.FloatTensor(observation).view(1, 1, -1)
            q_values, self.hidden_state = self.q_network(observation, hidden_state)

        self.q_network.train()
        self.epsilon = max(self.epsilon * 0.999, 0.05)

        if random.random() < self.epsilon:
            return random.randint(0, 4)
        else:
            return q_values.argmax().item()

    def train(self, batch_size, hidden_states):
        if len(self.memory) < batch_size:
            return

        # Sample experiences
        minibatch = random.sample(self.memory, batch_size)
        observations, actions, rewards, next_observations, dones = zip(*minibatch)

        # Convert to PyTorch tensors
        observations = torch.FloatTensor(observations)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_observations = torch.FloatTensor(next_observations)
        dones = torch.FloatTensor(dones)

        # Forward pass
        hx = (torch.zeros(batch_size, 1, LSTM_HIDDEN_DIM), torch.zeros(batch_size, 1, LSTM_HIDDEN_DIM))
        q_values, _ = self.q_network(observations, hx) # CHECK: hx or hidden_states
        q_values = q_values.gather(1, actions.view(-1, 1))

        # Compute target
        with torch.no_grad():
            next_q_values, _ = self.q_network(next_observations, hx)
            max_next_q_values = next_q_values.max(1)[0]
            target = rewards + (1 - dones) * 0.999 * max_next_q_values

        # Loss and backward pass
        loss = nn.MSELoss()(q_values, target.view(-1, 1))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()





### The Environment

class Environment:
    def __init__(self, size, agents: List[Agent]):
        self.size = size  # 3x3 grid, so size is 3
        self.agents = agents
    
    def turn(self):
        cooperative_reward = 0
        base_reward = 0
        total_reward = 0
        for agent in self.agents:

            obs = self.get_observation(agent)
            hidden_state = agent.hidden_state

            # Agent's move
            move = agent.select_action(obs, hidden_state)
            if not self.valid_move(agent, move):
                move = 0
            self.move_agent(agent, move)

            # Compute rewards
            reward = REWARD_DICT[self.count_agents_in_cell(agent.x, agent.y)]
            if agent.agent_type=="cooperative": cooperative_reward += reward
            else: base_reward += reward
            total_reward += reward

            # Store experience (s, a, r, s', done)
            next_obs = self.get_observation(agent)
            agent.memory.append((obs, move, reward, next_obs, 0))
            

            # Train agent
            agent.train(BATCH_SIZE, hidden_state)

            # Print summary
            # print(f"Agent #{agent} {obs} moves {MOVE_DICT[move]}")
        # print(f"Total Reward: {total_reward}")

        return total_reward, base_reward, cooperative_reward
        
    def get_observation(self, agent):
        """
        The baseline observation is a 1D vector of:
        - The agent's current (x, y) location
        - The number of players in the agent's cell
        - The number of players in the cells up, down, left, and right of the agent
        - The LSTM hidden state from the agent's previous timestep
        """

        obs = list()
        x, y = agent.x, agent.y
        obs.extend([x, y])
        obs.append(self.count_agents_in_cell(x, y))
        obs.append(self.count_agents_in_cell(x, y+1))
        obs.append(self.count_agents_in_cell(x+1, y))
        obs.append(self.count_agents_in_cell(x, y-1))
        obs.append(self.count_agents_in_cell(x-1, y))

        if agent.agent_type == "cooperative":
            for other_agent in self.agents:
                if other_agent.id != agent.id and other_agent.agent_type == "cooperative":
                    obs.extend([other_agent.x, other_agent.y])

        return obs

    def valid_move(self, agent, move):
        if ((move == 1 and agent.y == self.size - 1) or
            (move == 2 and agent.x == self.size - 1) or
            (move == 3 and agent.y == 0) or
            (move == 4 and agent.x == 0)): 
            return False
        return True
    
    def move_agent(self, agent, move):
        if move==1: agent.y += 1
        elif move==2: agent.x += 1
        elif move==3: agent.y -= 1
        elif move==4: agent.x -= 1
    
    def count_agents_in_cell(self, x, y) -> int:
        count = 0
        for agent in self.agents:
            if agent.x == x and agent.y == y:
                count += 1
        return count
    
    def display(self):
        # Create an empty grid
        grid = [["   " for _ in range(self.size)] for _ in range(self.size)]

        # Initialize a grid to hold agent IDs for each cell
        agent_grid = [[[] for _ in range(self.size)] for _ in range(self.size)]

        # Collect agent IDs for each cell
        for agent in self.agents:
            x, y = agent.x, agent.y
            agent_grid[self.size - 1 - y][x].append(agent.id)

        # Fill in the display grid based on agent IDs
        for y in range(self.size):
            for x in range(self.size):
                cell = list(grid[y][x])  # Convert string to list for easy modification
                for id in agent_grid[y][x]:
                    cell[id - 1] = str(id)  # ID - 1 is used because IDs start from 1 but list indices start from 0
                grid[y][x] = "".join(cell)

        # Print the grid row by row
        for row in grid:
            print(row)






### The Experiment

# Initialize environment and agents
input_dim = 7  # observation size
output_dim = 5  # action size

num_random_seeds = 3
num_turns = 2000
window_size = 128

# Initialize a 2D array to store rewards for each policy and each turn
all_rewards = np.zeros((num_random_seeds, num_turns))
all_base_rewards = np.zeros((num_random_seeds, num_turns))
all_cooperative_rewards = np.zeros((num_random_seeds, num_turns))

for policy in range(num_random_seeds):
    agents = [
        DQNAgent(id=1, input_dim=input_dim, output_dim=output_dim, agent_type="base"),
        DQNAgent(id=2, input_dim=input_dim, output_dim=output_dim, agent_type="cooperative"),
        DQNAgent(id=3, input_dim=input_dim, output_dim=output_dim, agent_type="cooperative")
    ]
    env = Environment(SIZE, agents)
    episode_reward = 0
    
    for turn in range(num_turns):
        total_reward, base_reward, cooperative_reward = env.turn()
        all_rewards[policy, turn] = total_reward
        all_base_rewards[policy, turn] = base_reward
        all_cooperative_rewards[policy, turn] = cooperative_reward
        episode_reward += total_reward

        if turn % 32 == 0:
            print(f"Policy #{policy+1}, Episode {turn} Reward: {episode_reward}")
            episode_reward = 0
            env.display()
            for agent in agents:
                agent.randomize_location()


# for policy in range(num_policies):
#     # # Plotting total rewards
#     # policy_rewards = all_rewards[policy, :]
#     # smoothed_rewards = np.convolve(policy_rewards, np.ones(window_size)/window_size, mode='valid')
#     # plt.plot(smoothed_rewards, label=f'Total Policy #{policy+1}')

#     # Plotting base rewards
#     policy_base_rewards = all_base_rewards[policy, :] 
#     smoothed_base_rewards = np.convolve(policy_base_rewards, np.ones(window_size)/window_size, mode='valid')
#     plt.plot(smoothed_base_rewards, '--', label=f'Base Policy #{policy+1}')

#     # Plotting cooperative rewards
#     # TODO: Hardcodes number of coop agents
#     policy_coop_rewards = all_cooperative_rewards[policy, :] / 2
#     smoothed_coop_rewards = np.convolve(policy_coop_rewards, np.ones(window_size)/window_size, mode='valid')
#     plt.plot(smoothed_coop_rewards, label=f'Cooperative Policy #{policy+1}')

# plt.xlabel('Turn')
# plt.ylabel('Reward')
# plt.title('Smoothed Reward Over Time for Each Policy')
# plt.legend()
# plt.grid(True)
# plt.show()


average_base_rewards = np.mean(all_base_rewards, axis=0)
average_cooperative_rewards = np.mean(all_cooperative_rewards, axis=0) / 2
smoothed_average_base_rewards = np.convolve(average_base_rewards, np.ones(window_size)/window_size, mode='valid')
smoothed_average_cooperative_rewards = np.convolve(average_cooperative_rewards, np.ones(window_size)/window_size, mode='valid')
plt.plot(smoothed_average_base_rewards, '--', label='Average Base Reward')
plt.plot(smoothed_average_cooperative_rewards, label='Average Cooperative Reward')

plt.xlabel('Turn')
plt.ylabel('Average Reward')
plt.title('Smoothed Average Reward Over Time')
plt.legend()
plt.grid(True)
plt.show()



# # Loop through each policy's rewards and plot them
# for policy in range(num_policies):
#     policy_rewards = all_rewards[policy, :]
#     smoothed_rewards = np.convolve(policy_rewards, np.ones(window_size)/window_size, mode='valid')
#     plt.plot(smoothed_rewards, label=f'Policy #{policy+1}')

# plt.xlabel('Turn')
# plt.ylabel('Total Reward')
# plt.title('Smoothed Reward Over Time for Each Policy')
# plt.legend()
# plt.grid(True)
# plt.show()


# # Average rewards over all policies
# avg_rewards = np.mean(all_rewards, axis=0)

# # Calculate the moving average of the averaged rewards
# smoothed_rewards = np.convolve(avg_rewards, np.ones(window_size)/window_size, mode='valid')

# # Plotting
# plt.figure(figsize=(10, 6))
# plt.plot(smoothed_rewards)
# plt.xlabel('Turn')
# plt.ylabel('Average Total Reward')
# plt.title('Smoothed Reward Over Time (Averaged Across All Policies)')
# plt.grid(True)
# plt.show()
