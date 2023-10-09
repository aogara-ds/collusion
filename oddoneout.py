import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import defaultdict
import matplotlib.pyplot as plt
import random
import numpy as np
import pdb

REWARD_DICT = {
    0: -3,
    1: 1,
    2: 0
}


#########################
###       Agents      ###
#########################

class Agent():
    def __init__(self, id):
        self.id = id
        self.location = np.zeros(2)
        self.location[random.choice([0,1])] = 1
    
    def __str__(self):
        return str(self.id)
    
    def random_policy(self, obs):
        x = np.zeros(2)
        x[random.choice([0,1])] = 1
        return x

#########################
###    Environment    ###
#########################

class Environment():
    def __init__(self, agents):
        self.agents = agents

    def round(self, round_type):
        # Request an action from each agent
        moves = []
        
        if round_type!="cooperative":
            for agent in self.agents:
                obs = self.position_of_agents()
                if round_type=="random": move=agent.random_policy(obs)
                elif round_type=="silent": move = agent.dqn_policy(obs)
                moves.append(move)
        
        else:
            # Cooperative rounds have a special structure
            assert ((type(self.agents[0]) == MoveSenderLearningAgent) and 
                (type(self.agents[1]) == MoveReceiverLearningAgent) and 
                (type(self.agents[2]) == LearningAgent)), "Communication must be ordered"
            
            for i, agent in enumerate(self.agents):
                positions = self.position_of_agents()

                # Sender sends a message before anyone moves
                if i == 0:
                    move = agent.dqn_policy(positions)
                    message = move
                    # print(f"(Move, Message): {(move, message)}")
                    
                # Receiver receives the message before moving
                if i == 1:
                    move = agent.dqn_policy(positions, message)
                
                # This agent cannot communicate
                if i == 2:
                    move = agent.dqn_policy(positions)

                moves.append(move)
        
        # Make all the moves simultaneously
        for agent, move in zip(self.agents, moves):
            new_location = np.zeros(2)
            new_location[move] = 1
            agent.location = new_location

    def count_agents_in_same_location(self, agent):
        num_agents_in_same_location = 0
        for other_agent in self.agents:
            if other_agent.id == agent.id: 
                continue
            elif np.argmax(other_agent.location) == np.argmax(agent.location): 
                num_agents_in_same_location += 1
        return num_agents_in_same_location
    
    def position_of_agents(self): 
        positions_v = np.concatenate([agent.location for agent in self.agents], axis=0)
        return positions_v
    
    def display(self):
        print(f"Room 0: {[agent.id for agent in self.agents if np.argmax(agent.location)==0]}")
        print(f"Room 1: {[agent.id for agent in self.agents if np.argmax(agent.location)==1]}")


#########################
###        DQNs       ###
#########################

class BaseDQN(nn.Module):
    def __init__(self):
        super(BaseDQN, self).__init__()
        self.fc1 = nn.Linear(6, 12)
        self.fc2 = nn.Linear(12, 2)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class MoveReceiverDQN(nn.Module):
    # The Receiver sees the Sender's move before choosing a move
    def __init__(self):
        super(MoveReceiverDQN, self).__init__()
        self.fc1 = nn.Linear(8, 12)
        self.fc2 = nn.Linear(12, 2)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


#########################
###  Learning Agent   ###
#########################

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)

class LearningAgent(Agent):
    def __init__(self, id, lr=0.01, discount_factor=0, buffer_size=1000, batch_size=64):
        super().__init__(id)
        self.replay_buffer = ReplayBuffer(buffer_size)
        self.batch_size = batch_size
        self.model = BaseDQN()
        self.target_model = BaseDQN()
        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()
        self.discount_factor = discount_factor
        self.epsilon = 1  # for ε-greedy strategy
        self.epsilon_decay = 0.9999
        self.epsilon_min = 0
        self.last_action = None
        self.last_obs = None
    
    def update_target_network(self):
        self.target_model.load_state_dict(self.model.state_dict())
    
    def dqn_policy(self, obs, verbose = False):
        with torch.no_grad(): 
            q_values = self.model(torch.FloatTensor(obs))
        if random.random() < self.epsilon: action_index = random.choice([0, 1])
        else: 
            # action_index = torch.argmax(q_values).item()
            probs = torch.nn.functional.softmax(q_values, dim=0)
            action_index = torch.multinomial(probs, 1).item()

        self.last_obs = obs
        self.last_action = action_index
        
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        if verbose: return q_values
        else: return action_index

    def train(self):
        if len(self.replay_buffer) <= self.batch_size: return
        experiences = self.replay_buffer.sample(self.batch_size)
        obs, actions, rewards, next_obs = zip(*experiences)
        obs_tensor = torch.FloatTensor(obs)
        actions_tensor = torch.LongTensor(actions)
        rewards_tensor = torch.FloatTensor(rewards)
        next_obs_tensor = torch.FloatTensor(next_obs)

        q_values = self.model(obs_tensor).gather(1, actions_tensor.unsqueeze(1)).squeeze(1)
        next_q_values = self.target_model(next_obs_tensor).max(1)[0]

        target = rewards_tensor + self.discount_factor * next_q_values.detach()
        loss = self.criterion(q_values, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

class MoveSenderLearningAgent(LearningAgent):
    pass

class MoveReceiverLearningAgent(LearningAgent):
    def __init__(self, id, lr=0.01, discount_factor=0.0):
        super().__init__(id, lr, discount_factor)
        self.model = MoveReceiverDQN()
        self.target_model = MoveReceiverDQN()
        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()

    def dqn_policy(self, obs, message, verbose = False):
        one_hot_message = np.zeros(2)
        one_hot_message[message] = 1
        obs_tensor = torch.cat((torch.FloatTensor(obs), torch.FloatTensor(one_hot_message)), dim=0)

        with torch.no_grad(): q_values = self.model(obs_tensor)
        if random.random() < self.epsilon: action_index = random.choice([0, 1])
        else: 
            # action_index = torch.argmax(q_values).item()
            probs = torch.nn.functional.softmax(q_values, dim=0)
            action_index = torch.multinomial(probs, 1).item()

        self.last_obs = (obs, message)
        self.last_action = action_index
        
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        if verbose: return q_values
        else: return action_index

    def train(self):
        if len(self.replay_buffer) <= self.batch_size: return

        experiences = self.replay_buffer.sample(self.batch_size)
        obs, actions, rewards, next_obs = zip(*experiences)

        # Reorganizing
        positions, messages = zip(*obs)
        next_positions, next_messages = zip(*next_obs)
        one_hot_messages = [np.concatenate((pos, [1 if i == msg else 0 for i in range(2)])) for pos, msg in zip(positions, messages)]
        one_hot_next_messages = [np.concatenate((pos, [1 if i == nm else 0 for i in range(2)])) for pos, nm in zip(next_positions, next_messages)]
        
        obs_tensor = torch.FloatTensor(one_hot_messages)
        next_obs_tensor = torch.FloatTensor(one_hot_next_messages)
        rewards_tensor = torch.FloatTensor(rewards)
        actions_tensor = torch.LongTensor(actions)

        q_values = self.model(obs_tensor).gather(1, actions_tensor.unsqueeze(1)).squeeze(1)
        next_q_values = self.model(next_obs_tensor).max(1)[0]

        target = rewards_tensor + self.discount_factor * next_q_values.detach()

        loss = self.criterion(q_values, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


#########################
###    Show Policy    ###
#########################

def print_policies(agents):
    positions = [
        [0, 1, 0, 1, 0, 1],
        [0, 1, 0, 1, 1, 0],
        [0, 1, 1, 0, 0, 1],
        [0, 1, 1, 0, 1, 0],
        [1, 0, 0, 1, 0, 1],
        [1, 0, 0, 1, 1, 0],
        [1, 0, 1, 0, 0, 1],
        [1, 0, 1, 0, 1, 0],
    ]

    messages = [0, 1]

    for agent in agents:
        print(type(agent))
        for position in positions:
            if type(agent) == MoveReceiverLearningAgent:
                for message in messages:
                    q_values = agent.dqn_policy(position, message, verbose=True)
                    action_probs = torch.softmax(q_values, dim=0)
                    q_values_rounded = [round(val, 4) for val in q_values.detach().tolist()]
                    action_probs_rounded = [round(val, 4) for val in action_probs.detach().tolist()]
                    print(f"{position}, {message}: {q_values_rounded}, {action_probs_rounded}") 
            else:
                q_values = agent.dqn_policy(position, verbose=True)
                action_probs = torch.softmax(q_values, dim=0)
                q_values_rounded = [round(val, 4) for val in q_values.detach().tolist()]
                action_probs_rounded = [round(val, 4) for val in action_probs.detach().tolist()]
                print(f"{position}: {q_values_rounded}, {action_probs_rounded}")


#########################
###      Training     ###
#########################

# Initialize agents and environment with configurable parameters
seeds = 1

# Initialize reward tracking
reward_history = {(i, init): [] for i in range(1, 4) for init in range(1, seeds+1)}

for initialization in range(1,seeds+1):
    agents = [
        LearningAgent(1),
        LearningAgent(2),
        LearningAgent(3)
    ]
    env = Environment(agents)

    # Just one long episode
    for episode in range(3000):
        for step in range(10):
            env.round("silent")

            for agent in agents:
                reward = REWARD_DICT[env.count_agents_in_same_location(agent)]
                reward_history[(agent.id, initialization)].append(reward)
                next_positions = env.position_of_agents()
                if type(agent) != MoveReceiverLearningAgent:
                    agent.replay_buffer.push(agent.last_obs, agent.last_action, reward, next_positions)
                else:
                    sender = agents[0]
                    assert type(sender == MoveSenderLearningAgent), "Agents must be ordered"
                    sender_next_obs = env.position_of_agents()
                    sender_next_move = sender.dqn_policy(sender_next_obs)
                    next_message = sender_next_move
                    agent.replay_buffer.push(agent.last_obs, agent.last_action, reward, (next_positions, next_message))

        if episode % 2 == 0:
            for agent in agents:
                agent.train()
        
        if episode % 10 == 0:
            for agent in agents:
                agent.update_target_network()

        if episode % 1000 == 0:
            print_policies(agents)

        # Reset agent location at the end of each episode
        for agent in env.agents:
            new_location = np.zeros(2)
            new_location[random.choice([0,1])] = 1
            agent.location = new_location

# Calculate and print the average reward for each agent and each initialization
for agent_id in range(1, 4):
    for initialization in range(1, seeds+1):
        avg_reward = sum(reward_history[(agent_id, initialization)]) / len(reward_history[(agent_id, initialization)])
        print(f"Agent {agent_id}, Initialization {initialization}: Average Reward = {avg_reward}")

#########################
###      Plotting     ###
#########################

colors_group = {
    1: ['#FF0000', '#FF6666', '#FFCCCC'],  # Reds
    2: ['#0000FF', '#6666FF', '#CCCCFF'],  # Blues
    3: ['#00FF00', '#66FF66', '#CCFFCC']   # Greens
}

for agent_id in range(1, 4):
    for initialization in range(1, seeds+1):
        rewards = reward_history[(agent_id, initialization)]
        if len(rewards) < 100:
            continue
        avg_rewards = [sum(rewards[i:i+100]) / 100 for i in range(len(rewards) - 99)]
        plt.plot(avg_rewards, label=f'Agent {agent_id}, Init {initialization}', 
                 color=colors_group[agent_id][initialization-1])

plt.title('Average Reward per Player per Turn Over Last 100 Steps')
plt.xlabel('Step')
plt.ylabel('Average Reward')
plt.legend()
plt.show()