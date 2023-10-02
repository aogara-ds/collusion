import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import defaultdict
import matplotlib.pyplot as plt
import random

REWARD_DICT = {
    0: 0,
    1: 1,
    2: 2
}


#########################
###       Agents      ###
#########################

class Agent():
    def __init__(self, id):
        self.id = id
        self.location = random.choice([0,1])
    
    def __str__(self):
        return str(self.id)
    
    def random_policy(self, obs):
        return random.choice([True, False])

class RandomSender(Agent):
    # Before the other players select their move, the Sender sends a message to the Receiver    
    def random_policy(self, obs):
        move = random.choice([True, False])
        message = random.random()
        return move, message

class RandomReceiver(Agent):
    # The Receiver gets a message from the Sender before choosing a move
    def random_policy(self, obs, message):
        return random.choice([True, False])



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
                obs = self.count_agents_in_same_location(agent)
                if round_type=="random": move=agent.random_policy(obs)
                elif round_type=="silent": move = agent.dqn_policy(obs)
                moves.append(move)
        
        else:
            # Cooperative rounds have a special structure
            assert ((type(self.agents[0]) == SenderLearningAgent) and 
                (type(self.agents[1]) == ReceiverLearningAgent) and 
                (type(self.agents[2]) == LearningAgent)), "Communication must be ordered"
            
            for i, agent in enumerate(self.agents):
                obs = self.count_agents_in_same_location(agent)

                # Sender sends a message before anyone moves
                if i == 0:
                    move, message = agent.dqn_policy(obs)
                    # print(f"(Move, Message): {(move, message)}")
                    
                # Receiver receives the message before moving
                if i == 1:
                    move = agent.dqn_policy(obs, message)
                
                # This agent cannot communicate
                if i == 2:
                    move = agent.dqn_policy(obs)

                moves.append(move)
        
        # Make all the moves simultaneously
        for agent, move in zip(self.agents, moves):
            agent.location = agent.location if not move else 1 - agent.location

    def count_agents_in_same_location(self, agent):
        num_agents_in_same_location = 0
        for other_agent in self.agents:
            if other_agent.id == agent.id: 
                continue
            elif other_agent.location == agent.location: 
                num_agents_in_same_location += 1
        return num_agents_in_same_location
    
    def display(self):
        print(f"Room 0: {[agent.id for agent in self.agents if agent.location==0]}")
        print(f"Room 1: {[agent.id for agent in self.agents if agent.location==1]}")


#########################
###        DQNs       ###
#########################

class BaseDQN(nn.Module):
    def __init__(self):
        super(BaseDQN, self).__init__()
        self.fc1 = nn.Linear(1, 8)
        self.fc2 = nn.Linear(8, 2)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class SenderDQN(nn.Module):
    # Before the other players select their move, the Sender sends a message to the Receiver
    def __init__(self):
        super(SenderDQN, self).__init__()
        self.fc1 = nn.Linear(1, 8)
        self.choose_move = nn.Linear(8, 2)
        self.choose_message = nn.Linear(8, 1)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        move = self.choose_move(x)
        message = self.choose_message(x)
        return move, message

class ReceiverDQN(nn.Module):
    # The Receiver gets a message from the Sender before choosing a move
    def __init__(self):
        super(ReceiverDQN, self).__init__()
        self.fc1 = nn.Linear(2, 8)
        self.fc2 = nn.Linear(8, 2)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x



#########################
###  Learning Agent   ###
#########################

class LearningAgent(Agent):
    def __init__(self, id, lr=0.01, discount_factor=0.99):
        super().__init__(id)
        self.model = BaseDQN()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()
        self.discount_factor = discount_factor
        self.epsilon = 1  # for ε-greedy strategy
        self.epsilon_decay = 0.99
        self.epsilon_min = 0.01
        self.last_action = None
        self.last_obs = None

        self.metrics = {
            'average_reward': 0,
            'action_count': defaultdict(int),
            'loss': [],
            'q_values': []
        }
    
    def dqn_policy(self, obs):
        self.last_obs = obs
        obs_tensor = torch.FloatTensor([obs]).view(-1, 1)
        with torch.no_grad():
            q_values = self.model(obs_tensor)
        # The action represents whether the agent will switch locations
        if random.random() < self.epsilon:
            action = random.choice([0, 1])
        else:
            action = torch.argmax(q_values).item()
        self.metrics['action_count'][action] += 1
        self.metrics['q_values'].append(q_values.tolist())
        self.last_action = action
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        return bool(action)
    
    def update_q_values(self, reward, next_obs):
        if self.last_action is None:
            return
        self._train(self.last_obs, self.last_action, reward, next_obs)

    # def _train(self, obs, action, reward, next_obs):
    #     if isinstance(obs, tuple):  # For ReceiverLearningAgent
    #         obs_tensor = torch.FloatTensor([obs]).view(-1, 2)
    #     else:  # For LearningAgent and SenderLearningAgent
    #         obs_tensor = torch.FloatTensor([obs]).view(-1, 1)

    def _train(self, obs, action, reward, next_obs):
        if isinstance(obs, tuple):  # For ReceiverLearningAgent
            obs_tensor = torch.FloatTensor([obs]).view(1, -1)
            next_obs_tensor = torch.FloatTensor([next_obs]).view(1, -1)
        else:  # For LearningAgent and SenderLearningAgent
            obs_tensor = torch.FloatTensor([obs]).view(-1, 1)
            next_obs_tensor = torch.FloatTensor([next_obs]).view(-1, 1)


        # next_obs_tensor = torch.FloatTensor([next_obs]).view(-1, 1)
        action_tensor = torch.LongTensor([action]).view(-1, 1)
        reward_tensor = torch.FloatTensor([reward]).view(-1, 1)

        if isinstance(self.model, SenderDQN):  # For SenderLearningAgent
            q_values, _ = self.model(obs_tensor)
            q_values = q_values.gather(1, action_tensor)
        else:  # For LearningAgent and ReceiverLearningAgent
            q_values = self.model(obs_tensor).gather(1, action_tensor)

        if isinstance(self.model, SenderDQN):  # For SenderLearningAgent
            next_q_values = self.model(next_obs_tensor)[0].max(1)[0].detach()
        else:  # For LearningAgent and ReceiverLearningAgent
            next_q_values = self.model(next_obs_tensor).max(1)[0].detach()

        target = reward_tensor + self.discount_factor * next_q_values.view(-1, 1)

        loss = self.criterion(q_values, target)
        self.metrics['loss'].append(loss.item())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.metrics['average_reward'] = (self.metrics['average_reward'] * len(self.metrics['loss']) + reward) / (len(self.metrics['loss']) + 1)

    def print_metrics(self):
        print(f"Agent {self.id} Metrics:")
        print(f"  Average Reward: {self.metrics['average_reward']:.2f}")
        print(f"  Action Count: {dict(self.metrics['action_count'])}")
        print(f"  Last Loss: {self.metrics['loss'][-1] if self.metrics['loss'] else 'N/A'}")
        print(f"  Last Q-Values: {self.metrics['q_values'][-1] if self.metrics['q_values'] else 'N/A'}")



class SenderLearningAgent(LearningAgent):
    def __init__(self, id, lr=0.01, discount_factor=0.99):
        super().__init__(id, lr, discount_factor)
        self.model = SenderDQN()
        
    def dqn_policy(self, obs):
        self.last_obs = obs
        obs_tensor = torch.FloatTensor([obs]).view(-1, 1)
        with torch.no_grad():
            move_q_values, message = self.model(obs_tensor)
        if random.random() < self.epsilon:
            action = random.choice([0, 1])
        else:
            action = torch.argmax(move_q_values).item()

        # Add random decaying noise to the message
        message_with_noise = message.item() + (random.random() - 0.5) * self.epsilon
        message_with_noise = min(max(message_with_noise, 0), 1)


        self.last_action = action
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        return bool(action), message



class ReceiverLearningAgent(LearningAgent):
    def __init__(self, id, lr=0.01, discount_factor=0.99):
        super().__init__(id, lr, discount_factor)
        self.model = ReceiverDQN()
        
    def dqn_policy(self, obs, message):
        self.last_obs = (obs, message)
        obs_tensor = torch.FloatTensor([obs, message]).view(-1, 2)
        with torch.no_grad():
            q_values = self.model(obs_tensor)
        if random.random() < self.epsilon:
            action = random.choice([0, 1])
        else:
            action = torch.argmax(q_values).item()
        self.last_action = action
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        return bool(action)
    

    def update_q_values(self, reward, next_obs):
        if self.last_action is None:
            return
        # Include the last message in the observation
        full_next_obs = (next_obs, self.last_obs[1])
        self._train(self.last_obs, self.last_action, reward, full_next_obs)

#########################
###      Training     ###
#########################

# Initialize agents and environment with configurable parameters
lr = 0.01
discount_factor = 0.99

# Initialize reward tracking
reward_history = {(i, init): [] for i in range(1, 4) for init in range(1, 4)}

for initialization in range(1,4):
    agents = [
        SenderLearningAgent(1),
        ReceiverLearningAgent(2),
        LearningAgent(3)
    ]
    env = Environment(agents)

    # Just one long episode
    for episode in range(100):
        for step in range(100):
            env.round("cooperative")

            for agent in agents:
                reward = REWARD_DICT[env.count_agents_in_same_location(agent)]
                next_obs = env.count_agents_in_same_location(agent)
                agent.update_q_values(reward, next_obs)
                reward_history[(agent.id, initialization)].append(reward)

        print(f"Episode #{episode}")
        env.display()
        print()

        # Reset agent location at the end of each episode
        for agent in env.agents:
            agent.location = random.choice([0, 1])

# Calculate and print the average reward for each agent and each initialization
for agent_id in range(1, 4):
    for initialization in range(1, 4):
        avg_reward = sum(reward_history[(agent_id, initialization)]) / len(reward_history[(agent_id, initialization)])
        print(f"Agent {agent_id}, Initialization {initialization}: Average Reward = {avg_reward}")


#########################
###      Plotting     ###
#########################

# # Plotting
# colors = ['b', 'g', 'r']
# patterns = ['-', '--', ':']

# for agent_id in range(1, 4):
#     for initialization in range(1, 4):
#         rewards = reward_history[(agent_id, initialization)]
#         if len(rewards) < 100:
#             continue
#         avg_rewards = [sum(rewards[i:i+100]) / 100 for i in range(len(rewards) - 99)]
#         plt.plot(avg_rewards, label=f'Agent {agent_id}, Init {initialization}', 
#                  linestyle=patterns[initialization-1], color=colors[agent_id-1])

# plt.title('Average Reward per Player per Turn Over Last 100 Steps')
# plt.xlabel('Step')
# plt.ylabel('Average Reward')
# plt.legend()
# plt.show()


# Plotting
colors_group = {
    1: ['#FF0000', '#FF6666', '#FF9999'],  # Reds
    2: ['#0000FF', '#6666FF', '#9999FF'],  # Blues
    3: ['#00FF00', '#66FF66', '#99FF99']   # Greens
}

for agent_id in range(1, 4):
    for initialization in range(1, 4):
        rewards = reward_history[(agent_id, initialization)]
        if len(rewards) < 1000:
            continue
        avg_rewards = [sum(rewards[i:i+1000]) / 1000 for i in range(len(rewards) - 999)]
        plt.plot(avg_rewards, label=f'Agent {agent_id}, Init {initialization}', 
                 color=colors_group[agent_id][initialization-1])

plt.title('Average Reward per Player per Turn Over Last 100 Steps')
plt.xlabel('Step')
plt.ylabel('Average Reward')
plt.legend()
plt.show()