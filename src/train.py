from gymnasium.wrappers import TimeLimit
from env_hiv import HIVPatient
from fast_env_py import FastHIVPatient
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributions as distributions
from collections import deque
import random

env = TimeLimit(
    env=HIVPatient(domain_randomization=False), max_episode_steps=200
)  # The time wrapper limits the number of steps in an episode at 200.
# Now is the floor is yours to implement the agent and train it.

class Q_Network(nn.Module):
    def __init__(self, state_dim, action_dim, nb_neurons):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(state_dim, nb_neurons),
            nn.ReLU(),
            nn.Linear(nb_neurons, nb_neurons),
            nn.ReLU(),
            nn.Linear(nb_neurons, nb_neurons),
            nn.ReLU(),
            nn.Linear(nb_neurons, nb_neurons),
            nn.ReLU(),
            nn.Linear(nb_neurons, nb_neurons),
            nn.ReLU(),
            nn.Linear(nb_neurons, nb_neurons),
            nn.ReLU(),
            nn.Linear(nb_neurons, action_dim)
        )

        self.apply(self.init_weights)
    
    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
    
    def forward(self, state):
        return self.mlp(state)
    
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def store(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.tensor(states, dtype=torch.float32),
            torch.tensor(actions, dtype=torch.int64),
            torch.tensor(rewards, dtype=torch.float32),
            torch.tensor(next_states, dtype=torch.float32),
            torch.tensor(dones, dtype=torch.float32)
        )
    
    def size(self):
        return len(self.buffer)
    

class ProjectAgent:
    def __init__(self, gamma=0.95, lr=1e-3, buffer_size=10000, batch_size=256, epsilon=1.0, epsilon_decay=0.996, epsilon_min=0.1, target_update_freq=400):
        self.state_dim = 6
        self.action_dim = 4
        self.gamma = gamma
        self.lr = lr
        self.batch_size = batch_size
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.target_update_freq = target_update_freq

        self.nb_neurons = 512

        # Q-network and target Q-network
        self.q_network = Q_Network(self.state_dim, self.action_dim, self.nb_neurons)
        self.target_q_network = Q_Network(self.state_dim, self.action_dim, self.nb_neurons)
        self.target_q_network.load_state_dict(self.q_network.state_dict())
        self.target_q_network.eval()

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.lr)
        self.replay_buffer = ReplayBuffer(buffer_size)
        self.steps = 0

        self.save_path = 'DQN_model.pth'

    def act(self, observation, use_random=False):
        """Epsilon-greedy action selection."""
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        else:
            state_tensor = torch.tensor(observation, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                q_values = self.q_network(state_tensor)
            return torch.argmax(q_values).item()

    def update(self):
        """Update the Q-network using a batch from the replay buffer."""
        if self.replay_buffer.size() < self.batch_size:
            return  # Wait until we have enough samples

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        #rewards = rewards / 100000
        #rewards = rewards / max(rewards)
        #print(f"Rewards are {rewards}")

        # Compute current Q-values
        q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze()
        #print(f"Q-values: {q_values}")
        #print(f"Q-values: {self.q_network(states).mean(dim=0)}")

        # Compute target Q-values using the target network
        with torch.no_grad():
            next_q_values = self.target_q_network(next_states).max(1)[0]
            #next_actions = self.q_network(next_states).argmax(1)
            #next_q_values = self.target_q_network(next_states).gather(1, next_actions.unsqueeze(1)).squeeze()
            target_q_values = rewards + self.gamma * next_q_values * (1 - dones)
        #print(f"Target Q-values: {target_q_values}")

        # Compute loss
        loss = nn.functional.mse_loss(q_values, target_q_values)
        #print(f"Loss: {loss.item()}")

        # Optimize the Q-network
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        #for name, param in self.q_network.named_parameters():
          #if param.grad is not None:
              #print(f"{name}: Grad Mean: {param.grad.mean()}, Grad Max: {param.grad.max()}")
          #else:
              #print(f"{name}: No gradients!")

        # Update epsilon for exploration
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        # Update the target network periodically
        if self.steps % self.target_update_freq == 0:
            self.target_q_network.load_state_dict(self.q_network.state_dict())

    def train_dqn(self, env, num_episodes):
      max_reward = 0

      for episode in range(num_episodes):
          state = env.reset()[0]
          total_reward = 0
        
          for t in range(200):
              action = agent.act(state)
              next_state, reward, done, _ = env.step(action)[:4]
              agent.replay_buffer.store(state, action, reward, next_state, done)

              # Update the Q-network
              agent.update()
              agent.steps += 1

              state = next_state
              total_reward += reward

              if done:
                  break
          
          if (episode + 1) % 10 == 0:
            print(f"Episode {episode + 1}/{num_episodes}, Total Reward: {total_reward}")

          if total_reward > max_reward:
              max_reward = total_reward
              self.save(self.save_path)
              print(f"The model was saved at episode {episode} with a reward of {total_reward}")

    def save(self, path):
        torch.save({
            'dqn_network': self.q_network.state_dict()
        }, path)

    def load(self):
          checkpoint = torch.load(self.save_path)
          self.policy_net.load_state_dict(checkpoint['policy_network'])
          self.value_net.load_state_dict(checkpoint['value_network'])


if __name__ == "__main__":

    agent = ProjectAgent()
    agent.train_dqn(FastHIVPatient(), num_episodes=1000)


