from gymnasium.wrappers import TimeLimit
from env_hiv import HIVPatient
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributions as distributions

env = TimeLimit(
    env=HIVPatient(domain_randomization=False), max_episode_steps=200
)  # The time wrapper limits the number of steps in an episode at 200.
# Now is the floor is yours to implement the agent and train it.


# You have to implement your own agent.
# Don't modify the methods names and signatures, but you can add methods.
# ENJOY!

class Policy_Network(nn.Module):
    
    def __init__(self, state_dim, action_dim):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )

    def forward(self, states):
        x = self.mlp(states)
        #print(f"input of softmax is {x}")
        return nn.functional.softmax(x, dim=-1)
    

class Value_Network(nn.Module):

    def __init__(self, state_dim):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, states):
        return self.mlp(states)

class ProjectAgent:

    def __init__(self, gamma=0.99, epsilon=0.2, beta=0.01):
        self.state_dim = 6
        self.action_dim = 4

        self.policy_net = Policy_Network(self.state_dim, self.action_dim)
        self.value_net = Value_Network(self.state_dim)
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=3e-4)
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=3e-4)
        self.gamma = gamma
        self.epsilon = epsilon
        self.beta = beta

        self.trajectories = {"states": [], "actions": [], "rewards": [], "log_probs": [], "dones": []}
        self.env = HIVPatient(domain_randomization=False)

        self.save_path = 'PPO_results'

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def act(self, observation, use_random=False):
        if use_random:
            return torch.randint(0, 4, (1,)).item()
        
        states = torch.tensor(observation, dtype=torch.float32)
        states = (states - states.mean(dim=0)) / (states.std(dim=0) + 1e-8)

        with torch.no_grad():
            action_probs = self.policy_net.forward(states)
            #print(f'Action probabilities = {action_probs}')
        m = distributions.Categorical(action_probs)
        action = m.sample().item()
        #print(f'Action = {action}')

        return action
    
    def store_experience(self, state, action, reward, log_prob, done):
        self.trajectories["states"].append(state)
        self.trajectories["actions"].append(action)
        self.trajectories["rewards"].append(reward)
        self.trajectories["log_probs"].append(log_prob)
        self.trajectories["dones"].append(done)

    def compute_returns_and_advantages(self, next_value):
        rewards = self.trajectories["rewards"]
        dones = self.trajectories["dones"]
        states = torch.tensor(self.trajectories["states"], dtype=torch.float32)
        values = self.value_net.forward(states).detach().numpy()

        # Bootstrap the final value
        values = np.append(values, next_value)
        advantages = []
        returns = []
        gae = 0

        # Compute advantages and returns (reverse loop for GAE)
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * values[t + 1] * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * 0.95 * gae * (1 - dones[t])
            advantages.insert(0, gae)
            returns.insert(0, gae + values[t])

        return torch.tensor(returns), torch.tensor(advantages)

    def update_policy(self):
        #print(self.trajectories["log_probs"])
        states = torch.tensor(np.array(self.trajectories["states"]), dtype=torch.float32)
        actions = torch.tensor(np.array(self.trajectories["actions"]), dtype=torch.float32)
        log_probs_old = torch.tensor(np.array(self.trajectories["log_probs"])).detach()
        
        # Compute returns and advantages
        next_state = torch.tensor(self.trajectories["states"][-1], dtype=torch.float32)
        next_value = self.value_net.forward(next_state).item()
        returns, advantages = self.compute_returns_and_advantages(next_value)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        for _ in range(4):
            action_probs = self.policy_net(states)
            dist = torch.distributions.Categorical(action_probs)
            new_log_probs = dist.log_prob(actions)
            entropy = dist.entropy().mean()

            ratio = torch.exp(new_log_probs - log_probs_old)
            #print(f"New Log Probs: {new_log_probs[:5]}")
            #print(f"Old Log Probs: {log_probs_old[:5]}")
            #print(f"Ratio: {ratio[:5]}")

            # Policy Loss
            surrogate1 = ratio * advantages
            surrogate2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * advantages
            policy_loss = -torch.min(surrogate1, surrogate2).mean()
            #print(f"Policy Loss: {policy_loss.item():.6f}")

            # Value loss
            values = self.value_net.forward(states).squeeze()
            value_loss = ((returns - values) ** 2).mean()

            # Total loss
            #loss = policy_loss + 0.5 * value_loss - self.beta * entropy

            # Policy optimization
            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            self.policy_optimizer.step()

            #print("Policy Network Gradients:")
            #for name, param in self.policy_net.named_parameters():
            #  if param.grad is not None:
            #      print(f"{name} - Mean: {param.grad.mean():.6f}, Std: {param.grad.std():.6f}, Max: {param.grad.max():.6f}")
            #  else:
            #      print(f"{name} - No gradient")

            # Value optimization
            self.value_optimizer.zero_grad()
            value_loss.backward()
            self.value_optimizer.step()

            #print("Value Network Gradients:")
            #for name, param in self.value_net.named_parameters():
            #    if param.grad is not None:
            #        print(f"{name} - Mean: {param.grad.mean():.6f}, Std: {param.grad.std():.6f}, Max: {param.grad.max():.6f}")
            #    else:
            #        print(f"{name} - No gradient")

        self.trajectories = {"states": [], "actions": [], "rewards": [], "log_probs": [], "dones": []}

    
    def train(self, num_episodes):
        for episode in range(num_episodes):
            print(episode)
            state = self.env.reset()[0]
            done = False
            total_reward = 0

            for k in range(200):
                action = self.act(state)
                action_tensor = torch.tensor(action, dtype=torch.float32).to(self.device)
                state_tensor = torch.tensor(state, dtype=torch.float32).to(self.device)
                state_tensor = (state_tensor - state_tensor.mean(dim=0)) / (state_tensor.std(dim=0) + 1e-8)

                action_prob = self.policy_net(state_tensor)[action].item()
                log_prob = np.log(action_prob)

                next_state, reward, done, _ = self.env.step(action)[:4]
                total_reward += reward
                self.store_experience(state, action, reward, log_prob, done)
                state = next_state

            self.update_policy()

            # Save periodically
            if (episode + 1) % 100 == 0 or (episode + 1) == num_episodes:
                self.save(self.save_path)
                print(f"Agent state saved after {episode + 1} episodes.")

            # Log progress
            if (episode + 1) % 10 == 0:
                print(f"Episode {episode + 1}/{num_episodes}, Total Reward: {total_reward}")

    def save(self, path):
        torch.save({
            'policy_network': self.policy_net.state_dict(),
            'value_network': self.value_net.state_dict()
        }, path)

    def load(self):
        checkpoint = torch.load(self.save_path)
        self.policy_net.load_state_dict(checkpoint['policy_network'])
        self.value_net.load_state_dict(checkpoint['value_network'])



if __name__ == "__main__":

    agent = ProjectAgent()
    agent.train(num_episodes=1000)


