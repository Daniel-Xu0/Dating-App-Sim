import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.distributions import Categorical
from helper.timing import timer


class IndividualPolicy(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(IndividualPolicy, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Softmax(dim=-1)
        )
        self.critic = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        return self.actor(x), self.critic(x)


class IndividualPPOAgent:
    def __init__(self, agent_id, input_dim, hidden_dim, output_dim, writer, learning_rate=3e-4, entropy_coeff=.005):
        self.agent_id = agent_id
        self.policy = IndividualPolicy(input_dim, hidden_dim, output_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)
        self.scheduler = lr_scheduler.StepLR(self.optimizer, step_size=20, gamma=0.5)
        self.memory = []
        self.loss_history = []
        self.entropy_coeff = entropy_coeff
        self.writer = writer
        self.attractiveness = None
        self.cumulative_reward = 0

    def act(self, observation):
        state = torch.FloatTensor(observation).view(1, -1)
        action_probs, _ = self.policy(state)
        dist = Categorical(action_probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action)

    def remember(self, state, action, log_prob, reward, done):
        self.memory.append((state, action, log_prob, reward, done))
        self.cumulative_reward += reward

    @timer
    def monte_carlo_update(self, episode, discount_factor=0.95):
        """
        Perform a Monte Carlo update using the rewards accumulated during the episode.
        """
        if len(self.memory) == 0:
            return

        # Unpack memory
        states, actions, log_probs, rewards, dones = zip(*self.memory)
        discounted_rewards = []
        cumulative_reward = 0

        # Compute discounted rewards in reverse
        for reward in reversed(rewards):
            cumulative_reward = reward + discount_factor * cumulative_reward
            discounted_rewards.insert(0, cumulative_reward)

        # Convert to tensors
        states = torch.FloatTensor(np.array(states))
        actions = torch.LongTensor(actions)
        old_log_probs = torch.FloatTensor(log_probs)
        discounted_rewards = torch.FloatTensor(discounted_rewards)

        # Normalize rewards
        discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-8)

        # Multiple optimization epochs
        for _ in range(10):
            # Get current policy outputs
            action_probs, values = self.policy(states)
            dist = Categorical(action_probs)
            new_log_probs = dist.log_prob(actions)

            # Calculate ratio and surrogate objectives
            ratio = (new_log_probs - old_log_probs).exp()
            unclipped_surrogate = ratio * discounted_rewards
            clipped_surrogate = torch.clamp(ratio, 1.0 - 0.2, 1.0 + 0.2) * discounted_rewards

            # Compute losses
            actor_loss = -torch.min(unclipped_surrogate, clipped_surrogate).mean()
            critic_loss = nn.MSELoss()(values.squeeze(), discounted_rewards)
            entropy = dist.entropy().mean()

            total_loss = actor_loss + 0.4 * critic_loss - self.entropy_coeff * entropy

            # Optimize
            self.optimizer.zero_grad()
            total_loss.backward()
            nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0)
            self.optimizer.step()

            self.writer.add_scalar(f"Agent_{self.agent_id}/Loss/Actor", actor_loss.item(), episode)
            self.writer.add_scalar(f"Agent_{self.agent_id}/Loss/Critic", critic_loss.item(), episode)
            self.writer.add_scalar(f"Agent_{self.agent_id}/Loss/Total", total_loss.item(), episode)
            self.writer.add_scalar(f"Agent_{self.agent_id}/Entropy", entropy.item(), episode)
            self.writer.add_scalar(f"Agent_{self.agent_id}/Average_Reward", discounted_rewards.mean().item(), episode)
            self.writer.add_scalar(f"Agent_{self.agent_id}/Cumulative_Reward", self.cumulative_reward, episode)
            
            if self.attractiveness is not None:
                self.writer.add_scalar(f"Agent_{self.agent_id}/Attractiveness_vs_Reward", 
                                     self.attractiveness * discounted_rewards.mean().item(), episode)

        self.scheduler.step()
        self.loss_history.append(total_loss.item())
        self.memory = []


def create_individual_ppo_agents(players, input_dim, hidden_dim, output_dim, writer, learning_rate):
    agents = {player: IndividualPPOAgent(player, input_dim, hidden_dim, output_dim, writer, learning_rate)
              for player in players}
    return agents