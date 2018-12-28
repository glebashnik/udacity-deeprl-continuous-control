import random
import copy
from collections import deque, namedtuple

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from unityagents import UnityEnvironment

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return -lim, lim


class Actor(nn.Module):
    def __init__(self, state_size, action_size, fc1_units, fc2_units):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.bn1 = nn.BatchNorm1d(fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.bn2 = nn.BatchNorm1d(fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)
        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):
        if state.dim() == 1:
            state = torch.unsqueeze(state, 0)

        x = self.bn1(F.relu(self.fc1(state)))
        x = self.bn2(F.relu(self.fc2(x)))
        return F.tanh(self.fc3(x))


class Critic(nn.Module):
    def __init__(self, state_size, action_size, fc1_units, fc2_units):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.bn1 = nn.BatchNorm1d(fc1_units)
        self.fc2 = nn.Linear(fc1_units + action_size, fc2_units)
        self.fc3 = nn.Linear(fc2_units, 1)
        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state, action):
        if state.dim() == 1:
            state = torch.unsqueeze(state, 0)

        x = self.bn1(F.relu(self.fc1(state)))
        x = F.relu(self.fc2(torch.cat((x, action), dim=1)))
        return self.fc3(x)


Experience = namedtuple('Experience', 'state action reward next_state done')


class Replay:
    def __init__(self, action_size, buffer_size, batch_size):
        self.action_size = action_size
        self.buffer = deque(maxlen=buffer_size)
        self.batch_size = batch_size

    def add(self, state, action, reward, next_state, done):
        experience = Experience(state, action, reward, next_state, done)
        self.buffer.append(experience)

    def sample(self):
        experiences = random.sample(self.buffer, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences])).float().to(DEVICE)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences])).float().to(DEVICE)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences])).float().to(DEVICE)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences])).float().to(DEVICE)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences]).astype(np.uint8)).float().to(DEVICE)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)


class Agent:
    def __init__(self, config):
        self.config = config

        self.online_actor = config.actor_fn().to(DEVICE)
        self.target_actor = config.actor_fn().to(DEVICE)
        self.actor_opt = config.actor_opt_fn(self.online_actor.parameters())

        self.online_critic = config.critic_fn().to(DEVICE)
        self.target_critic = config.critic_fn().to(DEVICE)
        self.critic_opt = config.critic_opt_fn(self.online_critic.parameters())

        self.noise = config.noise_fn()
        self.replay = config.replay_fn()

    def step(self, state, action, reward, next_state, done):
        self.replay.add(state, action, reward, next_state, done)

        if len(self.replay) > self.replay.batch_size:
            self.learn()

    def act(self, state, add_noise=True):
        state = torch.from_numpy(state).float().to(DEVICE)

        self.online_actor.eval()

        with torch.no_grad():
            action = self.online_actor(state).cpu().data.numpy()

        self.online_actor.train()

        if add_noise:
            action += self.noise.sample()

        return np.clip(action, -1, 1)

    def reset(self):
        self.noise.reset()

    def learn(self):
        states, actions, rewards, next_states, dones = self.replay.sample()

        # Update online critic model
        # Predict actions for the next states with the target actor model
        target_next_actions = self.target_actor(next_states)
        # Compute Q values for the next states and actions with the target critic model
        target_next_qs = self.target_critic(next_states, target_next_actions)
        # Compute target Q values for the current states using the Bellman equation
        target_qs = rewards + (self.config.discount * target_next_qs * (1 - dones))
        # Compute Q values for the current states and actions with the online critic model
        online_qs = self.online_critic(states, actions)
        # Compute and minimize the online critic loss
        critic_loss = F.mse_loss(online_qs, target_qs)
        self.critic_opt.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online_critic.parameters(), 1)
        self.critic_opt.step()

        # Update online actor model
        # Predict actions for current states from the online actor model
        online_actions = self.online_actor(states)
        # Compute and minimize the online actor loss
        actor_loss = -self.online_critic(states, online_actions).mean()
        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        # Update target critic and actor models
        self.soft_update(self.online_critic, self.target_critic)
        self.soft_update(self.online_actor, self.target_actor)

    def soft_update(self, online_model, target_model):
        for target_param, online_param in zip(target_model.parameters(), online_model.parameters()):
            target_param.data.copy_(self.config.target_mix * online_param.data + (1.0 - self.config.target_mix) * target_param.data)


class OrnsteinUhlenbeck:
    def __init__(self, size, mu, theta, sigma):
        self.state = None
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.reset()

    def reset(self):
        self.state = copy.copy(self.mu)

    def sample(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array([random.random() for _ in range(len(x))])
        self.state = x + dx
        return self.state


def run(agent):
    config = agent.config
    scores_deque = deque(maxlen=100)
    scores = []

    for episode in range(1, config.max_episodes + 1):
        agent.reset()
        score = 0

        env_info = config.env.reset(train_mode=True)[config.brain_name]
        state = env_info.vector_observations[0]

        for step in range(config.max_steps):
            action = agent.act(state)
            env_info = config.env.step(action)[config.brain_name]
            next_state = env_info.vector_observations[0]
            reward = env_info.rewards[0]
            done = env_info.local_done[0]

            agent.step(state, action, reward, next_state, done)

            score += reward
            state = next_state

            if done:
                break

        scores.append(score)
        scores_deque.append(score)
        mean_score = np.mean(scores_deque)

        print('\rEpisode {}\tAverage Score: {:.2f}\tScore: {:.2f}'.format(episode, mean_score, score))

        if mean_score >= config.goal_score:
            break

    torch.save(agent.online_actor.state_dict(), config.actor_path)
    torch.save(agent.online_critic.state_dict(), config.critic_path)

    fig, ax = plt.subplots()
    ax.plot(np.arange(1, len(scores) + 1), scores)
    ax.set_ylabel('Score')
    ax.set_xlabel('Episode #')
    fig.savefig(config.scores_path)
    plt.show()


class Config:
    def __init__(self, seed):
        self.seed = seed
        random.seed(seed)
        torch.manual_seed(seed)

        self.env = None
        self.brain_name = None
        self.state_size = None
        self.action_size = None
        self.actor_fn = None
        self.actor_opt_fn = None
        self.critic_fn = None
        self.critic_opt_fn = None
        self.replay_fn = None
        self.noise_fn = None
        self.discount = None
        self.target_mix = None

        self.max_episodes = None
        self.max_steps = None

        self.actor_path = None
        self.critic_path = None
        self.scores_path = None


def main():
    config = Config(seed=6)
    config.env = UnityEnvironment(file_name='Reacher_Linux/Reacher.x86_64')
    config.brain_name = config.env.brain_names[0]
    config.state_size = config.env.brains[config.brain_name].vector_observation_space_size
    config.action_size = config.env.brains[config.brain_name].vector_action_space_size

    config.actor_fn = lambda: Actor(config.state_size, config.action_size, fc1_units=256, fc2_units=256)
    config.actor_opt_fn = lambda params: optim.Adam(params, lr=3e-4)

    config.critic_fn = lambda: Critic(config.state_size, config.action_size, fc1_units=256, fc2_units=256)
    config.critic_opt_fn = lambda params: optim.Adam(params, lr=3e-4)

    config.replay_fn = lambda: Replay(config.action_size, buffer_size=int(1e6), batch_size=128)
    config.noise_fn = lambda: OrnsteinUhlenbeck(config.action_size, mu=0., theta=0.15, sigma=0.05)

    config.discount = 0.99
    config.target_mix = 1e-3

    config.max_episodes = int(1000)
    config.max_steps = int(1e6)
    config.goal_score = 30

    config.actor_path = 'actor.pth'
    config.critic_path = 'critic.pth'
    config.scores_path = 'scores.png'

    agent = Agent(config)
    run(agent)


if __name__ == '__main__':
    main()
