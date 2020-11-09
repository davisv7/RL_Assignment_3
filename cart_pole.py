import torch as T
import torch.nn as nn
import torch.nn.functional as F
# import torch.optim as optim
import torch_optimizer as optim
import numpy as np
from random import randint, sample
import gym
import matplotlib.pyplot as plt
from gym import wrappers
from collections import deque


class GenericNetwork(nn.Module):
    def __init__(self, learning_rate, input_dim, layer1_dim, layer2_dim, num_actions, actor=True):
        super(GenericNetwork, self).__init__()
        self.input_dim = input_dim
        self.layer1_dim = layer1_dim
        self.layer2_dim = layer2_dim
        self.num_actions = num_actions
        self.lr = learning_rate

        self.input_layer = nn.Linear(self.input_dim, self.layer1_dim)
        self.layer1 = nn.Linear(self.layer1_dim, self.layer2_dim)
        self.output_layer = nn.Linear(self.layer2_dim, self.num_actions)
        if actor:
            self.optimizer = optim.RAdam(self.parameters(), lr=self.lr)  # either radam or yogi
        else:
            self.optimizer = optim.RAdam(self.parameters(), lr=self.lr)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu:0')
        self.to(self.device)

    def forward(self, observation):
        state = T.tensor(observation).to(self.device).float()
        out_0 = F.relu(self.input_layer(state))
        out_1 = F.relu(self.layer1(out_0))
        out_final = self.output_layer(out_1)
        # out final is activated when action is selected

        return out_final


class Agent(object):
    def __init__(self, env, alpha, beta, input_dim, gamma, layer_1_size, layer_2_size, num_actions):
        self.env = env
        self.batch_size = 50
        self.obs_history = deque(maxlen=1000)

        self.gamma = gamma
        self.log_probs = None
        # policy is just a prob dist
        # updating the actor network with the gradient of the log of the policy
        # taking the log of the prob dist,
        # back propagating that to do loss minimization
        self.actor = GenericNetwork(alpha, input_dim, layer_1_size, layer_2_size, num_actions, actor=True)
        # attempts to approximate a policy and so it should have num action as output
        self.critic = GenericNetwork(beta, input_dim, layer_1_size, layer_2_size, num_actions=1, actor=False)
        # attempts to approximate q-value and so output should be 1
        # TODO: what happens if you have different structures for the networks?

    def choose_action(self, observation):
        # pass observation into actor network
        output = self.actor.forward(observation)
        # output is unactivated!!
        probabilities = F.softmax(output)
        # turn probabilities into a distribution to sample from
        action_probs = T.distributions.Categorical(probabilities)
        # ...and sample from that distribution to get an action
        action = action_probs.sample()
        # over time the agent will update those probabilities by backpropgating
        # and it will learn to take optimal actions
        self.log_probs = action_probs.log_prob(action)
        # action is a tensor so we have to get the action out of the tensor
        return action.item()

    def learn(self, state, reward, new_state, done):
        self.actor.optimizer.zero_grad()
        self.critic.optimizer.zero_grad()

        critic_value_s = self.critic.forward(state)
        critic_value_s_prime = self.critic.forward(new_state)

        # Take Temporal Difference Error
        delta = reward + self.gamma * critic_value_s_prime * (1 - int(done)) - critic_value_s

        # modify probabilities in the direction that will maximize future reward
        actor_loss = -self.log_probs * delta
        critic_loss = delta ** 2

        # back propagate
        (actor_loss + critic_loss).backward()
        # (actor_loss + critic_loss).backward(retain_graph=True)

        self.critic.optimizer.step()
        self.actor.optimizer.step()

    def do_training(self):
        done = False
        score = 0
        current_state = self.env.reset()
        while not done:
            action = self.choose_action(current_state)
            new_state, reward, done, _ = self.env.step(action)
            score += reward
            self.learn(current_state, reward, new_state, int(done))
            self.obs_history.append((current_state, reward, new_state, done))
            current_state = new_state
        # self.action_replay()
        return score

    def do_test(self):
        score = 0
        for i in range(10):
            done = False
            current_state = self.env.reset()
            while not done:
                action = self.choose_action(current_state)
                new_state, reward, done, _ = self.env.step(action)
                score += reward
                current_state = new_state
        return score / 10

    def action_replay(self):
        if len(self.obs_history) < self.batch_size:
            return
        else:
            batch = sample(self.obs_history, self.batch_size)
            for state, reward, new_state, done in batch:
                self.learn(state, reward, new_state, done)


def do_plotting(y1_values, y2_values):
    plt.plot(y1_values)
    plt.plot(y2_values)

    # forward_average = [sum(y1_values[:i]) / i for i in range(1, len(y1_values) + 1)]
    # plt.plot(forward_average)
    plt.xlabel("Episodes")
    plt.ylabel("Score")
    plt.legend(["Learning Scores", "Average Test Scores"])
    plt.show()


def main():
    env = gym.make("CartPole-v1")
    agent = Agent(
        env,
        alpha=0.001,
        beta=0.001,
        input_dim=4,
        gamma=0.99,
        num_actions=2,
        layer_1_size=256,
        layer_2_size=256)
    score_history = []
    test_score_history = []
    episodes = 300
    for i in range(episodes):
        score = agent.do_training()
        test_score = agent.do_test()
        print(f"Episode: {i}, Score: {score}")
        score_history.append(score)
        test_score_history.append(test_score)
    do_plotting(score_history, test_score_history)


if __name__ == '__main__':
    main()
