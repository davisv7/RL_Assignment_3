from network import GenericNetwork
from collections import deque
from random import sample
import torch as T
import torch.nn.functional as F


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
        critic_loss = (delta ** 2)

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
        scores = []
        for i in range(10):
            done = False
            current_state = self.env.reset()
            total_reward = 0
            while not done:
                action = self.choose_action(current_state)
                new_state, reward, done, _ = self.env.step(action)
                current_state = new_state
                total_reward += reward
            scores.append(total_reward)
        return scores

    def action_replay(self):
        if len(self.obs_history) < self.batch_size:
            return
        else:
            batch = sample(self.obs_history, self.batch_size)
            for state, reward, new_state, done in batch:
                self.learn(state, reward, new_state, done)
