from hyperopt import hp
from hyperopt import fmin, tpe
import gym
from cart_pole import Agent as CartPoleAgent
from cart_pole import Agent as LunarLanderAgent
from collections import deque
from functools import partial
from statistics import mean


def lunar_lander_objective(strategy, env):
    # 1500 time steps until convergence.  Here I have a looser definition of convergence.
    # For  me  that  is  just  receiving  on  average  greater  than  40  reward  per episode on average.
    # average over 10 tests
    alpha = strategy["alpha"]
    beta = strategy["beta"]
    gamma = strategy["gamma"]
    num_actions = env.action_space.n
    agent = LunarLanderAgent(env=env,
                             alpha=alpha,
                             beta=beta,
                             input_dim=8,
                             gamma=gamma,
                             num_actions=num_actions,
                             layer_1_size=128,
                             layer_2_size=128)
    score_history = deque(maxlen=10)
    episodes = 1510
    avg_performance = -float("inf")
    for i in range(episodes):
        done = False
        current_state = env.reset()
        while not done:
            action = agent.choose_action(current_state)
            new_state, reward, done, _ = env.step(action)
            agent.learn(current_state, reward, new_state, int(done))
            avg_performance = max(avg_performance, mean(agent.do_test()))
            current_state = new_state
        if avg_performance > 40:  # constraint according to assignment
            return i
        # lower -> better
    else:  # no break
        return 10000 - avg_performance
        # forces the objective score of a failed set of params to have a score higher than one with a successful
        # set of parameters. and one set of failed params leads to better results
        # than another set of failed params


# define an objective function
def cart_pole_objective(strategy, env):
    # CartPole - 300 episodes until convergence.
    # Convergence is defined in this environment as consistently receiving greater than 195 reward.
    # I will define consistently as 10 times in a row.
    alpha = strategy["alpha"]
    beta = strategy["beta"]
    gamma = strategy["gamma"]
    num_actions = env.action_space.n
    agent = CartPoleAgent(env=env,
                          alpha=alpha,
                          beta=beta,
                          input_dim=4,
                          gamma=gamma,
                          num_actions=num_actions,
                          layer_1_size=128,
                          layer_2_size=128)
    episodes = 310
    min_performance = 0
    for i in range(episodes):
        done = False
        current_state = env.reset()
        while not done:
            action = agent.choose_action(current_state)
            new_state, reward, done, _ = env.step(action)
            agent.learn(current_state, reward, new_state, int(done))
            min_performance = max(min_performance, min(agent.do_test()))
            current_state = new_state
        # print(f"Episode: {i}, Score: {score}")
        if min_performance > 195:  # constraint according to assignment
            return i
            # lower -> better
    else:  # no break
        return 1000 - min_performance
        # forces the objective score of a failed set of params to have a score higher than one with a successful
        # set of parameters. while still keeping sets of failed parameters comparable


def minimize_cartpole():
    # define a search space
    space = {
        'alpha': hp.randint('alpha', 1, 128) / 10000,  # alpha and beta between 0.0001 and 0.0256
        'beta': hp.randint('beta', 1, 128) / 10000,
        'gamma': hp.randint('gamma', 8192, 10000) / 10000,  # discount factor between .8192 and 1.0
    }

    # minimize the objective over the space
    def _objective(strategy):
        env = gym.make("CartPole-v1")
        return cart_pole_objective(strategy, env)

    best_configuration = fmin(_objective, space, algo=tpe.suggest, max_evals=100)

    print(best_configuration)


def minimize_lunarlander():
    # define a search space
    space = {
        'alpha': hp.randint('alpha', 1, 128) / 10000,
        'beta': hp.randint('beta', 1, 128) / 10000,
        'gamma': hp.randint('gamma', 8192, 10000) / 10000,
    }

    # minimize the objective over the space
    def _objective(strategy):
        env = gym.make("LunarLander-v2")
        return lunar_lander_objective(strategy, env)

    best_configuration = fmin(_objective, space, algo=tpe.suggest, max_evals=100)

    print(best_configuration)


if __name__ == '__main__':
    minimize_lunarlander()
    #minimize_cartpole()
