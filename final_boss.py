from hyperopt import hp
from hyperopt import fmin, tpe
import gym
from cart_pole import Agent as CartPoleAgent
from collections import deque
from functools import partial
from statistics import mean


def lunar_lander_objective(env, strategy):
    # 1500 time steps until convergence.  Here I have a looser definition of convergence.
    # For  me  that  is  just  receiving  on  average  greater  than  40  reward  per episode on average.
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
    score_history = deque(maxlen=10)
    episodes = 310
    min_performance = -float("inf")
    for i in range(episodes):
        done = False
        current_state = env.reset()
        while not done:
            action = agent.choose_action(current_state)
            new_state, reward, done, _ = env.step(action)
            agent.learn(current_state, reward, new_state, int(done))
            min_performance = max(min_performance, agent.do_test())
            current_state = new_state


# define an objective function
def cart_pole_objective(env, strategy):
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
    score_history = deque(maxlen=10)
    episodes = 310
    min_performance = 0
    for i in range(episodes):
        done = False
        current_state = env.reset()
        while not done:
            action = agent.choose_action(current_state)
            new_state, reward, done, _ = env.step(action)
            agent.learn(current_state, reward, new_state, int(done))
            min_performance = max(min_performance, agent.do_test())
            current_state = new_state
        # print(f"Episode: {i}, Score: {score}")
        if min_performance > 195:  # constraint according to assignment
            return i
            # lower -> better
    else:  # no break
        return 1000000 - min_performance
        # forces the objective score of a failed set of params to have a score higher than one with a successful
        # set of parameters. and one set of failed params led to better results
        # than another set of failed params


def minimize_cartpole():
    # define a search space
    space = {
        'alpha': hp.randint('alpha', 1, 256) / 10000,
        'beta': hp.randint('beta', 1, 256) / 10000,
        'gamma': hp.randint('gamma', 8192, 10000) / 10000,
    }
    # minimize the objective over the space
    env = gym.make("CartPole-v1")
    # make a partial so we can reuse the same environment
    cart_pole_partial = partial(cart_pole_objective, env)
    best_configuration = fmin(cart_pole_partial, space, algo=tpe.suggest, max_evals=100)

    print(best_configuration)


def minimize_lunarlander():
    # define a search space
    space = {
        'alpha': hp.randint('alpha', 1, 256) / 10000,
        'beta': hp.randint('beta', 1, 256) / 10000,
        'gamma': hp.randint('gamma', 8192, 10000) / 10000,
    }
    # minimize the objective over the space
    env = gym.make('LunarLander-v2')
    # make a partial so we can reuse the same environment
    lunar_landar_partial = partial(lunar_lander_objective, env)
    best_configuration = fmin(lunar_landar_partial, space, algo=tpe.suggest, max_evals=100)

    print(best_configuration)


if __name__ == '__main__':
    minimize_lunarlander()
