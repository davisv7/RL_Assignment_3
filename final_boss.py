from hyperopt import hp
from hyperopt import fmin, tpe
import gym
from cart_pole import Agent as CartPoleAgent
from collections import deque
from functools import partial
from statistics import mean

# define an objective function
def cart_pole_objective(env, strategy):
    # CartPole - 300 episodes until convergence.
    # Convergence is defined in this environment as consistently receiving greater than 195 reward.
    # I will define consistently as 10 times in a row.
    alpha = strategy["alpha"]
    beta = strategy["beta"]
    gamma = strategy["gamma"]
    num_actions = env.action_space.n
    agent = CartPoleAgent(alpha=alpha,
                          beta=beta,
                          input_dim=4,
                          gamma=gamma,
                          num_actions=num_actions,
                          layer_1_size=128,
                          layer_2_size=128)
    score_history = deque(maxlen=10)
    episodes = 310
    for i in range(episodes):
        done = False
        score = 0
        current_state = env.reset()
        while not done:
            action = agent.choose_action(current_state)
            new_state, reward, done, _ = env.step(action)
            score += reward
            agent.learn(current_state, reward, new_state, int(done))
            current_state = new_state
        # print(f"Episode: {i}, Score: {score}")
        score_history.append(score)
        if all([x > 195 for x in score_history]):
            break
    else:

        return 1000000-mean(score_history)
    # 0 <= x <= 310
    return -i


# define a search space
# agent = Agent(alpha, beta, input_dim=4, gamma, num_actions=2, layer_1_size=128, layer_2_size=128)
space = {
    'alpha': hp.randint('alpha', 1, 256)/10000,
    'beta': hp.randint('beta', 1, 256)/10000,
    'gamma': hp.randint('gamma', 8192, 10000)/10000,
}
# minimize the objective over the space
env = gym.make("CartPole-v1")
# make a partial so we can reuse the same environment
cart_pole_partial = partial(cart_pole_objective, env)
best = fmin(cart_pole_partial, space, algo=tpe.suggest, max_evals=100)

print(best)
