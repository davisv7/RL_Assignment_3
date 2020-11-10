import torch as T
import gym
import matplotlib.pyplot as plt
from agent import Agent
from statistics import mean
from util import do_plotting
def main():
    env = gym.make('LunarLander-v2')
    num_actions = env.action_space.n
    agent = Agent(
        env=env,
        alpha=0.001,
        beta=0.001,
        input_dim=8,
        gamma=0.99,
        num_actions=num_actions,
        layer_1_size=128,
        layer_2_size=128)
    score_history = []
    test_score_history = []
    episodes = 300
    max_score = -float("inf")
    max_score_episode = 0
    for i in range(episodes):
        score = agent.do_training()
        test_score = mean(agent.do_test())
        print(f"Episode: {i}, Score: {score} Test Score: {test_score}")
        if test_score > max_score:
            max_score = test_score
            max_score_episode = i
            T.save(agent.critic, "best_lander_actor.pt")
            T.save(agent.actor, "best_lander_critic.pt")
            print("saved new best model")
            if test_score == 500:
                print("saved THE best model")
                break
        score_history.append(score)
        test_score_history.append(test_score)
    do_plotting(score_history, test_score_history)
    print(f"Best Model found at episode {max_score_episode} with a Min. Test Score of {max_score}")





if __name__ == '__main__':
    main()
