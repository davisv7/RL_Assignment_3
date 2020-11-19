import torch as T
import gym
from agent import Agent
from util import do_plotting


def main():
    env = gym.make("CartPole-v1")
    agent = Agent(
        env,
        alpha=0.001,
        beta=0.001,
        input_dim=4,
        gamma=0.90,
        num_actions=2,
        layer_1_size=256,
        layer_2_size=256)
    score_history = []
    test_score_history = []
    episodes = 300
    max_score = 0
    max_score_episode = 0
    for i in range(episodes):
        score = agent.do_training()
        test_score = min(agent.do_test())
        print(f"Episode: {i}, Score: {score} Test Score: {test_score}")
        score_history.append(score)
        test_score_history.append(test_score)
        if test_score > max_score:
            max_score = test_score
            max_score_episode = i
            T.save(agent.critic, "best_cartpole_actor.pt")
            T.save(agent.actor, "best_cartpole_critic.pt")
            print("saved new best model")
            if test_score == 500:
                print("saved THE best model")
                break
    do_plotting(score_history, test_score_history)
    print(f"Best Model found at episode {max_score_episode} with a Min. Test Score of {max_score}")


if __name__ == '__main__':
    main()
