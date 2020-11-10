from agent import Agent
import gym


# import matplotlib as plt


def main():
    env = gym.make('LunarLander-v2')
    num_actions = env.action_space.n
    agent = Agent(env=env,
                  alpha=0.001,
                  beta=0.001,
                  input_dim=8,
                  gamma=0.99,
                  num_actions=num_actions,
                  layer_1_size=128,
                  layer_2_size=128)
    score_history = []
    episodes = 2500
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
        print(f"Episode: {i}, Score: {score}")
        score_history.append(score)


if __name__ == '__main__':
    main()
