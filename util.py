import matplotlib.pyplot as plt


def do_plotting(agent_scores, performance_scores):
    plt.plot(agent_scores)
    plt.plot(performance_scores)
    # forward_average = [sum(y1_values[:i]) / i for i in range(1, len(y1_values) + 1)]
    # plt.plot(forward_average)
    plt.xlabel("Episodes")
    plt.ylabel("Score")
    plt.legend(["Learning Scores", "Minimum Test Scores"])
    plt.show()
