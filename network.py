import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


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
            self.optimizer = optim.Adam(self.parameters(), lr=self.lr)  # either radam or yogi
        else:
            self.optimizer = optim.Adam(self.parameters(), lr=self.lr)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu:0')
        self.to(self.device)

    def forward(self, observation):
        state = T.tensor(observation).to(self.device).float()
        out_0 = F.relu(self.input_layer(state))
        out_1 = F.relu(self.layer1(out_0))
        out_final = self.output_layer(out_1)
        # out final is activated when action is selected

        return out_final
