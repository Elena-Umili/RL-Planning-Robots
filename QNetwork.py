import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import warnings

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=UserWarning)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Net(nn.Module):
    def __init__(self, code_size, n_hidden_nodes, n_outputs, bias, encoder):
        super().__init__()
        self.encoder = encoder
        # self.encoder.load_state_dict(torch.load('lunar_models/code100_enc.pt'))

        self.layers0 = nn.Linear(
            code_size,
            n_hidden_nodes,
            bias=bias).to(device)
        self.layers1 = nn.ReLU().to(device)

        self.layers2 = nn.Linear(
            n_hidden_nodes,
            # int(n_hidden_nodes/2), #<-----comment for maze
            n_outputs,
            bias=bias).to(device)

        self.layers3 = nn.ReLU()
        '''
        self.layers4 = nn.Linear(
                    int(n_hidden_nodes/2),
                    int(n_hidden_nodes/4),
                    bias=bias).to('cuda')
        self.layers5 = nn.ReLU().to('cuda')

        self.layers6 = nn.Linear(
                    int(n_hidden_nodes/4),
                    n_outputs,
                    bias=bias).to('cuda')
        self.layers7 = nn.ReLU()
        '''

    def forward(self, data):
        out = self.encoder(data.to(device))  # <-------------------------comment for Q-learning

        out = self.layers0(out.to(device))
        out = self.layers1(out).to(device)
        out = F.tanh(self.layers2(out).to(device))
        out = self.layers3(out).to(device)
        # out = self.layers4(out).to('cuda')
        # out = self.layers5(out).to('cuda')
        # out = self.layers6(out).to('cuda')
        # out = self.layers7(out)
        return out

    def enc_forw(self, enc_data):
        out = self.layers0(enc_data.to(device))
        out = self.layers1(out).to(device)
        out = self.layers2(out).to(device)
        out = self.layers3(out).to(device)
        # out = self.layers4(out).to('cuda')
        # out = self.layers5(out).to('cuda')
        # out = self.layers6(out).to('cuda')
        # out = self.layers7(out)
        return out


class QNetwork(nn.Module):

    def __init__(self, env, encoder, learning_rate=1e-4, n_hidden_nodes=128, bias=True, device='cpu', norm_out=False, epsilon=0.05):
        super(QNetwork, self).__init__()
        self.norm_out = norm_out
        self.device = device
        self.actions = np.arange(env.action_space.n)
        self.epsilon = epsilon
        n_outputs = env.action_space.n

        self.network = Net(encoder.code_size, n_hidden_nodes, n_outputs, bias, encoder)
        # Set device for GPU's
        if self.device == 'cuda':
            self.network.cuda()

        self.optimizer = torch.optim.Adam(self.parameters(),
                                          lr=learning_rate)

    def get_action(self, state):

        if np.random.random() < self.epsilon:
            action = np.random.choice(self.actions)
        else:
            action = self.greedy_action(state)
        return action

    def greedy_action(self, state):
        qvals, _ = self.get_qvals(state)
        return torch.max(qvals, dim=-1)[1].item()

    def get_qvals(self, state):
        if type(state) is tuple:
            # print("TUPLE!!!!!!!!!!!!!!!!!!!!")
            state = np.array([np.ravel(s) for s in state])
            state_t = torch.FloatTensor(state).to(device)
            # print(state_t)
        else:
            # print("NO TUPLE!")
            state_t = torch.from_numpy(np.asarray([state])).type(torch.FloatTensor).to(device)
            # print(state_t)
        out = self.network(state_t)
        # print("OUT = ", out)
        return out

    def get_enc_value(self, enc_state):
        return self.network.enc_forw(enc_state)