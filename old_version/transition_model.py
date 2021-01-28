import torch
import torch.nn as nn
import torch.optim as optim
from system_conf import DISCRETE_CODES, MARGIN


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TransitionDelta(nn.Module):

    def __init__(self, code_size, action_size):
        super().__init__()

        self.code_size = code_size
        self.action_size = action_size
        input_size = code_size + action_size

        self.layer1 = nn.Linear(input_size, input_size * 2).to(device)
        self.layer2 = nn.Linear(input_size * 2, code_size).to(device)

    def forward(self, z, action, discrete_codes = True):
        cat = torch.cat((z, action), -1)
        delta_z = torch.sigmoid(self.layer1(cat))
        delta_z = torch.tanh(self.layer2(delta_z))
        y = torch.ones(self.code_size).to(device).to('cuda')
        x = torch.zeros(self.code_size).to(device).to('cuda')

        t_pred = z + delta_z
        if discrete_codes:
            t_pred = t_pred.where(t_pred < 0.5, y)
            t_pred = t_pred.where(t_pred >= 0.5, x)
        return delta_z, t_pred


class Transition(nn.Module):

    def __init__(self, encoder, decoder, transition_delta):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.transition_delta = transition_delta
        self.optimizer = optim.Adam(self.parameters(), lr=0.0001)

    def forward_one_step(self, s, action):
        x = self.encoder(s, DISCRETE_CODES)
        _, x_prime_hat = self.transition_delta(x, action, False)

        return x_prime_hat

    def forward_two_step(self, s, a, a_prime):
        x_prime_hat = self.forward_one_step(s, a)
        _, x_prime_prime_hat = self.transition_delta(x_prime_hat, a_prime, False)
        return x_prime_prime_hat


    def one_step_loss(self, s, a, s_prime):
       lmse = nn.MSELoss()
       L1 = nn.L1Loss()
       x_prime = self.encoder(s_prime, DISCRETE_CODES)
       x_prime_hat = self.forward_one_step(s,a)

       error_x = lmse(x_prime, x_prime_hat)

       s_prime_hat = self.decoder(x_prime_hat)
       error_s = lmse(s_prime, s_prime_hat)

       return  error_x, error_s

    def two_step_loss(self, s, a, s_prime, a_prime, s_prime_prime):
       lmse = nn.MSELoss()

       one_step_loss = self.one_step_loss(s, a, s_prime)

       x_prime_prime = self.encoder(s_prime_prime, DISCRETE_CODES)
       x_prime_prime_hat = self.forward_two_step(s,a, a_prime)

       error_x = lmse(x_prime_prime, x_prime_prime_hat)

       s_prime_prime_hat = self.decoder(x_prime_prime_hat)
       error_s = lmse(s_prime_prime, s_prime_prime_hat)

       return one_step_loss + error_x + error_s

    def distant_codes_loss(self, s, s_prime, margin = MARGIN):
        distF = torch.nn.L1Loss()
        dist = distF(self.encoder(s), self.encoder(s_prime))
        return (1/margin) * torch.nn.functional.relu(- dist + margin)

    ############## TRIPLET LOSS per imparare uno spazio metrico
    # farla con solo l'encoder??
    def triplet_loss_encoder(self, s, s_prime, s_prime_prime, margin):
        dist = torch.nn.L1Loss()
        dist_pos = dist(self.encoder(s), self.encoder(s_prime))
        dist_neg = dist(self.encoder(s), self.encoder(s_prime_prime))
        current_margin = dist_neg - dist_pos
        return torch.nn.functional.relu(- current_margin + margin)

    #o con la one step loss??
    def triplet_loss_transition_ome_step(self, s, a, s_prime, a_prime, margin):
        dist = torch.nn.L1Loss()
        x = self.encoder(s)
        x_prime = self.forward_one_step(s,a)
        x_prime_prime = self.forward_one_step(s_prime, a_prime)
        dist_pos = dist(x, x_prime)
        dist_neg = dist(x, x_prime_prime)
        current_margin = dist_neg - dist_pos
        return torch.nn.functional.relu(- current_margin + margin)

    #o con la two step loss??
    def triplet_loss_transition_two_step(self, s, a, a_prime, margin):
        dist = torch.nn.L1Loss()
        x = self.encoder(s)
        x_prime = self.forward_one_step(s,a)
        x_prime_prime = self.forward_two_step(s, a, a_prime)
        dist_pos = dist(x, x_prime)
        dist_neg = dist(x, x_prime_prime)
        current_margin = dist_neg - dist_pos
        return torch.nn.functional.relu(- current_margin + margin)