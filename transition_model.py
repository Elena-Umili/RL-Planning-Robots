import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import L1Loss

from system_conf import CODE_SIZE, ACTION_SIZE, DISCRETE_CODES



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TransitionDelta(nn.Module):

    def __init__(self, code_size = CODE_SIZE, action_size = ACTION_SIZE):
        super().__init__()

        self.code_size = code_size
        self.action_size = action_size
        input_size = code_size + action_size

        self.layer1 = nn.Linear(input_size, input_size * 2).to(device)
        self.layer2 = nn.Linear(input_size * 2, code_size).to(device)

    def forward(self, x, action, discrete_codes = True):
        cat = torch.cat((x, action), -1)
        delta_x = torch.sigmoid(self.layer1(cat))
        delta_x = torch.tanh(self.layer2(delta_x))
        ones = torch.ones(self.code_size).to(device).to(device)
        zeros = torch.zeros(self.code_size).to(device).to(device)

        t_pred = x + delta_x

        if discrete_codes:
          t_pred = t_pred.where(t_pred < 0.5, ones)
          t_pred = t_pred.where(t_pred >= 0.5, zeros)

        return delta_x, t_pred


class Transition(nn.Module):

    def __init__(self, encoder, decoder, transition_delta):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.transition_delta = transition_delta
        self.optimizer = optim.Adam(self.parameters(), lr=0.0001)

    ######################################################## OLD
    def forward(self, x, action, x_prime):
        z = self.encoder(x)
        z_prime = self.encoder(x_prime)

        delta_z, _ = self.transition_delta(z, action)

        z_prime_hat = z + delta_z

        x_prime_hat = self.decoder(z_prime_hat)

        error_z = z_prime_hat - z_prime

        return error_z, x_prime_hat

    def loss_function_transition(self, error_z, x_prime, recon_x_prime):
        # BCE = F.binary_cross_entropy(recon_x, x.view(-1, state_size), size_average=False) / x.shape[0]
        l = nn.MSELoss()
        RE = l(recon_x_prime, x_prime)
        E = torch.norm(error_z)

        return RE + E
    ######################################################## OLD

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
       x_prime = self.encoder(s_prime, DISCRETE_CODES)
       x_prime_hat = self.forward_one_step(s,a)

       error_x = lmse(x_prime, x_prime_hat)

       s_prime_hat = self.decoder(x_prime_hat)
       error_s = lmse(s_prime, s_prime_hat)

       return  error_x + error_s

    def two_step_loss(self, s, a, s_prime, a_prime, s_prime_prime):
       lmse = nn.MSELoss()

       one_step_loss = self.one_step_loss(s, a, s_prime)

       x_prime_prime = self.encoder(s_prime_prime, DISCRETE_CODES)
       x_prime_prime_hat = self.forward_two_step(s,a, a_prime)

       error_x = lmse(x_prime_prime, x_prime_prime_hat)

       s_prime_prime_hat = self.decoder(x_prime_prime_hat)
       error_s = lmse(s_prime_prime, s_prime_prime_hat)

       return one_step_loss + error_x + error_s

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