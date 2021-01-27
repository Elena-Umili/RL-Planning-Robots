import torch
import torch.nn as nn
import torch.nn.functional as F
from system_conf import CODE_SIZE, INPUT_SIZE, DISCRETE_CODES

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Encoder(nn.Module):
    def __init__(self, code_size = CODE_SIZE):
        super().__init__()
        self.code_size = code_size
        self.enc_linear_1 = nn.Linear(INPUT_SIZE, INPUT_SIZE * 4).to('cuda')
        self.enc_linear_2 = nn.Linear(INPUT_SIZE * 4, INPUT_SIZE * 16).to('cuda')
        self.enc_linear_3 = nn.Linear(INPUT_SIZE * 16, INPUT_SIZE * 16).to('cuda')
        self.enc_linear_4 = nn.Linear(INPUT_SIZE * 16, code_size).to('cuda')


    def forward(self,data, discrete_codes = DISCRETE_CODES):
        code = (self.enc_linear_1(data.to('cuda')))
        code = F.selu(self.enc_linear_2(code)).to('cuda')
        code = F.selu(self.enc_linear_3(code)).to('cuda')
        code = F.sigmoid(self.enc_linear_4(code)).to('cuda')
        y = torch.ones(self.code_size).to(device).to('cuda')
        x = torch.zeros(self.code_size).to(device).to('cuda')
        if (discrete_codes):
            code = code.where(code < 0.5, y)
            code = code.where(code >= 0.5, x)

        return code

class Decoder(nn.Module):
    def __init__(self, code_size):
        super().__init__()
        self.dec_linear_1 = nn.Linear(code_size, INPUT_SIZE * 16).to('cuda')
        self.dec_linear_2 = nn.Linear(INPUT_SIZE * 16, INPUT_SIZE * 8).to('cuda')
        self.dec_linear_3 = nn.Linear(INPUT_SIZE * 8, INPUT_SIZE * 4).to('cuda')
        self.dec_linear_4 = nn.Linear(INPUT_SIZE * 4, INPUT_SIZE).to('cuda')


    def forward(self, code):
        out = (self.dec_linear_1(code)).to('cuda')
        out = F.selu(self.dec_linear_2(out)).to('cuda')
        out = F.selu(self.dec_linear_3(out)).to('cuda')
        out = (self.dec_linear_4(out)).to('cuda')
        return out

class AutoEncoder(nn.Module):

    def __init__(self, encoder, decoder, code_size):
        super().__init__()
        self.code_size = code_size
        self.encoder = encoder
        self.decoder = decoder
        self.optimizer = torch.optim.Adam(self.parameters(),
                                          lr=0.001)

    def forward(self, data, epoch, n_epochs):
        code = self.encoder(data, epoch, n_epochs).to('cuda')
        out = self.decoder(code).to('cuda')
        return out, code






