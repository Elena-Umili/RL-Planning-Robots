import torch.nn as nn

class PlannerNets(nn.Module):

    def __init__(self, transition, Qnet):
        super().__init__()
        self.transition = transition
        self.Qnet = Qnet
        self.optimizer = torch.optim.Adam(self.parameters(),
                                          lr=0.001)