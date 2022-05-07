import numpy as np
from algo import ValueFunctionWithApproximation

import torch
from torch import nn

import warnings
warnings.filterwarnings("error")

class Net(nn.Module):
    def __init__(self, state_dim):
        super(Net, self).__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(state_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.linear_relu_stack(x)

class ValueFunctionWithNN(ValueFunctionWithApproximation):

    def __init__(self,
                 state_dims):
        """
        state_dims: the number of dimensions of state space
        """
        # TODO: implement this method
        self.net = Net(state_dims)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=0.001, betas=(0.9, 0.999))
        self.loss_func = torch.nn.MSELoss()

    def __call__(self,s):
        # TODO: implement this method
        return self.eval_state(s).detach().numpy()[0]

    def eval_state(self,s):
        # TODO: implement this method
        self.net.eval()
        s_float32 = np.float32(s)
        v = self.net(torch.tensor(s_float32))
        return v

    def update(self,alpha,G,s_tau):
        # TODO: implement this method
        # Compute the loss
        if isinstance(G, float):
            G = torch.tensor([G])
        else:
            G = torch.tensor(G)
        # print(self.eval_state(s_tau), G)
        loss = self.loss_func(self.eval_state(s_tau), G)

        self.net.train()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return None

