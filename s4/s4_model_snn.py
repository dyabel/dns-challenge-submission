import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse

# from s4.s4 import S4Block as S4  # Can use full version instead of minimal S4D standalone below
from s4.s4d import S4D_SNN as S4D
from tqdm.auto import tqdm
from spikingjelly.clock_driven import neuron, encoding, functional,layer, surrogate
import spikingjelly.activation_based.layer as layer

import time
class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
config = dotdict(dict(ctx_len = 1024,        # ===> increase T_MAX in model.py if your ctx_len > 1024
n_layer = 24,
n_embd = 256))



# Dropout broke in PyTorch 1.11
if tuple(map(int, torch.__version__.split('.')[:2])) == (1, 11):
    print("WARNING: Dropout is bugged in PyTorch 1.11. Results may be worse.")
    dropout_fn = nn.Dropout
if tuple(map(int, torch.__version__.split('.')[:2])) >= (1, 12):
    dropout_fn = nn.Dropout1d
else:
    dropout_fn = nn.Dropout2d
    
class S4Model(nn.Module):

    def __init__(
        self,
        d_input,
        d_output=10,
        d_model=256,
        n_layers=4,
        # n_layers=4,
        dropout=0.2,
        prenorm=False,
        args=None
    ):
        super().__init__()

        self.prenorm = prenorm

        # Linear encoder (d_input = 1 for grayscale and 3 for RGB)
        tau = 2.
        v_threshold = 1.
        v_reset = 0.
        self.encoder = nn.Sequential(nn.Linear(d_input, d_model),
                        # neuron.MultiStepLIFNode(tau=2., surrogate_function=surrogate.ATan(alpha=2.0), backend='cupy'),
                        neuron.ParametricLIFNode()
                        # neuron.LIFNode(tau=2., surrogate_function=surrogate.ATan(alpha=2.0))
                                    #  neuron.MultiStepLIFNode(tau=tau, v_threshold=v_threshold, v_reset=v_reset)
                                    #  neuron.LIAFNode(act=F.relu, threshold_related=True, tau=tau, v_threshold=v_threshold, v_reset=v_reset)
        )

        # Stack S4 layers as residual blocks
        self.s4_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        self.lifs = nn.ModuleList()
        lr = args['lr'] if isinstance(args, dict) else args.lr
        self.n_layers = n_layers
        for _ in range(n_layers):
            self.s4_layers.append(
                S4D(d_model, dropout=dropout, transposed=True, lr=min(0.001, lr))
                )
            self.norms.append(nn.LayerNorm(d_model))
            self.dropouts.append(dropout_fn(dropout))
            self.lifs.append(
                        neuron.ParametricLIFNode()
                        # neuron.MultiStepLIFNode(tau=2., surrogate_function=surrogate.ATan(alpha=2.0), backend='cupy'),
                        # neuron.LIFNode(tau=2., surrogate_function=surrogate.ATan(alpha=2.0))
                                            # v_threshold=1.)
                        # nn.Identity()
                # neuron.LIAFNode(act=F.relu, threshold_related=True, tau=tau, v_threshold=v_threshold, v_reset=v_reset)
                )

        # Linear decoder
        # self.decoder = nn.Sequential(nn.Linear(d_model, d_output),
        #                             #  neuron.MultiStepLIFNode(tau=tau, v_threshold=v_threshold, v_reset=v_reset)
        #                             neuron.LIFNode()
        #                             #  neuron.LIAFNode(act=F.relu, threshold_related=True, tau=tau, v_threshold=v_threshold, v_reset=v_reset)
        #                              )
        # self.poisson_encoder = encoding.PoissonEncoder()
        # config = dotdict(dict(ctx_len = 1024,        # ===> increase T_MAX in model.py if your ctx_len > 1024
        #     n_layer = 2,
        #     n_embd = d_model))
        # self.blocks = nn.Sequential(*[Block(config, i)
        #                               for i in range(config.n_layer)])



    def forward(self, x):
        """
        Input x is shape (B, L, d_input)
        """
        # x = self.poisson_encoder(x)
        # print(x)
        x = self.encoder(x) # (B, L, d_input) -> (B, L, d_model)

        x = x.transpose(-1, -2)  # (B, L, d_model) -> (B, d_model, L)
        mid_features = []
        for i, (layer, lif, norm, dropout) in enumerate(zip(self.s4_layers, self.lifs, self.norms, self.dropouts)):
            # Each iteration of this loop will map (B, d_model, L) -> (B, d_model, L)

            z = x
            if self.prenorm:
                # Prenorm
                z = norm(z.transpose(-1, -2)).transpose(-1, -2)

            # Apply S4 block: we ignore the state input and output
            z, _ = layer(z)
            # Dropout on the output of the S4 block
            # if i < self.n_layers - 1:
            z = lif(z) 
            z = dropout(z)

            # Residual connection
            x = z + x

            if not self.prenorm:
                # Postnorm
                x = norm(x.transpose(-1, -2)).transpose(-1, -2)
            mid_features.append(x)
                

        x = x.transpose(-1, -2)

        # Pooling: average pooling over the sequence length

        # Decode the outputs
        # x = self.blocks(x)
        # x = self.decoder(x) # (B, d_model) -> (B, d_output)

        return x
        # return x, torch.stack(mid_features, dim=1)

class S4Layer(nn.Module):

    def __init__(
        self,
        d_input,
        d_output=10,
        d_model=256,
        n_layers=4,
        # n_layers=4,
        dropout=0.2,
        prenorm=False,
        args=None
    ):
        super().__init__()

        self.prenorm = prenorm

        # Stack S4 layers as residual blocks
        self.s4_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        for _ in range(n_layers):
            self.s4_layers.append(
                S4D(d_model, dropout=dropout, transposed=True, lr=min(0.001, args.lr))
            )
            self.norms.append(nn.LayerNorm(d_model))
            self.dropouts.append(dropout_fn(dropout))

    def forward(self, x):
        """
        Input x is shape (B, L, d_input)
        """

        x = x.transpose(-1, -2)  # (B, L, d_model) -> (B, d_model, L)
        for layer, norm, dropout in zip(self.s4_layers, self.norms, self.dropouts):
            # Each iteration of this loop will map (B, d_model, L) -> (B, d_model, L)

            z = x
            if self.prenorm:
                # Prenorm
                z = norm(z.transpose(-1, -2)).transpose(-1, -2)

            # Apply S4 block: we ignore the state input and output
            z, _ = layer(z)

            # Dropout on the output of the S4 block
            z = dropout(z)

            # Residual connection
            x = z + x

            if not self.prenorm:
                # Postnorm
                x = norm(x.transpose(-1, -2)).transpose(-1, -2)

        x = x.transpose(-1, -2)



        return x
# Model
# print('==> Building model..')
# d_input = 257
# d_output= 257
# model = S4Model(
#     d_input=d_input,
#     d_output=d_output,
#     d_model=args.d_model,
#     n_layers=args.n_layers,
#     dropout=args.dropout,
#     prenorm=args.prenorm,
# )

# model = model.to(device)
class S4ModelLayer(nn.Module):

    def __init__(
        self,
        d_input,
        d_output=10,
        d_model=256,
        n_layers=4,
        # n_layers=4,
        dropout=0.2,
        prenorm=False,
        args=None
    ):
        super().__init__()

        self.prenorm = prenorm

        # Linear encoder (d_input = 1 for grayscale and 3 for RGB)
        self.encoder = nn.Linear(d_input, d_model)

        # Stack S4 layers as residual blocks
        self.s4_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        for _ in range(n_layers):
            self.s4_layers.append(
                S4D(d_model, dropout=dropout, transposed=True, lr=min(0.001, args.lr))
            )
            self.norms.append(nn.LayerNorm(d_model))
            self.dropouts.append(dropout_fn(dropout))

        # Linear decoder
        self.decoder = nn.Linear(d_model, d_output)

    def forward(self, x):
        """
        Input x is shape (B, L, d_input)
        """
        b, c, h, T = x.size()
        x = torch.reshape(x, (b, c * h, T))
        x = torch.transpose(x, 1, 2)
        x = self.encoder(x)  # (B, L, d_input) -> (B, L, d_model)

        x = x.transpose(-1, -2)  # (B, L, d_model) -> (B, d_model, L)
        for layer, norm, dropout in zip(self.s4_layers, self.norms, self.dropouts):
            # Each iteration of this loop will map (B, d_model, L) -> (B, d_model, L)

            z = x
            if self.prenorm:
                # Prenorm
                z = norm(z.transpose(-1, -2)).transpose(-1, -2)

            # Apply S4 block: we ignore the state input and output
            z, _ = layer(z)

            # Dropout on the output of the S4 block
            z = dropout(z)

            # Residual connection
            x = z + x

            if not self.prenorm:
                # Postnorm
                x = norm(x.transpose(-1, -2)).transpose(-1, -2)

        x = x.transpose(-1, -2)

        # Pooling: average pooling over the sequence length
        # x = x.mean(dim=1)

        # Decode the outputs
        x = self.decoder(x)  # (B, d_model) -> (B, d_output)
        x = torch.transpose(x, 1, 2)
        x = torch.reshape(x, (b, c, h, T))

        return x

class S4ModelLayer_L1(nn.Module):

    def __init__(
        self,
        d_input,
        d_output=10,
        d_model=256,
        n_layers=1,
        # n_layers=4,
        dropout=0.2,
        prenorm=False,
        args=None
    ):
        super().__init__()

        self.prenorm = prenorm

        # Linear encoder (d_input = 1 for grayscale and 3 for RGB)
        self.encoder = nn.Linear(d_input, d_model)

        # Stack S4 layers as residual blocks
        self.s4_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        for _ in range(n_layers):
            self.s4_layers.append(
                S4D(d_model, dropout=dropout, transposed=True, lr=min(0.001, args.lr))
            )
            self.norms.append(nn.LayerNorm(d_model))
            self.dropouts.append(dropout_fn(dropout))

        # Linear decoder
        self.decoder = nn.Linear(d_model, d_output)

    def forward(self, x):
        """
        Input x is shape (B, L, d_input)
        """
        b, c, h, T = x.size()
        # x : [b,T,h,c,2]
        # print(x.shape)
        x = torch.transpose(x, 1, 3)
        x = torch.reshape(x, (b * T, h, c))
        x = self.encoder(x)  # (B, L, d_input) -> (B, L, d_model)

        x = x.transpose(-1, -2)  # (B, L, d_model) -> (B, d_model, L)
        for layer, norm, dropout in zip(self.s4_layers, self.norms, self.dropouts):
            # Each iteration of this loop will map (B, d_model, L) -> (B, d_model, L)

            z = x
            if self.prenorm:
                # Prenorm
                z = norm(z.transpose(-1, -2)).transpose(-1, -2)

            # Apply S4 block: we ignore the state input and output
            z, _ = layer(z)

            # Dropout on the output of the S4 block
            z = dropout(z)

            # Residual connection
            x = z + x

            if not self.prenorm:
                # Postnorm
                x = norm(x.transpose(-1, -2)).transpose(-1, -2)

        x = x.transpose(-1, -2)

        # Decode the outputs
        x = self.decoder(x)  # (B, d_model) -> (B, d_output)
        x = torch.reshape(x, (b, T, h, c))
        x = torch.transpose(x, 1, 3)

        return x