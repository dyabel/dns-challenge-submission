import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse

# from s4.s4 import S4Block as S4  # Can use full version instead of minimal S4D standalone below
from s4.s4d import S4D
from tqdm.auto import tqdm
import torch.nn.functional as F
from lava.lib.dl import slayer


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
        n_layers=2,
        dropout=0.2,
        prenorm=False,
        args=None
    ):
        super().__init__()

        self.prenorm = prenorm
        sigma_params = { # sigma-delta neuron parameters
            'threshold'     : 0.1,   # delta unit threshold
            'tau_grad'      : 0.1,    # delta unit surrogate gradient relaxation parameter
            'scale_grad'    : 0.8,  # delta unit surrogate gradient scale parameter
            'requires_grad' : False,  # trainable threshold
            'shared_param'  : True,   # layer wise threshold
        }
        sdnn_params = {
            **sigma_params,
            'activation'    : F.relu, # activation function
        }

        self.input_quantizer = lambda x: slayer.utils.quantize(x, step=1 / 64)
        self.input =  slayer.block.sigma_delta.Input(sdnn_params)
        self.input.pre_hook_fx = self.input_quantizer
        # Linear encoder (d_input = 1 for grayscale and 3 for RGB)
        self.encoder = slayer.block.sigma_delta.Dense(sdnn_params, d_input, d_model, weight_norm=False, delay=True, delay_shift=True)
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
        # self.decoder = nn.Linear(d_model, d_output)
        # self.decoder =  slayer.block.sigma_delta.Output(sdnn_params, d_model, d_output, weight_norm=False)
        max_delay = 64
        self.decoder = slayer.block.sigma_delta.Dense(sdnn_params, d_model, d_output, weight_norm=False, delay=True, delay_shift=True)
        self.encoder.delay.max_delay = max_delay
        self.decoder.delay.max_delay = max_delay
        # self.output = slayer.block.sigma_delta.Output(sdnn_params, d_model, d_output, weight_norm=False)

    def forward(self, x):
        """
        Input x is shape (B, L, d_input)
        """
        # x = self.input(x)
        x = self.encoder(self.input(x))  # (B, L, d_input) -> (B, L, d_model)
        # print(x)

        # x = x.transpose(-1, -2)  # (B, L, d_model) -> (B, d_model, L)
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

        # x = x.transpose(-1, -2)

        # Pooling: average pooling over the sequence length
        # x = x.mean(dim=1)

        # Decode the outputs
        x = self.decoder(x)  # (B, d_model) -> (B, d_output)
        # x = self.output(self.decoder(x))  # (B, d_model) -> (B, d_output)

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