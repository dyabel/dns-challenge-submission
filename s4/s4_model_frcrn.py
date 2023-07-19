import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse
from frcrn.conv_stft import ConviSTFT, ConvSTFT

# from s4.s4 import S4Block as S4  # Can use full version instead of minimal S4D standalone below
from s4.s4d import S4D
from tqdm.auto import tqdm


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


        self.win_len = 640
        self.win_inc = 320
        self.fft_len = 640
        self.win_type = 'hann'
        self.feat_dim = self.fft_len // 2 + 1

        fix = True
        self.stft = ConvSTFT(
            self.win_len,
            self.win_inc,
            self.fft_len,
            self.win_type,
            feature_type='complex',
            fix=fix)
        self.istft = ConviSTFT(
            self.win_len,
            self.win_inc,
            self.fft_len,
            self.win_type,
            feature_type='complex',
            fix=fix)
        
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