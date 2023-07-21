import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse

# from s4.s4 import S4Block as S4  # Can use full version instead of minimal S4D standalone below
from s4.s4d_complex import ComplexS4D
from tqdm.auto import tqdm
from frcrn import complex_nn


# Dropout broke in PyTorch 1.11
if tuple(map(int, torch.__version__.split('.')[:2])) == (1, 11):
    print("WARNING: Dropout is bugged in PyTorch 1.11. Results may be worse.")
    dropout_fn = nn.Dropout
if tuple(map(int, torch.__version__.split('.')[:2])) >= (1, 12):
    dropout_fn = nn.Dropout1d
else:
    dropout_fn = nn.Dropout2d

class SELayer(nn.Module):

    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        # self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc_r = nn.Sequential(
            nn.Linear(channel, channel // reduction), nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel), nn.Sigmoid())
        self.fc_i = nn.Sequential(
            nn.Linear(channel, channel // reduction), nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel), nn.Sigmoid())

    def forward(self, x):
        x_r = x[..., 0]
        x_i = x[..., 1]
        y_r = self.fc_r(x_r) - self.fc_i(x_i)
        y_i = self.fc_r(x_i) + self.fc_i(x_r)
        y = torch.stack([y_r, y_i], dim=-1)
        return x * y



class ComplexS4Model(nn.Module):

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
        self.encoder = complex_nn.ComplexLinear(d_input, d_model)
        self.se_layer_enc = SELayer(d_model, 8)

        # Stack S4 layers as residual blocks
        self.s4_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        self.se_layers = nn.ModuleList()
        for _ in range(n_layers):
            self.s4_layers.append(
                ComplexS4D(d_model, dropout=dropout, transposed=True, lr=min(0.001, args.lr))
            )
            self.norms.append(complex_nn.ComplexLayerNorm(d_model))
            self.dropouts.append(dropout_fn(dropout))
            # self.se_layers.append(SELayer(d_model, 8))
            self.se_layers.append(nn.Identity())

        # Linear decoder
        self.se_layer_dec = SELayer(d_model, 8)
        self.decoder = complex_nn.ComplexLinear(d_model, d_output)

    def forward(self, x):
        """
        Input x is shape (B, L, d_input)
        """
        x = self.encoder(x)  # (B, L, d_input) -> (B, L, d_model)
        # se_enc = self.se_layer_enc(x)

        x = x.transpose(-2, -3)  # (B, L, d_model) -> (B, d_model, L)
        for layer, norm, dropout, se in zip(self.s4_layers, self.norms, self.dropouts, self.se_layers):
            # Each iteration of this loop will map (B, d_model, L) -> (B, d_model, L)

            z = x
            if self.prenorm:
                # Prenorm
                z = norm(z.transpose(-2, -3)).transpose(-2, -3)

            # Apply S4 block: we ignore the state input and output
            z, _ = layer(z)


            # Dropout on the output of the S4 block
            z = torch.stack((dropout(z[...,0]), dropout(z[...,1])), dim=-1)
            # z = se(z.transpose(-2, -3)).transpose(-2, -3)

            # Residual connection
            x = z + x

            if not self.prenorm:
                # Postnorm
                x = norm(x.transpose(-2, -3)).transpose(-2, -3)

        x = x.transpose(-2, -3)

        # Decode the outputs
        # x = torch.cat((self.se_layer_dec(x), se_enc), dim=-2)
        # x = self.se_layer_dec(x) + x
        x = self.decoder(x)  # (B, d_model) -> (B, d_output)

        return x

class ComplexS4ModelLayer(nn.Module):

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
        self.encoder = complex_nn.ComplexLinear(d_input, d_model)
        self.se_layer_enc = SELayer(d_model, 8)

        # Stack S4 layers as residual blocks
        self.s4_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        self.se_layers = nn.ModuleList()
        for _ in range(n_layers):
            self.s4_layers.append(
                ComplexS4D(d_model, dropout=dropout, transposed=True, lr=min(0.001, args.lr))
            )
            self.norms.append(complex_nn.ComplexLayerNorm(d_model))
            self.dropouts.append(dropout_fn(dropout))
            # self.se_layers.append(SELayer(d_model, 8))
            self.se_layers.append(nn.Identity())

        # Linear decoder
        self.se_layer_dec = SELayer(d_model, 8)
        self.decoder = complex_nn.ComplexLinear(d_model, d_output)

    def forward(self, x):
        """
        Input x is shape (B, C, H, L, 2)
        """
        b, c, h, T, d = x.size()
        x = torch.reshape(x, (b, c * h, T, 2))
        # x: [b,h,T,2], [6, 256, 106, 2]
        x = torch.transpose(x, 1, 2)
        
        x = self.encoder(x)  # (B, L, d_input) -> (B, L, d_model)
        # se_enc = self.se_layer_enc(x)

        x = x.transpose(-2, -3)  # (B, L, d_model) -> (B, d_model, L)
        for layer, norm, dropout, se in zip(self.s4_layers, self.norms, self.dropouts, self.se_layers):
            # Each iteration of this loop will map (B, d_model, L) -> (B, d_model, L)

            z = x
            if self.prenorm:
                # Prenorm
                z = norm(z.transpose(-2, -3)).transpose(-2, -3)

            # Apply S4 block: we ignore the state input and output
            z, _ = layer(z)


            # Dropout on the output of the S4 block
            z = torch.stack((dropout(z[...,0]), dropout(z[...,1])), dim=-1)
            # z = se(z.transpose(-2, -3)).transpose(-2, -3)

            # Residual connection
            x = z + x

            if not self.prenorm:
                # Postnorm
                x = norm(x.transpose(-2, -3)).transpose(-2, -3)

        x = x.transpose(-2, -3)

        # Decode the outputs
        # x = torch.cat((self.se_layer_dec(x), se_enc), dim=-2)
        # x = self.se_layer_dec(x) + x
        x = self.decoder(x)  # (B, d_model) -> (B, d_output)
        
        x = torch.transpose(x, 1, 2)
        x = torch.reshape(x, (b, c, h, T, d))

        return x

class ComplexS4ModelLayer_L1(nn.Module):

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
        self.d_input = d_input
        self.encoder = complex_nn.ComplexLinear(d_input, d_model)
        self.se_layer_enc = SELayer(d_model, 8)

        # Stack S4 layers as residual blocks
        self.s4_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        self.se_layers = nn.ModuleList()
        for _ in range(n_layers):
            self.s4_layers.append(
                ComplexS4D(d_model, dropout=dropout, transposed=True, lr=min(0.001, args.lr))
            )
            self.norms.append(complex_nn.ComplexLayerNorm(d_model))
            self.dropouts.append(dropout_fn(dropout))
            # self.se_layers.append(SELayer(d_model, 8))
            self.se_layers.append(nn.Identity())

        # Linear decoder
        self.se_layer_dec = SELayer(d_model, 8)
        self.decoder = complex_nn.ComplexLinear(d_model, d_output)

    def forward(self, x):
        """
        Input x is shape (B, C, H, L, 2)
        """
        b, c, h, T, d = x.size()
        # x : [b,T,h,c,2]
        # print(x.shape)
        x = torch.transpose(x, 1, 3)
        x = torch.reshape(x, (b * T, h, c, d))
        
        x = self.encoder(x)  # (B, L, d_input) -> (B, L, d_model)
        # se_enc = self.se_layer_enc(x)

        x = x.transpose(-2, -3)  # (B, L, d_model) -> (B, d_model, L)
        for layer, norm, dropout, se in zip(self.s4_layers, self.norms, self.dropouts, self.se_layers):
            # Each iteration of this loop will map (B, d_model, L) -> (B, d_model, L)

            z = x
            if self.prenorm:
                # Prenorm
                z = norm(z.transpose(-2, -3)).transpose(-2, -3)

            # Apply S4 block: we ignore the state input and output
            z, _ = layer(z)


            # Dropout on the output of the S4 block
            z = torch.stack((dropout(z[...,0]), dropout(z[...,1])), dim=-1)
            # z = se(z.transpose(-2, -3)).transpose(-2, -3)

            # Residual connection
            x = z + x

            if not self.prenorm:
                # Postnorm
                x = norm(x.transpose(-2, -3)).transpose(-2, -3)

        x = x.transpose(-2, -3)

        # Decode the outputs
        # x = torch.cat((self.se_layer_dec(x), se_enc), dim=-2)
        # x = self.se_layer_dec(x) + x
        x = self.decoder(x)  # (B, d_model) -> (B, d_output)
        x = torch.reshape(x, (b, T, h, c, d))
        x = torch.transpose(x, 1, 3)
        

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