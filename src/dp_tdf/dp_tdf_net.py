import torch.nn as nn
import torch

from src.dp_tdf.modules import TFC_TDF_v1, TFC_TDF_v2, TFC_TDF_v3
from src.dp_tdf.bandsequence import BandSequenceModelModule

from src.layers import (get_norm)
from src.dp_tdf.abstract import AbstractModel

class DPTDFNet(AbstractModel):
    def __init__(self, num_blocks, l, g, k, bf, bias, bn_norm, bandsequence, block_type, **kwargs):
        # print(f"**kwargs: {kwargs}")
        super(DPTDFNet, self).__init__(**kwargs)
        # self.save_hyperparameters()

        self.num_blocks = num_blocks
        self.l = l
        self.g = g
        self.k = k
        self.bf = bf # bottleneck factor
        self.bias = bias

        self.n = num_blocks // 2
        scale = (2, 2)

        if block_type == "TFC_TDF":
            T_BLOCK = TFC_TDF_v1
        elif block_type == "TFC_TDF_Res1":
            T_BLOCK = TFC_TDF_v2
        elif block_type == "TFC_TDF_Res2":
            T_BLOCK = TFC_TDF_v3
        else:
            raise ValueError(f"Unknown block type {block_type}")

        self.first_conv = nn.Sequential(
            nn.Conv2d(in_channels=self.dim_c_in, out_channels=g, kernel_size=(1, 1)),
            get_norm(bn_norm, g),
            nn.ReLU(),
        )

        f = self.dim_f
        c = g
        self.encoding_blocks = nn.ModuleList()
        self.ds = nn.ModuleList()

        for i in range(self.n):
            c_in = c

            self.encoding_blocks.append(T_BLOCK(c_in, c, l, f, k, bf, bn_norm, bias=bias))
            self.ds.append(
                nn.Sequential(
                    nn.Conv2d(in_channels=c, out_channels=c + g, kernel_size=scale, stride=scale),
                    get_norm(bn_norm, c + g),
                    nn.ReLU()
                )
            )
            f = f // 2
            c += g

        self.bottleneck_block1 = T_BLOCK(c, c, l, f, k, bf, bn_norm, bias=bias)
        self.bottleneck_block2 = BandSequenceModelModule(
            **bandsequence,
            input_dim_size=c,
            hidden_dim_size=2*c
        )

        self.decoding_blocks = nn.ModuleList()
        self.us = nn.ModuleList()
        for i in range(self.n):
            # print(f"i: {i}, in channels: {c}")
            self.us.append(
                nn.Sequential(
                    nn.ConvTranspose2d(in_channels=c, out_channels=c - g, kernel_size=scale, stride=scale),
                    get_norm(bn_norm, c - g),
                    nn.ReLU()
                )
            )

            f = f * 2
            c -= g

            self.decoding_blocks.append(T_BLOCK(c, c, l, f, k, bf, bn_norm, bias=bias))

        self.final_conv = nn.Sequential(
            nn.Conv2d(in_channels=c, out_channels=self.dim_c_out, kernel_size=(1, 1)),
        )

    def forward(self, x):
        '''
        Args:
            x: (batch, c*2, 2048, 256)
        '''
        B, C, F, T = x.shape

        x = self.first_conv(x)

        x = x.transpose(-1, -2)

        skip_connections = []
        for i in range(self.n):
            x = self.encoding_blocks[i](x)
            skip_connections.append(x)
            x = self.ds[i](x)

        # print(f"bottleneck in: {x.shape}")
        x = self.bottleneck_block1(x)
        x = self.bottleneck_block2(x)

        for i in range(self.n):
            x = self.us[i](x)
            # print(f"us{i} in: {x.shape}")
            # print(f"ds{i} out: {ds_outputs[-i - 1].shape}")
            x = x * skip_connections[-i - 1]
            x = self.decoding_blocks[i](x)

        x = x.transpose(-1, -2)

        x = self.final_conv(x)

        return x