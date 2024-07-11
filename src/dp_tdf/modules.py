import torch
import torch.nn as nn

from src.layers import (get_norm)

class TFC(nn.Module):
    def __init__(self, c_in, c_out, l, k, bn_norm):
        super(TFC, self).__init__()

        self.H = nn.ModuleList()
        for i in range(l):
            if i == 0:
                c_in = c_in
            else:
                c_in = c_out
            self.H.append(
                nn.Sequential(
                    nn.Conv2d(in_channels=c_in, out_channels=c_out, kernel_size=k, stride=1, padding=k // 2),
                    get_norm(bn_norm, c_out),
                    nn.ReLU(),
                )
            )

    def forward(self, x):
        for h in self.H:
            x = h(x)
        return x


class DenseTFC(nn.Module):
    def __init__(self, c_in, c_out, l, k, bn_norm):
        super(DenseTFC, self).__init__()

        self.conv = nn.ModuleList()
        for i in range(l):
            self.conv.append(
                nn.Sequential(
                    nn.Conv2d(in_channels=c_in, out_channels=c_out, kernel_size=k, stride=1, padding=k // 2),
                    get_norm(bn_norm, c_out),
                    nn.ReLU(),
                )
            )

    def forward(self, x):
        for layer in self.conv[:-1]:
            x = torch.cat([layer(x), x], 1)
        return self.conv[-1](x)


class TFC_TDF_v1(nn.Module):
    def __init__(self, c_in, c_out, l, f, k, bn, bn_norm, dense=False, bias=True):

        super(TFC_TDF_v1, self).__init__()

        self.use_tdf = bn is not None

        self.tfc = DenseTFC(c_in, c_out, l, k, bn_norm) if dense else TFC(c_in, c_out, l, k, bn_norm)

        if self.use_tdf:
            if bn == 0:
                # print(f"TDF={f},{f}")
                self.tdf = nn.Sequential(
                    nn.Linear(f, f, bias=bias),
                    get_norm(bn_norm, c_out),
                    nn.ReLU()
                )
            else:
                # print(f"TDF={f},{f // bn},{f}")
                self.tdf = nn.Sequential(
                    nn.Linear(f, f // bn, bias=bias),
                    get_norm(bn_norm, c_out),
                    nn.ReLU(),
                    nn.Linear(f // bn, f, bias=bias),
                    get_norm(bn_norm, c_out),
                    nn.ReLU()
                )

    def forward(self, x):
        x = self.tfc(x)
        return x + self.tdf(x) if self.use_tdf else x


class TFC_TDF_v2(nn.Module):
    def __init__(self, c_in, c_out, l, f, k, bn, bn_norm, dense=False, bias=True):

        super(TFC_TDF_v2, self).__init__()

        self.use_tdf = bn is not None

        self.tfc = DenseTFC(c_in, c_out, l, k, bn_norm) if dense else TFC(c_in, c_out, l, k, bn_norm)

        self.res = TFC(c_in, c_out, 1, k, bn_norm)

        if self.use_tdf:
            if bn == 0:
                # print(f"TDF={f},{f}")
                self.tdf = nn.Sequential(
                    nn.Linear(f, f, bias=bias),
                    get_norm(bn_norm, c_out),
                    nn.ReLU()
                )
            else:
                # print(f"TDF={f},{f // bn},{f}")
                self.tdf = nn.Sequential(
                    nn.Linear(f, f // bn, bias=bias),
                    get_norm(bn_norm, c_out),
                    nn.ReLU(),
                    nn.Linear(f // bn, f, bias=bias),
                    get_norm(bn_norm, c_out),
                    nn.ReLU()
                )

    def forward(self, x):
        res = self.res(x)
        x = self.tfc(x)
        x = x + res
        return x + self.tdf(x) if self.use_tdf else x


class TFC_TDF_v3(nn.Module):
    def __init__(
            self,
            c_in=32,
            c_out=32,
            l=3,
            f=2048,
            k=3,
            bf=8,
            bn_norm='BN',
            dense=False,
            bias=True
    ):
        super(TFC_TDF_v3, self).__init__()

        self.use_tdf = bf is not None

        self.tfc1 = TFC(c_in, c_out, l, k, bn_norm)
        self.tfc2 = TFC(c_in, c_out, l, k, bn_norm)

        self.res = TFC(c_in, c_out, 1, k, bn_norm)

        if self.use_tdf:
            if bf == 0:
                # print(f"TDF={f},{f}")
                self.tdf = nn.Sequential(
                    nn.Linear(f, f, bias=bias),
                    nn.BatchNorm2d(c_out),
                    nn.ReLU()
                )
            else:
                # print(f"TDF={f},{f // bn},{f}")
                self.tdf = nn.Sequential(
                    nn.Linear(f, f // bf, bias=bias),
                    nn.BatchNorm2d(c_out),
                    nn.ReLU(),
                    nn.Linear(f // bf, f, bias=bias),
                    nn.BatchNorm2d(c_out),
                    nn.ReLU()
                )

    def forward(self, x):
        res = self.res(x)
        x = self.tfc1(x)
        if self.use_tdf:
            x = x + self.tdf(x)
        x = self.tfc2(x)
        x = x + res
        return x


class TFC_TDF(nn.Module):
    def __init__(self, in_c, c, l, f, bn):
        super().__init__()

        self.blocks = nn.ModuleList()
        for i in range(l):
            block = nn.Module()

            block.tfc1 = nn.Sequential(
                nn.BatchNorm2d(in_c),
                nn.ReLU(),
                nn.Conv2d(in_c, c, 3, 1, 1, bias=False),
            )

            block.tdf = nn.Sequential(
                nn.BatchNorm2d(c),
                nn.ReLU(),
                nn.Linear(f, f // bn, bias=False),
                nn.BatchNorm2d(c),
                nn.ReLU(),
                nn.Linear(f // bn, f, bias=False),
            )
            block.tfc2 = nn.Sequential(
                nn.BatchNorm2d(c),
                nn.ReLU(),
                nn.Conv2d(c, c, 3, 1, 1, bias=False),
            )
            block.shortcut = nn.Conv2d(in_c, c, 1, 1, 0, bias=False)

            self.blocks.append(block)
            in_c = c

    def forward(self, x):
        for block in self.blocks:
            s = block.shortcut(x)
            x = block.tfc1(x)
            x = x + block.tdf(x)
            x = block.tfc2(x)
            x = x + s
        return x


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    batch_size, k_subbands, t_timesteps, input_dim = 1, 96, 64, 512
    in_features = torch.rand(batch_size, k_subbands, t_timesteps, input_dim).to(device)
    net = TFC_TDF_v3()

    print(f"Total number of parameters: {sum([p.numel() for p in net.parameters()])}")
    print(f"tfc1: {sum([p.numel() for p in net.tfc1.parameters() if p.requires_grad])}")
    print(f"tfc2: {sum([p.numel() for p in net.tfc2.parameters() if p.requires_grad])}")
    print(f"res : {sum([p.numel() for p in net.res.parameters() if p.requires_grad])}")
    print(f"tdf : {sum([p.numel() for p in net.tdf.parameters() if p.requires_grad])}")
