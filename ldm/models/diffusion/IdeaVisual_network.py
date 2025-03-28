import torch
import torch.nn as nn


class Image2DResBlockWithTV(nn.Module):
    def __init__(self, dim, tdim, vdim):
        super().__init__()
        norm = lambda c: nn.GroupNorm(8, c)
        self.time_embed = nn.Conv2d(tdim, dim, 1, 1)
        self.view_embed = nn.Conv2d(vdim, dim, 1, 1)
        self.conv = nn.Sequential(
            norm(dim),
            nn.SiLU(True),
            nn.Conv2d(dim, dim, 3, 1, 1),
            norm(dim),
            nn.SiLU(True),
            nn.Conv2d(dim, dim, 3, 1, 1),
        )

    def forward(self, x, t, v):
        return x+self.conv(x+self.time_embed(t)+self.view_embed(v))


class NoisyTargetViewEncoder(nn.Module):
    def __init__(self, time_embed_dim, viewpoint_dim, run_dim=16, output_dim=8):
        super().__init__()

        self.init_conv = nn.Conv2d(4, run_dim, 3, 1, 1)
        self.out_conv0 = Image2DResBlockWithTV(run_dim, time_embed_dim, viewpoint_dim)
        self.out_conv1 = Image2DResBlockWithTV(run_dim, time_embed_dim, viewpoint_dim)
        self.out_conv2 = Image2DResBlockWithTV(run_dim, time_embed_dim, viewpoint_dim)
        self.final_out = nn.Sequential(
            nn.GroupNorm(8, run_dim),
            nn.SiLU(True),
            nn.Conv2d(run_dim, output_dim, 3, 1, 1)
        )

    def forward(self, x, t, v):
        B, DT = t.shape
        t = t.view(B, DT, 1, 1)
        B, DV = v.shape
        v = v.view(B, DV, 1, 1)

        x = self.init_conv(x)
        x = self.out_conv0(x, t, v)
        x = self.out_conv1(x, t, v)
        x = self.out_conv2(x, t, v)
        x = self.final_out(x)
        return x


class SpatialUpTimeBlock(nn.Module):
    def __init__(self, x_in_dim, t_in_dim, out_dim):
        super().__init__()
        norm_act = lambda c: nn.GroupNorm(8, c)
        self.t_conv = nn.Conv3d(t_in_dim, x_in_dim, 1, 1)
        self.norm = norm_act(x_in_dim)
        self.silu = nn.SiLU(True)
        self.conv = nn.ConvTranspose3d(x_in_dim, out_dim, kernel_size=3, padding=1, output_padding=1, stride=2)

    def forward(self, x, t):
        x = x + self.t_conv(t)
        return self.conv(self.silu(self.norm(x)))


class SpatialTimeBlock(nn.Module):
    def __init__(self, x_in_dim, t_in_dim, out_dim, stride):
        super().__init__()
        norm_act = lambda c: nn.GroupNorm(8, c)
        self.t_conv = nn.Conv3d(t_in_dim, x_in_dim, 1, 1)
        self.bn = norm_act(x_in_dim)
        self.silu = nn.SiLU(True)
        self.conv = nn.Conv3d(x_in_dim, out_dim, 3, stride=stride, padding=1)

    def forward(self, x, t):
        x = x + self.t_conv(t)
        return self.conv(self.silu(self.bn(x)))


class SpatialTime3DNet(nn.Module):
        def __init__(self, time_dim=256, input_dim=128, dims=(32, 64, 128, 256)):
            super().__init__()
            d0, d1, d2, d3 = dims
            dt = time_dim

            self.init_conv = nn.Conv3d(input_dim, d0, 3, 1, 1)  # 32
            self.conv0 = SpatialTimeBlock(d0, dt, d0, stride=1)

            self.conv1 = SpatialTimeBlock(d0, dt, d1, stride=2)
            self.conv2_0 = SpatialTimeBlock(d1, dt, d1, stride=1)
            self.conv2_1 = SpatialTimeBlock(d1, dt, d1, stride=1)

            self.conv3 = SpatialTimeBlock(d1, dt, d2, stride=2)
            self.conv4_0 = SpatialTimeBlock(d2, dt, d2, stride=1)
            self.conv4_1 = SpatialTimeBlock(d2, dt, d2, stride=1)

            self.conv5 = SpatialTimeBlock(d2, dt, d3, stride=2)
            self.conv6_0 = SpatialTimeBlock(d3, dt, d3, stride=1)
            self.conv6_1 = SpatialTimeBlock(d3, dt, d3, stride=1)

            self.conv7 = SpatialUpTimeBlock(d3, dt, d2)
            self.conv8 = SpatialUpTimeBlock(d2, dt, d1)
            self.conv9 = SpatialUpTimeBlock(d1, dt, d0)

        def forward(self, x, t):
            B, C = t.shape
            t = t.view(B, C, 1, 1, 1)

            x = self.init_conv(x)
            conv0 = self.conv0(x, t)

            x = self.conv1(conv0, t)
            x = self.conv2_0(x, t)
            conv2 = self.conv2_1(x, t)

            x = self.conv3(conv2, t)
            x = self.conv4_0(x, t)
            conv4 = self.conv4_1(x, t)

            x = self.conv5(conv4, t)
            x = self.conv6_0(x, t)
            x = self.conv6_1(x, t)

            x = conv4 + self.conv7(x, t)
            x = conv2 + self.conv8(x, t)
            x = conv0 + self.conv9(x, t)
            return x
