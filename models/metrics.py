import torch
from torch.nn import functional as F


def psnr(x, y): return -10 * torch.log10((x - y).square().mean().clamp_min(1e-12))


def ssim(x, y):
    coordinate = torch.arange(11, device=x.device, dtype=x.dtype) - 5
    kernel = torch.exp(-coordinate.square() / 4.5); kernel /= kernel.sum()
    kernel = (kernel[:, None] * kernel[None, :])[None, None].expand(3, 1, 11, 11)
    smooth = lambda v: F.conv2d(v, kernel, groups=3)
    mx, my = smooth(x), smooth(y)
    vx, vy = smooth(x.square()) - mx.square(), smooth(y.square()) - my.square()
    covariance = smooth(x * y) - mx * my
    return (((2 * mx * my + .01 ** 2) * (2 * covariance + .03 ** 2)) / (
        (mx.square() + my.square() + .01 ** 2) * (vx + vy + .03 ** 2))).mean()
