"""Reference-based image quality on GPU: PSNR, MS-SSIM, CW-SSIM (Workstream A).

PSNR follows RoDLA's compute_psnr (per-channel MSE, averaged PSNR). MS-SSIM uses
pytorch_msssim on RGB with data_range 255, as RoDLA does through pyiqa.

CW-SSIM (Wang & Simoncelli 2005) is implemented here because RoDLA's pyiqa
dependency would pull in a large dependency tree. Coefficients come from a complex
steerable filter bank built in the Fourier domain (log-cosine radial bands x
one-sided angular windows, so responses are analytic/complex). Per band:

    cw = (2|sum_w c1 c2*| + K) / (sum_w |c1|^2 + sum_w |c2|^2 + K)

over 7x7 local windows, averaged over the map, then over 4 scales x 8 orientations.
K stabilises blank paper regions where both coefficient energies vanish.
"""
import math

import torch
import torch.nn.functional as F
from pytorch_msssim import ms_ssim

N_SCALES, N_ORIENTS, WIN, K = 4, 8, 7, 1e-2
_bank = {}


def _filters(h, w, device):
    key = (h, w, str(device))
    if key in _bank:
        return _bank[key]
    fy = torch.fft.fftfreq(h, device=device)
    fx = torch.fft.fftfreq(w, device=device)
    FY, FX = torch.meshgrid(fy, fx, indexing="ij")
    r = torch.sqrt(FX ** 2 + FY ** 2).clamp_min(1e-9)
    th = torch.atan2(FY, FX)
    bank = []
    for k in range(N_SCALES):
        rc = 0.25 / 2 ** k                                   # octave band centres (cycles/px)
        u = torch.log2(r / rc)
        radial = torch.where(u.abs() < 1, torch.cos(math.pi * u / 2), torch.zeros_like(u))
        for j in range(N_ORIENTS):
            d = torch.remainder(th - math.pi * j / N_ORIENTS + math.pi, 2 * math.pi) - math.pi
            ang = torch.where(d.abs() < math.pi / 2, torch.cos(d) ** (N_ORIENTS - 1), torch.zeros_like(d))
            bank.append(radial * ang)
    _bank[key] = torch.stack(bank)                          # (B, h, w), one-sided -> complex output
    return _bank[key]


def _luma(x):
    """x: (N,3,H,W) BGR float 0..255 -> (N,H,W) luma."""
    return 0.114 * x[:, 0] + 0.587 * x[:, 1] + 0.299 * x[:, 2]


def cw_ssim(x, y):
    """x, y: (N,3,H,W) float tensors 0..255 (BGR). Returns (N,) CW-SSIM."""
    a, b = _luma(x), _luma(y)
    H = _filters(a.shape[-2], a.shape[-1], a.device)
    ca = torch.fft.ifft2(torch.fft.fft2(a)[:, None] * H)    # (N, B, h, w) complex
    cb = torch.fft.ifft2(torch.fft.fft2(b)[:, None] * H)
    n, nb, h, w = ca.shape

    def box(t):
        return F.avg_pool2d(t.reshape(n * nb, 1, h, w), WIN, stride=1).reshape(n, nb, h - WIN + 1, w - WIN + 1) * WIN * WIN

    cross = ca * cb.conj()
    num = 2 * torch.sqrt(box(cross.real) ** 2 + box(cross.imag) ** 2) + K
    den = box(ca.abs() ** 2) + box(cb.abs() ** 2) + K
    return (num / den).mean(dim=(2, 3)).mean(dim=1)


def psnr(x, y):
    """RoDLA's definition: PSNR from per-channel MSE, averaged over channels."""
    mse = ((x - y) ** 2).mean(dim=(2, 3))                   # (N, 3)
    p = 10 * torch.log10(255.0 ** 2 / mse.clamp_min(1e-12))
    return p.mean(dim=1)


@torch.no_grad()
def all_metrics(x, y):
    """x clean, y degraded: (N,3,H,W) float 0..255. Returns dict of (N,) tensors."""
    return {"psnr": psnr(x, y),
            "ms_ssim": ms_ssim(x, y, data_range=255, size_average=False),
            "cw_ssim": cw_ssim(x, y)}
