# -*- coding: utf-8 -*-
"""Minimal example of using FrierenFuse with [B,T,N,D] tokens."""
import torch
from src.core.frieren_fuse import FrierenFuse


def main():
    B, T, N, D = 2, 8, 49, 384
    x = torch.randn(B, T, N, D)
    mask = torch.ones(B, T, N)

    fusion = FrierenFuse(
        d_model=D,
        num_dirs=8,
        scales=(1, 2, 4),
        include_second=True,
        include_posneg=True,
        poly_order=2,
        bound_scale=2.5,
        beta_init=0.5,
        ortho_every_forward=True,
    )

    h, aux = fusion(x, valid_mask=mask)
    print('h shape:', h.shape)
    # Log-friendly summaries
    U = aux['U']
    W = aux['W']
    scales = aux['scales']
    fdim = aux['feature_dim']
    print('U mean:', U.mean().item())
    print('W mean:', W.mean().item())
    print('scales mean:', scales.float().mean().item())
    print('feature_dim:', int(fdim.mean().item()))


if __name__ == '__main__':
    main()
