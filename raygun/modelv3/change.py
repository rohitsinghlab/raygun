#Copyright © 2025 Kavi Haria Shah CC-BY-NC-SA 4.0

import torch.nn as nn
import torch
import torch.nn.functional as F
from einops import reduce, repeat, rearrange
import numpy as np
import os
import math
from esm.model.esm2 import TransformerLayer
from einops.layers.torch import Rearrange


import torch
import torch.nn as nn
import torch.nn.functional as F


class Change_length(nn.Module):
    """
    Resamples sequences (second dim) with 'area' interpolation and optionally adds noise.
    This Function can go from any sequence length or set of lengths to
    any other sequence length or set of lenghths, optionally adding noise.

    Is a reworking of ReductionAndExpansionAreaResamp(), simplified (no noise options - just returns stdev
    matrix) to slot into Raygun

    This class replaces reduction and repetition:
    1. removes the need for variable batching across length to keep whole numbers - keeping truly linear and
       removing edge effects
    2. Removes any hard limits in input seq length to Raygun irrespective of desired fixed lengths


    Uses - for going back and forth between variable and fixed/variable length reps.

    Features:
      - Uses F.interpolate(mode='area') for efficiency and correctness.

      - returns the original average output and the stdev matrix so that it can be dropped in place of
        reduction/repetition function

    #note that where input length is below output length, stdev will be minimal - that was the point of the
    original add absolute noise - this can be added here...
    # note that Bessel's correction is NOT applied (uses N, not than a N-1 in stdev- this makes sense for this
    application )

    """

    def __init__(self, eps = 1e-12, finallength = None):
        super().__init__()
        self.eps = eps  # small epsilon for numerical stability
        self.finallength = finallength



    # Helper function for stdev calculation: compute fractional-overlap weight matrix
    def _build_weight_matrix(self, L: int, T: int, device, dtype):
        """
        W[j,i] = overlap length between input cell [i, i+1)
                 and output bin [j*step, (j+1)*step)
        """
        if L == 0 or T == 0:
            return torch.zeros((T, L), device=device, dtype=dtype)
        step = L / T
        j = torch.arange(T, device=device, dtype=dtype)
        i = torch.arange(L, device=device, dtype=dtype)
        start = j * step
        end = start + step
        start_exp = start[:, None]
        end_exp = end[:, None]
        i_left = i[None, :]
        i_right = (i + 1.0)[None, :]
        overlap = (torch.minimum(i_right, end_exp) -
                   torch.maximum(i_left, start_exp)).clamp(min=0.0)
        return overlap

    def forward(self, x, finallength=None, mask=None, getstd = False):
        """
        Args:
            x: [B, L_in, D]
            finallength: int or [B]
            mask: optional [B, L_in] bool (False = pad) #note here this is the raygun convention
        Returns:
            padded_out: [B, L_out_max, D]
            out_mask: [B, L_out_max] (False = pad)
        """
        B, L_in, D = x.shape
        device, dtype = x.device, x.dtype

        if finallength is None and hasattr(self, "finallength") and self.finallength is not None:
            finallength = self.finallength
        elif finallength is None:
            raise ValueError("finallength must be specified either as argument or class attribute")


        if mask is None:
            mask = torch.ones((B, L_in), dtype=torch.bool, device=device)
        mask = mask.bool()

        if isinstance(finallength, int):
            finallengths = torch.full((B,), finallength, dtype=torch.long, device=device)
        else:
            finallengths = torch.as_tensor(finallength, dtype=torch.long, device=device)

        max_T = int(finallengths.max().item())
        padded_out = torch.zeros((B, max_T, D), dtype=dtype, device=device)
        out_mask = torch.zeros((B, max_T), dtype=torch.bool, device=device)

        if getstd:
            std_out= torch.zeros((B, max_T, D), dtype=dtype, device=device)

        for b, T in enumerate(finallengths.tolist()):
            seq = x[b, mask[b]]  # [L, D]
            L = seq.shape[0]
            if L == 0 or T == 0:
                continue

            # --- Base interpolation (fast and exact) ---
            seq_interp = seq.T.unsqueeze(0).unsqueeze(0)  # [1,1,D,L]
            out = F.interpolate(seq_interp, size=(D, T), mode="area")
            out = out.squeeze(0).squeeze(0).T  # [T, D]

            padded_out[b, :T] = out
            out_mask[b, :T] = True

            #Local noise
            if getstd:
                # Build fractional overlap weight matrix
                W = self._build_weight_matrix(L, T, device, dtype)  # [T, L]
                Wsum = W.sum(dim=1, keepdim=True).clamp(min=self.eps)  # [T,1]

                # Weighted variance computation
                seq_sq = seq * seq
                mean = (W @ seq) / Wsum                     # [T,D]
                mean_sq = (W @ seq_sq) / Wsum               # [T,D]
                var = (mean_sq - mean ** 2).clamp(min=self.eps)
                std = var.sqrt()                            # [T,D]
                std_out[b, :T] = std

        if getstd:

            return padded_out, out_mask, std_out



        return padded_out, out_mask


#generalised class - can be modified to output original plus stdev/var matrices and noise separately if desired
class ReductionAndExpansionAreaResamp(nn.Module):
    """
    Resamples sequences (second dim) with 'area' interpolation and optionally adds noise.
    This Function can go from any sequence length or set of lengths to
    any other sequence length or set of lenghths, optionally adding noise.

    Uses - for going back and forth between variable and fixed/variable length reps.

    Features:
      - Uses F.interpolate(mode='area') for efficiency and correctness.
      - When local noise is requested, computes exact weighted per-bin std
        using fractional overlaps (area-based weighting).
      - Global and absolute noise supported.
      - Optionally detaches std/var in noise computation for non-gradient noise.
    """

    def __init__(self, add_global_noise=False, add_local_noise=False,
                 input_noise_scale=0.1, abs_noise=0.0, eps=1e-12,
                 detach_std_for_noise=False):
        super().__init__()
        self.add_global_noise = add_global_noise
        self.add_local_noise = add_local_noise
        self.input_noise_scale = input_noise_scale
        self.abs_noise = abs_noise
        self.eps = eps
        self.detach_std_for_noise = detach_std_for_noise

    def _build_weight_matrix(self, L: int, T: int, device, dtype):
        """
        W[j,i] = overlap length between input cell [i, i+1)
                 and output bin [j*step, (j+1)*step)
        """
        if L == 0 or T == 0:
            return torch.zeros((T, L), device=device, dtype=dtype)
        step = L / T
        j = torch.arange(T, device=device, dtype=dtype)
        i = torch.arange(L, device=device, dtype=dtype)
        start = j * step
        end = start + step
        start_exp = start[:, None]
        end_exp = end[:, None]
        i_left = i[None, :]
        i_right = (i + 1.0)[None, :]
        overlap = (torch.minimum(i_right, end_exp) -
                   torch.maximum(i_left, start_exp)).clamp(min=0.0)
        return overlap

    def forward(self, x, finallength, padding_mask=None):
        """
        Args:
            x: [B, L_in, D]
            finallength: int or [B]
            padding_mask: optional [B, L_in] bool (True = pad)
        Returns:
            padded_out: [B, L_out_max, D]
            out_mask: [B, L_out_max] (False = valid)
        """
        B, L_in, D = x.shape
        device, dtype = x.device, x.dtype

        if padding_mask is None:
            padding_mask = torch.zeros((B, L_in), dtype=torch.bool, device=device)

        if isinstance(finallength, int):
            finallengths = torch.full((B,), finallength, dtype=torch.long, device=device)
        else:
            finallengths = torch.as_tensor(finallength, dtype=torch.long, device=device)

        max_T = int(finallengths.max().item())
        padded_out = torch.zeros((B, max_T, D), dtype=dtype, device=device)
        out_mask = torch.ones((B, max_T), dtype=torch.bool, device=device)

        for b, T in enumerate(finallengths.tolist()):
            seq = x[b, ~padding_mask[b]]  # [L, D]
            L = seq.shape[0]
            if L == 0 or T == 0:
                continue

            #Base interpolation
            seq_interp = seq.T.unsqueeze(0).unsqueeze(0)  # [1,1,D,L]
            out = F.interpolate(seq_interp, size=(D, T), mode="area")
            out = out.squeeze(0).squeeze(0).T  # [T, D]

            #Global noise
            if self.add_global_noise:
                std_global = seq.std(dim=0, keepdim=True)
                if self.detach_std_for_noise:
                    std_global = std_global.detach()
                eps = torch.randn_like(out).detach()
                out = out + eps * std_global * self.input_noise_scale

            #Local noise
            elif self.add_local_noise:
                # Build fractional overlap weight matrix
                W = self._build_weight_matrix(L, T, device, dtype)  # [T, L]
                Wsum = W.sum(dim=1, keepdim=True).clamp(min=self.eps)  # [T,1]

                # Weighted variance computation
                seq_sq = seq * seq
                mean = (W @ seq) / Wsum                     # [T,D]
                mean_sq = (W @ seq_sq) / Wsum               # [T,D]
                var = (mean_sq - mean ** 2).clamp(min=self.eps)
                std = var.sqrt()                            # [T,D]

                if self.detach_std_for_noise:
                    std = std.detach()

                eps = torch.randn_like(std).detach()
                out = out + eps * std * self.input_noise_scale

            #Absolute noise
            if self.abs_noise > 0.0:
                out = out + torch.randn_like(out).detach() * self.abs_noise

            padded_out[b, :T] = out
            out_mask[b, :T] = False

        return padded_out, out_mask



#Original simple class
class simple_ReductionAndExpansionAreaResamp(nn.Module):
    """
    Resamples sequences with variable input and output lengths using 'area' interpolation.
    Supports masking of padded input tokens and variable target lengths per batch.

    Args:
        None (can later add stdev/noise parameters)

    Inputs:
        x: Tensor of shape [B, L_max, D]
        padding_mask: Bool tensor [B, L_max], where True = padded (invalid)
        finallength: int | list[int] | tensor[int]
            Target length(s) per batch element.

    Outputs:
        padded_out: Tensor [B, L_out_max, D] (zero-padded)
        out_mask:   Bool tensor [B, L_out_max] (True = padded)
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        x: torch.Tensor,
        finallength,
        padding_mask: torch.Tensor = None,
    ):
        B, L_max, D = x.shape

        if padding_mask is None:
            padding_mask = torch.zeros(B, L_max, dtype=torch.bool, device=x.device)

        assert padding_mask.shape == (B, L_max), "padding_mask must be [B, L_max]"

        # Normalize finallength → Tensor[B]
        if isinstance(finallength, int):
            finallengths = torch.full((B,), finallength, dtype=torch.long, device=x.device)
        else:
            finallengths = torch.as_tensor(finallength, dtype=torch.long, device=x.device)
            assert finallengths.shape[0] == B, "finallength must have one value per batch element"

        # --- Preallocate outputs ---
        max_len_out = finallengths.max().item()
        padded_out = torch.zeros(B, max_len_out, D, dtype=x.dtype, device=x.device)
        out_mask = torch.ones(B, max_len_out, dtype=torch.bool, device=x.device)  # all padded initially

        # --- Single pass loop with enumerate ---
        for b, target_len in enumerate(finallengths.tolist()):
            # Extract valid (unpadded) region
            valid_len = (~padding_mask[b]).sum().item()
            seq = x[b, :valid_len]  # [L_b, D]

            # Interpolate to target length
            seq = seq.unsqueeze(0).transpose(1, 2).unsqueeze(1)  # [1, 1, D, L_b]
            out = F.interpolate(seq, size=(D, target_len), mode="area")
            out = out.squeeze(1).transpose(1, 2)  # [1, target_len, D]

            # Place in padded_out and mark valid positions in mask
            Lb = out.shape[1]
            padded_out[b, :Lb] = out
            out_mask[b, :Lb] = False  # False = valid, True = padded

        return padded_out, out_mask
