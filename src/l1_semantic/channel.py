"""Payload channel for WP1 (decision 2026-10-11, docs/DECISIONS.md).

The per-beat payload is a byte stream: the RR bytes first, then the representation's elements
in transmission order (for the semantic latent: time-major over the active channels; for a
waveform: sample order). Two impairments:

* Packet loss: the stream is cut into fragments of ``fragment_bytes`` bytes; each fragment is
  erased independently with probability p (an erased fragment removes every element that
  overlaps it). Erased elements are zeroed and flagged in a mask, so the gateway knows what
  is missing. RR features erased with their fragment are zeroed and flagged too.
* SNR: additive white Gaussian noise on the received analog-valued elements (latent or
  waveform), with the SNR defined per sample relative to the mean power of that sample's
  delivered elements. RR bytes are digital and are either delivered intact or erased.

The BLE PHY packet model (src/channel) is not used here; it enters in WP2/WP3.
Randomness comes from the global torch generator, which the entry point seeds.
"""
from __future__ import annotations

import torch


def _per_sample(v, n, device):
    if v is None:
        return None
    t = torch.as_tensor(v, dtype=torch.float32, device=device)
    return t.expand(n) if t.dim() == 0 else t


def fragment_keep(n: int, elem_bits: torch.Tensor, fragment_bytes: int, p: torch.Tensor,
                  device=None) -> torch.Tensor:
    """(n, E) mask: 1 where element e of sample i survives fragment erasure."""
    ends = torch.cumsum(elem_bits, 0)
    starts = ends - elem_bits
    frag_bits = 8 * fragment_bytes
    first = (starts // frag_bits).long()
    last = ((ends - 1) // frag_bits).long()
    n_frag = int(last.max().item()) + 1
    frag_ok = (torch.rand(n, n_frag, device=device) >= p[:, None]).float()
    keep = frag_ok[:, first]
    span = last > first
    if span.any():  # an element split across fragments needs both
        keep[:, span] = keep[:, span] * frag_ok[:, last[span]]
    return keep


class PayloadChannel:
    def __init__(self, fragment_bytes: int = 16, rr_bytes: int = 3):
        self.fragment_bytes = fragment_bytes
        self.rr_bytes = rr_bytes

    def __call__(self, z, mask, bits: int, snr_db=None, loss_p=None, order: str = "time_major"):
        """z, mask: (n, C, T); mask marks transmitted elements. Returns (z_rx, mask_rx, rr_keep)."""
        n, c, t = z.shape
        dev = z.device
        rr_keep = torch.ones(n, 1, device=dev)
        p = _per_sample(loss_p, n, dev)
        if p is not None:
            active = mask[0].bool()  # transmitted set is the same for every sample in a batch
            if order == "time_major":
                idx = active.t().nonzero()          # (E, 2): (t, c) in time-major order
                ci, ti = idx[:, 1], idx[:, 0]
            else:
                idx = active.nonzero()
                ci, ti = idx[:, 0], idx[:, 1]
            elem_bits = torch.cat([torch.full((self.rr_bytes,), 8.0, device=dev),
                                   torch.full((len(ci),), float(bits), device=dev)])
            keep = fragment_keep(n, elem_bits, self.fragment_bytes, p, dev)
            rr_keep = keep[:, :self.rr_bytes].prod(dim=1, keepdim=True)
            m = torch.zeros(n, c, t, device=dev)
            m[:, ci, ti] = keep[:, self.rr_bytes:]
            mask = mask * m
            z = z * mask
        snr = _per_sample(snr_db, n, dev)
        if snr is not None:
            power = (z.pow(2) * mask).sum(dim=(1, 2)) / mask.sum(dim=(1, 2)).clamp_min(1.0)
            # noise level is a property of the channel: no gradient through it (and none at power 0)
            std = torch.sqrt(power.detach().clamp_min(1e-12) / 10.0 ** (snr / 10.0))
            z = (z + torch.randn_like(z) * std[:, None, None]) * mask
        return z, mask, rr_keep
