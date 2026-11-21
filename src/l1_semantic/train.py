"""Training and evaluation of the semantic codec (single-label, MIT-BIH).

* Each batch draws one rate level (nested training), an SNR in dB and a packet-loss
  probability from the configured training ranges.
* Model selection uses the validation split only: mean macro-F1 over rate levels on a clean
  latent channel. The test split is touched once, by ``evaluate_grid`` after training.
* Full-state checkpoints (model, optimiser, epoch, best score, RNG states) are written
  atomically every epoch; a rerun with the same output directory resumes.
"""
from __future__ import annotations

import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .channel import PayloadChannel
from .metrics import assert_consistent, multiclass_metrics
from .model import RateLevel, SemanticCodec  # noqa: F401  (any method with the same interface)


def _batches(n, batch_size, gen: torch.Generator | None, shuffle: bool):
    idx = torch.randperm(n, generator=gen) if shuffle else torch.arange(n)
    for i in range(0, n, batch_size):
        yield idx[i:i + batch_size]


def augment(x: torch.Tensor, cfg, gen: torch.Generator) -> torch.Tensor:
    """Training-only waveform augmentation (shared by every method; per-beat z-score kept).

    amplitude scaling, low-frequency baseline wander, additive Gaussian noise and a small
    circular time shift. Disabled when cfg is None or cfg.enabled is false.
    """
    if cfg is None or not cfg.enabled:
        return x
    n, _, length = x.shape
    scale = torch.empty(n, 1, 1).uniform_(1 - cfg.scale, 1 + cfg.scale, generator=gen)
    t = torch.arange(length, dtype=torch.float32)[None, None, :] / length
    freq = torch.empty(n, 1, 1).uniform_(0.1, 1.0, generator=gen)
    phase = torch.empty(n, 1, 1).uniform_(0, 6.2832, generator=gen)
    amp = torch.empty(n, 1, 1).uniform_(0, cfg.wander, generator=gen)
    noise = torch.randn(x.shape, generator=gen) * cfg.noise
    out = x * scale + amp * torch.sin(6.2832 * freq * t + phase) + noise
    if cfg.shift > 0:
        k = int(torch.randint(-cfg.shift, cfg.shift + 1, (1,), generator=gen))
        out = torch.roll(out, k, dims=2)
    return out


def _atomic_save(obj, path: Path):
    tmp = path.with_suffix(".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def _rng_state():
    return {"torch": torch.get_rng_state(), "numpy": np.random.get_state(), "python": random.getstate()}


def _set_rng_state(s):
    torch.set_rng_state(s["torch"])
    np.random.set_state(s["numpy"])
    random.setstate(s["python"])


@torch.no_grad()
def predict(model: SemanticCodec, data, level: RateLevel, channel=None, snr_db=None, loss_p=None,
            batch_size: int = 2048, eval_seed: int | None = None) -> np.ndarray:
    model.eval()
    if eval_seed is not None:
        torch.manual_seed(eval_seed)
    x, rr = torch.from_numpy(data.x), torch.from_numpy(data.rr)
    out = []
    for b in _batches(len(x), batch_size, None, shuffle=False):
        logits, _ = model(x[b], rr[b], level, channel, snr_db=snr_db, loss_p=loss_p)
        out.append(torch.softmax(logits, dim=1))
    return torch.cat(out).numpy()


def train(model: SemanticCodec, train_data, val_data, cfg, class_names, out_dir: Path,
          class_w: np.ndarray, log=print) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt, best_path, log_path = out_dir / "last.pt", out_dir / "best.pt", out_dir / "train_log.jsonl"
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    gen = torch.Generator().manual_seed(cfg.seed)
    channel = PayloadChannel(cfg.channel.fragment_bytes, cfg.channel.rr_bytes)
    start, best, bad_epochs = 0, -1.0, 0
    if ckpt.exists():
        s = torch.load(ckpt, weights_only=False)
        model.load_state_dict(s["model"])
        opt.load_state_dict(s["opt"])
        gen.set_state(s["gen"])
        _set_rng_state(s["rng"])
        start, best, bad_epochs = s["epoch"] + 1, s["best"], s["bad_epochs"]
        log(f"resumed at epoch {start}")
    x, rr, y = (torch.from_numpy(a) for a in (train_data.x, train_data.rr, train_data.y))
    w = torch.from_numpy(class_w)
    levels = model.rate_levels
    for epoch in range(start, cfg.epochs):
        if bad_epochs >= cfg.patience:
            break
        model.train()
        t0, tot, n = time.time(), {"ce": 0.0, "kl": 0.0, "rate_bits": 0.0}, 0
        for b in _batches(len(x), cfg.batch_size, gen, shuffle=True):
            level = levels[int(torch.randint(len(levels), (1,), generator=gen))]
            nb = len(b)
            snr = torch.empty(nb).uniform_(*cfg.channel.train_snr_db, generator=gen) \
                if cfg.channel.train_snr_db else None
            if snr is not None and cfg.channel.clean_fraction > 0:  # some clean batches
                snr = torch.where(torch.rand(nb, generator=gen) < cfg.channel.clean_fraction,
                                  torch.full_like(snr, 100.0), snr)
            loss_p = torch.empty(nb).uniform_(0.0, cfg.channel.train_loss_max, generator=gen) \
                if cfg.channel.train_loss_max > 0 else None
            xb = augment(x[b], cfg.get("augment"), gen)
            logits, aux = model(xb, rr[b], level, channel, snr_db=snr, loss_p=loss_p)
            ce = F.cross_entropy(logits, y[b], weight=w)
            loss = ce + cfg.beta_kl * aux["kl"] + cfg.lambda_rate * aux["rate_bits"]
            if "extra_loss" in aux:  # e.g. the JSCC reconstruction term
                loss = loss + aux["extra_loss"]
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            tot["ce"] += ce.item() * nb
            tot["kl"] += aux["kl"].item() * nb
            tot["rate_bits"] += aux["rate_bits"].item() * nb
            n += nb
        val = {}
        for lv in levels:
            p = predict(model, val_data, lv)
            val[f"k{lv.channels}_b{lv.bits}"] = multiclass_metrics(val_data.y, p, class_names)["macro_f1"]
        score = float(np.mean(list(val.values())))
        improved = score > best + cfg.min_delta
        if improved:
            best, bad_epochs = score, 0
            _atomic_save({"model": model.state_dict(), "epoch": epoch, "val_score": score}, best_path)
        else:
            bad_epochs += 1
        rec = {"epoch": epoch, **{k: v / n for k, v in tot.items()}, "val_macro_f1": val,
               "val_score": score, "best": best, "seconds": round(time.time() - t0, 2)}
        with open(log_path, "a") as f:
            f.write(json.dumps(rec) + "\n")
        log(json.dumps(rec))
        if not np.isfinite(rec["ce"]):
            raise FloatingPointError("training loss is not finite")
        _atomic_save({"model": model.state_dict(), "opt": opt.state_dict(), "epoch": epoch,
                      "best": best, "bad_epochs": bad_epochs, "gen": gen.get_state(),
                      "rng": _rng_state()}, ckpt)
    return {"best_val_score": best, "best_epoch": torch.load(best_path, weights_only=False)["epoch"]}


def evaluate_grid(model: SemanticCodec, data, class_names, snr_grid, loss_grid, eval_seed: int,
                  fragment_bytes: int = 16, rr_bytes: int = 3) -> list[dict]:
    """Test-time sweep: every rate level on a clean channel, then SNR and loss sweeps.

    Every metric set passes the consistency check (errors raise; warnings are recorded).
    """
    channel = PayloadChannel(fragment_bytes, rr_bytes)
    rows = []
    conditions = [(None, None)] + [(s, None) for s in snr_grid] + [(None, p) for p in loss_grid]
    for li, lv in enumerate(model.rate_levels):
        for ci, (snr, p) in enumerate(conditions):
            probs = predict(model, data, lv, channel if (snr is not None or p) else None,
                            snr_db=snr, loss_p=p, eval_seed=eval_seed * 1000 + li * 100 + ci)
            m = multiclass_metrics(data.y, probs, class_names)
            rep = assert_consistent(m, data.y, probs)
            rows.append({"channels": lv.channels, "bits": lv.bits, "snr_db": snr, "loss_p": p,
                         "metrics": m, "consistency_warnings": rep["warnings"]})
    return rows
