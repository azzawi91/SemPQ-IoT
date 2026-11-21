"""WP1: train the semantic codec on MIT-BIH (one seed per call) and evaluate on DS2.

    python experiments/wp1_train.py seed=0
    python experiments/wp1_train.py -m seed=0,1,2,3,4

Writes results/wp1/<run_name>/seed<k>.json (metrics for every rate level x SNR / loss
condition, payload bytes, encoder cost model, provenance) and a training log; checkpoints go
to paths.out and are never committed.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import hydra  # noqa: E402
from omegaconf import DictConfig, OmegaConf  # noqa: E402

from src.utils.seed import pin_threads, set_seed  # noqa: E402


def _abs(p) -> Path:
    p = Path(p)
    return p if p.is_absolute() else REPO / p


def build_method(cfg, tr, window):
    import numpy as np

    from src.l1_semantic import baselines as bl
    from src.l1_semantic.model import RateLevel, SemanticCodec

    n_cls = len(cfg.classes)
    rr_dim, in_ch = tr.rr.shape[1], tr.x.shape[1]
    levels = [RateLevel(**lv) for lv in cfg.model.rate_levels]
    if cfg.method == "semantic":
        # optional encoder size (decision 2026-10-13: smaller on-node encoder chosen by DS1 CV)
        enc = cfg.model.get("encoder", None)
        enc_kw = {k: tuple(v) for k, v in enc.items()} if enc else None
        return SemanticCodec(n_cls, levels, cfg.model.latent_channels, cfg.model.clip,
                             rr_dim=rr_dim, variational=cfg.model.variational,
                             recon_weight=cfg.model.recon_weight, window=window, in_channels=in_ch,
                             encoder_kwargs=enc_kw)
    if cfg.method == "rr_only":  # ablation: RR bytes only, no waveform information
        return SemanticCodec(n_cls, [RateLevel(0, 8)], cfg.model.latent_channels, cfg.model.clip,
                             rr_dim=rr_dim, variational=cfg.model.variational, in_channels=in_ch)
    if cfg.method == "raw":
        return bl.RawBaseline(n_cls, cfg.baselines.raw_sample_bits, rr_dim=rr_dim, gateway_template=in_ch > 1)
    if cfg.method == "dwt":
        return bl.DWTBaseline(n_cls, tuple(cfg.baselines.dwt_coeff_levels), rr_dim=rr_dim,
                              gateway_template=in_ch > 1)
    if cfg.method == "jscc":
        return bl.JSCCBaseline(n_cls, levels, cfg.model.latent_channels, cfg.model.clip, window, rr_dim=rr_dim,
                               gateway_template=in_ch > 1)
    if cfg.method == "label":
        prior = np.bincount(tr.y, minlength=n_cls) / len(tr.y)  # training split only
        return bl.LabelBaseline(n_cls, prior, cfg.model.latent_channels, rr_dim=rr_dim, in_channels=in_ch)
    raise ValueError(cfg.method)


def device_cost(model, method, window):
    """Analytic on-device cost model (layer shapes); never a hardware measurement."""
    from src.l1_semantic.model import encoder_macs, encoder_memory_estimate

    note = "analytic estimate from layer shapes; not a hardware measurement"
    if method == "rr_only":
        return {"macs_per_beat": 0, "note": note + "; RR intervals only"}
    if method in ("semantic", "jscc"):
        return {"macs_per_beat": encoder_macs(model.encoder, window),
                **encoder_memory_estimate(model.encoder, window), "note": note}
    if method == "label":
        enc = encoder_macs(model.encoder, window)
        head = sum(m.weight.numel() * (window // 32) for m in model.head.conv if hasattr(m, "weight")) \
            + sum(m.weight.numel() for m in model.head.head if hasattr(m, "weight"))
        mem = encoder_memory_estimate(model.encoder, window)
        mem["flash_bytes"] += sum(p.numel() for p in model.head.parameters())
        return {"macs_per_beat": int(enc + head), **mem, "note": note + "; head convs at T=window/32"}
    return {"macs_per_beat": int(model.device_macs(window)), "note": note}


def run_cv(cfg, tr, va, window, rr_bytes, tcfg, out_dir):
    """Development cross-validation over DS1 patients (DS2 is never loaded).

    Deterministic patient folds; each fold trains for a fixed number of epochs (no early
    stopping, so the held-out fold is never used for selection) and predicts its held-out
    patients. Out-of-fold predictions are pooled over all 22 DS1 records.
    """
    import numpy as np
    import torch

    from src.data import mitbih
    from src.l1_semantic import data as l1data
    from src.l1_semantic.metrics import assert_consistent, multiclass_metrics
    from src.l1_semantic.channel import PayloadChannel
    from src.l1_semantic.train import predict, train
    from src.utils.provenance import provenance

    ds1 = l1data.concat_splits([tr, va])
    aami = [mitbih.CLASS_INDEX[c] for c in cfg.classes]
    counts = {}
    for r in np.unique(ds1.record):
        c = np.zeros(len(mitbih.AAMI_CLASSES), int)
        np.add.at(c, np.array(aami)[ds1.y[ds1.record == r]], 1)
        counts[int(r)] = c
    folds = mitbih.group_folds(counts, cfg.dev.cv_folds)
    ftcfg = OmegaConf.create(OmegaConf.to_container(tcfg))
    ftcfg.epochs, ftcfg.patience = cfg.dev.cv_epochs, 10 ** 6
    conds = [(None, None), (10.0, None), (None, 0.2)]
    oof = {}
    for fi, fold in enumerate(folds):
        ftr = l1data.subset_records(ds1, fold, keep=False)
        fho = l1data.subset_records(ds1, fold, keep=True)
        model = build_method(cfg, ftr, window)
        cw = l1data.class_weights(ftr.y, len(cfg.classes), cfg.train.class_weights)
        fdir = out_dir / f"cv{fi}"
        train(model, ftr, fho, ftcfg, cfg.classes, fdir, cw)
        model.load_state_dict(torch.load(fdir / "last.pt", weights_only=False)["model"])  # final epoch
        ch = PayloadChannel(cfg.train.channel.fragment_bytes, rr_bytes)
        for li, lv in enumerate(model.rate_levels):
            for ci, (snr, p) in enumerate(conds):
                probs = predict(model, fho, lv, ch if (snr is not None or p) else None, snr_db=snr,
                                loss_p=p, eval_seed=cfg.seed * 1000 + fi * 100 + li * 10 + ci)
                key = (lv.channels, lv.bits, snr, p)
                ys, ps = oof.get(key, ([], []))
                oof[key] = (ys + [fho.y], ps + [probs])
    grid = []
    for (k, b, snr, p), (ys, ps) in oof.items():
        y, pr = np.concatenate(ys), np.concatenate(ps)
        m = multiclass_metrics(y, pr, cfg.classes)
        rep = assert_consistent(m, y, pr)
        grid.append({"channels": k, "bits": b, "snr_db": snr, "loss_p": p, "metrics": m,
                     "consistency_warnings": rep["warnings"]})
    dres = {"provenance": provenance(OmegaConf.to_container(cfg, resolve=True), cfg.seed),
            "variant": cfg.dev.variant, "method": cfg.method, "cv_folds": [list(f) for f in folds],
            "training": {"best_epoch": cfg.dev.cv_epochs - 1, "mode": "fixed-epoch CV"},
            "val_grid": grid}
    dd = _abs(cfg.paths.dev_results) / cfg.dev.variant
    dd.mkdir(parents=True, exist_ok=True)
    (dd / f"{cfg.method}_seed{cfg.seed}.json").write_text(json.dumps(dres, indent=1) + "\n")
    for r in grid:
        if r["snr_db"] is None and r["loss_p"] is None:
            print(f"OOF {cfg.dev.variant} {cfg.method} k{r['channels']}_b{r['bits']} "
                  f"macroF1={r['metrics']['macro_f1']:.4f}")


@hydra.main(config_path="../configs", config_name="wp1_mitbih", version_base="1.3")
def main(cfg: DictConfig) -> None:
    # ablation aliases (configuration B): method name -> base method + config overrides
    abl = cfg.get("ablations") or {}
    if cfg.method in abl:
        spec = abl[cfg.method]
        OmegaConf.set_struct(cfg, False)
        for k, v in spec.overrides.items():
            OmegaConf.update(cfg, k, v, merge=False)
        alias = cfg.method
        cfg.run_name = f"{alias}_mitbih"
        cfg.ablation_of, cfg.run_label, cfg.method = spec.base, alias, spec.base
    set_seed(cfg.seed)
    pin_threads(cfg.threads)
    import torch  # after seeding

    from src.data import splits
    from src.l1_semantic import data as l1data
    from src.l1_semantic.model import count_params
    from src.l1_semantic.train import evaluate_grid, train
    from src.utils.provenance import provenance

    splits.verify_splits()
    root, cache = _abs(cfg.data.root), _abs(cfg.paths.cache)
    w = cfg.data.window
    prep = lambda s, excl=(): l1data.prepare_mitbih(  # noqa: E731
        root, s, w, cfg.classes, cfg.rr.step_s, excl, cache,
        template_alpha=cfg.features.template_alpha, rr_long_window=cfg.features.rr_long_window)
    tr, va = prep("train"), prep("val")
    cw = l1data.class_weights(tr.y, len(cfg.classes), cfg.train.class_weights)

    window = w.pre + w.post
    rr_bytes = cfg.rr.bytes + (1 if cfg.features.rr_long_window > 0 else 0)  # long-term RR byte
    if not cfg.dev.enabled and cfg.final.train_on == "ds1_all":
        # decision 2026-10-11: train on all 22 DS1 records for a fixed number of epochs (the
        # protocol selected by DS1 cross-validation); the 5 validation records are monitored only
        tr = l1data.concat_splits([tr, va])
    model = build_method(cfg, tr, window)
    levels = model.rate_levels
    out_dir = _abs(cfg.paths.out) / cfg.run_name / f"seed{cfg.seed}"
    tcfg = OmegaConf.create({**OmegaConf.to_container(cfg.train, resolve=True), "seed": cfg.seed})
    tcfg.channel.rr_bytes = rr_bytes
    if not cfg.dev.enabled and cfg.final.fixed_epochs > 0:
        tcfg.epochs, tcfg.patience = cfg.final.fixed_epochs, 10 ** 6

    if cfg.dev.enabled and cfg.dev.cv_folds > 0:
        return run_cv(cfg, tr, va, window, rr_bytes, tcfg, out_dir)

    summary = train(model, tr, va, tcfg, cfg.classes, out_dir, cw)

    if cfg.dev.enabled:  # validation-only development run: DS2 is never loaded
        model.load_state_dict(torch.load(out_dir / "best.pt", weights_only=False)["model"])
        vgrid = evaluate_grid(model, va, cfg.classes, cfg.eval.snr_grid_db, cfg.eval.loss_grid,
                              eval_seed=cfg.seed, fragment_bytes=cfg.train.channel.fragment_bytes,
                              rr_bytes=rr_bytes)
        dres = {"provenance": provenance(OmegaConf.to_container(cfg, resolve=True), cfg.seed),
                "variant": cfg.dev.variant, "method": cfg.method, "training": summary,
                "payload_bytes": {f"k{lv.channels}_b{lv.bits}": model.payload_bytes(lv, window, rr_bytes)
                                  for lv in levels},
                "val_grid": vgrid}
        dd = _abs(cfg.paths.dev_results) / cfg.dev.variant
        dd.mkdir(parents=True, exist_ok=True)
        (dd / f"{cfg.method}_seed{cfg.seed}.json").write_text(json.dumps(dres, indent=1) + "\n")
        for r in vgrid:
            if r["snr_db"] is None and r["loss_p"] is None:
                print(f"VAL {cfg.dev.variant} {cfg.method} k{r['channels']}_b{r['bits']} "
                      f"macroF1={r['metrics']['macro_f1']:.4f}")
        return

    te = prep("test")

    final_ckpt = "last.pt" if cfg.final.fixed_epochs > 0 else "best.pt"
    model.load_state_dict(torch.load(out_dir / final_ckpt, weights_only=False)["model"])
    grid = evaluate_grid(model, te, cfg.classes, cfg.eval.snr_grid_db, cfg.eval.loss_grid,
                         eval_seed=cfg.seed, fragment_bytes=cfg.train.channel.fragment_bytes,
                         rr_bytes=rr_bytes)
    te_s = prep("test", tuple(cfg.eval.sensitivity_exclude_test))
    sens = evaluate_grid(model, te_s, cfg.classes, [], [], eval_seed=cfg.seed, rr_bytes=rr_bytes)

    int8 = None
    if cfg.int8.enabled and hasattr(model, "encoder"):
        from src.l1_semantic.quantize import QuantizedEncoderAdapter, quantize_encoder_int8

        enc_in = model.encoder.body[0][0].in_channels  # baselines' encoders see channel 0 only
        calib = torch.from_numpy(tr.x[: cfg.int8.calib_beats, :enc_in])  # training beats only
        float_enc = model.encoder
        model.encoder = QuantizedEncoderAdapter(quantize_encoder_int8(float_enc, calib),
                                                cfg.model.latent_channels)
        int8 = {"calib_beats": int(len(calib)), "backend": "x86 (emulated INT8)",
                "grid": evaluate_grid(model, te, cfg.classes, [], [], eval_seed=cfg.seed)}
        model.encoder = float_enc

    result = {
        "provenance": provenance(OmegaConf.to_container(cfg, resolve=True), cfg.seed),
        "split_sha256": splits.verify_splits(),
        "n": {"train": len(tr.y), "val": len(va.y), "test": len(te.y), "test_sensitivity": len(te_s.y)},
        "class_weights": cw.tolist(),
        "training": summary,
        "payload_bytes": {f"k{lv.channels}_b{lv.bits}": model.payload_bytes(lv, window, rr_bytes)
                          for lv in levels},
        "method": cfg.get("run_label", cfg.method),
        "ablation_of": cfg.get("ablation_of"),
        "final_protocol": {"train_on": cfg.final.train_on, "fixed_epochs": cfg.final.fixed_epochs,
                           "checkpoint": final_ckpt},
        "device_cost_model": device_cost(model, cfg.method, window),
        "params_total": count_params(model),
        "test_grid": grid,
        "test_int8_encoder": int8,
        "test_without_records": {"records": list(cfg.eval.sensitivity_exclude_test), "grid": sens},
    }
    res = _abs(cfg.paths.results) / cfg.run_name
    res.mkdir(parents=True, exist_ok=True)
    (res / f"seed{cfg.seed}.json").write_text(json.dumps(result, indent=1) + "\n")
    (res / f"seed{cfg.seed}_train_log.jsonl").write_text((out_dir / "train_log.jsonl").read_text())
    if cfg.get("save_model", True):  # evaluated weights, for the Zenodo deposit (no optimizer state)
        torch.save({"model": model.state_dict(), "method": result["method"], "seed": cfg.seed,
                    "checkpoint": final_ckpt, "commit": result["provenance"]["commit"],
                    "config": OmegaConf.to_container(cfg, resolve=True)}, res / f"seed{cfg.seed}_model.pt")
    # per-beat DS2 predictions (clean channel) for record-level bootstrap, whole-record-loss and
    # end-to-end analyses (decision 2026-10-12); labels and record ids in DS2 beat order
    import numpy as np

    from src.l1_semantic.train import predict
    preds = {f"k{lv.channels}_b{lv.bits}": predict(model, te, lv).argmax(1).astype(np.int8) for lv in levels}
    np.savez_compressed(res / f"seed{cfg.seed}_pred.npz", y=te.y.astype(np.int8), record=te.record.astype(np.int16),
                        **preds)
    clean = [r for r in grid if r["snr_db"] is None and r["loss_p"] is None]
    for r in clean:
        m, key = r["metrics"], f"k{r['channels']}_b{r['bits']}"
        print(f"{key:>7} bytes={result['payload_bytes'][key]:>4} acc={m['accuracy']:.4f} "
              f"macroF1={m['macro_f1']:.4f} AUROC={m['macro_auroc']:.4f} "
              f"Se(S)={m['per_class']['S']['sensitivity']:.3f} Se(V)={m['per_class']['V']['sensitivity']:.3f}")

if __name__ == "__main__":
    main()
