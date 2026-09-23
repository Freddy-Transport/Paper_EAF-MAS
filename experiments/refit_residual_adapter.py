#!/usr/bin/env python3
"""Fit only a residual MLP from private fixed backbone predictions."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agents.audit_inputs import load_bundle, new_output_dir, digest, configuration_digest
from agents.event_adapter import EventResidualAdapter, save_adapter
from agents.formal_workflow import numerical_context, FEATURE_SCHEMA, FORMAL_FEATURES


def training_rows(records, memory, config):
    xs, ys = [], []
    for r in records:
        c = numerical_context(r, memory, config)
        actual = np.asarray(r["outcomes"], dtype=float)
        if actual.shape != c["raw"].shape or not np.isfinite(actual).all() or (actual < 0).any():
            raise ValueError("Invalid adapter training outcomes")
        target = np.clip((actual-c["raw"])/np.maximum(c["raw"], config["training"]["epsilon"]),
                         -config["rho_full"], config["rho_full"])
        mask = c["mask"]
        xs.append(c["features"][mask])
        ys.append(target[mask])
    if not xs or not sum(len(x) for x in xs):
        raise ValueError("No eligible historical training cells; no synthetic fallback")
    return torch.tensor(np.concatenate(xs), dtype=torch.float32), torch.tensor(np.concatenate(ys), dtype=torch.float32)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    manifest, train, memory = load_bundle(args.manifest, "train")
    _, val, _ = load_bundle(args.manifest, "val")
    config = json.loads(Path(args.config).read_text())
    t = config["training"]
    # All settings explicit; never select them using test outcomes.
    for key in ("seed", "epochs", "patience", "batch_size", "learning_rate", "weight_decay", "hidden_dim", "dropout", "max_correction", "epsilon"):
        if key not in t:
            raise ValueError(f"Missing locked training configuration: {key}")
    torch.manual_seed(t["seed"])
    device = config.get("device", "cpu")
    x, y = training_rows(train, memory, config)
    vx, vy = training_rows(val, memory, config)
    net = EventResidualAdapter(len(FORMAL_FEATURES), t["hidden_dim"], t["dropout"], t["max_correction"]).to(device)
    net.output_transform = "identity"
    opt = torch.optim.AdamW(net.parameters(), lr=t["learning_rate"], weight_decay=t["weight_decay"])
    loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x, y), batch_size=t["batch_size"], shuffle=True)
    best, state, stale, history = float("inf"), None, 0, []
    for epoch in range(t["epochs"]):
        net.train()
        for bx, by in loader:
            opt.zero_grad()
            loss = torch.nn.functional.mse_loss(net(bx.to(device)), by.to(device))
            loss.backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            total = sum(float(torch.nn.functional.mse_loss(net(vx[i:i+t["batch_size"]].to(device)), vy[i:i+t["batch_size"]].to(device), reduction="sum")) for i in range(0,len(vx),t["batch_size"]))
        score = total/len(vx)
        history.append(dict(epoch=epoch, validation_mse=score))
        if score < best:
            best, state, stale = score, deepcopy(net.state_dict()), 0
        else:
            stale += 1
        if stale >= t["patience"]:
            break
    if state is None:
        raise ValueError("Adapter fitting produced no valid checkpoint")
    net.load_state_dict(state)
    out = new_output_dir(args.output)
    adapter_config = dict(input_dim=len(FORMAL_FEATURES), hidden_dim=t["hidden_dim"], dropout=t["dropout"],
        max_correction=t["max_correction"], feature_names=FORMAL_FEATURES, feature_schema=FEATURE_SCHEMA,
        model_identity=manifest["model_identity"], output_transform="identity")
    save_adapter(net, out, adapter_config, dict(input_manifest_digest=digest(manifest), configuration_digest=configuration_digest(config), training=t,
        model_identity=manifest["model_identity"], train_requests=len(train), val_requests=len(val), history=history))


if __name__ == "__main__":
    main()
