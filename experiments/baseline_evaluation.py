"""Baseline 对比实验：Zero-shot / Linear Probing / GCA-LPT 微调.

支持两种运行方式：
  1. Colab / Jupyter：直接在 notebook 中 import 并调用 run_all()
  2. 终端命令行：python experiments/baseline_evaluation.py --device cuda:0

Colab 快速使用：
    from google.colab import drive
    drive.mount('/content/drive')
    import os; os.chdir('/content/drive/MyDrive/moment')
    !pip install transformers==4.33.3 huggingface-hub==0.24.0 safetensors -q
    !pip install -e . -q

    from experiments.baseline_evaluation import run_all
    results = run_all(forecast_horizon=48)
"""

import copy
import json
import math
import os
import random
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader
from tqdm import tqdm


# ---- 自动检测项目根目录 ----
def _detect_project_root():
    """兼容 Colab / 终端 / Jupyter 等多种运行环境."""
    try:
        return str(Path(__file__).resolve().parent.parent)
    except NameError:
        pass
    cwd = os.getcwd()
    if os.path.exists(os.path.join(cwd, "momentfm")):
        return cwd
    for candidate in ["/content/drive/MyDrive/moment", "/content/drive/MyDrive/0206moment"]:
        if os.path.exists(os.path.join(candidate, "momentfm")):
            return candidate
    return cwd

PROJECT_ROOT = _detect_project_root()
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from momentfm import MOMENTPipeline
from momentfm.data.informer_dataset import InformerDataset
from momentfm.utils.forecasting_metrics import get_forecasting_metrics


# ---- 是否在 notebook 环境中 ----
def _in_notebook():
    try:
        from IPython import get_ipython
        shell = get_ipython().__class__.__name__
        return shell in ("ZMQInteractiveShell", "Shell", "Google Colab")
    except Exception:
        return False

IN_NOTEBOOK = _in_notebook()

if not IN_NOTEBOOK:
    matplotlib.use("Agg")


def set_seed(seed: int = 42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_data(data_path, forecast_horizon, batch_size, seed=42):
    train_dataset = InformerDataset(
        dataset_name=data_path,
        data_split="train",
        random_seed=seed,
        forecast_horizon=forecast_horizon,
    )
    test_dataset = InformerDataset(
        dataset_name=data_path,
        data_split="test",
        random_seed=seed,
        forecast_horizon=forecast_horizon,
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    return train_loader, test_loader, train_dataset, test_dataset


@torch.no_grad()
def evaluate(model, test_loader, device, dtype=torch.float32):
    """对模型在测试集上进行评估，返回指标和预测结果."""
    model.eval()
    trues, preds, histories = [], [], []

    for timeseries, forecast, input_mask in test_loader:
        timeseries = timeseries.float().to(device)
        input_mask = input_mask.to(device)
        forecast = forecast.float()

        with torch.autocast(device_type=device.type, dtype=dtype, enabled=(device.type == "cuda")):
            output = model(x_enc=timeseries, input_mask=input_mask)

        preds.append(output.forecast.detach().cpu().numpy())
        trues.append(forecast.numpy())
        histories.append(timeseries.detach().cpu().numpy())

    trues = np.concatenate(trues, axis=0)
    preds = np.concatenate(preds, axis=0)
    histories = np.concatenate(histories, axis=0)

    metrics = get_forecasting_metrics(y=trues, y_hat=preds, reduction="mean")
    wape = np.sum(np.abs(trues - preds)) / np.sum(np.abs(trues)) * 100

    return {
        "mse": float(metrics.mse),
        "mae": float(metrics.mae),
        "rmse": float(metrics.rmse),
        "mape": float(metrics.mape),
        "smape": float(metrics.smape),
        "wape": float(wape),
    }, trues, preds, histories


# ============================================================
# Linear Probing 训练
# ============================================================
def train_linear_probing(model, train_loader, test_loader, device, max_epoch=5, lr=1e-4, dtype=torch.float32,
                         save_dir=None):
    """Linear Probing：冻结编码器，只训练预测头.

    Args:
        save_dir: 若指定，则将最优模型 state_dict 保存到 save_dir/lp_weights.pt。

    Returns:
        (model, best_metrics, best_results)：model 已加载最优权重。
    """
    model.train()
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
    )
    total_steps = len(train_loader) * max_epoch
    scheduler = OneCycleLR(optimizer, max_lr=lr, total_steps=total_steps, pct_start=0.3)

    best_mse = float("inf")
    best_metrics = None
    best_results = None
    best_state_dict = None

    for epoch in range(max_epoch):
        losses = []
        model.train()
        for timeseries, forecast, input_mask in tqdm(train_loader, desc=f"LP Epoch {epoch+1}/{max_epoch}"):
            timeseries = timeseries.float().to(device)
            input_mask = input_mask.to(device)
            forecast = forecast.float().to(device)

            with torch.autocast(device_type=device.type, dtype=dtype, enabled=(device.type == "cuda")):
                output = model(x_enc=timeseries, input_mask=input_mask)

            loss = criterion(output.forecast, forecast)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            losses.append(loss.item())

        avg_loss = np.mean(losses)
        metrics, trues, preds, histories = evaluate(model, test_loader, device, dtype)
        print(f"  LP Epoch {epoch+1}: Train Loss={avg_loss:.5f} | "
              f"MSE={metrics['mse']:.5f} MAE={metrics['mae']:.5f} MAPE={metrics['mape']:.2f}%")

        if metrics["mse"] < best_mse:
            best_mse = metrics["mse"]
            best_metrics = metrics
            best_results = (trues, preds, histories)
            best_state_dict = copy.deepcopy(model.state_dict())

    # 恢复最优权重
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    # 保存权重
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        weights_path = os.path.join(save_dir, "lp_weights.pt")
        torch.save(best_state_dict if best_state_dict is not None else model.state_dict(), weights_path)
        print(f"  LP model saved to: {weights_path}")

    return model, best_metrics, best_results


# ============================================================
# GCA-LPT 训练（移植自 lpt复现主函数.ipynb）
# ============================================================
def train_gca_lpt(
    model,
    train_loader,
    test_loader,
    device,
    dtype=torch.bfloat16,
    num_epochs=50,
    lr=1e-3,
    eval_steps=100,
    warmup_steps=100,
    max_grad_norm=2.0,
    save_dir=None,
    tunable="head,prompt_generator,graph_module,prompt_gate",
):
    """GCA-LPT 训练：冻结编码器，训练 head + prompt_generator + graph_module + prompt_gate."""
    from transformers import get_scheduler

    tunable_keys = [k.strip() for k in tunable.split(",")]
    no_decay = ["bias", "layer_norm.weight"]

    # ---- 冻结/解冻参数 ----
    total_frozen = 0
    total_trainable = 0
    for n, p in model.named_parameters():
        if any(key in n for key in tunable_keys):
            p.requires_grad = True
            total_trainable += p.numel()
        else:
            p.requires_grad = False
            total_frozen += p.numel()

    print(f"  Frozen parameters: {total_frozen:,}")
    print(f"  Trainable parameters: {total_trainable:,}")
    print(f"  Trainable modules: {tunable_keys}")

    # ---- 优化器（AdamW + weight_decay 分组）----
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters()
                       if p.requires_grad and not any(nd in n for nd in no_decay)],
            "weight_decay": 0.01,
        },
        {
            "params": [p for n, p in model.named_parameters()
                       if p.requires_grad and any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=lr)

    # ---- 学习率调度 ----
    num_update_steps_per_epoch = len(train_loader)
    max_train_steps = num_epochs * num_update_steps_per_epoch
    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=max_train_steps,
    )

    # ---- 训练循环 ----
    criterion = torch.nn.MSELoss()
    completed_steps = 0
    best_mse = float("inf")
    best_metrics = None
    best_results = None

    for epoch in range(num_epochs):
        losses = []
        model.train()
        for timeseries, forecast, input_mask in tqdm(train_loader, desc=f"GCA-LPT Epoch {epoch+1}/{num_epochs}"):
            timeseries = timeseries.float().to(device)
            input_mask = input_mask.long().to(device)
            forecast = forecast.float().to(device)

            with torch.autocast(device_type=device.type, dtype=dtype, enabled=(device.type == "cuda")):
                output = model(x_enc=timeseries, input_mask=input_mask)

            loss = criterion(output.forecast, forecast)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            completed_steps += 1
            losses.append(loss.item())

            # ---- 定期评估 ----
            if completed_steps % eval_steps == 0 or completed_steps == 5:
                metrics, trues, preds, histories = evaluate(model, test_loader, device, dtype)
                model.train()

                if metrics["mse"] < best_mse:
                    best_mse = metrics["mse"]
                    best_metrics = metrics
                    best_results = (trues, preds, histories)
                    if save_dir:
                        os.makedirs(save_dir, exist_ok=True)
                        model.save_pretrained(save_dir)

                print(f"  Step {completed_steps}: "
                      f"MSE={metrics['mse']:.5f} MAE={metrics['mae']:.5f} "
                      f"MAPE={metrics['mape']:.2f}% WAPE={metrics['wape']:.2f}%")

        avg_loss = np.mean(losses)
        print(f"  Epoch {epoch+1}: Train Loss={avg_loss:.5f}")

    # ---- 最终评估（如果训练中没有触发过评估）----
    if best_metrics is None:
        best_metrics, trues, preds, histories = evaluate(model, test_loader, device, dtype)
        best_results = (trues, preds, histories)

    return model, best_metrics, best_results


def _find_prompt_gate(model):
    """在模型参数中查找 prompt_gate，返回 (参数, 路径名) 或 (None, None)."""
    for name, param in model.named_parameters():
        if "prompt_gate" in name:
            return param, name
    return None, None


def print_gca_lpt_diagnostics(model):
    """打印 prompt_gate 值和学习到的邻接矩阵."""
    print("\n" + "-" * 40)
    print("GCA-LPT Diagnostics")
    print("-" * 40)

    # prompt_gate
    gate_param, gate_name = _find_prompt_gate(model)
    if gate_param is not None:
        gate_raw = gate_param.item()
        gate_val = torch.sigmoid(gate_param).item()
        print(f"  prompt_gate [{gate_name}]")
        print(f"    raw value: {gate_raw:.4f}")
        print(f"    sigmoid:   {gate_val:.4f}")
        if gate_val > 0.7:
            print(f"    -> Prompt 影响较强 ({gate_val:.1%})")
        elif gate_val < 0.3:
            print(f"    -> Prompt 影响较弱 ({gate_val:.1%})，模型倾向忽略 prompt")
        else:
            print(f"    -> Prompt 影响适中 ({gate_val:.1%})")
    else:
        print("  prompt_gate: not found")

    # 邻接矩阵
    if hasattr(model, "graph_module"):
        adj = model.graph_module.get_adjacency_matrix()
        print(f"\n  Learned Adjacency Matrix ({adj.shape[0]}x{adj.shape[1]}):")
        np.set_printoptions(precision=3, suppress=True)
        print(f"  {adj.cpu().numpy()}")
        np.set_printoptions()

        gate_graph = torch.sigmoid(model.graph_module.gate).item()
        print(f"  Graph gate (sigmoid): {gate_graph:.4f}")
    print("-" * 40)


# ============================================================
# Experiment runners
# ============================================================
def run_zero_shot(data_path, forecast_horizon, batch_size, device, dtype):
    """Zero-shot：加载原始 MOMENT，不做任何训练直接预测."""
    print("\n" + "=" * 60)
    print("Experiment 1: Zero-shot (no training)")
    print("=" * 60)

    model = MOMENTPipeline.from_pretrained(
        "AutonLab/MOMENT-1-large",
        model_kwargs={
            "task_name": "forecasting",
            "forecast_horizon": forecast_horizon,
        },
    )
    model.init()
    model.to(device)

    _, test_loader, _, _ = load_data(data_path, forecast_horizon, batch_size)
    metrics, trues, preds, histories = evaluate(model, test_loader, device, dtype)

    print(f"  MSE={metrics['mse']:.5f} | MAE={metrics['mae']:.5f} | "
          f"MAPE={metrics['mape']:.2f}% | WAPE={metrics['wape']:.2f}%")

    return metrics, trues, preds, histories


def run_linear_probing(data_path, forecast_horizon, batch_size, device, max_epoch, lr, dtype, save_dir=None):
    """Linear Probing：冻结编码器，训练线性预测头.

    Args:
        save_dir: 若指定，则保存最优权重到 save_dir/lp_weights.pt，供 agent 调用。

    Returns:
        (metrics, trues, preds, histories)
    """
    print("\n" + "=" * 60)
    print("Experiment 2: Linear Probing (freeze encoder, train head)")
    print("=" * 60)

    model = MOMENTPipeline.from_pretrained(
        "AutonLab/MOMENT-1-large",
        model_kwargs={
            "task_name": "forecasting",
            "forecast_horizon": forecast_horizon,
            "freeze_encoder": True,
            "freeze_embedder": True,
            "freeze_head": False,
        },
    )
    model.init()
    model.to(device)

    print("Trainable parameters:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"  {name}: {param.numel()}")

    train_loader, test_loader, _, _ = load_data(data_path, forecast_horizon, batch_size)
    model, metrics, (trues, preds, histories) = train_linear_probing(
        model, train_loader, test_loader, device, max_epoch, lr, dtype, save_dir=save_dir
    )

    print(f"\n  Best: MSE={metrics['mse']:.5f} | MAE={metrics['mae']:.5f} | "
          f"MAPE={metrics['mape']:.2f}% | WAPE={metrics['wape']:.2f}%")

    return metrics, trues, preds, histories


def train_and_save_lp_model(
    data_path,
    forecast_horizon,
    save_dir,
    batch_size=64,
    max_epoch=5,
    lr=1e-4,
    device=None,
    seed=42,
):
    """训练 Linear Probing 模型并保存权重，供 NumericalPredictionAgent 调用.

    当用户未上传自定义模型权重时，此函数作为自动回退方案，使用官方 MOMENT
    的 Linear Probing 策略在指定数据集上训练并将最优模型权重持久化。

    Args:
        data_path:        训练/评估数据集路径（InformerDataset 支持的格式）。
        forecast_horizon: 预测步长，需与后续 agent 调用保持一致。
        save_dir:         权重保存目录，会写入 lp_weights.pt 和 lp_config.json。
        batch_size:       训练批大小，默认 64。
        max_epoch:        训练轮数，默认 5 轮即可获得良好效果。
        lr:               学习率，默认 1e-4。
        device:           计算设备（None 时自动检测）。
        seed:             随机种子。

    Returns:
        (model, metrics): 训练好的 MOMENTPipeline 模型及最优指标字典。
    """
    set_seed(seed)

    if device is None:
        dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        dev = torch.device(device)
    dtype = torch.bfloat16 if dev.type == "cuda" else torch.float32

    if not Path(data_path).is_absolute():
        data_path = str(Path(PROJECT_ROOT) / data_path)

    print(f"\n[LP Fallback] 未检测到自定义模型权重，启动 Linear Probing 训练...")
    print(f"  Device: {dev} | Data: {data_path} | Horizon: {forecast_horizon} | Epochs: {max_epoch}")

    model = MOMENTPipeline.from_pretrained(
        "AutonLab/MOMENT-1-large",
        model_kwargs={
            "task_name": "forecasting",
            "forecast_horizon": forecast_horizon,
            "freeze_encoder": True,
            "freeze_embedder": True,
            "freeze_head": False,
        },
    )
    model.init()
    model.to(dev)

    train_loader, test_loader, _, _ = load_data(data_path, forecast_horizon, batch_size, seed)

    model, metrics, _ = train_linear_probing(
        model, train_loader, test_loader, dev, max_epoch, lr, dtype, save_dir=save_dir
    )

    # 同时保存配置元信息，供 agent 加载时使用
    os.makedirs(save_dir, exist_ok=True)
    meta = {
        "model_type": "lp",
        "forecast_horizon": forecast_horizon,
        "data_path": data_path,
        "max_epoch": max_epoch,
        "lr": lr,
        "seed": seed,
        "metrics": metrics,
    }
    with open(os.path.join(save_dir, "lp_config.json"), "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"[LP Fallback] 完成！权重已保存至 {save_dir}/lp_weights.pt")
    print(f"  Best MSE={metrics['mse']:.5f} | MAE={metrics['mae']:.5f} | WAPE={metrics['wape']:.2f}%")

    return model, metrics


def run_gca_lpt(data_path, forecast_horizon, batch_size, device, dtype, n_channels=2,
                gca_epochs=50, gca_lr=1e-3, save_dir=None, use_graph=None):
    """GCA-LPT：从预训练 MOMENT 开始训练，完成后直接评估.

    Args:
        use_graph: 是否启用 Graph Cross-Channel Module。
                   None=自动（通道数>2 时启用），True=强制启用，False=强制关闭。
    """
    if use_graph is None:
        use_graph = n_channels > 2

    label = "GCA-LPT" if use_graph else "CA-LPT (no graph)"
    print("\n" + "=" * 60)
    print(f"Experiment 3: {label} (train from scratch)")
    if not use_graph:
        print(f"  Graph module disabled (n_channels={n_channels}, "
              "recommend enabling when n_channels > 2)")
    print("=" * 60)

    from lpt_finetuning.moment_lpt import Moment4Forecast

    model = Moment4Forecast.from_pretrained(
        "AutonLab/MOMENT-1-large",
        model_kwargs={
            "task_name": "forecasting",
            "forecast_horizon": forecast_horizon,
            "num_prompt_tokens": 16,
            "pg_dropout_prob": 0.3,
            "max_channels": 128,
            "pg_num_heads": 4,
            "use_graph": use_graph,
            "n_channels": n_channels,
            "n_gcn_layers": 2,
            "graph_embed_dim": 32,
            "graph_dropout": 0.1,
        },
        torch_dtype=torch.bfloat16,
    )
    model = model.to(device)

    train_loader, test_loader, _, _ = load_data(data_path, forecast_horizon, batch_size)

    # 模型保存到独立子目录，便于 agent 调用
    model_save_dir = os.path.join(save_dir, "gca_lpt_model") if save_dir else None

    model, best_metrics, (trues, preds, histories) = train_gca_lpt(
        model, train_loader, test_loader, device, dtype,
        num_epochs=gca_epochs,
        lr=gca_lr,
        eval_steps=100,
        warmup_steps=100,
        save_dir=model_save_dir,
    )

    # 打印 prompt_gate 和邻接矩阵
    print_gca_lpt_diagnostics(model)

    if model_save_dir and os.path.exists(os.path.join(model_save_dir, "model.safetensors")):
        print(f"\n  Model saved to: {model_save_dir}")
        print(f"  (可用于 agent 调用: Moment4Forecast.from_pretrained('{model_save_dir}', ...))")

    print(f"\n  Best: MSE={best_metrics['mse']:.5f} | MAE={best_metrics['mae']:.5f} | "
          f"MAPE={best_metrics['mape']:.2f}% | WAPE={best_metrics['wape']:.2f}%")

    return best_metrics, trues, preds, histories


def plot_comparison(results, channel_names, save_dir):
    """绘制对比可视化图."""
    os.makedirs(save_dir, exist_ok=True)
    methods = list(results.keys())

    # --- 1. 指标对比柱状图 ---
    metric_names = ["mse", "mae", "rmse", "mape", "wape"]
    metric_labels = ["MSE", "MAE", "RMSE", "MAPE (%)", "WAPE (%)"]
    colors = ["#95a5a6", "#3498db", "#e74c3c", "#2ecc71", "#9b59b6"][:len(methods)]

    fig, axes = plt.subplots(1, len(metric_names), figsize=(4 * len(metric_names), 5))
    for i, (mname, mlabel) in enumerate(zip(metric_names, metric_labels)):
        values = [results[m]["metrics"][mname] for m in methods]
        bars = axes[i].bar(methods, values, color=colors)
        axes[i].set_title(mlabel, fontsize=14)
        axes[i].set_ylabel(mlabel, fontsize=12)
        for bar, val in zip(bars, values):
            axes[i].text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                         f"{val:.3f}", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "metrics_comparison.png"), dpi=150, bbox_inches="tight")
    if IN_NOTEBOOK:
        plt.show()
    plt.close()
    print(f"Saved: {save_dir}/metrics_comparison.png")

    # --- 2. 各通道预测对比图 ---
    for ch_idx, ch_name in enumerate(channel_names):
        fig, axes_arr = plt.subplots(len(methods), 1, figsize=(14, 4 * len(methods)), sharex=True)
        if len(methods) == 1:
            axes_arr = [axes_arr]

        sample_idx = 0
        for ax, method_name in zip(axes_arr, methods):
            data = results[method_name]
            trues = data["trues"]
            preds = data["preds"]
            histories = data["histories"]

            if ch_idx >= trues.shape[1]:
                continue

            history = histories[sample_idx, ch_idx, :]
            true = trues[sample_idx, ch_idx, :]
            pred = preds[sample_idx, ch_idx, :]

            t_hist = range(len(history))
            offset = len(history)
            t_pred = range(offset, offset + len(true))

            ax.plot(t_hist, history, label="History", color="darkblue", linewidth=1)
            ax.plot(t_pred, true, label="Ground Truth", color="darkblue", linestyle="--", alpha=0.6)
            ax.plot(t_pred, pred, label="Forecast", color="red", linestyle="--", linewidth=1.2)
            ax.set_title(f"{method_name} — {ch_name}", fontsize=13)
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)

        plt.xlabel("Time Steps", fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"forecast_{ch_name}.png"), dpi=150, bbox_inches="tight")
        if IN_NOTEBOOK:
            plt.show()
        plt.close()
        print(f"Saved: {save_dir}/forecast_{ch_name}.png")

    # --- 3. 多样本预测对比 ---
    n_samples = min(3, results[methods[0]]["trues"].shape[0])
    sample_indices = sorted(random.sample(range(results[methods[0]]["trues"].shape[0]), n_samples))

    for ch_idx, ch_name in enumerate(channel_names):
        fig, axes_arr = plt.subplots(n_samples, 1, figsize=(14, 4 * n_samples), sharex=False)
        if n_samples == 1:
            axes_arr = [axes_arr]

        for ax, s_idx in zip(axes_arr, sample_indices):
            for method_name in methods:
                pred = results[method_name]["preds"][s_idx, ch_idx, :]
                ax.plot(pred, label=method_name, linewidth=1.2)
            true = results[methods[0]]["trues"][s_idx, ch_idx, :]
            ax.plot(true, label="Ground Truth", color="black", linewidth=1.5, linestyle="--")
            ax.set_title(f"Sample {s_idx} — {ch_name}", fontsize=12)
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"multi_sample_{ch_name}.png"), dpi=150, bbox_inches="tight")
        if IN_NOTEBOOK:
            plt.show()
        plt.close()
        print(f"Saved: {save_dir}/multi_sample_{ch_name}.png")


def print_summary_table(results):
    """打印汇总对比表."""
    print("\n" + "=" * 80)
    print("SUMMARY TABLE")
    print("=" * 80)
    header = f"{'Method':<25} {'MSE':>10} {'MAE':>10} {'RMSE':>10} {'MAPE%':>10} {'WAPE%':>10}"
    print(header)
    print("-" * 80)
    for name, data in results.items():
        m = data["metrics"]
        row = f"{name:<25} {m['mse']:>10.5f} {m['mae']:>10.5f} {m['rmse']:>10.5f} {m['mape']:>10.2f} {m['wape']:>10.2f}"
        print(row)
    print("=" * 80)


# ============================================================
# Colab / Jupyter 直接调用入口
# ============================================================
def run_all(
    data_path="data/Brooklyn_Bacaly_two_modal.csv",
    forecast_horizon=192,
    batch_size=64,
    max_epoch=5,
    lr=1e-4,
    device=None,
    save_dir="experiments/baseline_results",
    channel_names=None,
    n_channels=2,
    seed=42,
    skip_gca_lpt=False,
    gca_epochs=50,
    gca_lr=1e-3,
    use_graph=None,
):
    """一键运行全部实验（Colab / Jupyter 直接调用此函数）.

    Args:
        data_path: 数据集路径。
        forecast_horizon: 预测步长。
        batch_size: 批大小。
        max_epoch: Linear Probing 训练轮数。
        lr: Linear Probing 学习率。
        n_channels: 数据集的通道/变量数。
        skip_gca_lpt: 是否跳过 GCA-LPT 训练（训练耗时较长）。
        gca_epochs: GCA-LPT 训练轮数（默认 50，来自 notebook）。
        gca_lr: GCA-LPT 学习率（默认 1e-3，来自 notebook）。
        use_graph: 是否启用 Graph 模块。None=自动（通道>2时启用）。

    Colab 示例：
        from experiments.baseline_evaluation import run_all
        results = run_all(forecast_horizon=48, n_channels=2)
    """
    if channel_names is None:
        plot_count = min(n_channels, 5)
        channel_names = [f"ch_{i}" for i in range(plot_count)]

    set_seed(seed)

    if device is None:
        dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        dev = torch.device(device)
    dtype = torch.bfloat16 if dev.type == "cuda" else torch.float32

    if not Path(data_path).is_absolute():
        data_path = str(Path(PROJECT_ROOT) / data_path)
    abs_save_dir = save_dir if Path(save_dir).is_absolute() else str(Path(PROJECT_ROOT) / save_dir)

    print(f"Device: {dev}")
    print(f"Data: {data_path}")
    print(f"Forecast Horizon: {forecast_horizon}")

    results = {}

    # Experiment 1: Zero-shot
    zs_metrics, zs_trues, zs_preds, zs_hist = run_zero_shot(
        data_path, forecast_horizon, batch_size, dev, dtype
    )
    results["Zero-shot"] = {"metrics": zs_metrics, "trues": zs_trues, "preds": zs_preds, "histories": zs_hist}

    # Experiment 2: Linear Probing
    lp_metrics, lp_trues, lp_preds, lp_hist = run_linear_probing(
        data_path, forecast_horizon, batch_size, dev, max_epoch, lr, dtype
    )
    results["Linear Probing"] = {"metrics": lp_metrics, "trues": lp_trues, "preds": lp_preds, "histories": lp_hist}

    # Experiment 3: GCA-LPT (train from scratch)
    if not skip_gca_lpt:
        ft_metrics, ft_trues, ft_preds, ft_hist = run_gca_lpt(
            data_path, forecast_horizon, batch_size, dev, dtype, n_channels,
            gca_epochs=gca_epochs, gca_lr=gca_lr, save_dir=abs_save_dir,
            use_graph=use_graph,
        )
        results["GCA-LPT"] = {"metrics": ft_metrics, "trues": ft_trues, "preds": ft_preds, "histories": ft_hist}

    # Summary
    print_summary_table(results)

    # Save metrics JSON
    os.makedirs(abs_save_dir, exist_ok=True)
    metrics_only = {k: v["metrics"] for k, v in results.items()}
    with open(os.path.join(abs_save_dir, "metrics.json"), "w") as f:
        json.dump(metrics_only, f, indent=2, ensure_ascii=False)
    print(f"\nMetrics saved to {abs_save_dir}/metrics.json")

    # Visualizations
    plot_comparison(results, channel_names, abs_save_dir)

    print("\nDone!")
    return results


# ============================================================
# 终端命令行入口
# ============================================================
def main():
    if IN_NOTEBOOK:
        run_all()
        return

    import argparse

    parser = argparse.ArgumentParser(description="Baseline Comparison Experiments")
    parser.add_argument("--data_path", type=str, default="data/Brooklyn_Bacaly_two_modal.csv")
    parser.add_argument("--forecast_horizon", type=int, default=192)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_epoch", type=int, default=5, help="Linear Probing training epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Linear Probing learning rate")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--save_dir", type=str, default="experiments/baseline_results")
    parser.add_argument("--channel_names", type=str, nargs="+", default=None,
                        help="Channel names for plots; defaults to ch_0..ch_N")
    parser.add_argument("--n_channels", type=int, default=2, help="Number of channels (variables)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip_gca_lpt", action="store_true", help="Skip GCA-LPT training")
    parser.add_argument("--gca_epochs", type=int, default=50, help="GCA-LPT training epochs")
    parser.add_argument("--gca_lr", type=float, default=1e-3, help="GCA-LPT learning rate")
    parser.add_argument("--use_graph", type=str, default=None, choices=["true", "false"],
                        help="Enable graph module (default: auto, enabled when n_channels>2)")
    args = parser.parse_args()

    use_graph = None
    if args.use_graph is not None:
        use_graph = args.use_graph == "true"

    run_all(
        data_path=args.data_path,
        forecast_horizon=args.forecast_horizon,
        batch_size=args.batch_size,
        max_epoch=args.max_epoch,
        lr=args.lr,
        device=args.device,
        save_dir=args.save_dir,
        channel_names=args.channel_names,
        n_channels=args.n_channels,
        seed=args.seed,
        skip_gca_lpt=args.skip_gca_lpt,
        gca_epochs=args.gca_epochs,
        gca_lr=args.gca_lr,
        use_graph=use_graph,
    )


if __name__ == "__main__":
    main()
