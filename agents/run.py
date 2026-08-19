"""多智能体交通预测系统 — 主入口脚本.

Usage:
    python -m agents.run \
        --data_path data/Brooklyn_Bacaly_two_modal.csv \
        --event_path data/text_feat_0802.24.csv \
        --forecast_horizon 192 \
        --device cuda:0
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List, Optional

# 将项目根目录加入 sys.path
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from agents import config as cfg
from agents.orchestrator import AgentOrchestrator


def parse_args():
    parser = argparse.ArgumentParser(description="Multi-Agent Traffic Prediction System")

    parser.add_argument(
        "--data_path",
        type=str,
        default="data/Brooklyn_Bacaly_two_modal.csv",
        help="Path to historical traffic data CSV",
    )
    parser.add_argument(
        "--event_path",
        type=str,
        default=None,
        help="Path to event data file (Excel/CSV/JSON)",
    )
    parser.add_argument(
        "--forecast_horizon",
        type=int,
        default=cfg.FORECAST_HORIZON,
        help="Number of forecast steps",
    )
    parser.add_argument(
        "--n_channels",
        type=int,
        default=cfg.N_CHANNELS,
        help="Number of data channels",
    )
    parser.add_argument(
        "--channel_names",
        type=str,
        nargs="+",
        default=None,
        help="Names for each channel, e.g. subway taxi",
    )
    parser.add_argument(
        "--channel_map",
        type=str,
        default=None,
        help="JSON channel map with Top-N station channel names",
    )
    parser.add_argument(
        "--event_relevance_threshold",
        type=float,
        default=0.45,
        help="Minimum event relevance score sent to LLM/RAG",
    )
    parser.add_argument(
        "--max_llm_events_per_window",
        type=int,
        default=80,
        help="Maximum high-relevance events analyzed by LLM in one forecast window",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=cfg.DEVICE,
        help="Compute device (cuda:0 / cpu)",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=cfg.MODEL_PATH,
        help="Path to fine-tuned model directory",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file path (optional)",
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )

    return parser.parse_args()


def plot_forecast(
    result,
    output_path: str,
    timestamps: Optional[List[str]] = None,
) -> None:
    """绘制预测值 vs 真实值对比折线图，并标注事件位置。"""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
        import pandas as pd
        import numpy as np
    except ImportError:
        logging.getLogger("agents.run").warning("matplotlib 未安装，跳过可视化。")
        return

    raw = result.raw_forecast           # (C, H)
    adj = result.adjusted_forecast      # (C, H)
    gt  = getattr(result, "ground_truth", None)   # (C, H) or None
    channel_names = getattr(result, "channel_names", None) or [
        f"channel_{i}" for i in range(len(raw))
    ]
    n_channels = len(raw)

    # 构建时间轴
    if timestamps:
        try:
            x = pd.to_datetime(timestamps)
        except Exception:
            x = np.arange(len(raw[0]))
    else:
        x = np.arange(len(raw[0]))

    # 收集事件标注信息
    events_in_window = result.events_considered or []

    fig, axes = plt.subplots(
        n_channels, 1,
        figsize=(14, 4 * n_channels),
        sharex=True,
        squeeze=False,
    )

    colors = {"gt": "#2196F3", "raw": "#FF9800", "adj": "#4CAF50"}
    event_colors = ["#E91E63", "#9C27B0", "#F44336", "#00BCD4", "#FF5722"]

    for ch_idx, ax in enumerate([axes[i][0] for i in range(n_channels)]):
        ch_name = channel_names[ch_idx] if ch_idx < len(channel_names) else f"ch{ch_idx}"

        raw_vals = raw[ch_idx]
        adj_vals = adj[ch_idx]

        if gt and ch_idx < len(gt):
            gt_vals = gt[ch_idx]
            ax.plot(x, gt_vals, color=colors["gt"], linewidth=1.8,
                    label="Ground Truth", zorder=3)

        ax.plot(x, raw_vals, color=colors["raw"], linewidth=1.5,
                linestyle="--", label="Raw Prediction", zorder=2)

        # 仅在有事件调整时才画调整后折线
        if raw_vals != adj_vals:
            ax.plot(x, adj_vals, color=colors["adj"], linewidth=1.5,
                    linestyle="-.", label="Adjusted Prediction", zorder=2)

        # 标注事件（垂直虚线，只用英文日期避免字体问题）
        for ev_idx, ev in enumerate(events_in_window):
            if not ev.event_time:
                continue
            try:
                ev_dt = pd.to_datetime(ev.event_time)
                c = event_colors[ev_idx % len(event_colors)]
                ax.axvline(ev_dt, color=c, linewidth=1.2, linestyle=":", alpha=0.8)
                label = ev_dt.strftime("Event %m/%d %H:00")
                ax.text(
                    ev_dt, ax.get_ylim()[1] * 0.95,
                    label,
                    color=c, fontsize=6, rotation=45,
                    va="top", ha="left",
                )
            except Exception:
                pass

        ax.set_title(f"{ch_name.capitalize()} — Forecast vs Ground Truth", fontsize=11)
        ax.set_ylabel("Count", fontsize=9)
        ax.legend(loc="upper left", fontsize=8)
        ax.grid(True, alpha=0.3)

        if hasattr(x, "dtype") and hasattr(x, "__len__"):
            try:
                ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d %H:00"))
                ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
                fig.autofmt_xdate(rotation=30)
            except Exception:
                pass

    axes[-1][0].set_xlabel("Time", fontsize=9)
    fig.suptitle("Multi-Channel Traffic Forecast", fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logging.getLogger("agents.run").info("Forecast plot saved to %s", output_path)


def load_channel_names_from_map(channel_map_path: str) -> List[str]:
    with open(channel_map_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    channels = payload.get("channels", payload if isinstance(payload, list) else [])
    return [str(row["channel_name"]) for row in channels if row.get("channel_name")]


def main():
    args = parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger = logging.getLogger("agents.run")

    # 处理相对路径
    data_path = args.data_path
    if not Path(data_path).is_absolute():
        data_path = str(Path(cfg.PROJECT_ROOT) / data_path)

    event_path = args.event_path
    if event_path and not Path(event_path).is_absolute():
        event_path = str(Path(cfg.PROJECT_ROOT) / event_path)

    logger.info("Data path: %s", data_path)
    logger.info("Event path: %s", event_path)
    channel_map = args.channel_map
    if channel_map and not Path(channel_map).is_absolute():
        channel_map = str(Path(cfg.PROJECT_ROOT) / channel_map)

    channel_names = args.channel_names
    if channel_map and channel_names is None:
        channel_names = load_channel_names_from_map(channel_map)
        args.n_channels = len(channel_names)

    logger.info("Device: %s", args.device)
    logger.info("Channels: %d", args.n_channels)

    # 初始化协调器
    orchestrator = AgentOrchestrator.from_config(
        model_path=args.model_path,
        device=args.device,
        channel_names=channel_names,
        n_channels=args.n_channels,
        forecast_horizon=args.forecast_horizon,
        channel_map_path=channel_map,
        event_relevance_threshold=args.event_relevance_threshold,
        max_llm_events_per_window=args.max_llm_events_per_window,
    )

    # 运行预测流程
    result = orchestrator.run(
        data_path=data_path,
        event_source=event_path,
        forecast_horizon=args.forecast_horizon,
        n_channels=args.n_channels,
        channel_map_path=channel_map,
        event_relevance_threshold=args.event_relevance_threshold,
        max_llm_events_per_window=args.max_llm_events_per_window,
    )

    # 输出结果
    result_dict = result.model_dump()

    if args.output:
        output_path = args.output
        if not Path(output_path).is_absolute():
            output_path = str(Path(cfg.PROJECT_ROOT) / output_path)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result_dict, f, ensure_ascii=False, indent=2)
        logger.info("Results saved to %s", output_path)

        # 生成可视化图表（与 JSON 同目录，同名 .png）
        plot_path = str(Path(output_path).with_suffix(".png"))
        plot_forecast(result, plot_path, timestamps=result.forecast_timestamps)
    else:
        print("\n" + "=" * 60)
        print("PREDICTION RESULTS")
        print("=" * 60)
        print(f"Channels: {len(result.raw_forecast)}")
        print(f"Forecast horizon: {len(result.raw_forecast[0]) if result.raw_forecast else 0}")
        print(f"Events considered: {len(result.events_considered)}")
        print(f"\n{result.explanation}")

        # 无指定输出路径时保存到项目根目录
        plot_path = str(Path(cfg.PROJECT_ROOT) / "forecast_plot.png")
        plot_forecast(result, plot_path, timestamps=result.forecast_timestamps)


if __name__ == "__main__":
    main()
