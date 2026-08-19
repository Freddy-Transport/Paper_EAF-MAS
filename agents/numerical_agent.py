"""数值预测智能体 — 封装 Moment4Forecast 或 MOMENTPipeline（LP 回退）模型.

加载优先级：
  1. 用户上传的 GCA-LPT 微调权重（MODEL_PATH/model.safetensors）
     → 加载 Moment4Forecast
  2. 已缓存的 Linear Probing 回退权重（LP_MODEL_PATH/lp_weights.pt）
     → 加载 MOMENTPipeline + 恢复权重，无需重新训练
  3. 以上均不存在
     → 自动触发 Linear Probing 训练（调用 baseline_evaluation.train_and_save_lp_model），
        训练完成后缓存到 LP_MODEL_PATH，再加载使用
"""

import json
import logging
import os
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from agents import config as cfg
from agents.schemas import NumericalPrediction, PredictionRequest

logger = logging.getLogger(__name__)

# 标记不同模型来源，方便日志输出
_MODEL_TYPE_GCA = "GCA-LPT (user fine-tuned)"
_MODEL_TYPE_LP_CACHED = "Linear Probing (cached)"
_MODEL_TYPE_LP_NEW = "Linear Probing (newly trained)"


class NumericalPredictionAgent:
    """将微调后的 Moment4Forecast 或 MOMENTPipeline 封装为可独立调用的预测智能体.

    当用户提供了 GCA-LPT 微调权重时，直接加载使用；否则自动回退到
    官方 MOMENT 的 Linear Probing 策略——先检查是否有已缓存的 LP 权重，
    若没有则在 lp_data_path 指定的数据集上训练若干 epoch 并保存缓存。
    """

    def __init__(
        self,
        model_path: str = None,
        device: str = None,
        forecast_horizon: int = None,
        n_channels: int = None,
        channel_names: Optional[List[str]] = None,
        # LP 回退参数（仅在缺少自定义权重时生效）
        lp_model_path: str = None,
        lp_data_path: str = None,
        lp_max_epoch: int = None,
        lp_lr: float = None,
        lp_batch_size: int = None,
        allow_lp_training_fallback: bool = True,
    ):
        self.device = device or cfg.DEVICE
        self.forecast_horizon = forecast_horizon or cfg.FORECAST_HORIZON
        self.n_channels = n_channels or cfg.N_CHANNELS
        self.channel_names = channel_names or [f"channel_{i}" for i in range(self.n_channels)]
        self.model_path = model_path or cfg.MODEL_PATH

        # LP 回退参数
        self.lp_model_path = lp_model_path or cfg.LP_MODEL_PATH
        self.lp_data_path = lp_data_path or cfg.LP_DATA_PATH
        self.lp_max_epoch = lp_max_epoch or cfg.LP_MAX_EPOCH
        self.lp_lr = lp_lr or cfg.LP_LR
        self.lp_batch_size = lp_batch_size or cfg.LP_BATCH_SIZE
        self.allow_lp_training_fallback = bool(allow_lp_training_fallback)

        self.model_type: str = ""
        self.model = self._load_model()

    # ------------------------------------------------------------------
    # 权重检测工具
    # ------------------------------------------------------------------

    def _has_gca_weights(self) -> bool:
        """检查 GCA-LPT 微调权重是否存在（model.safetensors 或 pytorch_model.bin）."""
        p = Path(self.model_path)
        return (p / "model.safetensors").exists() or (p / "pytorch_model.bin").exists()

    def _has_lp_weights(self) -> bool:
        """检查 Linear Probing 缓存权重是否存在."""
        return (Path(self.lp_model_path) / "lp_weights.pt").exists()

    # ------------------------------------------------------------------
    # 模型加载
    # ------------------------------------------------------------------

    def _load_model(self):
        """按优先级加载模型：GCA-LPT → LP 缓存 → LP 新训练."""

        if self._has_gca_weights():
            logger.info("检测到 GCA-LPT 权重，从 %s 加载 Moment4Forecast", self.model_path)
            self.model_type = _MODEL_TYPE_GCA
            return self._load_gca_model()

        if self._has_lp_weights():
            logger.info("未检测到 GCA-LPT 权重，从缓存 %s 加载 Linear Probing 模型", self.lp_model_path)
            self.model_type = _MODEL_TYPE_LP_CACHED
            return self._load_lp_model()

        if not self.allow_lp_training_fallback:
            raise FileNotFoundError(
                "No GCA-LPT or cached LP MOMENT weights found. "
                f"Checked model_path={self.model_path!r} and lp_model_path={self.lp_model_path!r}. "
                "Strict paper experiments require real cached weights and do not train "
                "or use deterministic fallback as numerical_only."
            )

        logger.info("未检测到任何模型权重，启动 Linear Probing 自动训练...")
        self.model_type = _MODEL_TYPE_LP_NEW
        return self._train_lp_fallback()

    def _load_gca_model(self):
        """加载用户微调的 Moment4Forecast（GCA-LPT）模型."""
        from lpt_finetuning.moment_lpt import Moment4Forecast

        config_path = Path(self.model_path) / "config.json"
        with open(config_path, "r") as f:
            base_config = json.load(f)

        model_kwargs = {
            "task_name": "forecasting",
            "forecast_horizon": self.forecast_horizon,
            "num_prompt_tokens": cfg.NUM_PROMPT_TOKENS,
            "pg_dropout_prob": cfg.PG_DROPOUT_PROB,
            "max_channels": cfg.MAX_CHANNELS,
            "pg_num_heads": cfg.PG_NUM_HEADS,
            "use_graph": cfg.USE_GRAPH,
            "n_channels": self.n_channels,
            "n_gcn_layers": cfg.N_GCN_LAYERS,
            "graph_embed_dim": cfg.GRAPH_EMBED_DIM,
            "graph_dropout": cfg.GRAPH_DROPOUT,
        }

        model = Moment4Forecast.from_pretrained(
            self.model_path,
            config=base_config,
            model_kwargs=model_kwargs,
        )
        model.to(self.device).eval()
        logger.info("Moment4Forecast loaded from %s on %s", self.model_path, self.device)
        return model

    def _load_lp_model(self):
        """加载已缓存的 Linear Probing 权重到 MOMENTPipeline."""
        from momentfm import MOMENTPipeline

        weights_path = os.path.join(self.lp_model_path, "lp_weights.pt")

        # 读取缓存的配置（forecast_horizon 等）
        meta_path = os.path.join(self.lp_model_path, "lp_config.json")
        if os.path.exists(meta_path):
            with open(meta_path, "r") as f:
                meta = json.load(f)
            saved_horizon = meta.get("forecast_horizon", self.forecast_horizon)
            if saved_horizon != self.forecast_horizon:
                logger.warning(
                    "缓存 LP 模型的 forecast_horizon=%d 与当前设置 %d 不一致，"
                    "将重新训练。",
                    saved_horizon, self.forecast_horizon,
                )
                return self._train_lp_fallback()

        import time
        _t0 = time.time()
        logger.info("[LP加载 1/4] 从本地缓存初始化 MOMENT-1-large 基础模型...")
        print("[LP加载 1/4] 初始化 MOMENT-1-large 基础模型（~1.3GB，请稍候）...", flush=True)
        model = MOMENTPipeline.from_pretrained(
            "AutonLab/MOMENT-1-large",
            model_kwargs={
                "task_name": "forecasting",
                "forecast_horizon": self.forecast_horizon,
                "freeze_encoder": True,
                "freeze_embedder": True,
                "freeze_head": False,
            },
        )
        print(f"[LP加载 2/4] 模型结构初始化 (init)... [{time.time()-_t0:.1f}s]", flush=True)
        model.init()

        weights_size_mb = os.path.getsize(weights_path) / 1024 / 1024
        print(f"[LP加载 3/4] 读取 LP 权重文件 ({weights_size_mb:.0f}MB)... [{time.time()-_t0:.1f}s]", flush=True)
        state_dict = torch.load(weights_path, map_location=self.device)

        print(f"[LP加载 4/4] 加载权重到模型并迁移至 {self.device}... [{time.time()-_t0:.1f}s]", flush=True)
        model.load_state_dict(state_dict)
        model.to(self.device).eval()
        print(f"[LP加载完成] 耗时 {time.time()-_t0:.1f}s，设备: {self.device}", flush=True)
        logger.info("LP MOMENTPipeline loaded from %s on %s (%.1fs)", weights_path, self.device, time.time()-_t0)
        return model

    def _train_lp_fallback(self):
        """调用 baseline_evaluation.train_and_save_lp_model 训练并缓存 LP 模型."""
        from experiments.baseline_evaluation import train_and_save_lp_model

        # 确保 data_path 使用绝对路径
        data_path = self.lp_data_path
        if not Path(data_path).is_absolute():
            from experiments.baseline_evaluation import PROJECT_ROOT
            data_path = str(Path(PROJECT_ROOT) / data_path)

        if not Path(data_path).exists():
            raise FileNotFoundError(
                f"LP 回退训练数据集不存在：{data_path}\n"
                f"请通过 lp_data_path 参数或 config.LP_DATA_PATH 指定有效的数据集路径。"
            )

        model, metrics = train_and_save_lp_model(
            data_path=data_path,
            forecast_horizon=self.forecast_horizon,
            save_dir=self.lp_model_path,
            batch_size=self.lp_batch_size,
            max_epoch=self.lp_max_epoch,
            lr=self.lp_lr,
            device=self.device,
        )
        model.to(self.device).eval()
        logger.info(
            "LP fallback training done. MSE=%.5f MAE=%.5f — saved to %s",
            metrics["mse"], metrics["mae"], self.lp_model_path,
        )
        return model

    # ------------------------------------------------------------------
    # 数据加载
    # ------------------------------------------------------------------

    def _load_dataset(self, data_path: str, split: str = "test"):
        from momentfm.data.informer_dataset import InformerDataset

        dataset = InformerDataset(
            dataset_name=data_path,
            forecast_horizon=self.forecast_horizon,
            data_split=split,
            data_stride_len=1,
            task_name="forecasting",
        )
        return dataset

    # ------------------------------------------------------------------
    # 预测接口
    # ------------------------------------------------------------------

    @staticmethod
    def _get_forecast_timestamps(data_path: str, dataset, n_steps: int):
        """从原始 CSV 中提取预测窗口对应的时间戳列表（ISO 格式字符串）."""
        try:
            import pandas as pd
            df = pd.read_csv(data_path)
            if "date" not in df.columns:
                return None
            _, test_slice = dataset._get_borders()
            test_start_abs = test_slice.start
            # 最后测试窗口的预测段 = test split 末尾 n_steps 行
            forecast_start_abs = test_start_abs + dataset.length_timeseries - n_steps
            dates = df["date"].iloc[forecast_start_abs: forecast_start_abs + n_steps]
            return [str(d) for d in dates.tolist()]
        except Exception as e:
            logger.warning("无法提取预测时间戳: %s", e)
            return None

    @staticmethod
    def _inverse_transform(scaler, forecast: np.ndarray) -> np.ndarray:
        """将归一化预测值还原为原始量纲.

        Args:
            scaler: sklearn StandardScaler，已在训练集上 fit。
            forecast: 形状 (C, H)，C=通道数，H=预测步长。
        Returns:
            形状 (C, H) 的反归一化预测值。
        """
        # scaler 期望输入形状 (N, C)，因此先转置为 (H, C) 再反变换，最后转回 (C, H)
        return scaler.inverse_transform(forecast.T).T

    @torch.no_grad()
    def predict_for_date(
        self,
        data_path: str,
        target_date: str,
        forecast_horizon: int = None,
    ) -> NumericalPrediction:
        """以指定日期为预测窗口起始，运行 MOMENT 推理并返回预测+真实值.

        直接从 CSV 按日期切片，不依赖 InformerDataset 内部索引计算，
        适用于任意合法日期的按需预测（论文在线自适应学习演示）。

        Args:
            data_path:        CSV 文件路径（含 date 列）。
            target_date:      预测窗口起始日（YYYY-MM-DD 或 YYYY-MM-DD HH:MM:SS）。
                              系统取 >= target_date 的第一个时间步作为预测起始点。
            forecast_horizon: 预测步数（默认使用 self.forecast_horizon）。

        Returns:
            NumericalPrediction，含 forecast、ground_truth、forecast_timestamps。

        Raises:
            ValueError: 若 target_date 之前历史不足 512 步，或之后真实值不足 1 步。
        """
        import pandas as pd
        from sklearn.preprocessing import StandardScaler

        h = forecast_horizon or self.forecast_horizon
        seq_len = 512   # MOMENT 固定输入长度（与 InformerDataset.seq_len 一致）
        n_train = 12 * 30 * 24  # = 8640，训练集行数（用于 scaler fit）

        # 1. 加载 CSV
        df = pd.read_csv(data_path)
        if "date" not in df.columns:
            raise ValueError(f"CSV 缺少 'date' 列：{data_path}")

        dates = pd.to_datetime(df["date"])
        target_dt = pd.to_datetime(target_date)

        # 2. 找预测起始行（forecast_start_idx）
        mask = dates >= target_dt
        if not mask.any():
            raise ValueError(
                f"target_date '{target_date}' 超出数据集末尾 ({dates.iloc[-1]})。"
            )
        forecast_start_idx = int(mask.idxmax())  # 第一个 >= target_date 的行

        # 3. 检查历史数据是否充足（需要 seq_len = 512 步）
        hist_start_idx = forecast_start_idx - seq_len
        if hist_start_idx < 0:
            raise ValueError(
                f"target_date '{target_date}' 距数据集起点不足 {seq_len} 步历史，"
                f"请选择 {dates.iloc[seq_len]} 之后的日期。"
            )

        # 4. 检查预测段是否有真实值（至少 1 步）
        forecast_end_idx = min(forecast_start_idx + h, len(df))
        actual_h = forecast_end_idx - forecast_start_idx
        if actual_h <= 0:
            raise ValueError(
                f"target_date '{target_date}' 已超出或贴近数据集末尾，无可用真实值。"
            )

        # 5. 提取数值列（去掉 date 列），使用三次插值填充缺失值（与 InformerDataset 一致）
        data_df = df.drop(columns=["date"]).infer_objects(copy=False).interpolate(method="cubic")
        data_np = data_df.values  # (N, C)

        # 6. 用训练集拟合 StandardScaler（与 InformerDataset._read_data 逻辑完全一致）
        scaler = StandardScaler()
        scaler.fit(data_np[:n_train])

        # 7. 归一化整列，切出 history 和 ground truth
        data_norm = scaler.transform(data_np)           # (N, C)
        history_norm = data_norm[hist_start_idx : forecast_start_idx]  # (seq_len, C)
        gt_norm = data_norm[forecast_start_idx : forecast_end_idx]      # (actual_h, C)

        # 8. 构建模型输入张量 (1, C, seq_len)
        timeseries = torch.tensor(
            history_norm.T[np.newaxis, :, :],  # (1, C, 512)
            dtype=torch.float32,
        ).to(self.device)
        input_mask = torch.ones(1, seq_len, dtype=torch.float32).to(self.device)

        # 9. 前向推理
        output = self.model(x_enc=timeseries, input_mask=input_mask)
        pred_norm = output.forecast.cpu().numpy()[0]  # (C, H)

        # 10. 反归一化
        pred = self._inverse_transform(scaler, pred_norm)   # (C, H)
        # gt_norm shape: (actual_h, C) → (C, actual_h)
        gt_ch = gt_norm.T                                   # (C, actual_h)
        gt = scaler.inverse_transform(gt_ch.T).T            # (C, actual_h)

        # 11. 时间戳列表（预测窗口）
        timestamps = [
            str(dates.iloc[i])
            for i in range(forecast_start_idx, forecast_end_idx)
        ]

        channel_names = self.channel_names[: pred.shape[0]]

        logger.info(
            "predict_for_date: target=%s | forecast=[%s ~ %s] | channels=%d",
            target_date, timestamps[0][:16], timestamps[-1][:16], len(channel_names),
        )
        return NumericalPrediction(
            forecast=pred.tolist(),
            channel_names=channel_names,
            ground_truth=gt.tolist() if actual_h > 0 else None,
            forecast_timestamps=timestamps,
        )

    @torch.no_grad()
    def predict(self, request: PredictionRequest) -> NumericalPrediction:
        """对给定数据执行单次预测（返回最后一个测试窗口的预测结果，已反归一化）.

        若 request.target_date 非空，转发至 predict_for_date() 以支持指定日期预测。
        """
        # ── 指定日期模式（论文实验接口）──────────────────────────────────
        if request.target_date:
            return self.predict_for_date(
                data_path=request.history_data_path,
                target_date=request.target_date,
                forecast_horizon=request.forecast_horizon or self.forecast_horizon,
            )

        # ── 默认模式：最后一个测试窗口 ────────────────────────────────────
        dataset = self._load_dataset(request.history_data_path, split="test")
        loader = DataLoader(dataset, batch_size=1, shuffle=False)

        all_forecasts = []
        last_gt_raw = None
        for batch in loader:
            timeseries, forecast_gt, input_mask = batch
            timeseries = timeseries.float().to(self.device)
            input_mask = input_mask.to(self.device)

            output = self.model(x_enc=timeseries, input_mask=input_mask)
            pred = output.forecast.cpu().numpy()  # (B, C, H)
            all_forecasts.append(pred)
            last_gt_raw = forecast_gt.cpu().numpy()  # (B, C, H) 保留最后一批的真实值

        forecasts = np.concatenate(all_forecasts, axis=0)
        last_forecast = forecasts[-1]  # (C, H)
        last_gt = last_gt_raw[-1] if last_gt_raw is not None else None  # (C, H)

        # 反归一化：还原为原始客流量纲
        last_forecast = self._inverse_transform(dataset.scaler, last_forecast)
        if last_gt is not None:
            last_gt = self._inverse_transform(dataset.scaler, last_gt)

        n_steps = last_forecast.shape[1]
        channel_names = self.channel_names[: last_forecast.shape[0]]
        timestamps = self._get_forecast_timestamps(request.history_data_path, dataset, n_steps)

        return NumericalPrediction(
            forecast=last_forecast.tolist(),
            channel_names=channel_names,
            ground_truth=last_gt.tolist() if last_gt is not None else None,
            forecast_timestamps=timestamps,
        )

    @torch.no_grad()
    def predict_all(self, request: PredictionRequest) -> List[NumericalPrediction]:
        """对所有测试窗口执行预测（每个窗口结果均已反归一化）."""
        dataset = self._load_dataset(request.history_data_path, split="test")
        loader = DataLoader(dataset, batch_size=8, shuffle=False)

        results = []
        for batch in loader:
            timeseries, forecast_gt, input_mask = batch
            timeseries = timeseries.float().to(self.device)
            input_mask = input_mask.to(self.device)

            output = self.model(x_enc=timeseries, input_mask=input_mask)
            preds = output.forecast.cpu().numpy()  # (B, C, H)

            for pred in preds:
                # 反归一化：还原为原始客流量纲
                pred = self._inverse_transform(dataset.scaler, pred)
                results.append(
                    NumericalPrediction(
                        forecast=pred.tolist(),
                        channel_names=self.channel_names[: pred.shape[0]],
                    )
                )
        return results

    def __repr__(self) -> str:
        return (
            f"NumericalPredictionAgent("
            f"model_type='{self.model_type}', "
            f"forecast_horizon={self.forecast_horizon}, "
            f"n_channels={self.n_channels}, "
            f"device='{self.device}')"
        )
