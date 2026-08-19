"""事件数据采集器 — 从本地文件或网络 API 获取事件."""

import json
import logging
from pathlib import Path
from typing import List, Optional

from agents.schemas import EventInfo

logger = logging.getLogger(__name__)


class EventFetcher:
    """统一的事件采集接口，支持 Excel / CSV / JSON / API 等多种来源."""

    def fetch(self, source: str, **kwargs) -> List[EventInfo]:
        """根据来源路径自动选择读取方式."""
        path = Path(source)
        if path.exists():
            suffix = path.suffix.lower()
            if suffix in (".xlsx", ".xls"):
                return self.fetch_from_excel(source, **kwargs)
            elif suffix == ".csv":
                return self._try_excel_or_csv(source, **kwargs)
            elif suffix == ".json":
                return self.fetch_from_json(source, **kwargs)
            else:
                logger.warning("Unsupported file format: %s", suffix)
                return []
        else:
            logger.warning("Source path does not exist: %s", source)
            return []

    def _try_excel_or_csv(self, path: str, **kwargs) -> List[EventInfo]:
        """有些 .csv 文件实际上是 Excel 格式（如 text_feat_0802.24.csv）."""
        try:
            return self.fetch_from_excel(path, **kwargs)
        except Exception:
            return self.fetch_from_csv(path, **kwargs)

    def fetch_from_excel(
        self,
        path: str,
        date_col: Optional[str] = None,
        title_col: Optional[str] = None,
        content_col: Optional[str] = None,
        type_col: Optional[str] = None,
    ) -> List[EventInfo]:
        """读取 Excel 格式的事件数据（如巴克莱中心活动）."""
        import re
        import pandas as pd

        df = pd.read_excel(path, engine="openpyxl")
        logger.info("Loaded %d rows from Excel: %s (columns: %s)", len(df), path, list(df.columns))

        col_names = [c.lower().strip() for c in df.columns]
        col_map = dict(zip(col_names, df.columns))

        date_col = date_col or self._find_col(col_map, ["date", "time", "event_date", "event_time", "时间", "日期"])
        title_col = title_col or self._find_col(col_map, ["event_name", "title", "name", "事件", "event", "标题"])
        content_col = content_col or self._find_col(col_map, ["description", "content", "detail", "desc", "内容", "描述"])
        type_col = type_col or self._find_col(col_map, ["type", "category", "event_type", "类型"])
        # 专门识别独立的"年份"列，用于补全无年份的日期字符串
        year_col = self._find_col(col_map, ["年份", "year"])

        events = []
        for _, row in df.iterrows():
            title = str(row.get(title_col, "")) if title_col else ""
            content = str(row.get(content_col, title)) if content_col else title
            raw_time = str(row.get(date_col, "")) if date_col else ""
            event_type = str(row.get(type_col, "")) if type_col else None

            if not title and not content:
                continue

            # 解析时间：格式可能是 "February 08 | 7:30PM"，年份在独立列
            event_time = self._parse_event_time(raw_time, row, year_col)

            events.append(
                EventInfo(
                    source="barclays_center",
                    title=title,
                    content=content if content else title,
                    event_time=event_time,
                    location="Barclays Center, Brooklyn",
                    event_type=event_type,
                )
            )
        logger.info("Parsed %d events from %s", len(events), path)
        return events

    @staticmethod
    def _parse_event_time(raw_time: str, row, year_col: Optional[str]) -> str:
        """将原始时间字符串（如 'February 08 | 7:30PM'）解析为 ISO 格式 datetime 字符串."""
        import re
        import pandas as pd

        if not raw_time or raw_time in ("nan", "None", ""):
            return ""

        # 尝试直接解析（标准格式）
        try:
            dt = pd.to_datetime(raw_time)
            if dt.year > 2000:
                return dt.strftime("%Y-%m-%d %H:%M:%S")
        except Exception:
            pass

        # 处理 "Month DD | HH:MMam/pm" 格式
        year = None
        if year_col and year_col in row.index:
            try:
                year = int(row[year_col])
            except Exception:
                pass

        # 去掉 " | " 分隔符，拼接年份后再解析
        cleaned = re.sub(r"\s*\|\s*", " ", raw_time).strip()
        if year:
            cleaned = f"{cleaned} {year}"
        try:
            dt = pd.to_datetime(cleaned, dayfirst=False)
            if dt.year < 2000 and year:
                dt = dt.replace(year=year)
            return dt.strftime("%Y-%m-%d %H:%M:%S")
        except Exception:
            pass

        return raw_time  # 解析失败则返回原始字符串

    def fetch_from_csv(self, path: str, **kwargs) -> List[EventInfo]:
        """读取标准 CSV 格式的事件数据."""
        import pandas as pd

        df = pd.read_csv(path)
        logger.info("Loaded %d rows from CSV: %s", len(df), path)

        col_names = [c.lower().strip() for c in df.columns]
        col_map = dict(zip(col_names, df.columns))

        date_col = self._find_col(col_map, ["date", "time", "event_date", "event_time"])
        title_col = self._find_col(col_map, ["event_name", "title", "name", "event"])
        content_col = self._find_col(col_map, ["description", "content", "detail"])

        events = []
        for _, row in df.iterrows():
            title = str(row.get(title_col, "")) if title_col else ""
            content = str(row.get(content_col, title)) if content_col else title
            event_time = str(row.get(date_col, "")) if date_col else ""

            if not title and not content:
                continue

            events.append(
                EventInfo(
                    source="csv",
                    title=title,
                    content=content if content else title,
                    event_time=event_time,
                )
            )
        return events

    def fetch_from_json(self, path: str, **kwargs) -> List[EventInfo]:
        """读取 JSON 格式的事件数据."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, dict):
            data = data.get("events", [data])
        if not isinstance(data, list):
            data = [data]

        events = []
        for item in data:
            events.append(
                EventInfo(
                    source=item.get("source", "json"),
                    title=item.get("title", ""),
                    content=item.get("content", item.get("title", "")),
                    event_time=item.get("event_time", item.get("date", "")),
                    location=item.get("location"),
                    event_type=item.get("event_type", item.get("type")),
                    event_id=item.get("event_id"),
                    station_complex_id=item.get("station_complex_id"),
                    station_complex=item.get("station_complex"),
                    channel_name=item.get("channel_name"),
                    distance_m=item.get("distance_m"),
                    venue_name=item.get("venue_name"),
                )
            )
        return events

    @staticmethod
    def _find_col(col_map: dict, candidates: List[str]) -> Optional[str]:
        """在列名映射中查找匹配的列."""
        for c in candidates:
            if c in col_map:
                return col_map[c]
        return None
