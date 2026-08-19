"""技能库 — 技能的持久化、检索与置信度管理（AutoSkill 核心存储层）."""

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from agents.schemas import Skill, SkillType

logger = logging.getLogger(__name__)


class SkillLibrary:
    """以 JSONL 文件为后端的技能持久化与检索引擎。

    每行存储一个 Skill 的 JSON 序列化对象；skill_id 作为唯一键，
    重复保存时覆盖旧记录（全量重写文件）。
    """

    def __init__(self, library_path: str):
        self.library_path = Path(library_path)
        self._skills: Dict[str, Skill] = {}
        self._load()

    # ------------------------------------------------------------------
    # 持久化层
    # ------------------------------------------------------------------

    def _load(self):
        """从 JSONL 文件加载所有技能到内存."""
        if not self.library_path.exists():
            os.makedirs(self.library_path.parent, exist_ok=True)
            logger.info("SkillLibrary: no existing file at %s, starting fresh.", self.library_path)
            return

        loaded, skipped = 0, 0
        with open(self.library_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    skill = Skill(**data)
                    self._skills[skill.skill_id] = skill
                    loaded += 1
                except Exception as e:
                    logger.warning("Skipping malformed skill record: %s", e)
                    skipped += 1

        logger.info(
            "SkillLibrary loaded %d skills (%d skipped) from %s",
            loaded, skipped, self.library_path,
        )

    def _flush(self):
        """将内存中所有技能全量写回 JSONL 文件."""
        os.makedirs(self.library_path.parent, exist_ok=True)
        with open(self.library_path, "w", encoding="utf-8") as f:
            for skill in self._skills.values():
                f.write(skill.model_dump_json() + "\n")

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def save(self, skill: Skill):
        """新增或更新技能（按 skill_id 去重）。"""
        skill.last_updated = datetime.now(timezone.utc).isoformat()
        self._skills[skill.skill_id] = skill
        self._flush()
        logger.info(
            "Skill saved: [%s] %s  confidence=%.2f  use=%d",
            skill.skill_type.value, skill.skill_id, skill.confidence, skill.use_count,
        )

    def get(self, skill_id: str) -> Optional[Skill]:
        """按 ID 查询单个技能."""
        return self._skills.get(skill_id)

    def load_all(self) -> List[Skill]:
        """返回全部技能列表."""
        return list(self._skills.values())

    def delete(self, skill_id: str):
        """删除指定技能."""
        if skill_id in self._skills:
            del self._skills[skill_id]
            self._flush()
            logger.info("Skill deleted: %s", skill_id)

    def __len__(self) -> int:
        return len(self._skills)

    # ------------------------------------------------------------------
    # 检索
    # ------------------------------------------------------------------

    def query(
        self,
        context: dict,
        skill_type: Optional[SkillType] = None,
        min_confidence: float = 0.2,
    ) -> List[Skill]:
        """按上下文字典与可选类型过滤匹配技能。

        匹配规则：技能 trigger 中的每个 key-value 均须在 context 中出现
        （字符串子串匹配 / 列表成员匹配）。
        返回按置信度降序排列的技能列表。
        """
        matched = []
        for skill in self._skills.values():
            if skill.confidence < min_confidence:
                continue
            if skill_type is not None and skill.skill_type != skill_type:
                continue
            if self._matches(skill.trigger, context):
                matched.append(skill)

        matched.sort(key=lambda s: s.confidence, reverse=True)
        return matched

    @staticmethod
    def _matches(trigger: dict, context: dict) -> bool:
        """判断 trigger 的所有条件是否都被 context 满足。"""
        for k, v in trigger.items():
            ctx_val = context.get(k)
            if ctx_val is None:
                return False
            if isinstance(ctx_val, list):
                # 列表成员：只要有一项包含 trigger 值即算匹配
                if not any(str(v).lower() in str(item).lower() for item in ctx_val):
                    return False
            else:
                if str(v).lower() not in str(ctx_val).lower():
                    return False
        return True

    # ------------------------------------------------------------------
    # 置信度更新（贝叶斯风格增量更新）
    # ------------------------------------------------------------------

    def update_confidence(self, skill_id: str, supported: bool):
        """根据新经验更新技能置信度。

        - 支持（实际误差方向与技能一致）→ 置信度 +0.05，上限 1.0
        - 反驳（实际误差与技能预期相反）→ 置信度 -0.10，下限 0.0
        """
        skill = self._skills.get(skill_id)
        if skill is None:
            logger.warning("update_confidence: skill '%s' not found.", skill_id)
            return

        skill.use_count += 1
        if supported:
            skill.support_count += 1
            skill.confidence = min(1.0, skill.confidence + 0.05)
        else:
            skill.confidence = max(0.0, skill.confidence - 0.10)

        self.save(skill)
        logger.debug(
            "Skill '%s' confidence updated: supported=%s → %.2f",
            skill_id, supported, skill.confidence,
        )
