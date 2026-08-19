"""RAG 管道 — 基于 FAISS 的历史事件知识库检索.

支持两种知识库模式：
  1. 原始专家知识库（存储绝对事件影响，legacy）
  2. 残差修正知识库（存储模型边际修正量，由 build_residual_rag.py 生成）

当 cfg.USE_RESIDUAL_KB=True 时：
  - 优先从残差知识库检索（返回 model correction 量级，更适合 MOMENT 微调模型）
  - 同时保留原始知识库作为补充参考
  - RAG 上下文中会清晰标注来源，LLM 可区分两类信息
"""

import logging
import os
from pathlib import Path
from typing import List, Optional

from agents import config as cfg
from agents.schemas import EventInfo

logger = logging.getLogger(__name__)


class RAGPipeline:
    """检索增强生成管道：从历史事件知识库中检索相似案例作为 LLM 上下文."""

    def __init__(
        self,
        knowledge_base_dir: str = None,
        embedding_model: str = None,
        top_k: int = None,
        use_residual_kb: bool = None,
    ):
        self.knowledge_base_dir = knowledge_base_dir or cfg.KNOWLEDGE_BASE_DIR
        self.embedding_model_name = embedding_model or cfg.EMBEDDING_MODEL
        self.top_k = top_k or cfg.RAG_TOP_K
        self.vectorstore = None
        self.embeddings = None

        # 残差知识库（双库模式）
        _use_residual = use_residual_kb if use_residual_kb is not None else getattr(cfg, "USE_RESIDUAL_KB", False)
        self._residual_kb_dir = getattr(cfg, "RESIDUAL_KB_DIR", None) if _use_residual else None
        self._residual_vectorstore = None

        self._init_embeddings()
        self._load_or_create_vectorstore()

    def _init_embeddings(self):
        try:
            from langchain_community.embeddings import HuggingFaceEmbeddings

            model_path = self._resolve_local_model(self.embedding_model_name)
            self.embeddings = HuggingFaceEmbeddings(
                model_name=model_path,
                model_kwargs={"device": cfg.DEVICE},
            )
            logger.info("Embedding model loaded: %s", model_path)
        except (ImportError, Exception) as e:
            logger.warning(
                "Embedding model unavailable (%s: %s). RAG retrieval will use fallback.",
                type(e).__name__, e,
            )
            self.embeddings = None

    @staticmethod
    def _resolve_local_model(model_name: str) -> str:
        """若 HF Hub 本地缓存中已有该模型，直接返回快照路径，避免联网验证。"""
        try:
            import os
            from huggingface_hub import constants as hf_c

            cache_dir = hf_c.HF_HUB_CACHE
            slug = "models--" + model_name.replace("/", "--")
            snapshots_dir = os.path.join(cache_dir, slug, "snapshots")
            if os.path.isdir(snapshots_dir):
                for snap in os.listdir(snapshots_dir):
                    snap_path = os.path.join(snapshots_dir, snap)
                    if os.path.isfile(os.path.join(snap_path, "config.json")):
                        logger.info("Using cached model at %s", snap_path)
                        return snap_path
        except Exception:
            pass
        return model_name

    def _load_vectorstore_from(self, directory: str):
        """从指定目录加载 FAISS 向量索引，失败返回 None."""
        index_path = Path(directory) / "index.faiss"
        if not index_path.exists() or self.embeddings is None:
            return None
        try:
            from langchain_community.vectorstores import FAISS
            vs = FAISS.load_local(directory, self.embeddings, allow_dangerous_deserialization=True)
            logger.info("FAISS index loaded from %s", directory)
            return vs
        except Exception as e:
            logger.warning("Failed to load FAISS index from %s: %s", directory, e)
            return None

    def _load_or_create_vectorstore(self):
        index_path = Path(self.knowledge_base_dir) / "index.faiss"
        if index_path.exists() and self.embeddings is not None:
            self.vectorstore = self._load_vectorstore_from(self.knowledge_base_dir)
            if self.vectorstore is None:
                logger.info("No existing FAISS index found. Call build_knowledge_base() to create one.")
        else:
            logger.info("No existing FAISS index found. Call build_knowledge_base() to create one.")

        # 加载残差知识库（如有配置）
        if self._residual_kb_dir:
            self._residual_vectorstore = self._load_vectorstore_from(self._residual_kb_dir)
            if self._residual_vectorstore is not None:
                n_docs = len(self._residual_vectorstore.index_to_docstore_id)
                logger.info("Residual KB loaded: %d docs from %s", n_docs, self._residual_kb_dir)
            else:
                logger.warning(
                    "Residual KB dir configured (%s) but FAISS index not found. "
                    "Run: python build_residual_rag.py",
                    self._residual_kb_dir,
                )

    def build_knowledge_base(self, documents: List[str], metadatas: Optional[List[dict]] = None):
        """从文档列表构建向量知识库."""
        if self.embeddings is None:
            logger.error("Cannot build knowledge base: embedding model not available.")
            return

        from langchain_community.vectorstores import FAISS
        from langchain_core.documents import Document

        docs = []
        for i, text in enumerate(documents):
            meta = metadatas[i] if metadatas and i < len(metadatas) else {}
            docs.append(Document(page_content=text, metadata=meta))

        self.vectorstore = FAISS.from_documents(docs, self.embeddings)

        os.makedirs(self.knowledge_base_dir, exist_ok=True)
        self.vectorstore.save_local(self.knowledge_base_dir)
        logger.info("Knowledge base built with %d documents, saved to %s", len(docs), self.knowledge_base_dir)

    def _search_vectorstore(
        self,
        vectorstore,
        query: str,
        fetch_k: int,
        cutoff_date: Optional[str],
        top_k: int,
    ) -> list:
        """从单个 vectorstore 检索，返回 (doc, score) 列表."""
        try:
            docs_and_scores = vectorstore.similarity_search_with_score(query, k=fetch_k)
        except Exception:
            docs = vectorstore.similarity_search(query, k=fetch_k)
            docs_and_scores = [(doc, None) for doc in docs]

        filtered = []
        skipped = 0
        for doc, score in docs_and_scores:
            meta = doc.metadata or {}
            if cutoff_date:
                ev_date = meta.get("event_date", "")
                if ev_date and ev_date >= cutoff_date:
                    skipped += 1
                    continue
            tier = meta.get("impact_tier", meta.get("tier", ""))
            if tier and tier not in ("A", "B"):
                skipped += 1
                continue
            filtered.append((doc, score))
        if skipped:
            logger.info("RAG filter: skipped %d docs (date/tier)", skipped)
        return filtered[:top_k]

    def retrieve_context(self, event: EventInfo, top_k: int = None,
                         cutoff_date: Optional[str] = None) -> str:
        """检索与事件相关的历史案例，返回带 score 和 metadata provenance 的上下文文本.

        当 USE_RESIDUAL_KB=True 时，同时从残差知识库和原始知识库检索：
          - 残差库文档（前缀 [Correction RAG]）告知 LLM 所需边际修正量（通常 0.5-3%）
          - 原始库文档（前缀 [ref=...]）提供历史事件参考（绝对影响，仅作背景）
          LLM prompt 被引导优先使用残差库的修正量。

        Args:
            event:       待分析的事件对象。
            top_k:       检索文档数量（默认使用 self.top_k）。
            cutoff_date: 时间截止日（YYYY-MM-DD）。防止测试期数据泄漏。
        """
        k = top_k or self.top_k
        fetch_k = k * 3 if cutoff_date else k

        query_parts = [event.title, event.content or "", event.event_time or ""]
        if event.location:
            query_parts.append(event.location)
        if event.event_type:
            query_parts.append(event.event_type)
        query = " ".join(p for p in query_parts if p).strip()

        context_parts = []
        ref_idx = 1

        # ── 1. 残差知识库（优先显示）─────────────────────────────────────────
        if self._residual_vectorstore is not None:
            res_docs = self._search_vectorstore(
                self._residual_vectorstore, query,
                fetch_k=min(fetch_k, 3),  # 残差库：最多3条，避免 prompt 过长
                cutoff_date=None,          # 残差库来自训练集，无需截止过滤
                top_k=min(k, 3),
            )
            for doc, score in res_docs:
                meta = doc.metadata or {}
                score_str = f"{score:.3f}" if score is not None else "N/A"
                med_corr = meta.get("median_correction")
                corr_hint = f" correction={med_corr:+.1%}" if med_corr is not None else ""
                header = (
                    f"[correction_rag={ref_idx} score={score_str}"
                    f" type={meta.get('event_type','?')}"
                    f" day={meta.get('day_type','?')}"
                    f" rank={meta.get('rank_group','?')}"
                    f"{corr_hint}]"
                )
                context_parts.append(f"{header}\n{doc.page_content}")
                ref_idx += 1
            logger.info(
                "Residual KB: retrieved %d docs for '%s'",
                len(res_docs), event.title[:40],
            )

        # ── 2. 原始专家知识库（背景参考）────────────────────────────────────
        if self.vectorstore is not None:
            orig_k = max(1, k - min(len(context_parts), 3))
            orig_docs = self._search_vectorstore(
                self.vectorstore, query,
                fetch_k=orig_k * 3 if cutoff_date else orig_k,
                cutoff_date=cutoff_date,
                top_k=orig_k,
            )
            for doc, score in orig_docs:
                meta = doc.metadata or {}
                score_str = f"{score:.3f}" if score is not None else "N/A"
                meta_tags = " ".join(
                    f"{kk}={v}"
                    for kk, v in [
                        ("type",    meta.get("event_type")),
                        ("borough", meta.get("borough")),
                        ("station", meta.get("station_complex_id")),
                        ("day",     meta.get("day_type")),
                    ]
                    if v is not None
                )
                header = f"[ref={ref_idx} score={score_str}" + (f" {meta_tags}" if meta_tags else "") + "]"
                context_parts.append(f"{header}\n{doc.page_content}")
                ref_idx += 1
            logger.info(
                "Original KB: retrieved %d docs for '%s'",
                len(orig_docs), event.title[:40],
            )

        if context_parts:
            return "\n---\n".join(context_parts)

        logger.warning("No vectorstore available. Returning empty context.")
        return ""

    def add_documents(self, documents: List[str], metadatas: Optional[List[dict]] = None):
        """向现有知识库追加文档."""
        if self.embeddings is None:
            logger.error("Cannot add documents: embedding model not available.")
            return

        from langchain_core.documents import Document

        docs = []
        for i, text in enumerate(documents):
            meta = metadatas[i] if metadatas and i < len(metadatas) else {}
            docs.append(Document(page_content=text, metadata=meta))

        if self.vectorstore is not None:
            self.vectorstore.add_documents(docs)
        else:
            from langchain_community.vectorstores import FAISS

            self.vectorstore = FAISS.from_documents(docs, self.embeddings)

        os.makedirs(self.knowledge_base_dir, exist_ok=True)
        self.vectorstore.save_local(self.knowledge_base_dir)
        logger.info("Added %d documents to knowledge base.", len(docs))
