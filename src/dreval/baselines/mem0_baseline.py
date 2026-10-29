"""Baseline: Mem0 (open-source ``mem0ai``) stores the record as memories; the model answers from what it retrieves.

Each episode is added in time order as one user message (Mem0 extracts memories from it with the same model
as the other baseline); a deleted source's memories are deleted when its deletion date passes. To answer, each
question is a Mem0 search (top 20), and the model answers from the retrieved memories only. Embeddings are
local (FastEmbed ``BAAI/bge-small-zh-v1.5``), the vector store a local Qdrant directory per persona and run.
"""

from __future__ import annotations

import logging
import os
import tempfile
import threading
from typing import Any

# Mem0 sends usage telemetry and keeps a shared store for it under ~/.mem0, which concurrent instances fight over.
os.environ.setdefault("MEM0_TELEMETRY", "False")
os.environ.setdefault("MEM0_DIR", tempfile.mkdtemp(prefix="dreval-mem0-home-"))

from mem0 import Memory  # noqa: E402
from mem0.llms.base import LLMBase

from dreval.baselines.common import ANSWER_SCHEMA, GUIDE, Question, ask_text
from dreval.baselines.fullcontext import PREDICT_GUIDE, PREDICT_SCHEMA
from dreval.llm import Codex

logging.getLogger("mem0").setLevel(logging.ERROR)

EMBEDDER = "BAAI/bge-small-zh-v1.5"
TOP_K = 20
INSTRUCTIONS = ("这些记录来自一个求职平台：简历、岗位、模拟面试与真实面试的问答、准备清单。请记住求职者在每个知识点上"
                "答到了什么程度、哪里答不上来、简历里的说法、经历的细节与数字、用户的更正，并保留日期。")

_models: dict[int, Codex] = {}
_samples: dict[int, int] = {}
_lock = threading.Lock()


class CodexLLM(LLMBase):
    """Mem0's LLM interface over the cached Codex client registered in ``slot``."""

    def __init__(self, slot: int):
        super().__init__(None)
        self.slot = slot

    def generate_response(self, messages, tools=None, tool_choice="auto", **kwargs):
        system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
        user = "\n\n".join(m["content"] for m in messages if m["role"] != "system")
        if kwargs.get("response_format"):
            system += "\n\n只返回一个 JSON 对象，不要输出 Markdown 或解释。"
        return _models[self.slot](system, user, None, _samples[self.slot])


class Mem0Baseline:
    name = "mem0"

    def __init__(self, model: Codex, sample: int) -> None:
        self.model, self.sample = model, sample
        with _lock:
            self.slot = len(_models)
            _models[self.slot], _samples[self.slot] = model, sample
        self.dir = tempfile.TemporaryDirectory(prefix="dreval-mem0-")
        self.memory = Memory.from_config({
            # Mem0 validates the provider name against a fixed list; the client is replaced right after.
            "llm": {"provider": "openai", "config": {"model": "unused", "api_key": "unused"}},
            "embedder": {"provider": "fastembed", "config": {"model": EMBEDDER}},
            "vector_store": {"provider": "qdrant", "config": {"path": f"{self.dir.name}/qdrant", "collection_name": "m",
                                                               "embedding_model_dims": 512, "on_disk": True}},
            "history_db_path": f"{self.dir.name}/history.db",
            "custom_instructions": INSTRUCTIONS,
        })
        self.memory.llm = CodexLLM(self.slot)
        self.user = "seeker"
        self.added: dict[str, list[str]] = {}  # source id → memory ids
        self.seen: set[str] = set()

    def feed(self, episodes, deleted: dict[str, str], now: str) -> None:
        """Bring the memory up to ``now``: add new episodes in time order, delete what was deleted by then."""
        for e in sorted(episodes, key=lambda e: (e.occurred_at, e.id)):
            if e.id in self.seen or e.occurred_at > now:
                continue
            self.seen.add(e.id)
            text = f"[{e.occurred_at[:16].replace('T', ' ')}] {e.kind}（来源 {e.source.type}/{e.source.id}）\n{e.text.strip()}"
            out = self.memory.add([{"role": "user", "content": text}], user_id=self.user,
                                  metadata={"source": e.source.id, "at": e.occurred_at})
            self.added.setdefault(e.source.id, []).extend(r["id"] for r in out.get("results", []) if r.get("id"))
        for source, at in deleted.items():
            if at <= now and self.added.get(source):
                for mid in self.added.pop(source):
                    self.memory.delete(mid)

    def recall(self, query: str) -> list[str]:
        hits = self.memory.search(query, filters={"user_id": self.user}, top_k=TOP_K).get("results", [])
        return [f"[{(h.get('metadata') or {}).get('at', '')[:10]}] {h['memory']}" for h in hits]

    def context(self, questions: list[Question]) -> list[str]:
        seen: dict[str, None] = {}
        for q in questions:
            for m in self.recall(q.text):
                seen.setdefault(m, None)
        return list(seen)

    def answer(self, questions: list[Question], day: str) -> tuple[dict[str, dict[str, Any]], str]:
        memories = self.context(questions)
        text = "\n".join(f"- {m}" for m in memories)
        user = f"<记忆>\n{text}\n</记忆>\n\n{ask_text(questions, day)}"
        guide = GUIDE.replace("全部记录", "记忆系统检索出的记忆").replace("从记录中逐字摘录", "从记忆中逐字摘录")
        out = self.model(guide, user, ANSWER_SCHEMA, self.sample)
        return {a["id"]: a for a in out.get("answers") or []}, text

    def predict(self, question: str, nodes: list[str], levels: list[int], day: str) -> tuple[dict[int, float], list[str], str]:
        text = "\n".join(f"- {m}" for m in self.recall(f"{'、'.join(nodes)} {question}"))
        user = (f"<记忆>\n{text}\n</记忆>\n\n现在是 {day}。下一道题考察：{'、'.join(nodes)}\n题目：{question}\n"
                f"请对层级 {', '.join(f'L{l}' for l in levels)} 各给一个通过概率（0–1）。")
        out = self.model(PREDICT_GUIDE.replace("已有记录", "检索出的记忆"), user, PREDICT_SCHEMA, self.sample)
        return {int(p["level"]): float(p["p_pass"]) for p in out.get("predictions") or []}, out.get("quotes") or [], text

    def close(self) -> None:
        try:
            self.memory.vector_store.client.close()
        except Exception:  # noqa: BLE001
            pass
        self.dir.cleanup()
