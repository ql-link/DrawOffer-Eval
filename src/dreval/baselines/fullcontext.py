"""Baseline: the whole record handed to the model at every question (no memory structure at all)."""

from __future__ import annotations

from typing import Any

from dreval.baselines.common import ANSWER_SCHEMA, GUIDE, Question, ask_text, render
from dreval.llm import Codex

PREDICT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "predictions": {"type": "array", "items": {"type": "object", "properties": {
            "level": {"type": "integer"}, "p_pass": {"type": "number"}}, "required": ["level", "p_pass"]}},
        "quotes": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["predictions", "quotes"],
}

PREDICT_GUIDE = GUIDE + """
现在要预测下一道面试题：给出求职者在这道题上能答到各个深度层级的概率。层级 L 通过是指面试官追问到 L
层时能答好（L1 是什么、L2 怎么做到与为什么、L3 为什么选它不选别的、L4 故障规模与极端情况、L5 换场景自行设计）。
只看已有记录，不要假设题目答案。"""


class FullContext:
    name = "full_context"

    def __init__(self, model: Codex, sample: int) -> None:
        self.model, self.sample = model, sample

    def answer(self, episodes, questions: list[Question], day: str) -> dict[str, dict[str, Any]]:
        user = f"<记录>\n{render(episodes)}\n</记录>\n\n{ask_text(questions, day)}"
        out = self.model(GUIDE, user, ANSWER_SCHEMA, self.sample)
        return {a["id"]: a for a in out.get("answers") or []}

    def context(self, episodes, questions: list[Question], day: str) -> str:
        """What the model saw, for checking its quotes."""
        return render(episodes)

    def predict(self, episodes, question: str, nodes: list[str], levels: list[int], day: str) -> tuple[dict[int, float], list[str], str]:
        record = render(episodes)
        user = (f"<记录>\n{record}\n</记录>\n\n现在是 {day}。下一道题考察：{'、'.join(nodes)}\n题目：{question}\n"
                f"请对层级 {', '.join(f'L{l}' for l in levels)} 各给一个通过概率（0–1）。")
        out = self.model(PREDICT_GUIDE, user, PREDICT_SCHEMA, self.sample)
        return {int(p["level"]): float(p["p_pass"]) for p in out.get("predictions") or []}, out.get("quotes") or [], record
