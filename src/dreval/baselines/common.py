"""What both LLM baselines share: the record as text, the questions, the answer schema, and grading.

Each baseline sees the same episodes the engine sees (the product's converted records, up to the checkpoint
date, deletions applied) rendered as plain text, and is asked about exactly the things each checkpoint
asserts, in the gold set's vocabulary. That gives the baselines a hint the engine never gets: the engine
has to raise every issue on its own, the baselines are asked "is there X about Y". The comparison is
therefore tilted toward the baselines.

The answers are wrapped in ``AnswerState`` (the gold set's ``MemoryState``), so the gold set's own
``evaluate`` grades engine and baselines with one judge.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from drawoffer.memory.gold.metrics import quote_key
from drawoffer.memory.gold.personas import (
    DEPTHS, INSUFFICIENT, ISSUE_STATES, LEVELS, READINESS_LEVELS, WEIGHTS, Checkpoint, Persona,
)

from dreval import engine

GUIDE = """你是一个求职记忆系统，负责根据一位求职者在平台上留下的全部记录，回答关于他当前状态的问题。
只依据记录作答，不臆测。每个判断都要从记录中逐字摘录支持它的原话（quotes），摘不出就说明证据不足。

术语：
- 能力等级（知识点或技术）：aware 了解（是什么、用来做什么，L1）；familiar 熟悉（怎么做到、为什么，L2）；
  proficient 掌握（为什么选它、不选别的，L3）；expert 精通（故障、规模、极端情况，L4 及以上）。
  证据不足以下结论（例如没被直接问过、只有一次很浅的回答、只有简历自述）时回答 insufficient。
- 问题种类：knowledge_gap 知识缺口（目标岗位要求的知识点答不到要求的深度）；review_due 复测到期（曾经掌握、
  很久没练，可能遗忘）；unclear_telling 经历讲不清（被追问背景、目标、行动、取舍、结果、边界时说不清）；
  inconsistent 前后不一（同一段经历在不同地方说法矛盾，如数字、角色）；overstated 简历说法夸大（如写“精通”
  但面试答不到）；understated 简历说法低估；accuracy_risk 回答有事实性错误的风险；supplement_fact 面试里
  讲出了简历上没写、值得补充的事实。
- 问题状态：appeared 出现；disclosed 已告知用户；practicing 练习中（用户勾选了对应的准备项）；resolved 已解决；
  relapsed 解决后复发；dormant 沉睡（对应目标已结束）。不存在或已不成立时 exists=false。
- 经历分量：W1 打杂或照着做；W2 独立完成明确的小模块；W3 负责一个完整模块并处理过它的问题；W4 负责跨模块
  或关键链路的方案；W5 定义方向或从零设计核心系统。经历深度 L1–L5 同上面的深度阶梯（技术深度、业务深度分开）。
- 准备度（针对某个求职目标）：low / medium / high，或证据不足时 insufficient；coverage 是目标要求中已有面试
  证据的比例（0–1）。
"""


def render(episodes) -> str:
    """The record in time order, one block per episode, with its ID so quotes and deletions can be traced."""
    blocks = []
    for e in sorted(episodes, key=lambda e: (e.occurred_at, e.id)):
        head = f"[{e.occurred_at[:16].replace('T', ' ')}] {e.kind} · 来源 {e.source.type}/{e.source.id}"
        blocks.append(f"{head}\n{e.text.strip()}")
    return "\n\n".join(blocks)


def entry_titles(persona: Persona) -> dict[str, str]:
    """resume:entry → how the persona's resume titles that entry (for 'experience' questions)."""
    titles = {}
    for event in persona.timeline:
        for section in (event.get("resume") or {}).get("sections") or []:
            for entry in section.get("entries") or []:
                text = " · ".join(str(entry.get(k)) for k in ("title", "org") if entry.get(k))
                titles[f"{event['ref']}:{entry['node']}"] = text or entry["node"]
    return titles


def describe(persona: Persona, about: str, cat) -> str:
    """A persona key in words the record uses: nodes by name, experiences by resume title, claims by text."""
    if about.startswith(("kp:", "tp:", "tech:", "dom:", "cmp:")):
        node = cat.nodes.get(about)
        return f"{node.name_zh}（{about}）" if node else about
    if about.startswith("exp:"):
        key = about[4:]
        titles = entry_titles(persona)
        refs = [ref for event in persona.timeline for section in (event.get("resume") or {}).get("sections") or []
                for entry in section.get("entries") or [] if entry.get("experience") == key
                for ref in [f"{event['ref']}:{entry['node']}"]]
        names = sorted({titles[r] for r in refs}) or [key]
        return f"经历“{' / '.join(names)}”"
    if about.startswith("claim:"):
        claim = next((c for c in persona.truth.get("claims") or [] if c.get("key") == about[6:]), None)
        return f"简历说法“{claim['text']}”" if claim else about
    if about.startswith("goal:"):
        ref = about.split(".", 1)[1] if "." in about else about
        if about.startswith("goal:job."):
            job = next((e.get("job") for e in persona.timeline if e.get("ref") == ref), None) or {}
            return f"求职目标：{job.get('company', '')} {job.get('title', ref)}"
        if about.startswith("goal:app."):
            app = next((e.get("application") for e in persona.timeline if (e.get("application") or {}).get("ref") == ref), None) or {}
            return f"求职目标：投递 {app.get('company', '')} {app.get('title', ref)}"
        return f"求职目标：{ref} 方向"
    return about


@dataclass(frozen=True)
class Question:
    id: str
    kind: str          # ability | issue | experience | readiness | facts | merged
    about: str          # the persona key asked about
    text: str
    issue_kind: str | None = None
    mentions: tuple[str, ...] = ()


def questions_for(persona: Persona, checkpoints: list[Checkpoint], cat) -> list[Question]:
    """One question per thing the checkpoints at a date assert; issue_absent without 'about' asks for a list."""
    found: dict[str, Question] = {}
    titles = entry_titles(persona)
    for cp in checkpoints:
        p = cp.params
        if cp.assertion == "ability_level":
            q = Question(f"ability:{p['node']}", "ability", p["node"], f"{describe(persona, p['node'], cat)} 现在的能力等级是什么？")
        elif cp.assertion in ("issue_present", "issue_state") or (cp.assertion == "issue_absent" and p.get("about")):
            q = Question(f"issue:{p['kind']}:{p['about']}", "issue", p["about"],
                         f"关于{describe(persona, p['about'], cat)}，现在是否存在 {p['kind']} 问题？如果存在，处于什么状态，何时解决（如已解决）？",
                         issue_kind=p["kind"])
        elif cp.assertion == "issue_absent":
            q = Question(f"issues:{p['kind']}", "issue_list", "", f"现在存在哪些 {p['kind']} 问题？逐条列出它们针对什么（知识点、经历或简历说法）。",
                         issue_kind=p["kind"])
        elif cp.assertion == "experience_assessment":
            q = Question(f"experience:{p['experience']}", "experience", p["experience"],
                         f"{describe(persona, p['experience'], cat)} 的分量（W1–W5）、技术深度、业务深度（L1–L5）各是什么？证据不足的项给 unknown。")
        elif cp.assertion == "readiness":
            q = Question(f"readiness:{p['goal']}", "readiness", p["goal"], f"{describe(persona, p['goal'], cat)} 的准备度是什么？coverage 是多少？")
        elif cp.assertion == "supplement_fact":
            q = Question(f"facts:{p['experience']}", "facts", p["experience"],
                         f"{describe(persona, p['experience'], cat)} 在面试中讲出了哪些简历上没写、值得补充进简历的事实？逐条列出。")
        elif cp.assertion == "experience_merged":
            mentions = tuple(p["mentions"]) + tuple(p.get("separate_from") or ())
            listed = "；".join(f"{m}（{titles.get(m, m)}）" for m in mentions)
            q = Question(f"merged:{'|'.join(mentions)}", "merged", "", f"下面这些简历条目里，哪些说的是同一段经历？给每条一个分组编号，同一段经历同号：{listed}",
                         mentions=mentions)
        else:
            continue
        found.setdefault(q.id, q)
    return list(found.values())


QUOTES = {"type": "array", "items": {"type": "string"}, "description": "记录中逐字摘录的原话"}

ANSWER_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "answers": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "level": {"type": "string", "enum": [*LEVELS, INSUFFICIENT]},
                    "exists": {"type": "boolean"},
                    "state": {"type": "string", "enum": list(ISSUE_STATES)},
                    "resolved_on": {"type": "string", "description": "YYYY-MM-DD"},
                    "targets": {"type": "array", "items": {"type": "string"}, "description": "issue_list：每条问题针对什么"},
                    "weight": {"type": "string", "enum": [*WEIGHTS, "unknown"]},
                    "tech_depth": {"type": "string", "enum": [*DEPTHS, "unknown"]},
                    "biz_depth": {"type": "string", "enum": [*DEPTHS, "unknown"]},
                    "readiness": {"type": "string", "enum": [*READINESS_LEVELS, INSUFFICIENT]},
                    "coverage": {"type": "number"},
                    "facts": {"type": "array", "items": {"type": "string"}},
                    "groups": {"type": "array", "items": {"type": "integer"}, "description": "merged：与题目中条目顺序对应的分组编号"},
                    "quotes": QUOTES,
                },
                "required": ["id", "quotes"],
            },
        }
    },
    "required": ["answers"],
}


def ask_text(questions: list[Question], day: str) -> str:
    lines = [f"现在是 {day} 结束时。请逐题回答，id 原样照抄；只填与题目相关的字段。"]
    for q in questions:
        lines.append(f"- id={q.id}（{q.kind}）：{q.text}")
    return "\n".join(lines)


@dataclass
class AnswerState:
    """Baseline answers through the gold set's ``MemoryState``, so ``evaluate`` grades them as it grades the engine."""

    answers: dict[str, dict[str, Any]]
    questions: dict[str, Question] = field(default_factory=dict)

    def _a(self, qid: str) -> dict[str, Any]:
        return self.answers.get(qid) or {}

    def ability_level(self, node: str) -> str:
        return self._a(f"ability:{node}").get("level") or INSUFFICIENT

    def issues(self) -> list[dict[str, Any]]:
        out = []
        for qid, q in self.questions.items():
            a = self._a(qid)
            if q.kind == "issue" and a.get("exists"):
                out.append({"kind": q.issue_kind, "about": q.about, "state": a.get("state") or "appeared",
                            "resolved_on": a.get("resolved_on")})
            elif q.kind == "issue_list":
                out += [{"kind": q.issue_kind, "about": t, "state": "appeared"} for t in a.get("targets") or []]
        return out

    def experience_of(self, mention: str) -> str | None:
        for qid, q in self.questions.items():
            if q.kind == "merged" and mention in q.mentions:
                groups = self._a(qid).get("groups") or []
                i = q.mentions.index(mention)
                return f"g{groups[i]}" if i < len(groups) else None
        return None

    def experience_assessment(self, experience: str) -> dict[str, str]:
        a = self._a(f"experience:{experience}")
        return {k: a[k] for k in ("weight", "tech_depth", "biz_depth") if a.get(k) and a[k] != "unknown"}

    def readiness(self, goal: str) -> dict[str, Any]:
        a = self._a(f"readiness:{goal}")
        return {"level": a.get("readiness") or INSUFFICIENT, "coverage": float(a.get("coverage") or 0.0)}

    def supplement_facts(self, experience: str) -> list[str]:
        return list(self._a(f"facts:{experience}").get("facts") or [])


def grounded(quotes: list[str], source_text: str) -> tuple[int, int]:
    """(quotes found verbatim in the record, quotes given); ellipses split a quote into pieces."""
    import re

    whole = quote_key(source_text)
    total = ok = 0
    for quote in quotes or []:
        pieces = [quote_key(x) for x in re.split(r"…+|\.{3,}", quote) if quote_key(x)]
        if not pieces:
            continue
        total += 1
        ok += all(piece in whole for piece in pieces)
    return ok, total


def by_day(persona: Persona) -> dict[str, list[Checkpoint]]:
    days: dict[str, list[Checkpoint]] = {}
    for cp in persona.checkpoints:
        if "m8" in (cp.params.get("requires") or []):
            continue
        days.setdefault(cp.at.isoformat(), []).append(cp)
    return days


def episodes_at(item: engine.Case, day: str):
    """What the engine reads at the end of ``day``: episodes that happened by then, deletions applied."""
    from drawoffer.memory.assemble import moment

    now = moment(day)
    return [e for e in item.replay.at(now) if e.occurred_at <= now]


def as_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)
