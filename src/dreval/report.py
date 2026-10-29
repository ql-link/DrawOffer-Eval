"""Render the run JSONs into one Markdown report (``reports/RESULTS.md``).

    python -m dreval.report --comparison runs/comparison-dev.json --stability runs/stability-dev.json \
        --performance runs/performance.json --out reports/RESULTS.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

NAMES = {"engine": "DrawOffer 引擎", "full_context": "整段记录交给大模型", "mem0": "Mem0 + 大模型"}
TYPES = {"ability_level": "能力等级", "issue_present": "问题存在", "issue_absent": "不误报问题", "issue_state": "问题状态",
         "experience_assessment": "经历评估", "experience_merged": "经历合并", "readiness": "准备度", "supplement_fact": "可补充事实"}


def pct(r: dict[str, Any] | None) -> str:
    if not r or r.get("rate") is None:
        return "—"
    ci = r.get("ci95")
    tail = f"（{r['passed']}/{r['total']}，95% 区间 {ci[0]:.2f}–{ci[1]:.2f}）" if ci else f"（{r['passed']}/{r['total']}）"
    return f"{r['rate'] * 100:.1f}%{tail}"


def comparison(d: dict[str, Any]) -> str:
    systems = [k for k in NAMES if k in d["systems"]]
    pred = d.get("prediction", {})
    boot = pred.get("bootstrap", {})
    rows = ["| 指标 | " + " | ".join(NAMES[k] for k in systems) + " |", "| --- |" + " --- |" * len(systems)]

    def auc(k):
        p = pred.get(k) or {}
        if p.get("auc") is None:
            return "—"
        ci = boot.get(k)
        return f"{p['auc']:.3f}" + (f"（{ci[0]:.2f}–{ci[1]:.2f}）" if ci else "") + f"，{p['steps']} 步"
    rows.append("| 预测下一题 AUC | " + " | ".join(auc(k) for k in systems) + " |")
    rows.append("| 预测下一题 log loss（越低越好） | " + " | ".join(
        f"{(pred.get(k) or {}).get('log_loss') or '—'}" for k in systems) + " |")
    rows.append("| 能力判断准确率 | " + " | ".join(pct(d["systems"][k]["accuracy"]["ability"]) for k in systems) + " |")
    rows.append("| 全部检查点通过率 | " + " | ".join(pct(d["systems"][k]["accuracy"]["all_checkpoints"]) for k in systems) + " |")
    rows.append("| 对人的判断能追溯到原始记录里的原话 | " + " | ".join(pct(d["systems"][k]["grounded"]["conclusions"]) for k in systems) + " |")
    rows.append("| 原话能在模型看到的上下文里找到 | " + " | ".join(
        pct(d["systems"][k]["grounded"].get("in_model_context")) if k != "engine" else "（同上）" for k in systems) + " |")
    rows.append(f"| 同一输入 {d['runs']} 次结论一致 | " + " | ".join(
        pct(d["systems"][k]["consistency"].get("same_verdict")) for k in systems) + " |")
    rows.append(f"| 同一输入 {d['runs']} 次答案逐字相同 | " + " | ".join(
        pct(d["systems"][k]["consistency"].get("same_answer")) for k in systems) + " |")
    out = ["\n".join(rows)]
    diff = [f"- 引擎 − {NAMES[k]} 的 AUC 差，95% 区间 {boot[f'engine-{k}'][0]:.3f}–{boot[f'engine-{k}'][1]:.3f}"
            for k in systems if f"engine-{k}" in boot]
    if diff:
        out.append(f"按题重抽样 {boot.get('resamples')} 次（{boot.get('questions')} 道题）：\n" + "\n".join(diff))
    types = sorted({t for k in systems for t in d["systems"][k]["accuracy"]["by_type"]})
    trows = ["| 检查点类型 | " + " | ".join(NAMES[k] for k in systems) + " |", "| --- |" + " --- |" * len(systems)]
    for t in types:
        trows.append(f"| {TYPES.get(t, t)} | " + " | ".join(pct(d["systems"][k]["accuracy"]["by_type"].get(t)) for k in systems) + " |")
    out.append("按检查点类型（第 1 次运行）：\n\n" + "\n".join(trows))
    g = d["systems"].get("engine", {}).get("grounded", {})
    if g.get("derived"):
        out.append(f"引擎另有派生结论（准备度、被问过的题、岗位要求）{g['derived']['total']} 条，其中 {g['derived']['passed']} 条带来源；"
                   "准备度与“被问过的题”在产品数据里没有挂接依据，这是待补的缺口。逐类：\n\n"
                   + "\n".join(f"- `{k}`：{v['ok']}/{v['total']}" for k, v in g["by_predicate"].items()))
    return "\n\n".join(out)


def stability(d: dict[str, Any]) -> str:
    det, dele, q = d["determinism"], d["deletion"], d["quotes"]
    return "\n".join([
        "| 测试 | 结果 |", "| --- | --- |",
        f"| 同一输入重复推断（进程内 {det['repeats_in_process']} 次） | {det['identical_in_process']}/{det['personas']} 份剧本、"
        f"{det['moments']} 个时点的结论逐字节相同 |",
        f"| 换 {det['fresh_processes']} 个新进程、不同哈希种子再推断 | {det['identical_across_processes']}/{det['personas']} 份剧本逐字节相同 |",
        f"| 逐一删除来源（{dele['deletions']} 次） | 只依赖被删来源的结论 {dele['dependent_conclusions']} 条，删除后残留 "
        f"**{dele['dependent_survived']}** 条；仍引用被删来源的结论 **{dele['residue_conclusions']}** 条；残留观察 {dele['residue_observations']} 条 |",
        f"| 结论引用的原话 | {q['quotes']} 条中 {q['found_by_piece']} 条在原文中找到（评分器用省略号拼接的摘录按段核对；"
        f"整句直接匹配 {q['found_in_source']} 条） |",
    ])


def performance(d: dict[str, Any]) -> str:
    rows = ["| 历史长度 | 观察条数 | 引擎 1 全量 P50 / P95 | 引擎 2 全量 P50 | 引擎 2 增量 P50 | 增量与全量一致 |",
            "| --- | --- | --- | --- | --- | --- |"]
    for r in d["rows"]:
        rows.append(f"| {r['sessions']} 场 × {r['questions_per_session']} 题 = {r['answers']} 次作答 | {r['observations_mean']:.0f} | "
                    f"{r['engine1_full_seconds']['p50']:.3f} / {r['engine1_full_seconds']['p95']:.3f} 秒 | "
                    f"{r['engine2_full_seconds']['p50']:.3f} 秒 | {r['engine2_incremental_seconds']['p50']:.3f} 秒 | "
                    f"{r['incremental_equals_full']} |")
    m = d["machine"]
    rows.append(f"\n机器：{m['system']}，Python {m['python']}，NumPy {m['numpy']}；默认引擎 {d['default_engine']}；"
                f"每档 {d['rows'][0]['learners']} 个模拟学生 × {d['rows'][0]['repeats']} 次计时。")
    return "\n".join(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m dreval.report")
    parser.add_argument("--comparison", type=Path, action="append", default=[])
    parser.add_argument("--stability", type=Path, action="append", default=[])
    parser.add_argument("--performance", type=Path)
    parser.add_argument("--out", type=Path, default=Path("reports/RESULTS.md"))
    args = parser.parse_args(argv)
    parts = ["# 测评结果\n\n由 `python -m dreval.report` 从 `runs/` 下的结果生成，请勿手改。方法与限制见 [README](../README.md)。"]
    for path in args.comparison:
        d = json.loads(path.read_text(encoding="utf-8"))
        parts.append(f"## 1. 与通用记忆正面对比（{d['split']}，{len(d['personas'])} 份剧本）\n\n{comparison(d)}")
    for path in args.stability:
        d = json.loads(path.read_text(encoding="utf-8"))
        parts.append(f"## 3. 稳定性与可追溯性（{d['split']}）\n\n{stability(d)}")
    if args.performance:
        parts.append(f"## 6. 性能\n\n{performance(json.loads(args.performance.read_text(encoding='utf-8')))}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n\n".join(parts) + "\n", encoding="utf-8")
    print(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
