"""Benchmark 1: DrawOffer's engine against Mem0 and against the whole record in context, on the same scripts.

Four measures, the same judge for all three:

- **AUC** of "passes depth L on the next question", over every depth step the engine predicts (``dreval.steps``);
  each baseline gets the record before that question, the question, the knowledge point and the levels.
- **Ability accuracy**: the gold set's ``ability_level`` checkpoints (one level of tolerance; "insufficient"
  must be said where evidence is thin). Also reported: every checkpoint, by type.
- **Grounded**: the share of conclusions backed by a quote found verbatim in the record. For the engine a
  conclusion is grounded when the observations it rests on carry a quote found in their episode, or it is
  built only from structured records (a resume line, a job, a grader's depth) whose source it names.
- **Consistency**: the same inputs run 5 times with fresh model samples; a checkpoint is consistent when all
  5 answers agree. For the engine that means the observers' labels re-sampled too (end to end); the
  inference itself is deterministic (benchmark 3).

    python -m dreval.benchmarks.comparison --split dev --runs 5 --jobs 4 --out runs/comparison-dev.json
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from drawoffer.memory.gold.personas import evaluate, tallies, wilson
from drawoffer.memory.quality.metrics import auc, log_loss

from dreval import engine
from dreval.baselines.common import AnswerState, by_day, episodes_at, grounded, questions_for, render
from dreval.baselines.fullcontext import FullContext
from dreval.baselines.mem0_baseline import Mem0Baseline
from dreval.fingerprint import conclusions
from dreval.llm import Codex
from dreval.steps import steps


def _question_text(item: engine.Case, episode: str) -> str:
    e = next((e for e in item.replay.episodes if e.id == episode), None)
    return (e.context.get("question") or e.text.split("\n", 1)[0]) if e else ""


def _before(item: engine.Case, at: str):
    return [e for e in item.replay.at(at) if e.occurred_at < at]


# ---------------------------------------------------------------------------
# Engine

def engine_runs(items, cat, cache: Path, runs: int) -> list[dict[str, Any]]:
    structured = engine.labeller(cache)
    samples = engine.label_samples(items, cat, structured, list(range(1, runs + 1)))
    usage = {n: m.summary() for n, m in structured.meters.items()}
    return [{**engine_run(items, cat, labels), "usage": usage} for labels in samples]


def engine_run(items, cat, labels) -> dict[str, Any]:
    rows, grounded_rows = {}, []
    for item, lab in zip(items, labels):
        verdicts, states = engine.verdicts(item, cat, lab)
        rows[item.persona.id] = {(v.checkpoint): {"passed": v.passed, "detail": v.detail, "assertion": v.assertion}
                                 for v in verdicts}
        for day, ps in states.items():
            st = ps.state
            texts = {e.id: e.text for e in st.prepared.episodes}
            by_key = {o.key: o for o in st.prepared.observations}
            for c in conclusions(st, cat):
                quotes = [(q.text, q.episode_id) for s in c["support"] if tuple(s) in by_key
                          for f in by_key[tuple(s)].findings for q in f.quotes]
                ok = sum(grounded([t], texts.get(ep, ""))[0] for t, ep in quotes)
                cites = [e.get("ref") for e in c.get("evidence") or () if isinstance(e, dict)]
                grounded_rows.append({"persona": item.persona.id, "day": day, "key": c["key"], "predicate": c["predicate"],
                                      "quotes": len(quotes), "quotes_ok": ok,
                                      "sources": bool(cites) and all(ep in texts for ep in cites),
                                      "supported": bool(c["support"])})
    return {"verdicts": rows, "grounded": grounded_rows, "labels": labels}


# ---------------------------------------------------------------------------
# Baselines

def baseline_run(kind: str, item: engine.Case, cat, model: Codex, sample: int) -> dict[str, Any]:
    persona = item.persona
    days = by_day(persona)
    verdicts, quote_rows = {}, []
    mem = Mem0Baseline(model, sample) if kind == "mem0" else None
    full = FullContext(model, sample) if kind == "full_context" else None
    try:
        for day in sorted(days):
            cps = days[day]
            qs = questions_for(persona, cps, cat)
            eps = episodes_at(item, day)
            if mem is not None:
                mem.feed(item.replay.episodes, item.replay.deleted, f"{day}T23:59:59")
                answers, seen = mem.answer(qs, day)
            else:
                answers, seen = full.answer(eps, qs, day), render(eps)
            state = AnswerState(answers, {q.id: q for q in qs})
            for cp in cps:
                v = evaluate(persona, cp, state)
                verdicts[cp.id] = {"passed": v.passed, "detail": v.detail, "assertion": v.assertion}
            record = render(eps)
            for qid, a in answers.items():
                ok, total = grounded(a.get("quotes") or [], record)
                # Where the model's quotes come from: for Mem0 its retrieved memories, which are rewritten text.
                in_seen, _ = grounded(a.get("quotes") or [], seen)
                quote_rows.append({"persona": persona.id, "day": day, "id": qid, "quotes": total, "quotes_ok": ok,
                                   "quotes_in_context": in_seen})
    finally:
        if mem is not None:
            mem.close()
    return {"verdicts": verdicts, "grounded": quote_rows}


def baseline_predict(kind: str, item: engine.Case, step_rows, model: Codex) -> list[dict[str, Any]]:
    """P(pass) for every engine step, grouped by question: one call per question per baseline (run 1 only)."""
    by_q: dict[str, list] = defaultdict(list)
    for s in step_rows:
        by_q[s.episode].append(s)
    out = []
    mem = Mem0Baseline(model, 0) if kind == "mem0" else None
    full = FullContext(model, 0) if kind == "full_context" else None
    try:
        for episode in sorted(by_q, key=lambda e: by_q[e][0].at):
            group = by_q[episode]
            at = group[0].at
            nodes = sorted({n for s in group for n in s.nodes})
            names = [f"{cat_name(n)}（{n}）" for n in nodes]
            levels = sorted({s.level for s in group})
            question, day = _question_text(item, episode), at[:10]
            if mem is not None:
                mem.feed(_before(item, at), item.replay.deleted, at)
                probs, quotes, seen = mem.predict(question, names, levels, day)
            else:
                probs, quotes, seen = full.predict(_before(item, at), question, names, levels, day)
            for s in group:
                p = probs.get(s.level)
                out.append({"persona": item.persona.id, "episode": episode, "level": s.level, "y": s.y,
                            "p": None if p is None else min(1.0, max(0.0, p)), "engine_p": s.engine_p})
    finally:
        if mem is not None:
            mem.close()
    return out


_catalog = None


def cat_name(node: str) -> str:
    n = _catalog.nodes.get(node) if _catalog else None
    return n.name_zh if n else node


# ---------------------------------------------------------------------------
# Summaries

def _rate(passed: int, total: int) -> dict[str, Any]:
    return {"passed": passed, "total": total, "rate": round(passed / total, 4) if total else None,
            "ci95": wilson(passed, total)}


def accuracy(runs: list[dict[str, dict]]) -> dict[str, Any]:
    """Run 1's verdicts: ability accuracy and every checkpoint metric of the gold set."""
    first = runs[0]
    flat = [(pid, cid, r) for pid, rows in first.items() for cid, r in rows.items()]
    by_type: dict[str, list[bool]] = defaultdict(list)
    for _, _, r in flat:
        by_type[r["assertion"]].append(r["passed"])
    return {"ability": _rate(sum(by_type["ability_level"]), len(by_type["ability_level"])),
            "all_checkpoints": _rate(sum(r["passed"] for *_, r in flat), len(flat)),
            "by_type": {k: _rate(sum(v), len(v)) for k, v in sorted(by_type.items())}}


def consistency(runs: list[dict[str, dict]]) -> dict[str, Any]:
    """A checkpoint is consistent when every run gives the same answer (the detail string, not just pass/fail)."""
    if len(runs) < 2:
        return {"runs": len(runs)}
    same_verdict = same_answer = total = 0
    for pid, rows in runs[0].items():
        for cid in rows:
            got = [r.get(pid, {}).get(cid) for r in runs]
            if any(g is None for g in got):
                continue
            total += 1
            same_verdict += len({g["passed"] for g in got}) == 1
            same_answer += len({g["detail"] for g in got}) == 1
    return {"runs": len(runs), "checkpoints": total, "same_verdict": _rate(same_verdict, total),
            "same_answer": _rate(same_answer, total)}


# Conclusions computed from other conclusions or from a record's structure, not judgements read from what the person
# said: readiness aggregates the skills under a goal, an asked question is the fact that it was asked. Reported apart.
DERIVED = ("readiness", "readiness_insufficient", "asked_question", "requirement")


def grounding_engine(rows) -> dict[str, Any]:
    def ok(r):
        return (r["quotes"] and r["quotes_ok"] == r["quotes"]) or (not r["quotes"] and r["sources"])
    judged = [r for r in rows if r["predicate"] not in DERIVED]
    derived = [r for r in rows if r["predicate"] in DERIVED]
    quoted = [r for r in rows if r["quotes"]]
    by = defaultdict(lambda: [0, 0])
    for r in rows:
        by[r["predicate"]][0] += bool(ok(r))
        by[r["predicate"]][1] += 1
    return {"conclusions": _rate(sum(map(ok, judged)), len(judged)),
            "derived": _rate(sum(map(ok, derived)), len(derived)),
            "all": _rate(sum(map(ok, rows)), len(rows)),
            "by_predicate": {k: {"ok": v[0], "total": v[1]} for k, v in sorted(by.items())},
            "with_quotes": len(quoted), "quotes_all_found": sum(r["quotes_ok"] == r["quotes"] for r in quoted),
            "no_quote_but_source_cited": sum(1 for r in rows if not r["quotes"] and r["sources"]),
            "unsupported": sum(1 for r in rows if not r["supported"])}


def grounding_baseline(rows) -> dict[str, Any]:
    ok = sum(1 for r in rows if r["quotes"] and r["quotes_ok"] == r["quotes"])
    seen = sum(1 for r in rows if r["quotes"] and r.get("quotes_in_context") == r["quotes"])
    return {"conclusions": _rate(ok, len(rows)), "in_model_context": _rate(seen, len(rows)),
            "no_quote": sum(1 for r in rows if not r["quotes"]),
            "quotes": sum(r["quotes"] for r in rows), "quotes_found": sum(r["quotes_ok"] for r in rows)}


def prediction(rows, key: str) -> dict[str, Any]:
    pairs = [(r[key], r["y"]) for r in rows if r.get(key) is not None]
    missing = sum(1 for r in rows if r.get(key) is None)
    area, loss = auc(pairs), log_loss(pairs)
    return {"steps": len(pairs), "missing": missing, "auc": None if area is None else round(area, 4),
            "log_loss": None if loss is None else round(loss, 4)}


def bootstrap_auc(rows, keys: list[str], n: int = 2000, seed: int = 7) -> dict[str, Any]:
    """95% interval of each AUC and of engine − baseline, resampling whole questions (steps of one question move together)."""
    import random

    groups = defaultdict(list)
    for r in rows:
        if all(r.get(k) is not None for k in keys):
            groups[(r["persona"], r["episode"])].append(r)
    names = list(groups)
    rng = random.Random(seed)
    draws = defaultdict(list)
    for _ in range(n):
        pick = [r for _ in names for r in groups[names[rng.randrange(len(names))]]]
        values = {k: auc([(r[k], r["y"]) for r in pick]) for k in keys}
        for k, v in values.items():
            if v is not None:
                draws[k].append(v)
        for k in keys[1:]:
            if values[keys[0]] is not None and values[k] is not None:
                draws[f"{keys[0]}-{k}"].append(values[keys[0]] - values[k])

    def ci(xs):
        xs = sorted(xs)
        return [round(xs[int(0.025 * len(xs))], 4), round(xs[int(0.975 * len(xs)) - 1], 4)] if xs else None

    return {"questions": len(names), "resamples": n, **{k: ci(v) for k, v in draws.items()}}


# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    global _catalog
    parser = argparse.ArgumentParser(prog="python -m dreval.benchmarks.comparison")
    parser.add_argument("--split", default="dev", choices=("dev", "test", "all"))
    parser.add_argument("--final", action="store_true", help="required to open the sealed test split")
    parser.add_argument("--persona", action="append")
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--systems", default="engine,full_context,mem0")
    parser.add_argument("--no-predict", action="store_true", help="skip the next-question AUC (costly for baselines)")
    parser.add_argument("--engine-cache", type=Path, default=Path("cache/engine-labels.jsonl"))
    parser.add_argument("--baseline-cache", type=Path, default=Path("cache/baseline-llm.jsonl"))
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    if args.split != "dev" and not args.final:
        parser.error("the test split is sealed: add --final")
    systems = args.systems.split(",")
    _catalog = cat = engine.catalog()
    items = [engine.case(p) for p in engine.personas(args.split, args.persona)]
    model = Codex(args.baseline_cache)
    started = time.time()
    result: dict[str, Any] = {"split": args.split, "personas": [i.persona.id for i in items], "runs": args.runs,
                              "systems": {}}

    # Engine: run 1 from the shared label cache, runs 2..N with fresh model samples.
    eng = engine_runs(items, cat, args.engine_cache, args.runs) if "engine" in systems else []
    step_rows = {i.persona.id: steps(i, cat, lab) for i, lab in zip(items, eng[0]["labels"])} if eng else \
        {i.persona.id: steps(i, cat, l) for i, l in zip(items, engine.label_all(items, cat, engine.labeller(args.engine_cache)))}
    if eng:
        result["systems"]["engine"] = {
            "accuracy": accuracy([e["verdicts"] for e in eng]), "consistency": consistency([e["verdicts"] for e in eng]),
            "grounded": grounding_engine(eng[0]["grounded"]), "usage": eng[0]["usage"]}
        print("engine done", round(time.time() - started), "s", flush=True)

    pred_rows = [{"persona": s.persona, "episode": s.episode, "level": s.level, "y": s.y, "engine": s.engine_p}
                 for rows in step_rows.values() for s in rows]
    with ThreadPoolExecutor(args.jobs) as pool:
        for kind in (k for k in ("full_context", "mem0") if k in systems):
            futures = {(i.persona.id, r): pool.submit(baseline_run, kind, i, cat, model, r)
                       for r in range(args.runs) for i in items}
            pf = {} if args.no_predict else {i.persona.id: pool.submit(baseline_predict, kind, i, step_rows[i.persona.id], model)
                                             for i in items}
            runs = [{pid: futures[(pid, r)].result()["verdicts"] for pid in (i.persona.id for i in items)}
                    for r in range(args.runs)]
            quote_rows = [row for i in items for row in futures[(i.persona.id, 0)].result()["grounded"]]
            entry = {"accuracy": accuracy(runs), "consistency": consistency(runs), "grounded": grounding_baseline(quote_rows)}
            if pf:
                got = {(r["persona"], r["episode"], r["level"]): r["p"] for f in pf.values() for r in f.result()}
                for row in pred_rows:
                    row[kind] = got.get((row["persona"], row["episode"], row["level"]))
            result["systems"][kind] = entry
            print(kind, "done", round(time.time() - started), "s", model.summary(), flush=True)

    keys = [k for k in ("engine", "full_context", "mem0") if k == "engine" or (k in systems and not args.no_predict)]
    result["prediction"] = {k: prediction(pred_rows, k) for k in keys}
    if len(keys) > 1:
        result["prediction"]["bootstrap"] = bootstrap_auc(pred_rows, keys)
    result["baseline_usage"] = model.summary()
    result["seconds"] = round(time.time() - started, 1)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
        args.out.with_suffix(".steps.jsonl").write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in pred_rows), encoding="utf-8")
    print(json.dumps({k: {m: v.get(m) for m in ("accuracy", "consistency", "grounded")}
                      for k, v in result["systems"].items()}, ensure_ascii=False, indent=1, default=str))
    print(json.dumps(result["prediction"], ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
