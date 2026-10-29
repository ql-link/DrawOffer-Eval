"""Benchmark 3: determinism, deletion with no residue, and quotes that really are in the source.

All three run on fixed model labels (read from the cache), so they test the engine itself: the part that
decides, given what the observers saw. End-to-end agreement with fresh model samples is in the comparison.

    python -m dreval.benchmarks.stability --split dev --repeats 5 --out runs/stability.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Any

from drawoffer.memory.gold.metrics import quote_key

from dreval import engine
from dreval.fingerprint import conclusions, fingerprint


def _final_day(item: engine.Case) -> str:
    return max(c.at.isoformat() for c in item.persona.checkpoints)


# ---------------------------------------------------------------------------
# 1. Determinism

def determinism(items, cat, labels, repeats: int, cache: Path, split: str) -> dict[str, Any]:
    """Same inputs inferred ``repeats`` times in this process, and twice more in fresh processes with
    different hash seeds (set and dict order would show up there)."""
    rows = []
    for item, lab in zip(items, labels):
        days = sorted({c.at.isoformat() for c in item.persona.checkpoints})
        prints = {day: {fingerprint(engine.state(item, cat, lab, day), cat)["all"] for _ in range(repeats)}
                  for day in days}
        rows.append({"persona": item.persona.id, "moments": len(days),
                     "identical": all(len(v) == 1 for v in prints.values()),
                     "prints": {d: sorted(v)[0] for d, v in prints.items()}})
    outside = []
    for seed in ("1", "4242"):
        env = {**os.environ, "PYTHONHASHSEED": seed}
        done = subprocess.run([sys.executable, "-m", "dreval.benchmarks.stability", "--prints-only", "--split", split,
                               "--cache", str(cache)], capture_output=True, text=True, env=env, check=True)
        outside.append(json.loads(done.stdout))
    across = sum(all(o[r["persona"]] == r["prints"] for o in outside) for r in rows)
    moments = sum(r["moments"] for r in rows)
    return {"personas": len(rows), "moments": moments, "repeats_in_process": repeats, "fresh_processes": len(outside),
            "identical_in_process": sum(r["identical"] for r in rows), "identical_across_processes": across,
            "rate": (sum(r["identical"] for r in rows) + across) / (2 * len(rows)) if rows else None,
            "rows": [{k: v for k, v in r.items() if k != "prints"} for r in rows]}


# ---------------------------------------------------------------------------
# 2. Deletion

def _episodes_of(item: engine.Case, source: str) -> set[str]:
    return {e.id for e in item.replay.episodes if e.source.id == source}


def _cites(conclusion: dict[str, Any], episodes: set[str], observations: set[tuple]) -> bool:
    refs = {str(e.get("ref")) for e in conclusion.get("evidence") or () if isinstance(e, dict)}
    return bool(refs & episodes) or any(tuple(s) in observations for s in conclusion["support"])


def deletion(items, cat, labels) -> dict[str, Any]:
    """Delete each source in turn (as the product does: its episodes are gone from that moment on) and look
    for anything left that still rests on it."""
    rows = []
    for item, lab in zip(items, labels):
        day = _final_day(item)
        before_state = engine.state(item, cat, lab, day)
        before = {c["key"]: c for c in conclusions(before_state, cat)}
        for source in sorted({e.source.id for e in item.replay.episodes}):
            episodes = _episodes_of(item, source)
            gone_obs = {o.key for o in before_state.prepared.observations if set(o.episode_ids) & episodes}
            only = {k for k, c in before.items()
                    if c["support"] and all(tuple(s) in gone_obs for s in c["support"])}
            cut = replace(item.replay, deleted={**item.replay.deleted, source: "0000-00-00"})
            after_state = engine.state(engine.Case(item.persona, cut), cat, lab, day)
            after = {c["key"]: c for c in conclusions(after_state, cat)}
            residue_obs = [o.key for o in after_state.prepared.observations if set(o.episode_ids) & episodes]
            residue = [k for k, c in after.items() if _cites(c, episodes, gone_obs)]
            survived = sorted(only & set(after))
            rows.append({"persona": item.persona.id, "source": source, "episodes": len(episodes),
                         "conclusions_before": len(before), "conclusions_after": len(after),
                         "dependent": len(only), "dependent_survived": survived,
                         "residue_observations": len(residue_obs), "residue_conclusions": residue})
    return {
        "deletions": len(rows),
        "dependent_conclusions": sum(r["dependent"] for r in rows),
        "dependent_survived": sum(len(r["dependent_survived"]) for r in rows),
        "residue_observations": sum(r["residue_observations"] for r in rows),
        "residue_conclusions": sum(len(r["residue_conclusions"]) for r in rows),
        "rows": rows,
    }


# ---------------------------------------------------------------------------
# 3. Quotes

def quotes(items, cat, labels) -> dict[str, Any]:
    """Every quote an observation carries, looked up in the episode it names (spacing, punctuation and width
    ignored, as the observers' own check does)."""
    total, found, segmented, by_prop, misses = 0, 0, 0, Counter(), []
    for item, lab in zip(items, labels):
        st = engine.state(item, cat, lab, _final_day(item))
        texts = {e.id: e.text for e in st.prepared.episodes}
        for o in st.prepared.observations:
            for f in o.findings:
                for q in f.quotes:
                    total += 1
                    ok = bool(quote_key(q.text)) and quote_key(q.text) in quote_key(texts.get(q.episode_id, ""))
                    found += ok
                    # The interview grader joins several excerpts with an ellipsis; each piece must be there.
                    pieces = [quote_key(x) for x in re.split(r"…+|\.{3,}", q.text) if quote_key(x)]
                    whole = quote_key(texts.get(q.episode_id, ""))
                    seg = bool(pieces) and all(piece in whole for piece in pieces)
                    segmented += seg
                    by_prop[(f.proposition, ok)] += 1
                    if not seg and len(misses) < 20:
                        misses.append({"persona": item.persona.id, "observer": o.observer, "quote": q.text[:120]})
    return {"quotes": total, "found_in_source": found, "rate": found / total if total else None,
            "found_by_piece": segmented, "rate_by_piece": segmented / total if total else None,
            "by_proposition": {f"{p}:{'ok' if ok else 'miss'}": n for (p, ok), n in sorted(by_prop.items())},
            "misses": misses}


# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m dreval.benchmarks.stability")
    parser.add_argument("--split", default="dev", choices=("dev", "test", "all"))
    parser.add_argument("--persona", action="append")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--cache", type=Path, default=Path("cache/engine-labels.jsonl"))
    parser.add_argument("--out", type=Path)
    parser.add_argument("--prints-only", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    cat = engine.catalog()
    items = [engine.case(p) for p in engine.personas(args.split, args.persona)]
    labels = engine.label_all(items, cat, engine.labeller(args.cache, cache_only=True))
    if args.prints_only:
        print(json.dumps({i.persona.id: {d: fingerprint(engine.state(i, cat, l, d), cat)["all"]
                                         for d in sorted({c.at.isoformat() for c in i.persona.checkpoints})}
                          for i, l in zip(items, labels)}))
        return 0
    result = {"split": args.split, "determinism": determinism(items, cat, labels, args.repeats, args.cache, args.split),
              "deletion": deletion(items, cat, labels), "quotes": quotes(items, cat, labels)}
    text = json.dumps(result, ensure_ascii=False, indent=1)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
    d, x, q = result["determinism"], result["deletion"], result["quotes"]
    print(f"确定性：{d['identical_in_process']}/{d['personas']} 进程内 {d['repeats_in_process']} 次一致，"
          f"{d['identical_across_processes']}/{d['personas']} 跨 {d['fresh_processes']} 个新进程一致")
    print(f"删除：{x['deletions']} 次删除，依赖结论 {x['dependent_conclusions']} 条，残留 {x['dependent_survived']} 条；"
          f"残留观察 {x['residue_observations']}，仍引用被删来源的结论 {x['residue_conclusions']}")
    print(f"原话：{q['found_in_source']}/{q['quotes']} 整句在原文中找到，{q['found_by_piece']}/{q['quotes']} 按省略号分段后每段都找到")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
