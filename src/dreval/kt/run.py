"""Benchmark 2: 5-fold cross-validation by student on ASSISTments 2009.

    python -m dreval.kt.run --data ~/Downloads/kt-data/assist09 --out runs/kt-assist09.json

Each fold: every model is fitted on the other four folds' students and predicts each interaction of this fold's
students from that student's earlier interactions. AUC, RMSE and log loss are pooled over a fold's interactions;
the report gives their mean and standard deviation over the folds, and the per-fold AUC difference to the engine.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import log_loss, mean_squared_error, roc_auc_score

from dreval.kt import baselines, data, engine_kt

MODELS = ("skill_mean", "engine", "engine_difficulty", "engine_no_forgetting", "engine_problem", "irt", "pfa", "bkt", "dkt")
GAP_DAYS = 1.0
ROOTS = {"assist09": "assist09", "ednet": "ednet", "junyi": "junyi"}


def students_of(dataset: str, root: Path, seed: int) -> list[data.Student]:
    """The dataset's students, pickled under runs/kt-cache so parallel cells do not each re-read the raw files."""
    import pickle

    cache = Path("runs/kt-cache") / f"{dataset}-seed{seed}.pkl"
    if cache.exists():
        return pickle.loads(cache.read_bytes())
    students = data.load_dataset(dataset, root, seed)
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_bytes(pickle.dumps(students))
    return students


def after_gap(test: list[data.Student], days: float = GAP_DAYS) -> np.ndarray:
    """Interactions whose skills were all last practised at least ``days`` earlier (and at least once): where
    forgetting, if a model has it, changes the prediction."""
    mask = []
    for st in test:
        last: dict[str, float] = {}
        for it in st.items:
            seen = [last.get(s) for s in it.skills]
            mask.append(it.at is not None and all(t is not None for t in seen)
                        and all(it.at - t >= days * 86400 for t in seen))
            if it.at is not None:
                for s in it.skills:
                    last[s] = it.at
    return np.array(mask, dtype=bool)


def score(y: np.ndarray, p: np.ndarray, gap: np.ndarray | None = None) -> dict:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    out = {"auc": float(roc_auc_score(y, p)), "rmse": float(np.sqrt(mean_squared_error(y, p))),
           "log_loss": float(log_loss(y, p)), "n": int(len(y))}
    if gap is not None and gap.sum() > 100 and len(set(y[gap])) == 2:
        out["after_gap"] = score(y[gap], p[gap])
    return out


def run_fold(students, folds, f: int, models, jobs: int, seed: int):
    test = [students[i] for i in folds[f]]
    train = [students[i] for g, idx in enumerate(folds) if g != f for i in idx]
    skills = data.skills_of(students)
    y = np.concatenate([[it.correct for it in st.items] for st in test])
    gap = after_gap(test)
    result, seconds = {}, {}
    for name in models:
        started = time.time()
        if name == "skill_mean":
            p = baselines.skill_mean(train, test)
        elif name == "engine":
            p = engine_kt.engine(test, skills, None, jobs)
        elif name == "engine_difficulty":
            p = engine_kt.engine(test, skills, engine_kt.difficulties(train, skills), jobs)
        elif name == "engine_no_forgetting":
            p = engine_kt.engine(test, skills, engine_kt.difficulties(train, skills), jobs, forgetting=False)
        elif name == "engine_problem":
            by_problem, by_skill = engine_kt.problem_difficulties(train, skills)
            p = engine_kt.engine(test, skills, by_skill, jobs, by_problem)
        elif name == "irt":
            p = baselines.irt(train, test)
        elif name == "pfa":
            p = baselines.pfa(train, test)
        elif name == "bkt":
            p = baselines.bkt(train, test, seed)
        elif name == "dkt":
            p = baselines.dkt(train, test, seed)
        else:
            raise ValueError(name)
        flat = np.concatenate(p)
        if len(flat) != len(y):
            raise RuntimeError(f"{name}: {len(flat)} predictions for {len(y)} interactions")
        result[name] = score(y, flat, gap)
        seconds[name] = round(time.time() - started, 1)
        print(f"fold {f} {name}: auc {result[name]['auc']:.4f} rmse {result[name]['rmse']:.4f} ({seconds[name]} s)",
              flush=True)
    return result, seconds


def summarise(per_fold: list[dict], models) -> dict:
    out = {}
    for name in models:
        entry = {}
        for metric in ("auc", "rmse", "log_loss"):
            values = [fold[name][metric] for fold in per_fold if name in fold]
            entry[metric] = {"mean": statistics.mean(values), "sd": statistics.stdev(values) if len(values) > 1 else 0.0}
        gaps = [fold[name]["after_gap"] for fold in per_fold if "after_gap" in fold.get(name, {})]
        if len(gaps) == len(per_fold):
            entry["after_gap"] = {m: statistics.mean(g[m] for g in gaps) for m in ("auc", "rmse", "log_loss")}
            entry["after_gap"]["n"] = sum(g["n"] for g in gaps)
        if name != "engine" and all("engine" in fold for fold in per_fold):
            diff = [fold[name]["auc"] - fold["engine"]["auc"] for fold in per_fold]
            entry["auc_minus_engine"] = {"mean": statistics.mean(diff), "min": min(diff), "max": max(diff)}
        out[name] = entry
    return out


def describe(dataset: str, students: list[data.Student]) -> dict:
    n = sum(len(st.items) for st in students)
    names = {"assist09": ("ASSISTments 2009-2010 skill builder (corrected)", data.URL, data.SHA256),
             "ednet": ("EdNet-KT1", data.EDNET_URL, data.EDNET_SHA256),
             "junyi": ("Junyi Academy math practice log", data.JUNYI_URL, data.JUNYI_SHA256)}
    name, url, sha = names[dataset]
    out = {"id": dataset, "name": name, "url": url, "sha256": sha, "students": len(students), "interactions": n,
           "multi_skill": sum(len(it.skills) > 1 for st in students for it in st.items),
           "skills": len(data.skills_of(students)),
           "correct_rate": sum(it.correct for st in students for it in st.items) / n,
           "after_gap": int(after_gap(students).sum()), "gap_days": GAP_DAYS}
    if dataset != "assist09":
        out["sample"] = {"students": data.SAMPLE, "min_items": data.MIN_ITEMS, "max_items": data.MAX_ITEMS}
    return out


def merge(cells: Path, out: Path, k: int, models, dataset: str = "assist09", root: Path | None = None,
          seed: int = 0) -> int:
    """One result from the per-(fold, model) cells written by parallel ``--cell-dir`` runs."""
    found = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(cells.glob("fold*-*.json"))]
    per_fold = [{c["model"]: c["score"] for c in found if c["fold"] == f} for f in range(k)]
    seconds = [{c["model"]: c["seconds"] for c in found if c["fold"] == f} for f in range(k)]
    missing = [(f, m) for f in range(k) for m in models if m not in per_fold[f]]
    if missing:
        raise SystemExit(f"missing cells: {missing}")
    students = students_of(dataset, root or Path.home() / "Downloads/kt-data" / ROOTS[dataset], seed)
    result = {
        "dataset": describe(dataset, students),
        "folds": k, "evaluated_folds": list(range(k)), "seed": found[0]["seed"], "models": list(models),
        "per_fold": per_fold, "seconds": seconds, "summary": summarise(per_fold, models),
        "machine": {"system": platform.platform(), "python": platform.python_version()},
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(result["summary"], indent=1))
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m dreval.kt.run")
    parser.add_argument("--dataset", default="assist09", choices=sorted(ROOTS))
    parser.add_argument("--data", type=Path, help="raw data directory (default ~/Downloads/kt-data/<dataset>)")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--only-folds", type=str, default="")
    parser.add_argument("--models", default=",".join(MODELS))
    parser.add_argument("--students", type=int, default=0, help="use the first N students only (smoke test)")
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("runs/kt-assist09.json"))
    parser.add_argument("--cell-dir", type=Path, help="write one JSON per (fold, model) here instead of one result")
    parser.add_argument("--merge", type=Path, help="merge the cells of this directory into --out")
    args = parser.parse_args(argv)
    models = args.models.split(",")
    root = args.data or Path.home() / "Downloads/kt-data" / ROOTS[args.dataset]
    if args.merge:
        return merge(args.merge, args.out, args.folds, models, args.dataset, root, args.seed)
    students = students_of(args.dataset, root, args.seed)
    if args.students:
        students = students[: args.students]
    folds = data.folds(students, args.folds, args.seed)
    chosen = [int(x) for x in args.only_folds.split(",")] if args.only_folds else list(range(args.folds))
    started = time.time()
    per_fold, seconds = [], []
    for f in chosen:
        if args.cell_dir:
            args.cell_dir.mkdir(parents=True, exist_ok=True)
            for name in models:
                r, s = run_fold(students, folds, f, [name], args.jobs, args.seed)
                (args.cell_dir / f"fold{f}-{name}.json").write_text(json.dumps(
                    {"fold": f, "model": name, "score": r[name], "seconds": s[name], "seed": args.seed,
                     "students": len(students)}), encoding="utf-8")
            continue
        r, s = run_fold(students, folds, f, models, args.jobs, args.seed)
        per_fold.append(r)
        seconds.append(s)
    if args.cell_dir:
        return 0
    result = {
        "dataset": describe(args.dataset, students),
        "folds": args.folds, "evaluated_folds": chosen, "seed": args.seed, "models": models,
        "per_fold": per_fold, "seconds": seconds, "summary": summarise(per_fold, models),
        "machine": {"system": platform.platform(), "python": platform.python_version()},
        "total_seconds": round(time.time() - started, 1),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(result["summary"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
