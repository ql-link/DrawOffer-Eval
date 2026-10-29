"""Benchmark 6: inference time on long histories.

Simulated learners from the engine's own generative world (``drawoffer.memory.sim``), each with a long
history of mock interviews over three goal technologies. For each history length we time

- full replay: ``infer`` over every observation, what a cold read costs;
- incremental: the same read resumed from a checkpoint stored one session earlier (production keeps
  checkpoints, so a new interview costs only its own questions).

The incremental result is checked against the full one (same posterior, bit for bit) before it is timed.

    python -m dreval.benchmarks.performance --sessions 25,50,100,200 --learners 5 --out runs/performance.json
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from pathlib import Path
from typing import Any

import numpy as np
from drawoffer.memory.inference.checkpoint import Checkpoints, Slot
from drawoffer.memory.inference.engine import active_engine, infer
from drawoffer.memory.sim.bench import sim_params
from drawoffer.memory.sim.learner import draw_learner
from drawoffer.memory.sim.spec import ARCHETYPES, WorldSpec
from drawoffer.memory.sim.world import Asked, World, moment

from dreval import engine


def history(cat, index: int, sessions: int, questions: int, seed_name: str = "prior"):
    """A learner answering ``questions`` questions per session on leaves of its goal, at random."""
    params = sim_params()
    world = WorldSpec(sessions=sessions, questions=questions, spacing_days=2.0)
    learner = draw_learner(cat, ARCHETYPES[seed_name], index, world=world, params=params)
    simulation = World(learner, world, params)
    rng = np.random.default_rng(1000 + index)
    leaves = sorted({leaf for root in learner.roots for leaf in cat.leaves(root)})
    observations, times, bounds = [], {}, []
    for session in range(sessions):
        for number in range(questions):
            at = moment(world, session, number)
            leaf = leaves[int(rng.integers(len(leaves)))]
            style = world.styles[int(rng.integers(len(world.styles)))]
            ep = f"{learner.id}:s{session}:q{number}"
            observation, answer = simulation.ask(Asked({leaf: 1.0}, world.start_level, style), session, at, ep)
            observations += [observation, simulation.self_view(answer)]
            times[ep] = at.isoformat()
        bounds.append(len(observations))
    return learner, params, observations, times, moment(world, sessions).isoformat(), bounds, world


def _timed(fn, repeats: int) -> tuple[Any, list[float]]:
    out, seconds = None, []
    for _ in range(repeats):
        started = time.perf_counter()
        out = fn()
        seconds.append(time.perf_counter() - started)
    return out, seconds


def run(sessions_list: list[int], learners: int, questions: int, repeats: int) -> dict[str, Any]:
    cat = engine.catalog()
    rows = []
    for sessions in sessions_list:
        full_s, full2_s, inc_s, same, steps, leaves_n = [], [], [], 0, [], []
        for index in range(learners):
            learner, params, obs, times, now, bounds, world = history(cat, index, sessions, questions)
            roots = list(learner.roots)
            _, seconds = _timed(lambda: infer(cat, obs, times, now, params, extra_nodes=roots, engine="1"), repeats)
            full_s += seconds
            full, seconds = _timed(lambda: infer(cat, obs, times, now, params, extra_nodes=roots, engine="2"), repeats)
            full2_s += seconds
            steps.append(len(obs))
            leaves_n.append(len(full.scope.index))
            # A checkpoint stored after all but the last session, then a read with the last session added.
            store = Checkpoints()
            cut = bounds[-2] if len(bounds) > 1 else 0
            earlier = obs[:cut]
            infer(cat, earlier, {k: v for k, v in times.items()}, moment(world, sessions - 1).isoformat(), params,
                  extra_nodes=roots, engine="2", checkpoint=Slot(store, "main"))
            store.stored, store.fresh = list(store.fresh), []
            inc, seconds = _timed(lambda: infer(cat, obs, times, now, params, extra_nodes=roots, engine="2",
                                                checkpoint=Slot(Checkpoints(stored=list(store.stored)), "main")),
                                  repeats)
            inc_s += seconds
            same += bool(np.array_equal(inc.mean, full.mean) and np.array_equal(inc.cov, full.cov))
        rows.append({
            "sessions": sessions, "questions_per_session": questions, "answers": sessions * questions,
            "observations_mean": statistics.mean(steps), "leaves_mean": statistics.mean(leaves_n),
            "learners": learners, "repeats": repeats,
            "engine1_full_seconds": _summary(full_s), "engine2_full_seconds": _summary(full2_s),
            "engine2_incremental_seconds": _summary(inc_s),
            "incremental_equals_full": f"{same}/{learners}",
        })
        r = rows[-1]
        print(f"{sessions} 场 × {questions} 题（{sessions * questions} 次作答）：引擎 1 全量 P50 {r['engine1_full_seconds']['p50']}s "
              f"P95 {r['engine1_full_seconds']['p95']}s；引擎 2 全量 P50 {r['engine2_full_seconds']['p50']}s，"
              f"增量 P50 {r['engine2_incremental_seconds']['p50']}s；"
              f"增量与全量一致 {same}/{learners}", flush=True)
    return {"default_engine": active_engine(), "catalog_version": cat.version, "machine": _machine(), "rows": rows}


def _summary(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    pick = lambda q: ordered[min(len(ordered) - 1, int(q * len(ordered)))]
    return {"p50": round(pick(0.5), 4), "p95": round(pick(0.95), 4), "max": round(ordered[-1], 4), "n": len(values)}


def _machine() -> dict[str, str]:
    return {"python": platform.python_version(), "system": f"{platform.system()} {platform.machine()}",
            "processor": platform.processor(), "numpy": np.__version__}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m dreval.benchmarks.performance")
    parser.add_argument("--sessions", default="10,25,50,100,200")
    parser.add_argument("--questions", type=int, default=6)
    parser.add_argument("--learners", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    result = run([int(s) for s in args.sessions.split(",")], args.learners, args.questions, args.repeats)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
