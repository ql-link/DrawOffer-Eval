"""The DrawOffer engine on a knowledge-tracing sequence.

The dataset's skills become knowledge points under one technology root, so the engine's prior gives every skill
the shared root factor (``root_share``) and nothing else: the dataset has no topic tree. Each interaction is one
direct ``ability`` observation on its skills (equal shares when it has several), discrimination ``a`` = 1 and pass
bar ``b``; ``on_evidence`` gives the probability of a correct answer just before the engine absorbs it, which is the
prediction scored. Interactions are spaced one minute apart (the file has no times), so forgetting plays no part.

Two variants:

- ``engine``: the product's parameters and ``b`` = 0 for every skill. Nothing is learnt from the training students.
- ``engine_difficulty``: each skill's ``b`` set so that the engine's prior prediction equals that skill's rate of
  correct answers in the training students (smoothed toward the overall rate). Only these 123 numbers
  come from the training data.
- ``engine_problem`` (a control): each problem's ``b`` from the Rasch fit of the training students, the same
  difficulties IRT uses, mapped so the engine's prior prediction on a problem equals Rasch's for an average student.
  It asks how much of the gap to IRT is the per-problem difficulty, which the product does not estimate.
"""

from __future__ import annotations

import math
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timedelta, timezone

import numpy as np
from drawoffer.memory.catalog.loader import assemble
from drawoffer.memory.catalog.model import CatalogNode
from drawoffer.memory.inference.engine import infer
from drawoffer.memory.inference.params import active
from drawoffer.memory.observers.record import Finding, Observation, ability

from dreval.kt.data import Student

ROOT = "tech:assist09"
START = datetime(2009, 9, 1, tzinfo=timezone.utc)


def catalog(skills: list[str]):
    nodes = [CatalogNode(ROOT, "technology", "ASSISTments 2009")]
    nodes += [CatalogNode(f"kp:s{s}", "knowledge_point", f"skill {s}", parent=ROOT) for s in skills]
    return assemble("assist09", "assist09", nodes, [])


def _prior_var(skills: list[str]) -> float:
    """A leaf's prior variance in this catalog (every leaf the same: root share plus its own part)."""
    from drawoffer.memory.inference.prior import build_scope

    scope = build_scope(catalog(skills), [f"kp:s{skills[0]}"], active())
    return float(scope.prior_cov[0, 0])


def difficulties(train: list[Student], skills: list[str], a: float = 1.0, smooth: float = 10.0) -> dict[str, float]:
    """b per skill so that σ(a(μ − b)/√(1 + πa²v/8)) under the prior equals the skill's smoothed training rate."""
    params, v = active(), _prior_var(skills)
    right, total = defaultdict(float), defaultdict(float)
    for st in train:
        for it in st.items:
            for s in it.skills:
                right[s] += it.correct
                total[s] += 1
    overall = sum(right.values()) / sum(total.values())
    scale = math.sqrt(1 + math.pi * a * a * v / 8)
    out = {}
    for s in skills:
        p = min(0.99, max(0.01, (right[s] + smooth * overall) / (total[s] + smooth)))
        out[s] = params.mu_base - scale * math.log(p / (1 - p)) / a
    return out


def problem_difficulties(train: list[Student], skills: list[str], a: float = 1.0) -> tuple[dict[str, float], dict[str, float]]:
    """b per problem (and per skill, for problems never seen in training) from the Rasch fit of the training
    students: the engine's prior prediction on a problem equals the Rasch prediction for an average student."""
    from dreval.kt.baselines import _rasch

    beta, mu, var = _rasch(train)
    params, v = active(), _prior_var(skills)
    scale = math.sqrt(1 + math.pi * a * a * v / 8)
    rasch_scale = math.sqrt(1 + math.pi * var / 8)
    b = {problem: params.mu_base - scale * (mu - value) / rasch_scale / a for problem, value in beta.items()}
    by_skill: dict[str, list[float]] = defaultdict(list)
    for st in train:
        for it in st.items:
            for s in it.skills:
                by_skill[s].append(b[it.problem])
    return b, {s: float(np.mean(v)) for s, v in by_skill.items()}


def predict_student(args) -> np.ndarray:
    st, skills, b = args[:3]
    problems = args[3] if len(args) > 3 else None
    cat = catalog(skills)
    observations, times = [], {}
    for number, it in enumerate(st.items):
        episode = f"{st.id}:{number:05d}"
        share = 1.0 / len(it.skills)
        u = {f"kp:s{s}": share for s in it.skills}
        if problems is not None and it.problem in problems:
            bar = problems[it.problem]
        else:
            bar = float(np.mean([b.get(s, 0.0) for s in it.skills])) if b else 0.0
        digest = f"{st.id}-{number}"
        observations.append(Observation("kt", 1, digest, (episode,), "real_interview",
                                        (Finding("θ", ability(u, 1.0, bar, float(it.correct))),)))
        times[episode] = (START + timedelta(minutes=number)).isoformat()
    predictions: list[float] = []
    infer(cat, observations, times, (START + timedelta(minutes=len(st.items))).isoformat(),
          on_evidence=lambda _e, p: predictions.append(p), engine="1")
    if len(predictions) != len(st.items):
        raise RuntimeError(f"{st.id}: {len(predictions)} predictions for {len(st.items)} interactions")
    return np.array(predictions)


def engine(test: list[Student], skills: list[str], b: dict[str, float] | None = None, jobs: int = 8,
           problems: dict[str, float] | None = None) -> list[np.ndarray]:
    with ProcessPoolExecutor(jobs) as pool:
        return list(pool.map(predict_student, [(st, skills, b, problems) for st in test], chunksize=8))
