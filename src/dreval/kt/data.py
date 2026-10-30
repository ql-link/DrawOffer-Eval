"""ASSISTments 2009–2010 skill builder (corrected), read into one sequence of interactions per student.

The file has one row per (interaction, skill): a problem tagged with two skills appears twice with the same
``order_id``. An interaction here is one ``order_id`` with the set of its skills; rows without a skill are dropped,
as in the usual preprocessing. The file has no timestamps, so ``order_id`` gives the order within a student.
"""

from __future__ import annotations

import csv
import hashlib
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

URL = "http://base.ustc.edu.cn/data/ASSISTment/2009_skill_builder_data_corrected.zip"
SHA256 = "10c4fa76ae9410d8577cf9604cf21f8eed029552717a4b281d64945cf44ebe46"
CSV = "skill_builder_data_corrected.csv"


@dataclass(frozen=True)
class Interaction:
    order: int
    problem: str
    skills: tuple[str, ...]
    correct: int
    # Seconds since the epoch; None when the dataset has no times (ASSISTments 2009).
    at: float | None = None


@dataclass(frozen=True)
class Student:
    id: str
    items: tuple[Interaction, ...]


def fetch(root: Path) -> Path:
    """The CSV under ``root``, downloaded from the mirror and checked against its hash when missing."""
    path = root / CSV
    if path.exists():
        return path
    root.mkdir(parents=True, exist_ok=True)
    archive = root / "assist09.zip"
    if not archive.exists():
        urllib.request.urlretrieve(URL, archive)
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    if digest != SHA256:
        raise ValueError(f"{archive}: sha256 {digest}, expected {SHA256}")
    with zipfile.ZipFile(archive) as z:
        z.extract(CSV, root)
    return path


def load(path: Path, min_items: int = 2) -> list[Student]:
    rows: dict[str, dict[int, dict]] = {}
    with path.open(encoding="latin-1", newline="") as f:
        for row in csv.DictReader(f):
            if not row["skill_id"]:
                continue
            order = int(row["order_id"])
            item = rows.setdefault(row["user_id"], {}).setdefault(
                order, {"problem": row["problem_id"], "skills": [], "correct": int(row["correct"])})
            if row["skill_id"] not in item["skills"]:
                item["skills"].append(row["skill_id"])
    students = []
    for user in sorted(rows, key=int):
        items = tuple(Interaction(order, v["problem"], tuple(sorted(v["skills"], key=int)), v["correct"])
                      for order, v in sorted(rows[user].items()))
        if len(items) >= min_items:
            students.append(Student(user, items))
    return students


def folds(students: list[Student], k: int, seed: int = 0) -> list[list[int]]:
    """Student indices of each of ``k`` folds, at random with a fixed seed."""
    order = np.random.default_rng(seed).permutation(len(students))
    return [sorted(int(i) for i in order[f::k]) for f in range(k)]


def skills_of(students: list[Student]) -> list[str]:
    found = {s for st in students for it in st.items for s in it.skills}
    return sorted(found, key=lambda s: (0, int(s), "") if s.isdigit() else (1, 0, s))


# -- EdNet-KT1 and Junyi: real times, sampled students -----------------------------------------------------------

EDNET_URL = "http://base.ustc.edu.cn/data/EdNet/EdNet-KT1.zip"
EDNET_SHA256 = "34c9c12cea3a7a41afcfac4a2ba3f950273e31654068d17d11c47a2f5d7ebdf3"
EDNET_CONTENTS_URL = "http://base.ustc.edu.cn/data/EdNet/EdNet-Contents.zip"
EDNET_CONTENTS_SHA256 = "aa910a0436d9dbac0ba232f55e27b37ad8d39da285fcf15f1c1eb5d064be98e7"
JUNYI_URL = "http://base.ustc.edu.cn/data/JunyiAcademy_Math_Practicing_Log/junyi.rar"
JUNYI_SHA256 = "7335386710db473a475ac47f6049ab8f1031bb703e6937f2d5529b46293ff35e"

SAMPLE = 5000       # students drawn per dataset
MIN_ITEMS = 20      # a student needs this many interactions to be drawn
MAX_ITEMS = 500     # and only their first interactions are kept


def _check(path: Path, sha: str) -> None:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    if h.hexdigest() != sha:
        raise ValueError(f"{path}: sha256 {h.hexdigest()}, expected {sha}")


def _draw(candidates: list[str], seed: int, n: int) -> list[str]:
    order = np.random.default_rng(seed).permutation(len(candidates))
    return sorted(candidates[i] for i in order[:n])


def load_ednet(root: Path, seed: int = 0, n: int = SAMPLE) -> list[Student]:
    """EdNet-KT1 (Choi et al. 2020): one CSV per user. A question's skills are its tags in ``questions.csv``;
    correct when ``user_answer`` equals the question's ``correct_answer``; time from ``timestamp`` (ms)."""
    archive = root / "EdNet-KT1.zip"
    contents = root / "EdNet-Contents.zip"
    _check(archive, EDNET_SHA256)
    _check(contents, EDNET_CONTENTS_SHA256)
    with zipfile.ZipFile(contents) as z:
        name = next(x for x in z.namelist() if x.endswith("questions.csv"))
        questions = {r["question_id"]: (r["correct_answer"], tuple(sorted(t for t in r["tags"].split(";") if t and t != "-1")))
                     for r in csv.DictReader(z.read(name).decode("utf-8").splitlines())}
    students = []
    with zipfile.ZipFile(archive) as z:
        files = sorted(x for x in z.namelist() if x.endswith(".csv"))
        # Draw from all users, then keep those long enough; more are drawn than needed to fill the sample.
        for name in _draw(files, seed, len(files)):
            rows = list(csv.DictReader(z.read(name).decode("utf-8").splitlines()))
            items = []
            for number, r in enumerate(sorted(rows, key=lambda r: (int(r["timestamp"]), int(r["solving_id"])))):
                q = questions.get(r["question_id"])
                if q is None or not q[1]:
                    continue
                items.append(Interaction(number, r["question_id"], q[1], int(r["user_answer"] == q[0]),
                                         int(r["timestamp"]) / 1000.0))
            if len(items) >= MIN_ITEMS:
                students.append(Student(name.rsplit("/", 1)[-1][1:-4], tuple(items[:MAX_ITEMS])))
                if len(students) == n:
                    break
    return sorted(students, key=lambda s: int(s.id))


def load_junyi(root: Path, seed: int = 0, n: int = SAMPLE) -> list[Student]:
    """Junyi Academy math practice log (Chang et al. 2015): the skill and the problem are the exercise, as in
    the usual exercise-level preprocessing; ``correct`` is the first attempt without hints; time from ``time_done`` (µs)."""
    path = root / "junyi_ProblemLog_original.csv"
    rows: dict[str, list[tuple[float, int, str, int]]] = {}
    with path.open(encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            rows.setdefault(r["user_id"], []).append(
                (int(r["time_done"]) / 1e6, int(r["problem_number"] or 0), r["exercise"], int(r["correct"] == "true")))
    candidates = sorted(u for u, v in rows.items() if len(v) >= MIN_ITEMS)
    students = []
    for user in _draw(candidates, seed, n):
        items = tuple(Interaction(k, ex, (ex,), y, at) for k, (at, _, ex, y) in enumerate(sorted(rows[user])))
        students.append(Student(user, items[:MAX_ITEMS]))
    return sorted(students, key=lambda s: int(s.id))


def load_dataset(name: str, root: Path, seed: int = 0) -> list[Student]:
    if name == "assist09":
        return load(fetch(root))
    if name == "ednet":
        return load_ednet(root, seed)
    if name == "junyi":
        return load_junyi(root, seed)
    raise ValueError(name)
