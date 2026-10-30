import numpy as np

from dreval.kt import data, engine_kt


def _student(sid, answers, skill="1"):
    return data.Student(sid, tuple(data.Interaction(n, f"p{n % 3}", (skill,), y) for n, y in enumerate(answers)))


def test_one_prediction_per_interaction_before_the_answer():
    st = _student("1", [1, 1, 1, 1, 0])
    p = engine_kt.predict_student((st, ["1"], None))
    assert len(p) == 5
    # Right answers raise the next prediction; the first one only sees the prior.
    assert p[0] < p[1] < p[2] < p[3]


def test_skill_difficulty_matches_the_training_rate_at_the_prior():
    train = [_student(str(i), [1, 1, 1, 0]) for i in range(50)]
    b = engine_kt.difficulties(train, ["1"], smooth=0.0)
    first = engine_kt.predict_student((_student("x", [1]), ["1"], b))[0]
    assert np.isclose(first, 0.75, atol=1e-3)


def test_folds_cover_every_student_once():
    students = [_student(str(i), [1, 0]) for i in range(23)]
    folds = data.folds(students, 5)
    assert sorted(i for f in folds for i in f) == list(range(23))
