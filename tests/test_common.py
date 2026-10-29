from dreval.baselines.common import AnswerState, Question, grounded


def test_grounded_ignores_spacing_punctuation_and_splits_ellipsis():
    record = "答：acks=0 不等确认；acks=all 要等 ISR 全部写入。"
    assert grounded(["acks = 0 不等确认", "acks=0……ISR 全部写入"], record) == (2, 2)
    assert grounded(["acks=1 只等 leader"], record) == (0, 1)


def test_answer_state_reads_issues_and_levels():
    qs = {"issue:overstated:claim:x": Question("issue:overstated:claim:x", "issue", "claim:x", "", issue_kind="overstated"),
          "ability:kp:a": Question("ability:kp:a", "ability", "kp:a", "")}
    state = AnswerState({"issue:overstated:claim:x": {"exists": True, "state": "practicing"},
                         "ability:kp:a": {"level": "familiar"}}, qs)
    assert state.issues() == [{"kind": "overstated", "about": "claim:x", "state": "practicing", "resolved_on": None}]
    assert state.ability_level("kp:a") == "familiar"
    assert state.ability_level("kp:b") == "insufficient"
