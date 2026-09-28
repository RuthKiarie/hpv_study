import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SM = json.loads((ROOT / "step_functions" / "statemachine.json").read_text())
STATES = SM["States"]

def _targets(state):
    out = []
    if "Next" in state:
        out.append(state["Next"])
    if "Default" in state:
        out.append(state["Default"])
    out += [c["Next"] for c in state.get("Choices", [])]
    out += [c["Next"] for c in state.get("Catch", [])]
    return out

def test_all_transitions_point_at_real_states():
    assert SM["StartAt"] in STATES
    for name, st in STATES.items():
        for t in _targets(st):
            assert t in STATES, f"{name} -> unknown state {t}"

def test_no_unreachable_states():
    seen, stack = set(), [SM["StartAt"]]
    while stack:
        cur = stack.pop()
        if cur in seen:
            continue
        seen.add(cur)
        stack.extend(_targets(STATES[cur]))
    assert seen == set(STATES), f"unreachable: {set(STATES) - seen}"

def test_every_state_ends_or_continues():
    for name, st in STATES.items():
        if st["Type"] in ("Fail", "Succeed", "Choice"):
            continue
        assert "Next" in st or st.get("End") is True, f"{name} has no Next/End"

def test_every_task_has_a_catch():
    for name, st in STATES.items():
        if st["Type"] == "Task":
            assert st.get("Catch"), f"{name} has no Catch"

def test_retrain_choice_reads_execution_input():
    choice = STATES["RetrainDue"]["Choices"][0]
    assert choice["Variable"] == "$$.Execution.Input.retrain_requested"
