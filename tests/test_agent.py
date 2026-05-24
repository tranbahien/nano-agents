"""Tests for the nanoAgent reference implementation."""

from __future__ import annotations

from collections import Counter

import numpy as np

from nano_agents.agent import (
    Costs,
    NanoAgent,
    SelfModel,
    SimulatedWorld,
    run_agent,
)


def test_world_capabilities_valid():
    w = SimulatedWorld(n_questions=200, n_answers=5, seed=0)
    rng = np.random.default_rng(0)
    for qid in range(50):
        ans, conf = w.generate(qid, rng)
        assert 0 <= ans < 5 and 0.0 <= conf <= 1.0
        assert 0 <= w.use_tool(qid, rng) < 5
        assert 0 <= w.retrieve(qid, rng) < 5
        flagged, new = w.reflect(qid, ans, rng)
        assert isinstance(flagged, bool) and 0 <= new < 5


def test_tool_helps_on_calculation():
    w = SimulatedWorld(n_questions=4000, seed=0)
    rng = np.random.default_rng(1)
    calc = [q for q in range(w.n_questions) if w.types[q] == "calculation"]
    tool_correct = np.mean([w.is_correct(q, w.use_tool(q, rng)) for q in calc])
    assert tool_correct > 0.9          # tool near-perfect on its home type


def test_retrieval_helps_on_knowledge():
    w = SimulatedWorld(n_questions=4000, seed=0)
    rng = np.random.default_rng(1)
    know = [q for q in range(w.n_questions) if w.types[q] == "knowledge"]
    ret_correct = np.mean([w.is_correct(q, w.retrieve(q, rng)) for q in know])
    assert ret_correct > 0.8


def test_agent_beats_answer_only():
    w = SimulatedWorld(n_questions=3000, overconfidence=1.0, seed=0)
    agent = NanoAgent(calibration_T=1.0)
    answer_only = NanoAgent(calibration_T=1.0, max_steps=0)
    acc, _, _ = run_agent(w, agent, seed=1)
    acc0, _, _ = run_agent(w, answer_only, seed=1)
    assert acc > acc0 + 0.2            # orchestration is a big lift


def test_agent_routes_by_type():
    w = SimulatedWorld(n_questions=3000, overconfidence=1.0, seed=0)
    agent = NanoAgent(calibration_T=1.0)
    _, _, traces = run_agent(w, agent, seed=1)
    first = {t: Counter() for t in ("knowledge", "calculation", "reasoning")}
    for qid, tr in enumerate(traces):
        first[w.types[qid]][tr.actions[0] if tr.actions else "answer"] += 1
    # Each type's most-common first action is its home capability.
    assert first["calculation"].most_common(1)[0][0] == "tool"
    assert first["knowledge"].most_common(1)[0][0] == "retrieve"
    assert first["reasoning"].most_common(1)[0][0] == "reflect"


def test_calibration_improves_control():
    # An overconfident belief makes the agent stop too early -> lower accuracy.
    w = SimulatedWorld(n_questions=4000, overconfidence=2.5, seed=0)
    naive = NanoAgent(calibration_T=1.0)      # trusts inflated confidence
    calibrated = NanoAgent(calibration_T=2.5)  # corrects it
    acc_naive, cost_naive, _ = run_agent(w, naive, seed=1)
    acc_cal, cost_cal, _ = run_agent(w, calibrated, seed=1)
    assert acc_cal > acc_naive                 # calibration helps accuracy
    assert cost_naive < cost_cal               # overconfident agent is lazier


def test_adaptive_compute():
    # Harder questions (lower accuracy) should draw more actions.
    w = SimulatedWorld(n_questions=4000, overconfidence=1.0, seed=0)
    agent = NanoAgent(calibration_T=1.0)
    _, _, traces = run_agent(w, agent, seed=1)
    nact = np.array([len(t.actions) for t in traces])
    order = np.argsort(w.difficulty)           # higher difficulty = harder
    easy = nact[order[: len(order) // 3]]
    hard = nact[order[-len(order) // 3:]]
    assert hard.mean() > easy.mean()


def test_voi_stopping_when_confident():
    # If the agent already believes the answer is almost certainly right,
    # no information action's gain beats its cost: it should answer immediately.
    agent = NanoAgent(calibration_T=1.0)
    action, gain = agent._best_action(belief=0.99, qtype="reasoning")
    assert action == "answer"


def test_budget_respected():
    w = SimulatedWorld(n_questions=500, seed=0)
    agent = NanoAgent(calibration_T=1.0, max_steps=2)
    _, _, traces = run_agent(w, agent, seed=1)
    assert all(len(t.actions) <= 2 for t in traces)


def test_reuses_curriculum_models():
    # The controller's value estimates are the earlier posts' analytic results.
    from nano_agents.multiagent import condorcet_accuracy
    from nano_agents.reflection import one_round_accuracy
    agent = NanoAgent()
    assert abs(agent._expected_after("reflect", 0.6, "reasoning")
               - one_round_accuracy(0.6, 0.75, 0.20, 0.60)) < 1e-9
    assert abs(agent._expected_after("vote", 0.55, "reasoning")
               - condorcet_accuracy(0.55, 1 + agent.m.vote_n)) < 1e-9
