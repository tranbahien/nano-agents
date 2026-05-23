"""Tests for the tools subpackage."""

from __future__ import annotations

import numpy as np

from nano_agents.tools import (
    ArithmeticProblem,
    KnowledgeGraph,
    NoisyArithmeticModel,
    build_example_graph,
    make_calculator,
    make_lookup,
    run_react_noisy,
    run_react_oracle,
    sample_problem,
)


def test_arithmetic_problem_answer():
    assert ArithmeticProblem(3, "+", 4).answer == 7
    assert ArithmeticProblem(10, "-", 4).answer == 6
    assert ArithmeticProblem(6, "*", 7).answer == 42


def test_difficulty_monotone():
    easy = ArithmeticProblem(2, "+", 3)
    hard = ArithmeticProblem(1234, "*", 5678)
    assert hard.difficulty > easy.difficulty


def test_calculator_tool_exact():
    calc = make_calculator()
    result, ok = calc((12, "*", 12))
    assert ok and result == 144


def test_calculator_unsupported_op_errors():
    calc = make_calculator()
    result, ok = calc((1, "/", 2))
    assert not ok


def test_internal_accuracy_decays():
    model = NoisyArithmeticModel()
    p_easy = model.internal_accuracy(2.0)
    p_hard = model.internal_accuracy(12.0)
    assert p_easy > p_hard
    assert 0.0 <= p_hard <= p_easy <= 1.0


def test_internal_solver_correct_on_easy():
    """On trivial problems, internal accuracy is near base; most solves right."""
    model = NoisyArithmeticModel()
    rng = np.random.default_rng(0)
    p = ArithmeticProblem(2, "+", 3)
    correct = sum(model.solve_internally(p, rng)[1] for _ in range(200))
    assert correct > 180  # ~99% expected


def test_knowledge_graph_query():
    kg = KnowledgeGraph()
    kg.add("A", "next", "B")
    kg.add("B", "value", 42)
    assert kg.query("A", "next") == ("B", True)
    assert kg.query("B", "value") == (42, True)
    assert kg.query("A", "missing") == (None, False)


def test_react_oracle_solves_multihop():
    kg, task = build_example_graph(np.random.default_rng(0), chain_len=3, seed=0)
    lookup = make_lookup(kg)
    trace = run_react_oracle(kg, task, lookup)
    assert trace.correct
    assert trace.final_answer == task.answer
    # Oracle makes exactly chain_len + 1 (the 'value' hop) tool calls.
    assert trace.n_tool_calls == len(task.chain)


def test_react_noisy_perfect_when_no_noise():
    kg, task = build_example_graph(np.random.default_rng(1), chain_len=3, seed=1)
    lookup = make_lookup(kg)
    trace = run_react_noisy(kg, task, lookup, rng=np.random.default_rng(0),
                             wrong_relation_prob=0.0)
    assert trace.correct


def test_sample_problem_in_range():
    rng = np.random.default_rng(0)
    for _ in range(50):
        p = sample_problem(rng, max_digits=3)
        assert 1 <= abs(p.a) < 1000
        assert 1 <= abs(p.b) < 1000
        assert p.op in ("+", "-", "*")
