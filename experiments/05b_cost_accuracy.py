"""Figure: the compute-accuracy frontier.

Three ways to spend compute: do nothing (answer immediately), do everything
(run a capability plus a reflect pass on every question), or let the
controller decide. We plot accuracy against the mean number of actions taken
(compute spent). Sweeping the controller's cost sensitivity traces a frontier:
tuned well it nearly matches -- even beats -- "do everything" at far less
compute, because it only acts where the expected gain is worth it. Pushed to
act too eagerly (cost set too low), it over-acts and accuracy *falls*: the
agent reflects correct answers away, the over-correction of Post 4b.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from nano_agents.agent import Costs, NanoAgent, SimulatedWorld, run_agent


def do_everything(world, seed=1):
    rng = np.random.default_rng(seed)
    correct = 0
    for qid in range(world.n_questions):
        ans, _ = world.generate(qid, rng)
        if world.types[qid] == "calculation":
            ans = world.use_tool(qid, rng)
        elif world.types[qid] == "knowledge":
            ans = world.retrieve(qid, rng)
        _, ans = world.reflect(qid, ans, rng)
        correct += world.is_correct(qid, ans)
    return correct / world.n_questions, 2.0


def main() -> None:
    world = SimulatedWorld(n_questions=6000, overconfidence=1.0, seed=0)
    acc_none, _, _ = run_agent(world, NanoAgent(max_steps=0), seed=1)
    acc_all, act_all = do_everything(world)

    factors = [0.2, 0.4, 0.7, 1.0, 1.5, 2.5, 4.0]
    fr_acc, fr_act = [], []
    for fac in factors:
        c = Costs(tool=0.06 * fac, retrieve=0.06 * fac, reflect=0.05 * fac,
                  vote=0.04 * fac)
        a, _, tr = run_agent(world, NanoAgent(costs=c, calibration_T=1.0), seed=1)
        fr_acc.append(a)
        fr_act.append(np.mean([len(t.actions) for t in tr]))

    fig, ax = plt.subplots(figsize=(10, 5.8))
    ax.plot(fr_act, fr_acc, "o-", color="#55a467", linewidth=2, markersize=7,
            label="nanoAgent (varying cost sensitivity)")
    ax.annotate("acts too eagerly →\nover-reflects (4b over-correction)",
                (fr_act[0], fr_acc[0]), textcoords="offset points",
                xytext=(10, 14), fontsize=8.5, color="#c44e52")
    ax.scatter([0], [acc_none], color="#888888", s=130, zorder=5,
               label=f"do nothing ({acc_none:.2f})")
    ax.scatter([act_all], [acc_all], color="#c44e52", s=130, marker="D",
               zorder=5, label=f"do everything ({acc_all:.2f})")
    ax.set_xlabel("mean actions per question (compute spent)")
    ax.set_ylabel("accuracy")
    ax.set_title(
        "The compute-accuracy frontier. Tuned well, the controller matches or "
        "beats\n'do-everything' at far less compute; too eager, it over-acts "
        "and hurts.", fontsize=11)
    ax.legend(loc="lower right"); ax.grid(alpha=0.3); ax.set_axisbelow(True)

    print(f"  none=(0,{acc_none:.3f}) all=({act_all},{acc_all:.3f})")
    print(f"  frontier acc={np.round(fr_acc,3)} actions={np.round(fr_act,2)}")
    fig.tight_layout()
    out = Path("figures/05b_cost_accuracy.png")
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
