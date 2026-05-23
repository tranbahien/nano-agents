# Notebooks

Interactive companions to the [blog posts](../posts/). Same algorithms,
different reading mode:

| Posts                | Notebooks                       |
| -------------------- | ------------------------------- |
| Read top to bottom   | Run cell by cell                |
| Full math derivations| Brief math, then run the code   |
| Pre-rendered figures | Figures generated live          |
| Polished prose       | Compact prose + "try this" hints |
| ~400 lines / post    | ~10–20 cells / notebook         |

Both formats are first-class. The post is for the *narrative* read; the
notebook is for the *play with it* read. They share the same library
code in `src/nano_agents/`.

## Running

```bash
pip install -e ".[dev]"
pip install jupyter           # if you don't have it
jupyter lab notebooks/        # or: jupyter notebook
```

Then open any `.ipynb` file. Each notebook is self-contained — start at
the top, work down. Total runtime ≈ 1–3 minutes per notebook.

## Available notebooks

### Topic 1 — Foundations

| Notebook                                                     | Companion post                                                          |
| ------------------------------------------------------------ | ----------------------------------------------------------------------- |
| [`01a-bandits.ipynb`](01a-bandits.ipynb)                     | [Post 1a: Multi-armed bandits](../posts/01a-multi-armed-bandits.qmd)    |
| [`01b-contextual-bandits.ipynb`](01b-contextual-bandits.ipynb) | [Post 1b: Contextual bandits](../posts/01b-contextual-bandits.qmd)      |
| [`01c-mdps-and-bellman.ipynb`](01c-mdps-and-bellman.ipynb)   | [Post 1c: MDPs and Bellman](../posts/01c-mdps-and-bellman.qmd)          |
| [`01d-pomdps-and-q-learning.ipynb`](01d-pomdps-and-q-learning.ipynb) | [Post 1d: POMDPs and Q-learning](../posts/01d-pomdps-and-q-learning.qmd) |

### Topic 2 — Policy gradient methods

| Notebook                                                       | Companion post                                                                  |
| -------------------------------------------------------------- | ------------------------------------------------------------------------------- |
| [`02a-reinforce.ipynb`](02a-reinforce.ipynb)                   | [Post 2a: REINFORCE](../posts/02a-reinforce-and-policy-gradient.qmd)            |
| [`02b-actor-critic.ipynb`](02b-actor-critic.ipynb)             | [Post 2b: Actor-Critic](../posts/02b-actor-critic.qmd)                          |
| [`02c-ppo.ipynb`](02c-ppo.ipynb)                               | [Post 2c: TRPO and PPO](../posts/02c-trpo-and-ppo.qmd)                          |
| [`02d-rlhf-dpo-grpo.ipynb`](02d-rlhf-dpo-grpo.ipynb)           | [Post 2d: RLHF, DPO, and GRPO](../posts/02d-rlhf-dpo-grpo.qmd)                  |

### Topic 3 — Inference-time computation

| Notebook                                       | Companion post                                                          |
| ---------------------------------------------- | ----------------------------------------------------------------------- |
| [`03a-decoding.ipynb`](03a-decoding.ipynb)     | [Post 3a: Decoding as Inference](../posts/03a-decoding-as-inference.qmd) |
| [`03b-tool-use.ipynb`](03b-tool-use.ipynb)     | [Post 3b: Tool Use as Actions](../posts/03b-tool-use.qmd)               |
| [`03c-tree-of-thoughts.ipynb`](03c-tree-of-thoughts.ipynb) | [Post 3c: Tree of Thoughts and MCTS](../posts/03c-tree-of-thoughts.qmd) |

### Topic 4 — Memory, reflection, calibration

| Notebook                                       | Companion post                                                                          |
| ---------------------------------------------- | --------------------------------------------------------------------------------------- |
| [`04a-rag.ipynb`](04a-rag.ipynb)               | [Post 4a: Retrieval as Bayesian Conditioning](../posts/04a-rag-as-bayesian-conditioning.qmd) |

More notebooks coming as Topic 4 progresses.

More notebooks coming with Topic 4 (memory, retrieval, and reflection).

## Building your own

Each notebook is generated programmatically by a `build_notebook_*.py`
script at the repo root. The scripts use `nbformat` to assemble cells —
easier to maintain than hand-editing JSON. To add a notebook for a new
post, copy any existing build script and adapt.

To rebuild and re-verify everything:

```bash
for f in build_notebook_*.py; do python "$f"; done
for nb in notebooks/*.ipynb; do
  jupyter nbconvert --to notebook --execute "$nb" --output _check.ipynb
  rm notebooks/_check.ipynb
done
```
