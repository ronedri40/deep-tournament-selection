# Deep Tournament Selection (DTS) for EC-KitY

A learned, RL-trained **selection** operator for genetic algorithms, packaged as an
[EC-KitY](https://github.com/EC-KitY/EC-KitY) `SelectionMethod`. This repo follows the same
structure as [BERT-Mutation-GA](https://github.com/EC-KitY/BERT-Mutation-GA): one custom genetic
operator wrapped in a thin EC-KitY adapter, plus runnable experiments.

Instead of a fixed rule (tournament, roulette, …), DTS *learns* which individuals to select. A
Transformer encoder embeds the whole population, a self-attention pointer network picks tournament
winners, and the policy is trained online with REINFORCE using the population's fitness improvement
as reward.

## Layout

```
deep_tournament_selection/
  selection/
    eckity_adapter.py        # DeepTournamentSelection(SelectionMethod)  <-- the EC-KitY glue
    deep_neural_selection.py # core DTS: REINFORCE training loop
    population_to_vec_transformer.py / self_attention_pointer.py / ...  # the neural networks
    tournament_utils.py      # standalone tournament helper (teacher-forcing baseline)
  experiments/
    common.py                # build_dts_operator(...) -> ready-to-use SelectionMethod
    one_max.py               # OneMax demo (built-in EC-KitY problem) using DTS
```

This is a standalone repository that depends on EC-KitY (it does not modify the EC-KitY source
tree). Install it next to EC-KitY and it works as a drop-in selection operator.

## How it plugs in

`DeepTournamentSelection` extends EC-KitY's `SelectionMethod`. Its `select(source_inds, dest_inds)`
converts the `Individual` objects into the numpy `(population, fitness_dict)` form the learned
operator expects, runs the policy, then clones the chosen individuals back into `dest_inds`.

```python
from deep_tournament_selection.experiments.common import build_dts_operator

dts = build_dts_operator(population_size=300, vocab_size=2)  # vocab_size = max gene value + 1
# ... then in your Subpopulation:
selection_methods=[(dts, 1)]
```

## Install & run

```bash
pip install -e .            # or: pip install -r requirements.txt
python -m deep_tournament_selection.experiments.one_max          # full run (pop=300, len=100)
# quick smoke test:
python -m deep_tournament_selection.experiments.one_max --population-size 60 --length 40 --generations 30
```

You should see best fitness climb toward `length`, with periodic `loss: ..., reward: ...` lines
printed by the DTS training loop — confirming the operator is learning inside EC-KitY's evolution.
