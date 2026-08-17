# hyperon-ecan

A **research implementation** of Economic Attention Networks for [OpenCog Hyperon](https://github.com/trueagi-io/hyperon-experimental).

Classic OpenCog had ECAN. iCog ported the Classic MindAgents to MeTTa as
[metta-attention](https://github.com/iCog-Labs-Dev/metta-attention) (PeTTa).
This repo is a different slice: Iklé 2009 Variant 1 on
hyperon-experimental, plus embedding Hebbian links and attention-gated
inference. Example is in [metta-examples/ecan](https://github.com/trueagi-io/metta-examples/tree/main/ecan).
See [COMPARISON.md](COMPARISON.md).

## What is actually new

| Piece | Status in Hyperon before this | This repo |
| --- | --- | --- |
| Classic ECAN MindAgents in MeTTa | [metta-attention](https://github.com/iCog-Labs-Dev/metta-attention) (PeTTa) | Not that |
| ECAN Variant 1 (Iklé, Pitt, Goertzel, Sellman 2009) on hyperon-experimental | Not shipped | Running: STI/LTI currencies, rent, wages, Hebbian update, left-stochastic diffusion, forgetting |
| Neural Space as LLM prompt wrapper | Toy example, 2023 | Not that |
| Implicit Hebbian from embeddings | Does not exist | Cosine links mixed into the diffusion matrix |
| Attention-gated inference loop | Described in the 2009 paper, never a Hyperon module | Perceive → attend → infer → update |

The neural part is the research increment, not a wrapper around an LLM. Atoms get hashed n-gram vectors (or any vector you inject). Those vectors create *implicit* Hebbian weights. Attention can jump from `dog` to `wolf` with no symbolic `Inheritance` edge.

## Install

```bash
python -m pip install -e ".[dev]"
```

Optional MeTTa bindings:

```bash
python -m pip install -e ".[hyperon]"
```

## Run the two experiments

```bash
python examples/associative_memory.py
python examples/attention_gated_inference.py
python -m pytest
```

`associative_memory.py` imprints a sparse pattern (`dog bark leash park`), lets it decay, then cues `dog` + `bark`. The rest of the pattern should re-enter attentional focus. Distractors should not.

`attention_gated_inference.py` proves `(Inheritance dog animal)` while an unrelated oak/tree/plant subgraph sits in the same Atomspace. `wolf` gets STI through embeddings; `oak` does not.

## Use it as a library

```python
from hyperon_ecan import ECAN, AttentionChainer, CognitiveCycle, Triple

net = ECAN()
chainer = AttentionChainer(net)
chainer.assert_fact(Triple("Inheritance", "dog", "mammal"))
chainer.assert_fact(Triple("Inheritance", "mammal", "animal"))
net.stimulate(["dog"])
cycle = CognitiveCycle(net, chainer)
log = cycle.run(["dog"], Triple("Inheritance", "dog", "animal"))
print(log[-1].proved, log[-1].focus)
```

## MeTTa walkthrough

```bash
metta examples/concept_attention.metta
```

That script adds facts, clusters `dog`/`wolf` vs `oak`/`tree`, stimulates `dog`, then prints focus, neural neighbors, STI, and one inference step. `wolf` should pick up STI; `oak` should not. `!(ecan-infer)` should return `(Inheritance dog animal)`.

## How to contribute this upstream

People *do* contribute to OpenCog. The project is just split:

- **OpenCog Classic** (`opencog/atomspace`, `learn`, `sensory`) — Linas Vepstas still accepts work. The org README literally says HELP WANTED.
- **OpenCog Hyperon** (`trueagi-io/hyperon-experimental`, `trueagi-io/PLN`, `trueagi-io/metta-examples`) — this is where new cognitive algorithms belong.

A research module like this is not a drive-by docstring PR. The path that Hyperon already uses:

1. Keep the experiment in its own repo (this one).
2. Open an issue on [`trueagi-io/metta-examples`](https://github.com/trueagi-io/metta-examples) or `hyperon-experimental` describing the module and linking the code.
3. If they want it in-tree, send a PR that adds `examples/ecan/` plus a short note in their README.
4. Join the MeTTa study group / OpenCog Discord / Matrix and point at the issue. Do not wait for permission to start the research.

See [RESEARCH.md](RESEARCH.md) for the equations, what was left unimplemented, and what to measure next.

## References

- Iklé, Pitt, Goertzel, Sellman. *Economic Attention Networks: Associative Memory and Resource Allocation for General Intelligence.* AGI 2009.
- Goertzel et al. *OpenCog Hyperon: A Framework for AGI at the Human Level and Beyond.* arXiv:2310.18318.
- OpenCog wiki, *Economic attention allocation* (Classic, now marked obsolete).
