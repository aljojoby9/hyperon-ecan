# hyperon-ecan vs metta-attention

Bitseat pointed at [iCog-Labs-Dev/metta-attention](https://github.com/iCog-Labs-Dev/metta-attention)
on Hyperon issue #1083. That repo is a real MeTTa port of Classic ECAN.
This note is what differs, not a claim that one replaces the other.

## What metta-attention is

A Classic OpenCog attention port:

- MindAgents in MeTTa: HebbianCreation, HebbianUpdating,
  ImportanceDiffusion, RentCollection, Forgetting
- Attention values stored as MeTTa `AV` types on atoms
- WordNet / ConceptNet experiment (insect → poison attention shift)
- Runs on [PeTTa](https://github.com/trueagi-io/PeTTa) (sibling checkout,
  Linux/macOS, SWI-Prolog ≥ 9.3.25)

It is the agent-by-agent rewrite of [singnet/attention](https://github.com/singnet/attention).
No embedding / neural Hebbian path. No attention-gated inference loop.

## What this repo is

A small Hyperon-experimental library:

- Iklé et al. 2009 Variant 1 in one cycle (rent, wages, conjunction
  Hebbian, left-stochastic diffusion, forget)
- Implicit Hebbian from embeddings (`neural_mix`)
- Attention-gated Inheritance chaining (last section of that paper)
- Python core + grounded atoms: `!(import! &self hyperon_ecan)`
- Example now in [metta-examples/ecan](https://github.com/trueagi-io/metta-examples/tree/main/ecan)

It is not a MindAgent port and not MeTTa-native.

## Side by side

| | metta-attention | hyperon-ecan |
| --- | --- | --- |
| Host | PeTTa | hyperon-experimental |
| Language | MeTTa agents | Python + grounded atoms |
| Classic agents | Yes, split out | No, one `cycle()` |
| Neural / embeddings | No | Yes |
| Gated inference | No | Yes |
| Scale | ConceptNet / WordNet experiment | Tiny pattern + inheritance demos |

## If both live under trueagi-io

Keep them separate. metta-attention is the Classic port. This repo is
the 2009-equation + neural + infer prototype on the current Hyperon
interpreter. A later step is wiring STI into `trueagi-io/PLN`, which
metta-attention does not do.
