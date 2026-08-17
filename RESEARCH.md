# Research note: ECAN + Neural Space for Hyperon

## Why this exists

OpenCog Hyperon is a ground-up rewrite of OpenCog. The 2023 Hyperon paper is explicit that the old cognitive algorithms should come back **as MeTTa programs**, not as a C++ port of Classic:

> if we want to implement different components like PLN, MOSES, ECAN, etc. for Hyperon in MeTTa

As of 2026:

- MeTTa / `hyperon-experimental` is real and installable.
- Distributed Atomspace is real.
- PLN has `trueagi-io/PLN` and `pln-experimental`.
- Classic ECAN is marked obsolete. iCog's [metta-attention](https://github.com/iCog-Labs-Dev/metta-attention) ports the MindAgents to MeTTa on PeTTa.
- hyperon-experimental still had no Iklé-2009 + neural + gated-infer module.
- “Neural Space” in 2023 Hyperon talks was an LLM-calling `Space.query` demo, not an embedding geometry that participates in attention.

See [COMPARISON.md](COMPARISON.md). The remaining hole on this interpreter is not “another Classic agent port.”

## What we implemented

### Variant 1 of Iklé, Pitt, Goertzel & Sellman (2009)

Each atom holds two currencies:

- **STI** — short-term importance, working memory, attentional focus.
- **LTI** — long-term importance, whether the atom is worth keeping.

A central bank holds the rest. Stimulus is a transfer from the bank to an atom. Rent is a transfer back.

Rent (paper, `s_i > 0`):

```
rent_i = <Rent> * max(0, log(20 s_i / recentMaxSTI) / 2)
```

Consequence, used as a test: rent is exactly 0 when `s_i ≤ recentMaxSTI / 20`. That is the same cutoff the 2009 convergence argument relies on.

Hebbian update (conjunction form):

```
norm_i = s_i / recentMaxSTI          (s_i ≥ 0)
conj   = norm_i * norm_j
c'_ij  = <ConjDecay> * conj + (1 - conj) * c_ij
```

Diffusion: build a left-stochastic matrix `D` from the Hebbian weights, cap off-diagonal mass at `<MaxSpread>`, send a min-max normalized STI vector through `D`, rescale. Total STI in `{atoms ∪ bank}` is conserved under stimulus and rent. Diffusion redistributes; it does not mint.

Attentional focus = atoms with `STI ≥ <FocusBoundary>`.

Forgetting = drop a fraction of the lowest-LTI atoms that are outside the focus and not marked VLTI.

### Extension that did not exist: neural Hebbian

Classic ECAN only spreads along explicit `HebbianLink` / `InverseHebbianLink` atoms. We mix in an implicit graph:

```
c_neural(i, j) = max(0, cos(e_i, e_j) - θ)
c_hybrid       = (1 - λ) c_symbolic + λ c_neural
```

Default `e_i` is a deterministic signed hashed character n-gram vector. That is enough to put `dog` near `dogs` / `wolf` without a trained model. Any vector can be injected (`set_embedding`) so this is also the attachment point for a real Hyperon Neural Space (LLM embeddings, BioDAS vectors, etc.).

This is not MeTTa-Motto. MeTTa-Motto asks an LLM questions. This makes the **attention currency itself** flow through a vector geometry.

### Extension that was described and never shipped: attention-gated chaining

Last section of the 2009 paper:

1. Choose an inference step among atoms that LTI has kept in RAM, weighted by STI.
2. Pay the atoms in that step.
3. Run ECAN.
4. Repeat.

`AttentionChainer` + `CognitiveCycle` is that loop for `Inheritance` transitivity. The oak/tree/plant subgraph in the same space is the control: if attention works, `(Inheritance dog animal)` is derived and `(Inheritance oak plant)` is not.

## Claims the tests actually check

- Rent vanishes below `recentMaxSTI / 20`.
- Stimulus is a bank transfer, not a free increment.
- Co-attended atoms grow a Hebbian weight.
- Low-LTI atoms outside the focus can be forgotten.
- `dog` is closer to `dogs` than to `invoice` under the default embedding.
- With `neural_mix = 1` and no symbolic edge, STI still moves from `dog` to `dogs` more than to `invoice`.
- After imprint → decay → cue, a stored pattern re-enters focus and distractors do not.
- The chainer proves `(Inheritance dog animal)` and does not expand the unfocused oak subgraph.
- `wolf` (neural neighbor) outranks `oak` (unrelated) on STI after a `dog` percept.

## What this is not

- Not a line-for-line port of Classic `opencog/attention`. Classic had more MindAgents, InverseHebbian, and a different scheduler.
- Not Variant 2 (AVDIFF / Lyapunov welfare). That is the obvious next coding step.
- Not a MeTTa-native rewrite of the integrator. The Python core is the reference dynamics. `hyperon_ecan/ext.py` is a grounded-atom façade so MeTTa can drive it. A pure-MeTTa ECAN would be a follow-up PR once these equations are accepted.
- Not a claim of AGI. It is a resource allocator and an associative memory.

## What to do next (real research, still uncoded)

1. **Variant 2** with AVDIFF as a measured Lyapunov-like quantity. The 2009 paper sketches it; nobody published a Hyperon run.
2. **Wire this into `trueagi-io/PLN`** so the backward chainer samples clauses by STI instead of uninformed order. That is the experiment the paper asked for in 2009.
3. **Replace hashed n-grams with a Hyperon Neural Space** that stores embeddings as grounded atoms and answers `match` by cosine. The Space API already exists (`AbstractSpace.query`).
4. **DAS-backed LTI.** Very-long-term importance should decide what is paged to Distributed Atomspace, not just what is deleted from a dict.
5. **Hebbian links as first-class MeTTa expressions** so the connection matrix is inspectable and rewritable by MeTTa itself.

## How to talk about this to the Hyperon team

Lead with the gap, not with a rewrite of their README.

> Hyperon still has no ECAN. I implemented Variant 1 from Iklé 2009, added implicit Hebbian spreading from embeddings, and ran the associative-memory and attention-gated-inference experiments the paper described. Code: \<url\>. Happy to reshape it as a `metta-examples` tree or as grounded atoms against your Space API.

That is a contribution. A typo fix in `CONTRIBUTING.md` is also a contribution, but it is not the one this repo is for.
