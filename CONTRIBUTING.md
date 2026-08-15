# Contributing

This repo is itself a contribution *to* OpenCog Hyperon. If you are extending it:

1. Keep the 2009 equations recognizable. If you change rent, wages, or diffusion, say so in the commit and add a test that would fail on the old rule.
2. New research (Variant 2, PLN coupling, DAS paging) belongs in a new module plus a paragraph in `RESEARCH.md`.
3. Do not require an LLM or a network call on the default path. Neural spreading has to run offline.

## Sending this upstream

1. Fork [trueagi-io/metta-examples](https://github.com/trueagi-io/metta-examples).
2. Add `ecan/` with this package (or a submodule) and a short README that points at `RESEARCH.md`.
3. Open a PR. In the body, link Iklé et al. 2009 and state what is *not* a port of Classic.
4. In parallel, open an issue on [trueagi-io/hyperon-experimental](https://github.com/trueagi-io/hyperon-experimental) titled along the lines of “ECAN / attention space prototype” so the interpreter team sees it.

Hyperon contributing rules that apply if you touch their tree: one branch per change, tests green, commit subject under 73 characters, `Fixes #N` in the PR body not the subject. See their `docs/CONTRIBUTING.md`.
