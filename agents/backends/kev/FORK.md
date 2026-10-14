# Local Kev Fork

Upstream: https://github.com/jaredpalmer/kev

Base tag: `kev-1.0`

Base commit: `6b719c3c3f367295f6ef336f4f751cf5ff970abc`

This directory vendors the upstream `kev/` package, `tests/`, package metadata,
README and Apache-2.0 license. It is tracked by the assistant repository, not a
nested Git repository. Upstream examples, evaluation datasets and deployment
scripts are not included.

## Local Changes

- NumPy requirement: `>=2.5.3` becomes `>=1.26,<2.3` to overlap with Moshi.
- Torch requirement: `>=2.6,<2.9` becomes `>=2.6,<2.10` to include the assistant's
  existing `torch==2.9.1` pin.

These are experimental compatibility changes, not upstream support guarantees.
No model weights are included. The assistant downloads its configured checkpoint
from Hugging Face and runs inference in-process.

## Validation

Dependency resolution and runtime compatibility must be checked separately.
The upstream test suite includes tests requiring evaluation datasets or scripts
that are not vendored here; those tests cannot run from this subset alone.

The Hub conflict is addressed by the separate Moshi fork in `voice/moshi`.
`uv sync` and `uv pip check` now pass. Both Moshi and
Kev inference modules import with Torch 2.9.1, NumPy 2.2.6, Transformers 5.19.0,
Hub 1.33.0 and Safetensors 0.8.0. The Kev dependency is restricted to Python 3.12/3.13
to match upstream's supported range without raising the assistant's minimum.

Moshi's small reference-model/audio tests and focused assistant tests pass;
this does not validate Kev's full checkpoint inference. No model weights were
downloaded for these compatibility checks.