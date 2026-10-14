# Local Moshi Fork

Upstream: https://github.com/kyutai-labs/moshi

Base release: `moshi==0.2.13`, the same release previously installed here.

Source archive: https://pypi.org/project/moshi/0.2.13/#files

Archive filename: `moshi-0.2.13.tar.gz`

SHA-256: `f0c83aea037b1e2e3aefe5f9c2e77a0d6b5ec9b7d4e986e259990f6636706eac`

The source distribution is vendored here without generated egg-info or PKG-INFO
files. Upstream tests, fixtures, README and both MIT license files are preserved.
This is not a nested Git repository. No model checkpoints are included.

## Local Changes

The `huggingface-hub` requirement changes from `>=0.24,<1.0.0` to `>=1.5,<2.0`
to overlap with Transformers 5.17 used by the local Kev fork. Torch remains
`>=2.2.0,<2.10`, including the assistant's `torch==2.9.1` pin. Model code is
unchanged.

The `safetensors` requirement changes from `>=0.4.0,<0.8.0` to `>=0.8.0,<0.9.0`
to satisfy Transformers while bounding the compatibility experiment to 0.8.x.
The upstream `requirements.txt` is preserved as historical reference; install
this fork through its `pyproject.toml`, not that frozen requirements file.

The Hub calls in Moshi use `hf_hub_download` and `EntryNotFoundError`. Dependency
resolution, loader error handling and runtime tests must be checked separately;
relaxing metadata alone does not establish full TTS/STT checkpoint compatibility.

## Validation

Tested on Linux, Python 3.12 with Torch 2.9.1+cu128, NumPy 2.2.6, Hub 1.33.0,
Safetensors 0.8.0 and Transformers 5.19.0:

- `uv sync`: passes; both local forks and test dependencies are installed.
- `uv pip check`: passes.
- Moshi LM numerical reference and audio module tests: 34 pass.
- Assistant TTS/cache/Hub-loader tests: 9 pass.
- Assistant main/task tests: 87 pass.
- Moshi loader/TTS and Kev checkpoint/model imports: pass.

Run the upstream tests explicitly from the assistant root:

```bash
NO_TORCH_COMPILE=1 .venv/bin/python -m pytest --import-mode=importlib \
	voice/moshi/tests/test_lm.py \
	voice/moshi/moshi/modules/conv_test.py \
	voice/moshi/moshi/modules/seanet_test.py -q
```

These tests use CPU inference and the small upstream fixtures, not full Kyutai
TTS/STT checkpoints. Real audio, CUDA graph/compile execution, Windows and other
Python versions remain unverified. The root pytest configuration collects only
the assistant's `tests/` by default, not the vendored upstream suites.