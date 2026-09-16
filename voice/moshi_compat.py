"""Runtime compatibility settings for the Kyutai Moshi package.

Moshi lazily wraps a few inference kernels with ``torch.compile``. The
Inductor backend used by that path requires Triton, which is not available in
the native Windows environment supported by this application. Moshi exposes
the ``NO_TORCH_COMPILE`` environment switch for its lazy compiler wrapper.

This module must be imported before any Moshi model modules are imported.
"""

import os
import sys

if sys.platform == "win32":
    # Do not overwrite an explicit user choice. On Windows the default is
    # eager execution because native Triton support is unavailable.
    os.environ.setdefault("NO_TORCH_COMPILE", "1")
