# Pretrained DPA

<!-- [![PyPI - Version](https://img.shields.io/pypi/v/pretrained-dpa)](https://pypi.org/p/pretrained-dpa) -->

`pretrained-dpa` provides CLI tools for managing pretrained DPA model files.

## CLI

### Download a model

```bash
pretrained-dpa download DPA-3.2-5M
pretrained-dpa download DPA-3.1-3M
```

This command will download the model file from Hugging Face to:

```text
~/.cache/pretrained-dpa/models/DPA-3.2-5M.pt
```

The CLI validates SHA256 against metadata in `pretrained_dpa/models.json`.
If an existing cached file fails verification, it is deleted and re-downloaded.
If a fresh download fails checksum verification, it is deleted and the command exits with an error.

Each model can provide multiple download sources (`urls`).
The CLI probes sources in parallel and tries the fastest reachable one first,
then automatically falls back to slower/failed sources if needed.

## DeePMD-kit integration: `*.pretrained` alias

This package also registers a `deepmd.backend` entrypoint backend named `pretrained`.

After installing `pretrained-dpa`, you can directly pass a pretrained alias file name to DeepMD:

```python
from deepmd.infer import DeepPot

dp = DeepPot("DPA-3.2-5M.pretrained")
```

Behavior:

- parse alias `DPA-3.2-5M.pretrained` → model name `DPA-3.2-5M`
- resolve to cached model path `~/.cache/pretrained-dpa/models/DPA-3.2-5M.pt`
- if missing (or checksum mismatch), auto-download and verify SHA256
- delegate evaluation to the real backend based on resolved model suffix (`.pt`)
