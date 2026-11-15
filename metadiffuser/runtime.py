import json
import os
import platform
import random
import hashlib
import pickle
import types
from pathlib import Path
import numpy as np
import torch


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def rng_state():
    numpy_state = list(np.random.get_state())
    numpy_state[1] = numpy_state[1].tolist()
    return {"python": random.getstate(), "numpy": tuple(numpy_state),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def restore_rng(state):
    random.setstate(state["python"])
    numpy_state = list(state["numpy"])
    numpy_state[1] = np.asarray(numpy_state[1], dtype=np.uint32)
    np.random.set_state(tuple(numpy_state))
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])


def atomic_save(payload, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


class _NumpyPortableUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        # NumPy 2 pickles refer to numpy._core; legacy MuJoCo needs NumPy 1.23.
        # numpy.core is provided by both versions for these ndarray primitives.
        if module == "numpy._core" or module.startswith("numpy._core."):
            module = module.replace("numpy._core", "numpy.core", 1)
        return super().find_class(module, name)


_checkpoint_pickle = types.ModuleType("checkpoint_pickle")
_checkpoint_pickle.__dict__.update(vars(pickle))
_checkpoint_pickle.Unpickler = _NumpyPortableUnpickler


def load_checkpoint(path, device="cpu"):
    # Only load trusted, locally produced checkpoints (optimizer/RNG use pickle).
    return torch.load(path, map_location=device, weights_only=False, pickle_module=_checkpoint_pickle)


def json_write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def log_record(path, record):
    with Path(path).open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, allow_nan=False) + "\n")


def system_info():
    return {"python": platform.python_version(), "platform": platform.platform(),
            "torch": torch.__version__, "numpy": np.__version__, "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None}


def snapshot_source(run):
    """Preserve the exact local implementation for every start/resume segment."""
    package = Path(__file__).resolve().parent
    sources = {p.name: p.read_bytes() for p in sorted(package.glob("*.py"))}
    hashes = {name: hashlib.sha256(content).hexdigest() for name, content in sources.items()}
    version = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    root = Path(run) / "source_snapshots" / version
    root.mkdir(parents=True, exist_ok=True)
    for name, content in sources.items():
        destination = root / name
        if not destination.exists():
            destination.write_bytes(content)
    json_write(root / "sha256.json", hashes)
    log_record(Path(run) / "source_history.jsonl", {"source_sha256": version,
               "note": "Source at this process start/resume; source changes are retained as separate snapshots."})
    return version

