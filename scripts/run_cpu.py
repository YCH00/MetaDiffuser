"""Run modules in the isolated CPU fallback without changing the CUDA venv."""
from pathlib import Path
import runpy
import sys

root = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(root / ".runtime" / "cpu"), str(root)]
module = sys.argv[1]
sys.argv = [module, *sys.argv[2:]]
runpy.run_module(module, run_name="__main__")

