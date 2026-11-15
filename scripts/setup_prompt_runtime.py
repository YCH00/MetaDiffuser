"""Install numerical Prompt-DT dependencies into the invoking Python environment.

Transformers 4.5.1 is used unchanged. Its optional text tokenization backend has
no compatible old wheel for our Python 3.11 runtime and is not used by Prompt-DT.
"""
import subprocess
import sys

subprocess.run([sys.executable, "-m", "pip", "install", "transformers==4.5.1", "--no-deps"], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "requests", "regex", "sacremoses", "tqdm", "packaging", "filelock"], check=True)
