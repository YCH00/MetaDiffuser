"""Download official legacy engines and public activation file into workspace."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import tarfile
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1] / ".runtime" / "legacy"
SOURCES = {
    "mjpro131_linux.zip": "https://roboti.us/download/mjpro131_linux.zip",
    "mujoco200_linux.zip": "https://roboti.us/download/mujoco200_linux.zip",
    "mjkey.txt": "https://roboti.us/file/mjkey.txt",
    "license.html": "https://roboti.us/license.html",
    "uv-linux.tar.gz": "https://github.com/astral-sh/uv/releases/download/0.5.31/uv-x86_64-unknown-linux-gnu.tar.gz",
}


def fetch(item):
    name, url = item
    path = ROOT / name
    if not path.exists():
        temporary = path.with_suffix(path.suffix + ".part")
        with urllib.request.urlopen(url, timeout=60) as response, temporary.open("wb") as stream:
            while block := response.read(1024 * 1024):
                stream.write(block)
        temporary.replace(path)
    return name, {"url": url, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "bytes": path.stat().st_size}


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=4) as executor:
        records = dict(executor.map(fetch, SOURCES.items()))
    for filename in ("mjpro131_linux.zip", "mujoco200_linux.zip"):
        with zipfile.ZipFile(ROOT / filename) as archive:
            for entry in archive.infolist():
                if not (ROOT / entry.filename).resolve().is_relative_to(ROOT.resolve()):
                    raise ValueError("Invalid archive member")
            archive.extractall(ROOT)
    with tarfile.open(ROOT / "uv-linux.tar.gz") as archive:
        for entry in archive.getmembers():
            if entry.issym() or entry.islnk() or not (ROOT / entry.name).resolve().is_relative_to(ROOT.resolve()):
                raise ValueError("Invalid archive member")
        archive.extractall(ROOT)
    (ROOT / "downloads.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    print(json.dumps({name: record["bytes"] for name, record in records.items()}, indent=2))


if __name__ == "__main__":
    main()

