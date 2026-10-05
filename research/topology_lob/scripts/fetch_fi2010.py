from __future__ import annotations
import argparse
import hashlib
import json
import urllib.request
import zipfile
from pathlib import Path

DEFAULT_URL = "https://raw.githubusercontent.com/zcakhaa/DeepLOB-Deep-Convolutional-Neural-Networks-for-Limit-Order-Books/master/data/data.zip"

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch FI-2010 benchmark archive")
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--out", default="data/external/fi2010.zip")
    parser.add_argument("--extract", action="store_true")
    args = parser.parse_args()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(args.url, out)
    result = {"url": args.url, "archive": str(out), "sha256": sha256(out)}
    if args.extract:
        root = out.parent / "fi2010"
        root.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(out) as archive:
            archive.extractall(root)
        result["extract_root"] = str(root)
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()