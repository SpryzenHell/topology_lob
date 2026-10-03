from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fi2010 import convert_file


def main():
    parser = argparse.ArgumentParser(
        description="Convert one FI-2010 raw text split to canonical Topology-LOB CSV"
    )
    parser.add_argument("--input", required=True, help="FI-2010 .txt split")
    parser.add_argument("--output", required=True, help="canonical L2 CSV")
    parser.add_argument(
        "--allow-normalized",
        action="store_true",
        help="allow ZScore/MinMax files (not recommended for price-volume topology)",
    )
    args = parser.parse_args()

    result = convert_file(
        args.input,
        args.output,
        require_decpre=not args.allow_normalized,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()