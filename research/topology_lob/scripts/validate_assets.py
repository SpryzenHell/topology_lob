from __future__ import annotations

from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets"

EXPECTED = {
    "main.png": (1200, 675),
    "figures/experimental_validation_dashboard.png": None,
    "figures/robustness_dashboard.png": None,
    "figures/data_diagnostics_dashboard.png": None,
    "figures/fi2010_validation_dashboard.png": None,
    "screenshots/report_preview.png": None,
    "screenshots/terminal_demo_run.png": None,
}


def check(path: Path, expected_size: tuple[int, int] | None) -> dict:
    with Image.open(path) as image:
        image.verify()
    with Image.open(path) as image:
        size = image.size
        mode = image.mode
        extrema = image.convert("RGB").getextrema()
        channels = [hi - lo for lo, hi in extrema]
        return {
            "size": size,
            "mode": mode,
            "channel_ranges": channels,
            "nontrivial": max(channels) > 20,
            "expected_size": expected_size,
            "size_ok": expected_size is None or size == expected_size,
        }


def main() -> None:
    failures = []
    for relative, expected in EXPECTED.items():
        path = ASSETS / relative
        if not path.is_file():
            failures.append(f"missing: {relative}")
            continue
        result = check(path, expected)
        print(relative, result)
        if not result["nontrivial"] or not result["size_ok"]:
            failures.append(f"bad image: {relative}: {result}")

    root_main = ROOT.parent.parent / "main.png"
    if not root_main.is_file():
        failures.append("missing: root main.png")
    else:
        root_result = check(root_main, (1200, 675))
        print("root/main.png", root_result)
        if not root_result["nontrivial"] or not root_result["size_ok"]:
            failures.append(f"bad root main.png: {root_result}")

    if failures:
        raise SystemExit("Asset validation failed:\n" + "\n".join(failures))
    print("Asset validation: PASS")


if __name__ == "__main__":
    main()
