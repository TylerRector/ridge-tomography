from __future__ import annotations

import json
from pathlib import Path

def scaled(value, scale):
    return int(round(value * scale))

def main():
    root = Path(__file__).parent.parent
    report = json.loads((root / "results" / "benchmark.json").read_text(encoding="utf-8"))
    methods = {entry["method"]: entry for entry in report["methods"]}
    weighted = methods["Weighted FBP"]
    iterative = methods["Iterative curve-endpoint"]
    expected = {
        "weighted_psnr_millidb": 17773,
        "iterative_psnr_millidb": 23743,
        "target_psnr_millidb": 16010,
        "weighted_false_skeleton_centipx": 50125,
        "iterative_false_skeleton_centipx": 2025,
    }
    actual = {
        "weighted_psnr_millidb": scaled(weighted["mean_psnr_db"], 1000),
        "iterative_psnr_millidb": scaled(iterative["mean_psnr_db"], 1000),
        "target_psnr_millidb": scaled(report["target_psnr_db"], 1000),
        "weighted_false_skeleton_centipx": scaled(weighted["mean_false_skeleton_length_px"], 100),
        "iterative_false_skeleton_centipx": scaled(iterative["mean_false_skeleton_length_px"], 100),
    }
    if actual != expected:
        raise SystemExit(json.dumps({"expected": expected, "actual": actual}, indent=2))
    print("spec cross-check passed")

if __name__ == "__main__":
    main()
