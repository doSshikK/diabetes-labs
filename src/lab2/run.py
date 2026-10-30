from __future__ import annotations

import argparse
import json

from .pipeline import run_lab2


def main() -> None:
    parser = argparse.ArgumentParser(description="Run laboratory work 2 pipeline")
    parser.add_argument("--config", default="configs/lab2.yaml")
    args = parser.parse_args()
    summary = run_lab2(args.config)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
