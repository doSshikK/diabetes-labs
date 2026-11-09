from __future__ import annotations

import argparse
import json

from .pipeline import run_lab3


def main() -> None:
    parser = argparse.ArgumentParser(description="Run laboratory work 3")
    parser.add_argument("--config", default="configs/lab3.yaml")
    args = parser.parse_args()
    print(json.dumps(run_lab3(args.config), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
