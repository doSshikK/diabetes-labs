from __future__ import annotations

import argparse
import json

from .generator import run_lab5


def main() -> None:
    parser = argparse.ArgumentParser(description="Run laboratory work 5")
    parser.add_argument("--config", default="configs/lab5.yaml")
    args = parser.parse_args()
    print(json.dumps(run_lab5(args.config), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
