from __future__ import annotations

import argparse
import json

from .pipeline import run_end_to_end


def main() -> None:
    parser = argparse.ArgumentParser(description="Run laboratory work 8")
    parser.add_argument("--config", default="configs/lab8.yaml")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run a short functional drill; omit for the full real-time load test",
    )
    args = parser.parse_args()
    print(
        json.dumps(
            run_end_to_end(args.config, quick=args.quick),
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
