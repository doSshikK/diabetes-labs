from __future__ import annotations

import argparse
import json


def _parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", default="configs/lab8.yaml")
    parser.add_argument("--source", choices=("real", "synthetic"), default="real")
    return parser


def train_ml_command() -> None:
    args = _parser("Train the classical model").parse_args()
    from .config import load_config
    from .pipeline import train_ml

    result = train_ml(load_config(args.config), args.source)
    print(json.dumps(result, indent=2, ensure_ascii=False))


def train_dl_command() -> None:
    args = _parser("Train the neural model").parse_args()
    from .config import load_config
    from .pipeline import train_dl

    result = train_dl(load_config(args.config), args.source)
    print(json.dumps(result, indent=2, ensure_ascii=False))


def train_validator_command() -> None:
    args = _parser("Train the input validator").parse_args()
    from .config import load_config
    from .pipeline import train_validator

    result = train_validator(load_config(args.config), args.source)
    print(json.dumps(result, indent=2, ensure_ascii=False))


def generate_command() -> None:
    args = _parser("Generate a controlled dataset").parse_args()
    from .config import load_config
    from .pipeline import generate_data

    result = generate_data(load_config(args.config), args.source)
    print(json.dumps(result, indent=2, ensure_ascii=False))


def evaluate_command() -> None:
    args = _parser("Evaluate all models for one source").parse_args()
    from .config import load_config
    from .pipeline import evaluate_source

    result = evaluate_source(load_config(args.config), args.source)
    print(json.dumps(result, indent=2, ensure_ascii=False))


def monitor_command() -> None:
    args = _parser("Run alert drills").parse_args()
    from .config import load_config
    from .pipeline import run_alert_drill

    result = run_alert_drill(load_config(args.config), args.source)
    print(json.dumps(result, indent=2, ensure_ascii=False))


def serve_command() -> None:
    parser = _parser("Start the prediction service")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    import uvicorn

    from .config import load_config
    from .pipeline import create_service_app

    config = load_config(args.config)
    app = create_service_app(config, args.source)
    if args.dry_run:
        print(json.dumps({"status": "ready", "source": args.source}, ensure_ascii=False))
        return
    uvicorn.run(app, host=args.host, port=args.port)
