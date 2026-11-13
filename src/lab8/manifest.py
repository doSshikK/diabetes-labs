from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

IGNORED_PARTS = {".git", ".venv", ".pytest_cache", "__pycache__", "mlruns"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def current_commit(root: Path) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()


def included_files(root: Path, output: Path) -> list[Path]:
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file()
        and path != output
        and path.name != ".DS_Store"
        and not IGNORED_PARTS.intersection(path.relative_to(root).parts)
    )


def build_manifest(root: Path, output: Path) -> dict[str, Any]:
    files = [
        {
            "path": path.relative_to(root).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in included_files(root, output)
    ]
    return {
        "variant": "05 Diabetes",
        "code_commit": current_commit(root),
        "file_count": len(files),
        "files": files,
    }


def write_manifest(root: Path, output: Path) -> dict[str, Any]:
    payload = build_manifest(root, output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return payload


def verify_manifest(root: Path, output: Path) -> list[str]:
    payload = json.loads(output.read_text(encoding="utf-8"))
    errors: list[str] = []
    for item in payload["files"]:
        path = root / item["path"]
        if not path.exists():
            errors.append(f"missing: {item['path']}")
        elif path.stat().st_size != item["size_bytes"]:
            errors.append(f"size mismatch: {item['path']}")
        elif sha256_file(path) != item["sha256"]:
            errors.append(f"hash mismatch: {item['path']}")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description="Build and verify artifact manifest")
    parser.add_argument("--root", default=".")
    parser.add_argument("--output", default="reports/LAB8/artifacts_manifest.json")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    output = (root / args.output).resolve()
    if args.verify:
        errors = verify_manifest(root, output)
        print(json.dumps({"passed": not errors, "errors": errors}, ensure_ascii=False))
        raise SystemExit(bool(errors))
    payload = write_manifest(root, output)
    print(json.dumps({"file_count": payload["file_count"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
