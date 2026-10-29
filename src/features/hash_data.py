
import hashlib
import json
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_PATH = BASE_DIR / "data" / "raw" / "diabetes.csv"
MANIFEST_PATH = BASE_DIR / "reports" / "LAB1" / "hash_manifest.json"


def calculate_hash(path):
    sha256 = hashlib.sha256()

    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(65536), b""):
            sha256.update(chunk)

    return sha256.hexdigest()


def create_manifest():
    manifest = {
        "dataset": "Diabetes",
        "source": "sklearn.datasets.load_diabetes",
        "algorithm": "sha256",
        "files": [
            {
                "path": "data/raw/diabetes.csv",
                "sha256": calculate_hash(DATA_PATH),
                "size_bytes": DATA_PATH.stat().st_size,
            }
        ],
    }

    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(MANIFEST_PATH, "w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=4)

    print("Манифест создан!")
    print("SHA256:", manifest["files"][0]["sha256"])
    print("Размер:", manifest["files"][0]["size_bytes"], "байт")


def verify_manifest():
    with open(MANIFEST_PATH, encoding="utf-8") as file:
        manifest = json.load(file)

    all_valid = True

    for entry in manifest["files"]:
        path = BASE_DIR / entry["path"]

        if not path.is_file():
            print("ОШИБКА: файл отсутствует:", path)
            all_valid = False
            continue

        hash_matches = calculate_hash(path) == entry["sha256"]
        size_matches = path.stat().st_size == entry["size_bytes"]

        if hash_matches and size_matches:
            print("СОВПАДАЕТ:", entry["path"])
        else:
            print("ОШИБКА: файл изменён:", entry["path"])
            all_valid = False

    if not all_valid:
        sys.exit(1)


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ("create", "verify"):
        print("Использование: python src/features/hash_data.py create|verify")
        sys.exit(2)

    if sys.argv[1] == "create":
        create_manifest()
    else:
        verify_manifest()
