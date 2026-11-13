# Diabetes Labs Variant 05

Репозиторий лабораторных работ по варианту 05 Diabetes и проблеме
мультиколлинеарности.

## Установка

Создайте окружение и установите зависимости:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

## Запуск работ

```bash
PYTHONPATH=src python -m lab2.run --config configs/lab2.yaml
PYTHONPATH=src python -m lab3.run --config configs/lab3.yaml
PYTHONPATH=src python -m lab4.run --config configs/lab4.yaml
PYTHONPATH=src python -m lab5.run --config configs/lab5.yaml
PYTHONPATH=src python -m lab6.run --config configs/lab6.yaml
PYTHONPATH=src python -m lab7.run --config configs/lab7.yaml
lab8-run --config configs/lab8.yaml
```

Команды ЛР8 принимают `--config` и `--source real|synthetic`:

```bash
train_ml --config configs/lab8.yaml --source real
train_dl --config configs/lab8.yaml --source real
train_validator --config configs/lab8.yaml --source real
generate --config configs/lab8.yaml --source real
serve --config configs/lab8.yaml --source real
evaluate --config configs/lab8.yaml --source real
monitor --config configs/lab8.yaml --source real
```

Проверка:

```bash
PYTHONPATH=src pytest -q
```

Результаты находятся в `reports/LAB2/`–`reports/LAB8/`. Синтетические данные
ЛР5 сохраняются в `data/synthetic/`.
