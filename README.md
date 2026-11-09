# Diabetes Labs Variant 05

Репозиторий лабораторных работ по варианту 05 Diabetes и проблеме
мультиколлинеарности.

## Установка

Создайте окружение и установите зависимости:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Запуск работ

```bash
PYTHONPATH=src python -m lab2.run --config configs/lab2.yaml
PYTHONPATH=src python -m lab3.run --config configs/lab3.yaml
PYTHONPATH=src python -m lab4.run --config configs/lab4.yaml
PYTHONPATH=src python -m lab5.run --config configs/lab5.yaml
```

Проверка:

```bash
PYTHONPATH=src pytest -q
```

Результаты находятся в `reports/LAB2/`–`reports/LAB5/`. Синтетические данные
ЛР5 сохраняются в `data/synthetic/`.
