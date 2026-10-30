# Diabetes Labs Variant 05

Репозиторий лабораторных работ по варианту 05 Diabetes и проблеме
мультиколлинеарности.

## Лабораторная работа 2

Создайте окружение и установите зависимости:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Запуск воспроизводимого ML-пайплайна:

```bash
PYTHONPATH=src python -m lab2.run --config configs/lab2.yaml
```

Проверка негативных контролей и защиты от утечки:

```bash
PYTHONPATH=src pytest -q
```

Основные результаты находятся в `reports/LAB2/`: отчёт, таблица метрик,
модель, журнал невалидных конфигураций, проверка утечки и проверка
ограниченной памяти чанкового режима.
