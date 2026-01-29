# Time Series Connectivity Toolkit

Минимальный набор для демонстрации анализа связей между временными рядами и экспорта отчёта в Excel.

## Установка

```bash
pip install -r requirements.txt
```

### Опциональные зависимости

- Для методов transfer entropy (TE) требуется пакет `pyinform`:

```bash
pip install pyinform
```

## Как запустить (один гарантированный путь)

CLI:

```bash
python tool.py demo.csv --output demo_output.xlsx
```

Streamlit:

```bash
streamlit run app.py
```

## Что считается стабильным, а что экспериментальным

**Стабильные методы (используются по умолчанию в UI):**
- correlation_full / correlation_partial
- coherence_full
- granger_full

**Экспериментальные/чувствительные методы (выключены по умолчанию):**
- mutinf_full / mutinf_partial (KNN MI/CMI)
- te_full / te_partial / te_directed
- ah_full / ah_partial / ah_directed

## Ограничения корректности (важно знать)

- Данные должны быть синхронны по частоте дискретизации (равный шаг по времени).
- Для лаговых методов нужно достаточно точек: рекомендуется минимум 10–20 точек на лаг.
- Для устойчивых оценок на практике требуется хотя бы 200–300 наблюдений на ряд.

## Демо-артефакты

- `demo.csv` — входные данные для смоук-теста.
- `demo_output.xlsx` — готовый пример отчёта, который создаёт CLI.
