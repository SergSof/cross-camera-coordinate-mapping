# Coordinate Mapping Between Cameras

Решение для тестового задания по маппингу координаты `x, y, source -> x', y'`, где `source` ∈ `{top, bottom}`, а выход — координата на камере `door2`.

## Идея решения

Итоговый подход:

1. Для каждого источника (`top`, `bottom`) обучается отдельный baseline `poly3 + ridge`, который по `(x, y)` предсказывает координату на `door2`.
2. Затем обучается residual MLP, которая по координатным признакам предсказывает поправку к baseline.
3. Финальный inference использует только:
   - `x`
   - `y`
   - `source`

То есть итоговое API строго соответствует ограничению ТЗ: без использования изображений на inference.

## Почему выбран именно этот вариант

В ходе экспериментов лучшим оказался `control_residual_mlp`:

- `med_top = 128.72`
- `med_bottom = 140.50`
- `med_all = 134.42`

Mixture-of-Experts в текущем виде прироста не дал и оказался хуже по `med_all`, особенно по `bottom`, поэтому в финальную версию не включён.

## Ограничение постановки

После анализа изображений, можно предположить, что камеры: top, bottom, видимо фиксированны. А фронтальная камера door2, рандомно гуляет (подозреваю, что снимок, сделан телефоном). Тогда, в этом сдучае, только по координатам, мы не сможем добиться хорошей точности. Мы просто получаем какую то усредненную ошибку в пределах 130 - 140 пикселей.
Практический вывод такой:
- модель уже вышла на сильный уровень;
- дальнейший прирост в рамках текущего ТЗ, скорее всего, будет небольшим;
- значительного снижения ошибки без дополнительных признаков или без использования изображения на inference ожидать не стоит.

Поэтому финальный вариант строится вокруг лучшего воспроизводимого решения, а не вокруг дальнейшего перебора архитектур.

## Структура репозитория

```text
.
├── README.md
├── requirements.txt
├── train.py
├── predict.py
└── .gitignore
```

После обучения создаётся папка `artifacts/`:

```text
artifacts/
├── baseline_top_poly3_ridge.joblib
├── baseline_bottom_poly3_ridge.joblib
├── feature_scaler.joblib
├── control_residual_mlp.pt
└── metrics.json
```

## Требования к данным

Ожидается исходный датасет следующего вида:

```text
coord_data/
├── split.json
├── train/
└── val/
```

В ТЗ указано, строго соблюдать разбиение: обучение только на train, оценка только на val.

## Установка

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Обучение

Пример запуска:

```bash
python train.py \
  --data-root /path/to/coord_data \
  --artifacts-dir artifacts \
  --epochs 250 \
  --batch-size 256
```

Что делает `train.py`:

1. Загружает `split.json` и собирает point-level dataframe.
2. Обучает baseline `poly3 + ridge` отдельно для `top` и `bottom`.
3. Строит residual target.
4. Генерирует residual features.
5. Обучает `control_residual_mlp`.
6. Считает метрики на `val`.
7. Сохраняет артефакты в `artifacts/`.

## Inference

Пример использования:

```bash
python predict.py \
  --artifacts-dir artifacts \
  --x 1200 \
  --y 350 \
  --source top
```

Или из Python:

```python
from solution.predict import CoordinatePredictor

predictor = CoordinatePredictor(artifacts_dir="artifacts")
xp, yp = predictor.predict(1200, 350, "top")
print(xp, yp)
```

## Финальные метрики

На текущем лучшем решении:

- `top = 128.72 px`
- `bottom = 140.50 px`
- `med_all = 134.42 px`
