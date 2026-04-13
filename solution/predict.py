
from __future__ import annotations

# Нужен для корректной работы аннотаций типов в Python 3.10+
import argparse

# Path удобен для работы с путями к файлам и папкам
from pathlib import Path

# Literal ограничивает допустимые значения source: только "top" или "bottom"
# Tuple нужен для аннотации возвращаемого значения predict(...)
from typing import Literal, Tuple

# joblib используем для загрузки sklearn-артефактов:
# baseline моделей и scaler
import joblib

# numpy нужен для формирования признаков и работы с массивами
import numpy as np

# torch нужен для загрузки residual MLP и выполнения inference
import torch
import torch.nn as nn


# Размеры кадров зафиксированы в ТЗ и используются
# для нормализации / денормализации координат
IMG_W = 3200.0
IMG_H = 1800.0

# Допустимые значения источника координаты
SourceType = Literal["top", "bottom"]


class ControlResidualRegressor(nn.Module):
    # Это residual MLP, которая предсказывает поправку
    # к baseline-предсказанию отдельно для top и bottom
    def __init__(self, in_dim: int):
        super().__init__()

        # Общий trunk: сначала извлекаем общее скрытое представление признаков
        self.trunk = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.SiLU(),
            nn.Dropout(0.10),
            nn.Linear(128, 128),
            nn.SiLU(),
            nn.Dropout(0.10),
            nn.Linear(128, 64),
            nn.SiLU(),
        )

        # Голова для source="top"
        self.head_top = nn.Sequential(
            nn.Linear(64, 64),
            nn.SiLU(),
            nn.Linear(64, 2),
        )

        # Голова для source="bottom"
        self.head_bottom = nn.Sequential(
            nn.Linear(64, 64),
            nn.SiLU(),
            nn.Linear(64, 2),
        )

    def forward(self, X: torch.Tensor, source_id: torch.Tensor):
        # Пропускаем признаки через общий trunk
        h = self.trunk(X)

        # Получаем две альтернативные residual-поправки:
        # одну для top, вторую для bottom
        res_top = self.head_top(h)
        res_bottom = self.head_bottom(h)

        # source_id:
        # 0 -> top
        # 1 -> bottom
        # Маской выбираем нужную голову
        mask_bottom = (source_id == 1).float().unsqueeze(1)

        # Если source=top -> берем res_top
        # Если source=bottom -> берем res_bottom
        return res_top * (1.0 - mask_bottom) + res_bottom * mask_bottom


def build_residual_features(src_x: float, src_y: float, base_x: float, base_y: float) -> np.ndarray:
    # Нормализуем исходную координату top/bottom в [0, 1]
    x = np.array([[src_x / IMG_W]], dtype=np.float32)
    y = np.array([[src_y / IMG_H]], dtype=np.float32)

    # Нормализуем baseline-предсказание на door2 в [0, 1]
    bx = np.array([[base_x / IMG_W]], dtype=np.float32)
    by = np.array([[base_y / IMG_H]], dtype=np.float32)

    # Полиномиальные признаки по src
    x2 = x * x
    y2 = y * y
    xy = x * y

    # Полиномиальные признаки по baseline
    bx2 = bx * bx
    by2 = by * by
    bxby = bx * by

    # Кубические признаки
    x3 = x2 * x
    y3 = y2 * y
    x2y = x2 * y
    xy2 = x * y2

    # Синус / косинус — дополнительные нелинейные признаки
    sinx = np.sin(np.pi * x)
    cosx = np.cos(np.pi * x)
    siny = np.sin(np.pi * y)
    cosy = np.cos(np.pi * y)

    # То же самое для baseline-координат
    sinbx = np.sin(np.pi * bx)
    cosbx = np.cos(np.pi * bx)
    sinby = np.sin(np.pi * by)
    cosby = np.cos(np.pi * by)

    # Признаки относительно центра кадра
    dx_from_center = x - 0.5
    dy_from_center = y - 0.5
    dbx_from_center = bx - 0.5
    dby_from_center = by - 0.5

    # Квадрат расстояния до центра для src и baseline
    r2_src = dx_from_center**2 + dy_from_center**2
    r2_base = dbx_from_center**2 + dby_from_center**2

    # Итоговый вектор признаков должен совпадать с тем,
    # что использовался при обучении в train.py
    feats = np.concatenate(
        [
            x, y, bx, by,
            x2, y2, xy,
            bx2, by2, bxby,
            x3, y3, x2y, xy2,
            sinx, cosx, siny, cosy,
            sinbx, cosbx, sinby, cosby,
            dx_from_center, dy_from_center,
            dbx_from_center, dby_from_center,
            r2_src, r2_base,
        ],
        axis=1,
    ).astype(np.float32)

    return feats


class CoordinatePredictor:
    # Главный класс inference:
    # загружает артефакты и умеет делать predict(x, y, source)
    def __init__(self, artifacts_dir: str | Path = "artifacts", device: str | None = None):
        self.artifacts_dir = Path(artifacts_dir)

        # Если явно не задан device, используем cuda при наличии,
        # иначе обычный cpu
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Загружаем baseline-модели отдельно для top и bottom
        self.baseline_top = joblib.load(self.artifacts_dir / "baseline_top_poly3_ridge.joblib")
        self.baseline_bottom = joblib.load(self.artifacts_dir / "baseline_bottom_poly3_ridge.joblib")

        # Загружаем scaler для residual features
        self.feature_scaler = joblib.load(self.artifacts_dir / "feature_scaler.joblib")

        # Загружаем checkpoint residual MLP
        ckpt = torch.load(
            self.artifacts_dir / "control_residual_mlp.pt",
            map_location=self.device,
            weights_only=False,
        )

        # Размер входного признакового вектора
        feature_dim = int(ckpt["feature_dim"])

        # Воссоздаем архитектуру модели
        self.model = ControlResidualRegressor(in_dim=feature_dim).to(self.device)

        # Загружаем веса
        self.model.load_state_dict(ckpt["model_state_dict"])

        # Переводим модель в eval-режим
        self.model.eval()

    def predict(self, x: float, y: float, source: SourceType) -> Tuple[float, float]:
        # Проверяем корректность source
        if source not in {"top", "bottom"}:
            raise ValueError("source must be 'top' or 'bottom'")

        # Формируем вход для baseline
        xy = np.array([[float(x), float(y)]], dtype=np.float32)

        # Выбираем нужную baseline-модель
        baseline_model = self.baseline_top if source == "top" else self.baseline_bottom

        # Получаем baseline-предсказание на координату door2
        base_pred = baseline_model.predict(xy).astype(np.float32)[0]
        base_x, base_y = float(base_pred[0]), float(base_pred[1])

        # Строим residual-признаки на основе:
        # - исходной координаты
        # - baseline-предсказания
        feats_raw = build_residual_features(x, y, base_x, base_y)

        # Применяем scaler, который был обучен на train
        feats = self.feature_scaler.transform(feats_raw).astype(np.float32)

        # Конвертируем признаки в torch.Tensor
        X = torch.tensor(feats, dtype=torch.float32, device=self.device)

        # Кодируем source:
        # top -> 0
        # bottom -> 1
        source_id = torch.tensor([0 if source == "top" else 1], dtype=torch.long, device=self.device)

        # Предсказываем residual-поправку
        with torch.no_grad():
            pred_res = self.model(X, source_id).cpu().numpy()[0]

        # Складываем baseline + residual в нормализованном пространстве
        pred_x_n = np.clip(base_x / IMG_W + pred_res[0], 0.0, 1.0)
        pred_y_n = np.clip(base_y / IMG_H + pred_res[1], 0.0, 1.0)

        # Переводим обратно в пиксели
        pred_x = float(pred_x_n * IMG_W)
        pred_y = float(pred_y_n * IMG_H)

        return pred_x, pred_y


def parse_args() -> argparse.Namespace:
    # CLI-интерфейс для быстрого запуска predict.py из терминала
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--x", type=float, required=True)
    parser.add_argument("--y", type=float, required=True)
    parser.add_argument("--source", type=str, choices=["top", "bottom"], required=True)
    return parser.parse_args()


if __name__ == "__main__":
    # Точка входа при запуске файла из консоли
    args = parse_args()

    # Создаем predictor и делаем одно предсказание
    predictor = CoordinatePredictor(artifacts_dir=args.artifacts_dir)
    pred_x, pred_y = predictor.predict(args.x, args.y, args.source)

    # Печатаем результат в удобном виде
    print(f"pred_x={pred_x:.3f}")
    print(f"pred_y={pred_y:.3f}")
