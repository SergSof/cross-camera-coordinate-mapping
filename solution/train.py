
from __future__ import annotations

# Нужен для корректной поддержки аннотаций типов в Python 3.10+
import argparse
import json
import math
import random

# dataclass используем для аккуратной упаковки параметров обучения
from dataclasses import dataclass

# Path удобен для работы с файловой системой
from pathlib import Path

# Типы для лучшей читаемости кода
from typing import Dict, List, Tuple

# joblib используем для сохранения sklearn-моделей и scaler
import joblib

# numpy и pandas нужны для работы с координатами и таблицами
import numpy as np
import pandas as pd

# torch нужен для обучения residual MLP
import torch
import torch.nn as nn
import torch.nn.functional as F

# sklearn нужен для baseline-модели и нормализации признаков
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

# DataLoader / Dataset нужны для подачи данных в PyTorch
from torch.utils.data import DataLoader, Dataset


# Размеры изображения фиксированы по условию задачи
IMG_W = 3200.0
IMG_H = 1800.0


def set_seed(seed: int = 42) -> None:
    # Фиксируем seed для воспроизводимости результатов
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_json(path: Path):
    # Универсальная загрузка json-файлов
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def infer_source_name(coords_filename: str) -> str:
    # По имени json-файла определяем источник:
    # coords_top.json -> top
    # coords_bottom.json -> bottom
    name = coords_filename.lower()
    if "top" in name:
        return "top"
    if "bottom" in name:
        return "bottom"
    raise ValueError(f"Cannot infer source from filename: {coords_filename}")


def pair_points_by_number(image1_coordinates, image2_coordinates):
    #
    # image1_coordinates = door2
    # image2_coordinates = top/bottom
    #
    # Возвращаем:
    # - src_pts: точки из камеры top/bottom
    # - dst_pts: соответствующие точки на door2
    # - numbers: номера общих точек
    #
    door2_map = {p["number"]: (float(p["x"]), float(p["y"])) for p in image1_coordinates}
    src_map = {p["number"]: (float(p["x"]), float(p["y"])) for p in image2_coordinates}
    common_numbers = sorted(set(door2_map.keys()) & set(src_map.keys()))
    src_pts = np.array([src_map[n] for n in common_numbers], dtype=np.float32)
    dst_pts = np.array([door2_map[n] for n in common_numbers], dtype=np.float32)
    return src_pts, dst_pts, common_numbers


def denormalize_xy_array(xy_norm: np.ndarray) -> np.ndarray:
    # Переводим координаты из [0, 1] обратно в пиксели
    out = np.asarray(xy_norm, dtype=np.float32).copy()
    out[..., 0] *= IMG_W
    out[..., 1] *= IMG_H
    return out


def evaluate_predictions(y_true_px: np.ndarray, y_pred_px: np.ndarray) -> Tuple[float, float]:
    # MED = среднее евклидово расстояние между истинной и предсказанной координатой
    distances = np.linalg.norm(y_true_px - y_pred_px, axis=1)
    med = float(distances.mean())

    # RMSE считаем как дополнительную метрику
    rmse = float(math.sqrt(mean_squared_error(y_true_px, y_pred_px)))
    return med, rmse


def collect_unified_point_dataframe(dataset_root: Path, split_json_path: Path) -> pd.DataFrame:
    # Загружаем split.json
    split = load_json(split_json_path)

    # Здесь копим строки для общей point-level таблицы
    rows: List[Dict] = []

    # Строго соблюдаем split из ТЗ:
    # train используется только для обучения,
    # val используется только для оценки
    for split_name in ["train", "val"]:
        for session_rel in split[split_name]:
            session_dir = dataset_root / session_rel
            if not session_dir.exists():
                print(f"WARNING: session dir not found: {session_dir}")
                continue

            # Для каждой сессии отдельно обрабатываем top и bottom
            for coords_file_name in ["coords_top.json", "coords_bottom.json"]:
                coords_path = session_dir / coords_file_name
                if not coords_path.exists():
                    continue

                source = infer_source_name(coords_file_name)
                source_id = 0 if source == "top" else 1
                items = load_json(coords_path)

                for item_idx, item in enumerate(items):
                    src_pts, dst_pts, numbers = pair_points_by_number(
                        item["image1_coordinates"],
                        item["image2_coordinates"],
                    )
                    if len(src_pts) == 0:
                        continue

                    # Каждая точка становится отдельной строкой таблицы
                    for k, number in enumerate(numbers):
                        src_x, src_y = map(float, src_pts[k])
                        dst_x, dst_y = map(float, dst_pts[k])

                        rows.append(
                            {
                                "split": split_name,
                                "session_rel": session_rel,
                                "coords_file": coords_file_name,
                                "source": source,
                                "source_id": source_id,
                                "item_idx": int(item_idx),
                                "point_number": int(number),
                                "src_x": src_x,
                                "src_y": src_y,
                                "dst_x": dst_x,
                                "dst_y": dst_y,
                                "src_x_n": src_x / IMG_W,
                                "src_y_n": src_y / IMG_H,
                                "dst_x_n": dst_x / IMG_W,
                                "dst_y_n": dst_y / IMG_H,
                            }
                        )

    return pd.DataFrame(rows)


def make_poly_model(degree: int = 3, alpha: float = 1.0) -> Pipeline:
    # Baseline-модель:
    # сначала StandardScaler,
    # потом PolynomialFeatures,
    # затем Ridge-регрессия
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("poly", PolynomialFeatures(degree=degree, include_bias=True)),
            ("reg", Ridge(alpha=alpha)),
        ]
    )


def fit_source_baselines(train_df: pd.DataFrame, val_df: pd.DataFrame):
    # Здесь храним:
    # - baseline-модели для top и bottom
    # - сводную таблицу их метрик
    rows = []
    models = {}

    for source_name in ["top", "bottom"]:
        tr = train_df[train_df["source"] == source_name]
        va = val_df[val_df["source"] == source_name]

        # В baseline используем только (src_x, src_y) -> (dst_x, dst_y)
        X_train = tr[["src_x", "src_y"]].values.astype(np.float32)
        Y_train = tr[["dst_x", "dst_y"]].values.astype(np.float32)
        X_val = va[["src_x", "src_y"]].values.astype(np.float32)
        Y_val = va[["dst_x", "dst_y"]].values.astype(np.float32)

        model = make_poly_model(degree=3, alpha=1.0)
        model.fit(X_train, Y_train)

        train_pred = model.predict(X_train)
        val_pred = model.predict(X_val)

        train_med, train_rmse = evaluate_predictions(Y_train, train_pred)
        val_med, val_rmse = evaluate_predictions(Y_val, val_pred)

        models[source_name] = model
        rows.append(
            {
                "source": source_name,
                "model": "separate_poly3_ridge",
                "train_med": train_med,
                "train_rmse": train_rmse,
                "val_med": val_med,
                "val_rmse": val_rmse,
            }
        )

    return models, pd.DataFrame(rows)


def add_baseline_predictions(df_part: pd.DataFrame, baseline_models: Dict[str, Pipeline]) -> pd.DataFrame:
    # Добавляем к каждой строке:
    # - baseline-предсказание
    # - residual target = истинное - baseline
    df_out = df_part.copy()
    baseline_pred_x = np.zeros(len(df_out), dtype=np.float32)
    baseline_pred_y = np.zeros(len(df_out), dtype=np.float32)

    for source_name in ["top", "bottom"]:
        mask = df_out["source"].values == source_name
        X = df_out.loc[mask, ["src_x", "src_y"]].values.astype(np.float32)
        pred = baseline_models[source_name].predict(X).astype(np.float32)
        baseline_pred_x[mask] = pred[:, 0]
        baseline_pred_y[mask] = pred[:, 1]

    df_out["base_x"] = baseline_pred_x
    df_out["base_y"] = baseline_pred_y
    df_out["base_x_n"] = df_out["base_x"] / IMG_W
    df_out["base_y_n"] = df_out["base_y"] / IMG_H

    # Residual target — это поправка к baseline в нормализованных координатах
    df_out["res_dx_n"] = df_out["dst_x_n"] - df_out["base_x_n"]
    df_out["res_dy_n"] = df_out["dst_y_n"] - df_out["base_y_n"]
    return df_out


def build_residual_features(df_part: pd.DataFrame) -> np.ndarray:
    # Исходные нормализованные координаты source
    x = df_part["src_x_n"].values.astype(np.float32).reshape(-1, 1)
    y = df_part["src_y_n"].values.astype(np.float32).reshape(-1, 1)

    # Baseline-предсказание в нормализованном виде
    bx = df_part["base_x_n"].values.astype(np.float32).reshape(-1, 1)
    by = df_part["base_y_n"].values.astype(np.float32).reshape(-1, 1)

    # Полиномиальные признаки
    x2 = x * x
    y2 = y * y
    xy = x * y
    bx2 = bx * bx
    by2 = by * by
    bxby = bx * by

    # Кубические признаки
    x3 = x2 * x
    y3 = y2 * y
    x2y = x2 * y
    xy2 = x * y2

    # Тригонометрические признаки
    sinx = np.sin(np.pi * x)
    cosx = np.cos(np.pi * x)
    siny = np.sin(np.pi * y)
    cosy = np.cos(np.pi * y)
    sinbx = np.sin(np.pi * bx)
    cosbx = np.cos(np.pi * bx)
    sinby = np.sin(np.pi * by)
    cosby = np.cos(np.pi * by)

    # Признаки относительно центра
    dx_from_center = x - 0.5
    dy_from_center = y - 0.5
    dbx_from_center = bx - 0.5
    dby_from_center = by - 0.5

    # Квадрат расстояния до центра
    r2_src = dx_from_center**2 + dy_from_center**2
    r2_base = dbx_from_center**2 + dby_from_center**2

    # Итоговый вектор residual features
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


def get_residual_target(df_part: pd.DataFrame) -> np.ndarray:
    # Возвращаем target для residual MLP:
    # поправка по x и y в нормализованном пространстве
    return df_part[["res_dx_n", "res_dy_n"]].values.astype(np.float32)


class ResidualDataset(Dataset):
    # Dataset для PyTorch:
    # хранит residual features, residual target,
    # source_id и baseline-предсказание
    def __init__(self, X: np.ndarray, y_res: np.ndarray, source_id: np.ndarray, base_xy_norm: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y_res = torch.tensor(y_res, dtype=torch.float32)
        self.source_id = torch.tensor(source_id, dtype=torch.long)
        self.base_xy_norm = torch.tensor(base_xy_norm, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        return {
            "X": self.X[idx],
            "y_res": self.y_res[idx],
            "source_id": self.source_id[idx],
            "base_xy_norm": self.base_xy_norm[idx],
        }


class ControlResidualRegressor(nn.Module):
    # Основная итоговая модель:
    # общий trunk + две головы для top/bottom
    def __init__(self, in_dim: int):
        super().__init__()

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

        self.head_top = nn.Sequential(
            nn.Linear(64, 64),
            nn.SiLU(),
            nn.Linear(64, 2),
        )

        self.head_bottom = nn.Sequential(
            nn.Linear(64, 64),
            nn.SiLU(),
            nn.Linear(64, 2),
        )

    def forward(self, X: torch.Tensor, source_id: torch.Tensor):
        # Получаем общее скрытое представление
        h = self.trunk(X)

        # Считаем residual-поправки отдельно для top и bottom
        res_top = self.head_top(h)
        res_bottom = self.head_bottom(h)

        # Маской выбираем нужную голову
        mask_bottom = (source_id == 1).float().unsqueeze(1)
        res = res_top * (1.0 - mask_bottom) + res_bottom * mask_bottom
        return res


def regression_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    # Используем smooth L1 loss — он устойчивее к выбросам, чем MSE
    return F.smooth_l1_loss(pred, target)


@torch.no_grad()
def evaluate_model(model: nn.Module, loader: DataLoader, df_ref: pd.DataFrame, device: str):
    # Оценка модели на val
    model.eval()

    preds_final = []
    true_final = []
    total_loss = 0.0
    total_count = 0

    for batch in loader:
        X = batch["X"].to(device)
        y_res = batch["y_res"].to(device)
        source_id = batch["source_id"].to(device)
        base_xy_norm = batch["base_xy_norm"].to(device)

        # Предсказываем residual
        pred_res = model(X, source_id)

        # Считаем loss именно по residual target
        loss = regression_loss(pred_res, y_res)

        # Финальная координата = baseline + residual
        pred_final = torch.clamp(base_xy_norm + pred_res, 0.0, 1.0)

        bs = X.size(0)
        total_loss += loss.item() * bs
        total_count += bs

        preds_final.append(pred_final.detach().cpu().numpy())
        true_final.append((base_xy_norm + y_res).detach().cpu().numpy())

    preds_final = np.vstack(preds_final)
    true_final = np.vstack(true_final)

    # Переводим нормализованные координаты обратно в пиксели
    pred_px = denormalize_xy_array(preds_final)
    true_px = denormalize_xy_array(true_final)

    result_df = df_ref.reset_index(drop=True).copy()
    result_df["pred_x"] = pred_px[:, 0]
    result_df["pred_y"] = pred_px[:, 1]
    result_df["true_x"] = true_px[:, 0]
    result_df["true_y"] = true_px[:, 1]
    result_df["dist"] = np.sqrt(
        (result_df["pred_x"] - result_df["true_x"]) ** 2
        + (result_df["pred_y"] - result_df["true_y"]) ** 2
    )

    # Общие метрики
    metrics = {
        "loss": total_loss / max(total_count, 1),
        "med_all": float(result_df["dist"].mean()),
        "rmse_all": float(
            math.sqrt(
                mean_squared_error(
                    result_df[["true_x", "true_y"]].values,
                    result_df[["pred_x", "pred_y"]].values,
                )
            )
        ),
    }

    # Метрики отдельно по source
    for source_name in ["top", "bottom"]:
        sub = result_df[result_df["source"] == source_name]
        metrics[f"med_{source_name}"] = float(sub["dist"].mean())
        metrics[f"rmse_{source_name}"] = float(
            math.sqrt(
                mean_squared_error(
                    sub[["true_x", "true_y"]].values,
                    sub[["pred_x", "pred_y"]].values,
                )
            )
        )

    return metrics, result_df


@dataclass
class TrainConfig:
    # Параметры обучения и путей
    data_root: Path
    artifacts_dir: Path
    seed: int = 42
    epochs: int = 250
    batch_size: int = 256
    lr: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 25
    num_workers: int = 0


def train(config: TrainConfig) -> None:
    # Фиксируем seed
    set_seed(config.seed)

    # Выбираем устройство
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pin_memory = device == "cuda"

    # Проверяем наличие split.json
    split_path = config.data_root / "split.json"
    if not split_path.exists():
        raise FileNotFoundError(f"split.json not found: {split_path}")

    # Создаем папку для артефактов
    config.artifacts_dir.mkdir(parents=True, exist_ok=True)

    # Собираем единую таблицу всех точек
    df = collect_unified_point_dataframe(config.data_root, split_path)

    # Строго разделяем train и val
    train_df = df[df["split"] == "train"].reset_index(drop=True).copy()
    val_df = df[df["split"] == "val"].reset_index(drop=True).copy()

    # Обучаем baseline-модели
    baseline_models, baseline_df = fit_source_baselines(train_df, val_df)

    # Сохраняем baseline-модели
    joblib.dump(baseline_models["top"], config.artifacts_dir / "baseline_top_poly3_ridge.joblib")
    joblib.dump(baseline_models["bottom"], config.artifacts_dir / "baseline_bottom_poly3_ridge.joblib")

    # Добавляем baseline-предсказания и residual target
    train_df2 = add_baseline_predictions(train_df, baseline_models)
    val_df2 = add_baseline_predictions(val_df, baseline_models)

    # Строим признаки и target для residual MLP
    X_train_raw = build_residual_features(train_df2)
    X_val_raw = build_residual_features(val_df2)
    y_train_res = get_residual_target(train_df2)
    y_val_res = get_residual_target(val_df2)

    train_source_id = train_df2["source_id"].values.astype(np.int64)
    val_source_id = val_df2["source_id"].values.astype(np.int64)
    train_base_xy_norm = train_df2[["base_x_n", "base_y_n"]].values.astype(np.float32)
    val_base_xy_norm = val_df2[["base_x_n", "base_y_n"]].values.astype(np.float32)

    # Обучаем scaler только на train и применяем к train/val
    feature_scaler = StandardScaler()
    X_train = feature_scaler.fit_transform(X_train_raw).astype(np.float32)
    X_val = feature_scaler.transform(X_val_raw).astype(np.float32)

    # Сохраняем scaler
    joblib.dump(feature_scaler, config.artifacts_dir / "feature_scaler.joblib")

    # Создаем Dataset-объекты
    train_dataset = ResidualDataset(X_train, y_train_res, train_source_id, train_base_xy_norm)
    val_dataset = ResidualDataset(X_val, y_val_res, val_source_id, val_base_xy_norm)

    # DataLoader для train
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=pin_memory,
    )

    # DataLoader для val
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=pin_memory,
    )

    # Создаем модель
    model = ControlResidualRegressor(in_dim=X_train.shape[1]).to(device)

    # Оптимизатор
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )

    # Переменные для отслеживания лучшей модели
    best_metric = float("inf")
    best_epoch = -1
    best_metrics = None
    history = []
    patience_counter = 0

    for epoch in range(1, config.epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_count = 0

        for batch in train_loader:
            X = batch["X"].to(device)
            y_res = batch["y_res"].to(device)
            source_id = batch["source_id"].to(device)

            optimizer.zero_grad()
            pred_res = model(X, source_id)
            loss = regression_loss(pred_res, y_res)
            loss.backward()
            optimizer.step()

            bs = X.size(0)
            train_loss_sum += loss.item() * bs
            train_count += bs

        train_loss = train_loss_sum / max(train_count, 1)

        # Оценка на val после каждой эпохи
        val_metrics, _ = evaluate_model(model, val_loader, val_df2, device)

        row = {"epoch": epoch, "train_loss": train_loss, **val_metrics}
        history.append(row)

        print(
            f"epoch={epoch:03d} train_loss={train_loss:.6f} "
            f"val_med_all={val_metrics['med_all']:.4f} "
            f"top={val_metrics['med_top']:.4f} bottom={val_metrics['med_bottom']:.4f}"
        )

        # Если текущая модель лучшая по med_all — сохраняем checkpoint
        if val_metrics["med_all"] < best_metric:
            best_metric = val_metrics["med_all"]
            best_epoch = epoch
            best_metrics = val_metrics.copy()
            patience_counter = 0

            ckpt = {
                "run_name": "control_residual_mlp",
                "epoch": epoch,
                "feature_dim": X_train.shape[1],
                "model_state_dict": model.state_dict(),
                "best_metric": best_metric,
                "best_metrics": best_metrics,
            }
            torch.save(ckpt, config.artifacts_dir / "control_residual_mlp.pt")
        else:
            patience_counter += 1

        # Early stopping, если нет улучшения нужное число эпох
        if patience_counter >= config.patience:
            print(f"Early stopping at epoch {epoch}. Best epoch = {best_epoch}")
            break

    # Сохраняем сводные метрики
    baseline_metrics = baseline_df.to_dict(orient="records")
    metrics = {
        "best_epoch": best_epoch,
        "best_metrics": best_metrics,
        "baseline_metrics": baseline_metrics,
        "train_points": int(len(train_df)),
        "val_points": int(len(val_df)),
    }

    with open(config.artifacts_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    # Историю обучения сохраняем в csv
    pd.DataFrame(history).to_csv(config.artifacts_dir / "history.csv", index=False)

    print(f"Saved artifacts to: {config.artifacts_dir}")


def parse_args() -> argparse.Namespace:
    # Аргументы командной строки
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True, help="Path to coord_data")
    parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    # Точка входа при запуске train.py из терминала
    args = parse_args()

    # Собираем конфиг и запускаем обучение
    train(
        TrainConfig(
            data_root=args.data_root,
            artifacts_dir=args.artifacts_dir,
            seed=args.seed,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            patience=args.patience,
            num_workers=args.num_workers,
        )
    )
