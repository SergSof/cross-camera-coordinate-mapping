from __future__ import annotations

import argparse
from pathlib import Path
from typing import Literal, Tuple

import joblib
import numpy as np
import torch
import torch.nn as nn


IMG_W = 3200.0
IMG_H = 1800.0
SourceType = Literal["top", "bottom"]


class ControlResidualRegressor(nn.Module):
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
        h = self.trunk(X)
        res_top = self.head_top(h)
        res_bottom = self.head_bottom(h)
        mask_bottom = (source_id == 1).float().unsqueeze(1)
        return res_top * (1.0 - mask_bottom) + res_bottom * mask_bottom


def build_residual_features(src_x: float, src_y: float, base_x: float, base_y: float) -> np.ndarray:
    x = np.array([[src_x / IMG_W]], dtype=np.float32)
    y = np.array([[src_y / IMG_H]], dtype=np.float32)
    bx = np.array([[base_x / IMG_W]], dtype=np.float32)
    by = np.array([[base_y / IMG_H]], dtype=np.float32)

    x2 = x * x
    y2 = y * y
    xy = x * y
    bx2 = bx * bx
    by2 = by * by
    bxby = bx * by
    x3 = x2 * x
    y3 = y2 * y
    x2y = x2 * y
    xy2 = x * y2
    sinx = np.sin(np.pi * x)
    cosx = np.cos(np.pi * x)
    siny = np.sin(np.pi * y)
    cosy = np.cos(np.pi * y)
    sinbx = np.sin(np.pi * bx)
    cosbx = np.cos(np.pi * bx)
    sinby = np.sin(np.pi * by)
    cosby = np.cos(np.pi * by)
    dx_from_center = x - 0.5
    dy_from_center = y - 0.5
    dbx_from_center = bx - 0.5
    dby_from_center = by - 0.5
    r2_src = dx_from_center**2 + dy_from_center**2
    r2_base = dbx_from_center**2 + dby_from_center**2

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
    def __init__(self, artifacts_dir: str | Path = "artifacts", device: str | None = None):
        self.artifacts_dir = Path(artifacts_dir)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.baseline_top = joblib.load(self.artifacts_dir / "baseline_top_poly3_ridge.joblib")
        self.baseline_bottom = joblib.load(self.artifacts_dir / "baseline_bottom_poly3_ridge.joblib")
        self.feature_scaler = joblib.load(self.artifacts_dir / "feature_scaler.joblib")

        ckpt = torch.load(self.artifacts_dir / "control_residual_mlp.pt", map_location=self.device)
        feature_dim = int(ckpt["feature_dim"])

        self.model = ControlResidualRegressor(in_dim=feature_dim).to(self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()

    def predict(self, x: float, y: float, source: SourceType) -> Tuple[float, float]:
        if source not in {"top", "bottom"}:
            raise ValueError("source must be 'top' or 'bottom'")

        xy = np.array([[float(x), float(y)]], dtype=np.float32)
        baseline_model = self.baseline_top if source == "top" else self.baseline_bottom
        base_pred = baseline_model.predict(xy).astype(np.float32)[0]
        base_x, base_y = float(base_pred[0]), float(base_pred[1])

        feats_raw = build_residual_features(x, y, base_x, base_y)
        feats = self.feature_scaler.transform(feats_raw).astype(np.float32)

        X = torch.tensor(feats, dtype=torch.float32, device=self.device)
        source_id = torch.tensor([0 if source == "top" else 1], dtype=torch.long, device=self.device)

        with torch.no_grad():
            pred_res = self.model(X, source_id).cpu().numpy()[0]

        pred_x_n = np.clip(base_x / IMG_W + pred_res[0], 0.0, 1.0)
        pred_y_n = np.clip(base_y / IMG_H + pred_res[1], 0.0, 1.0)

        pred_x = float(pred_x_n * IMG_W)
        pred_y = float(pred_y_n * IMG_H)
        return pred_x, pred_y


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--x", type=float, required=True)
    parser.add_argument("--y", type=float, required=True)
    parser.add_argument("--source", type=str, choices=["top", "bottom"], required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    predictor = CoordinatePredictor(artifacts_dir=args.artifacts_dir)
    pred_x, pred_y = predictor.predict(args.x, args.y, args.source)
    print(f"pred_x={pred_x:.3f}")
    print(f"pred_y={pred_y:.3f}")
