"""Проба H6: классификация символов по прототипам легенды при идеальной локализации.

Вырезки берутся по рамкам разметки, прототипы: эталоны легенды в 4 поворотах. Сравниваются
эмбеддинги DINOv2-small (CLS + среднее патчей, косинус) и наивное NCC растянутой вырезки.
Запуск: .venv/bin/python m_dino_probe.py
"""

from __future__ import annotations

from collections import Counter

import numpy as np
import torch
from common import labeled_plans, legend_crops, load_sheets
from PIL import Image
from transformers import AutoModel

DEV = "mps" if torch.backends.mps.is_available() else "cpu"
MEAN = np.array([0.485, 0.456, 0.406])
STD = np.array([0.229, 0.224, 0.225])


def prep(g: np.ndarray, size: int = 112, pad: float = 0.15) -> np.ndarray:
    """Квадратная белая подложка с полем, ресайз, нормализация ImageNet."""
    h, w = g.shape
    s = max(h, w)
    p = int(s * pad)
    c = np.full((s + 2 * p, s + 2 * p), 255, np.uint8)
    y0, x0 = p + (s - h) // 2, p + (s - w) // 2
    c[y0 : y0 + h, x0 : x0 + w] = g
    im = np.asarray(Image.fromarray(c).resize((size, size), Image.BILINEAR), np.float32) / 255
    im = np.repeat(im[..., None], 3, 2)
    return ((im - MEAN) / STD).transpose(2, 0, 1)


@torch.no_grad()
def embed(model: AutoModel, batch: list[np.ndarray]) -> np.ndarray:
    x = torch.tensor(np.stack(batch), dtype=torch.float32, device=DEV)
    o = model(pixel_values=x).last_hidden_state
    f = torch.cat([o[:, 0], o[:, 1:].mean(1)], 1)
    return torch.nn.functional.normalize(f, dim=1).cpu().numpy()


def ncc(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float32)
    b = np.asarray(Image.fromarray(b).resize(a.shape[::-1], Image.BILINEAR), np.float32)
    a, b = a - a.mean(), b - b.mean()
    return float((a * b).sum() / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-6))


def main() -> None:
    model = AutoModel.from_pretrained("facebook/dinov2-small").to(DEV).eval()
    sheets = load_sheets()
    for fam in ("worcester", "usc"):
        legend = legend_crops(fam)
        proto_raw, proto_id = [], []
        for k, crop in legend.items():
            for r in range(4):
                proto_raw.append(np.rot90(crop, r).copy())
                proto_id.append(k)
        protos = embed(model, [prep(g) for g in proto_raw])
        crops, labels = [], []
        for s in labeled_plans(sheets):
            if s.family != fam:
                continue
            img = s.load_gray()
            for o in s.objects:
                x, y, w, h = (int(round(v)) for v in o.bbox)
                crops.append(img[y : y + h, x : x + w].copy())
                labels.append(o.legend_id)
        feats = np.concatenate(
            [embed(model, [prep(c) for c in crops[i : i + 64]]) for i in range(0, len(crops), 64)]
        )
        pred_dino = [proto_id[j] for j in (feats @ protos.T).argmax(1)]
        pred_ncc = [proto_id[int(np.argmax([ncc(c, g) for g in proto_raw]))] for c in crops]
        for name, pred in (("DINOv2", pred_dino), ("NCC", pred_ncc)):
            pairs = list(zip(labels, pred, strict=True))
            acc = np.mean([lab == p for lab, p in pairs])
            errs = Counter((lab, p) for lab, p in pairs if lab != p).most_common(6)
            print(f"{fam} n={len(labels)} {name} acc={acc:.3f} errors(gt,pred)={errs}")


if __name__ == "__main__":
    main()
