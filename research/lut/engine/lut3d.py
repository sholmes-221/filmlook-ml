"""3D LUT container: lattice storage, .cube reading/writing, validation, trilinear application (numpy only)."""
from __future__ import annotations

from pathlib import Path

import numpy as np


class LUT3D:
    """lattice[r, g, b] -> output RGB, all in [0, 1]; indices run over the input domain [0, 1] in `size` steps."""

    def __init__(self, lattice: np.ndarray, name: str = "", domain_min=(0, 0, 0), domain_max=(1, 1, 1), notes: str = ""):
        lat = np.asarray(lattice, dtype=np.float32)
        if lat.ndim != 4 or not (lat.shape[0] == lat.shape[1] == lat.shape[2]) or lat.shape[3] != 3:
            raise ValueError(f"lattice must be (n, n, n, 3), got {lat.shape}")
        self.lattice = lat
        self.size = lat.shape[0]
        self.name = name
        self.domain_min = np.asarray(domain_min, np.float32)
        self.domain_max = np.asarray(domain_max, np.float32)
        self.notes = notes

    # ---- validation -------------------------------------------------------------------------------------------
    def validate(self) -> dict:
        lat = self.lattice
        g = np.linspace(0, 1, self.size)
        diag = lat[np.arange(self.size), np.arange(self.size), np.arange(self.size)]
        slopes = np.diff(diag, axis=0)
        rep = {"size": int(self.size), "finite": bool(np.isfinite(lat).all()), "in_range": bool((lat >= -1e-4).all() and (lat <= 1 + 1e-4).all()),
               "grey_monotone": bool((slopes >= -1e-4).all()), "grey_max_step": float(np.abs(slopes).max()),
               "identity_rms": float(np.sqrt(((lat - np.stack(np.meshgrid(g, g, g, indexing="ij"), -1)) ** 2).mean())),
               "black": [float(v) for v in lat[0, 0, 0]], "white": [float(v) for v in lat[-1, -1, -1]],
               "domain": [self.domain_min.tolist(), self.domain_max.tolist()]}
        rep["ok"] = rep["finite"] and rep["in_range"] and rep["size"] >= 2 and rep["grey_max_step"] < 0.5
        return rep

    # ---- application ------------------------------------------------------------------------------------------
    def apply(self, rgb) -> np.ndarray:
        """float array (..., 3) in [0, 1] (or uint8) -> float32 (..., 3) in [0, 1] via trilinear interpolation."""
        x = np.asarray(rgb)
        if x.dtype == np.uint8:
            x = x.astype(np.float32) / 255
        x = (np.clip(x, 0, 1) - self.domain_min) / np.maximum(self.domain_max - self.domain_min, 1e-6)
        x = np.clip(x, 0, 1) * (self.size - 1)
        i0 = np.clip(np.floor(x).astype(np.int32), 0, self.size - 2)
        f = (x - i0).astype(np.float32)
        out = np.zeros(x.shape[:-1] + (3,), np.float32)
        for dr in (0, 1):
            wr = f[..., 0] if dr else 1 - f[..., 0]
            for dg in (0, 1):
                wg = f[..., 1] if dg else 1 - f[..., 1]
                for db in (0, 1):
                    wb = f[..., 2] if db else 1 - f[..., 2]
                    out += (wr * wg * wb)[..., None] * self.lattice[i0[..., 0] + dr, i0[..., 1] + dg, i0[..., 2] + db]
        return np.clip(out, 0, 1)

    def apply_u8(self, img_u8: np.ndarray) -> np.ndarray:
        return (self.apply(img_u8) * 255 + 0.5).astype(np.uint8)

    def grey_ramp(self, n: int = 33) -> np.ndarray:
        g = np.linspace(0, 1, n, dtype=np.float32)
        return self.apply(np.stack([g, g, g], -1))

    # ---- .cube I/O --------------------------------------------------------------------------------------------
    @staticmethod
    def from_cube(path) -> "LUT3D":
        """Adobe/Resolve .cube: LUT_3D_SIZE n, optional DOMAIN_MIN/MAX, then n^3 'r g b' rows with RED varying fastest."""
        size, dmin, dmax, title, rows = None, (0, 0, 0), (1, 1, 1), Path(path).stem, []
        for line in Path(path).read_text(errors="replace").splitlines():
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            key = s.split()[0].upper()
            if key == "TITLE":
                title = s[5:].strip().strip('"')
            elif key == "LUT_3D_SIZE":
                size = int(s.split()[1])
            elif key == "LUT_1D_SIZE":
                raise ValueError("1D .cube files are not supported")
            elif key == "DOMAIN_MIN":
                dmin = tuple(float(v) for v in s.split()[1:4])
            elif key == "DOMAIN_MAX":
                dmax = tuple(float(v) for v in s.split()[1:4])
            else:
                parts = s.split()
                if len(parts) >= 3:
                    rows.append([float(v) for v in parts[:3]])
        if size is None or len(rows) != size ** 3:
            raise ValueError(f"{path}: expected {size}^3 rows, got {len(rows)}")
        arr = np.array(rows, np.float32).reshape(size, size, size, 3)   # index order (b, g, r) because red is fastest
        lattice = np.transpose(arr, (2, 1, 0, 3))                         # -> [r, g, b]
        return LUT3D(lattice, title, dmin, dmax, notes="from .cube")

    def to_cube(self, path, title: str | None = None) -> None:
        n = self.size
        lines = [f'TITLE "{title or self.name}"', f"LUT_3D_SIZE {n}",
                 "DOMAIN_MIN " + " ".join(f"{v:.6f}" for v in self.domain_min), "DOMAIN_MAX " + " ".join(f"{v:.6f}" for v in self.domain_max)]
        lat = np.transpose(self.lattice, (2, 1, 0, 3)).reshape(-1, 3)   # red fastest
        lines += [" ".join(f"{v:.6f}" for v in row) for row in lat]
        Path(path).write_text("\n".join(lines) + "\n")

    def resample(self, size: int) -> "LUT3D":
        g = np.linspace(0, 1, size, dtype=np.float32)
        grid = np.stack(np.meshgrid(g, g, g, indexing="ij"), -1)
        return LUT3D(self.apply(grid), self.name, self.domain_min, self.domain_max, self.notes)
