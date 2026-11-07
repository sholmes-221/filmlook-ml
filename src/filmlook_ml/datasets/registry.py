"""Dataset registry + license allowlist enforcement."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from filmlook_ml.paths import CONFIG_DIR


class LicenseError(ValueError):
    pass


@dataclass
class Registry:
    allowlist: set[str]
    denylist: set[str]
    datasets: dict[str, dict]

    @classmethod
    def load(cls, path: Path | None = None) -> "Registry":
        cfg = yaml.safe_load(Path(path or CONFIG_DIR / "datasets.yaml").read_text())
        return cls(set(cfg["license_allowlist"]), set(cfg.get("license_denylist", [])),
                   {d["id"]: d for d in cfg["datasets"]})

    def dataset_allowed(self, dataset_id: str) -> bool:
        d = self.datasets.get(dataset_id)
        return bool(d and d.get("training_allowed") is True and d.get("license") in self.allowlist)

    def check_record(self, dataset_id: str, license_id: str | None) -> tuple[bool, str]:
        """Return (allowed, reason). Per-record license must ALSO be allowlisted."""
        if dataset_id not in self.datasets:
            return False, f"dataset '{dataset_id}' not in registry"
        if not self.dataset_allowed(dataset_id):
            return False, f"dataset '{dataset_id}' not approved for training"
        lic = license_id or "unknown"
        if lic in self.denylist or lic not in self.allowlist:
            return False, f"license '{lic}' not in allowlist"
        return True, "ok"

    def require(self, dataset_id: str, license_id: str | None) -> None:
        ok, why = self.check_record(dataset_id, license_id)
        if not ok:
            raise LicenseError(why)


def normalize_license(url_or_name: str | None) -> str:
    s = (url_or_name or "").lower()
    if "publicdomain/zero" in s or s in {"cc0", "cc0-1.0"}:
        return "CC0-1.0"
    if "publicdomain/mark" in s or s in {"pdm", "pdm-1.0"}:
        return "PDM-1.0"
    if "by-nc" in s:
        return "CC-BY-NC-4.0"
    if "licenses/by/4.0" in s:
        return "CC-BY-4.0"
    if "licenses/by/2.0" in s:
        return "CC-BY-2.0"
    return url_or_name or "unknown"
