"""紧凑产物 IO：策略表与模型权重的 JSON 存取。

产物放在 data/artifacts/（git 忽略，按需生成），桌面与浏览器共用。
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional, Tuple

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ARTIFACT_DIR = os.path.join(_ROOT, "data", "artifacts")


def artifact_path(name: str) -> str:
    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    return os.path.join(ARTIFACT_DIR, name)


def save_json(name: str, obj: Any) -> str:
    p = artifact_path(name)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False)
    return p


def load_json(name: str, default: Any = None) -> Any:
    p = artifact_path(name)
    if not os.path.exists(p):
        return default
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def save_policy(name: str, policy: Dict[str, Dict[str, float]],
                meta: Optional[Dict[str, Any]] = None) -> str:
    return save_json(name, {"meta": meta or {}, "policy": policy})


def load_policy(name: str) -> Tuple[Dict[str, Dict[str, float]], Dict[str, Any]]:
    blob = load_json(name)
    if blob is None:
        return {}, {}
    return blob.get("policy", {}), blob.get("meta", {})


# ── 新状态 (qa,qp) 策略表：元组键 ↔ "qa,qp" 字符串 ──

def save_policy_table(name: str,
                      policy: Dict[Tuple[int, int], Dict[str, float]],
                      meta: Optional[Dict[str, Any]] = None) -> str:
    flat = {f"{qa},{qp}": probs for (qa, qp), probs in policy.items()}
    return save_json(name, {"meta": meta or {}, "policy": flat})


def load_policy_table(name: str
                      ) -> Tuple[Dict[Tuple[int, int], Dict[str, float]], Dict[str, Any]]:
    blob = load_json(name)
    if blob is None:
        return {}, {}
    pol: Dict[Tuple[int, int], Dict[str, float]] = {}
    for k, probs in blob.get("policy", {}).items():
        qa, qp = (int(x) for x in k.split(","))
        pol[(qa, qp)] = {m: float(p) for m, p in probs.items()}
    return pol, blob.get("meta", {})
