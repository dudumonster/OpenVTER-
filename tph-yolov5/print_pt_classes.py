import argparse
import sys
from pathlib import Path


def _print_env():
    print(f"python: {sys.executable}")
    try:
        import numpy as np  # noqa: F401

        print(f"numpy: {np.__version__}")
    except Exception as exc:
        print(f"numpy: <import failed> ({type(exc).__name__}: {exc})")

    try:
        import pandas as pd  # noqa: F401

        print(f"pandas: {pd.__version__}")
    except Exception as exc:
        print(f"pandas: <import failed> ({type(exc).__name__}: {exc})")


def _normalize_names(names):
    if names is None:
        return None
    if isinstance(names, (list, tuple)):
        return list(names)
    if isinstance(names, dict):
        # common: {0: 'car', 1: 'bus', ...}
        try:
            keys = sorted(int(k) for k in names.keys())
            return [names[k] if k in names else names[str(k)] for k in keys]
        except Exception:
            pass
        try:
            return [v for _, v in sorted(names.items(), key=lambda kv: str(kv[0]))]
        except Exception:
            return None
    return None


def _extract_names_from_checkpoint(ckpt):
    if hasattr(ckpt, "names"):
        return getattr(ckpt, "names")
    if not isinstance(ckpt, dict):
        return None
    for key in ("model", "ema"):
        if key in ckpt and hasattr(ckpt[key], "names"):
            return getattr(ckpt[key], "names")
    return None


def main():
    parser = argparse.ArgumentParser(description="Print class names/order from a YOLOv5 .pt checkpoint.")
    parser.add_argument(
        "--weights",
        required=True,
        help=r"Path to .pt weights, e.g. .\weights\yolov5l-xs-1.pt",
    )
    args = parser.parse_args()

    weights = Path(args.weights).expanduser()
    if not weights.exists():
        print(f"[ERROR] weights not found: {weights}")
        sys.exit(2)

    _print_env()

    try:
        import torch
    except Exception as exc:
        print(f"[ERROR] torch import failed: {type(exc).__name__}: {exc}")
        sys.exit(3)

    try:
        ckpt = torch.load(str(weights), map_location="cpu")
    except Exception as exc:
        msg = f"{type(exc).__name__}: {exc}"
        print(f"[ERROR] torch.load failed: {msg}")
        if "numpy.dtype size changed" in msg or "binary incompatibility" in msg:
            print(
                "\n[HINT] 这是 numpy/pandas 二进制不兼容导致的（常见于 numpy 升到 2.x 以后）。\n"
                "建议在当前 conda 环境执行：\n"
                "  conda install -y numpy=1.24.4 pandas=1.5.3\n"
                "然后再运行本脚本。"
            )
        sys.exit(4)

    raw_names = _extract_names_from_checkpoint(ckpt)
    names = _normalize_names(raw_names)

    if names is None:
        print("[ERROR] 未能从权重中读取到 model.names。")
        print("可能原因：权重不是常规 YOLOv5 checkpoint，或该 checkpoint 未保存 names。")
        sys.exit(5)

    print("\n=== class order (model id -> name) ===")
    for i, name in enumerate(names):
        print(f"{i}: {name}")


if __name__ == "__main__":
    main()

