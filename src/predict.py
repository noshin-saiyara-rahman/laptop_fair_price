"""Prediction helpers shared by the Streamlit app and the notebooks.

Loads the saved model + metadata, turns a handful of user-friendly inputs into the
exact 20 features the model was trained on, and runs the fair price check.
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"

# Resolutions that exist in the training data (width x height)
RESOLUTIONS = {
    "1366 x 768 (HD)": (1366, 768),
    "1920 x 1080 (Full HD)": (1920, 1080),
    "1920 x 1200 (WUXGA)": (1920, 1200),
    "2560 x 1440 (QHD)": (2560, 1440),
    "2560 x 1600 (WQXGA)": (2560, 1600),
    "2880 x 1800 (2.8K)": (2880, 1800),
    "3200 x 2000": (3200, 2000),
    "3840 x 2400 (4K)": (3840, 2400),
}

# RTX models that appear in the training data in reasonable numbers
RTX_MODELS = [2050, 3050, 4050, 4060, 4070, 4080, 4090]

GPU_KINDS = ["Integrated", "NVIDIA RTX (dedicated)", "Other dedicated"]


def load_artifacts():
    model = joblib.load(MODELS / "laptop_price_model.joblib")
    with open(MODELS / "meta.json", encoding="utf-8") as f:
        meta = json.load(f)
    return model, meta


def default_cpu_brand(cpu_series):
    if cpu_series.startswith("Core") or cpu_series == "Celeron":
        return "Intel"
    if cpu_series.startswith("Ryzen"):
        return "AMD"
    if cpu_series == "Snapdragon":
        return "Qualcomm"
    return "Other"


def build_features(*, brand, cpu_brand, cpu_series, cores, threads, ram_gb, storage_gb,
                   storage_type, screen_in, resolution, touch, gpu_kind,
                   rtx_model=0, vram_gb=0, gpu_brand="Intel", os_name="Windows 11",
                   warranty=1):
    """Return a dict with every feature column the model expects."""
    res_w, res_h = RESOLUTIONS[resolution]

    if gpu_kind == "Integrated":
        gpu_type, gpu_family, gpu_num, vram, gpu_brand_value = "Integrated", "Integrated", 0, 0, gpu_brand
    elif gpu_kind == "NVIDIA RTX (dedicated)":
        gpu_type, gpu_family, gpu_num, vram, gpu_brand_value = "Dedicated", "RTX", int(
            rtx_model), vram_gb, "NVIDIA"
    else:  # other dedicated GPU
        gpu_type, gpu_family, gpu_num, vram, gpu_brand_value = "Dedicated", "Other dedicated", 0, vram_gb, gpu_brand

    return {
        "brand_name": brand,
        "processor_brand": cpu_brand,
        "iter": cpu_series,
        "threads_num": float(threads),
        "core_num": float(cores),
        "ram_num": float(ram_gb),
        "memory_type": storage_type,
        "memory_size": float(storage_gb),
        "gpu_brand": gpu_brand_value,
        "gpu_type": gpu_type,
        "os": os_name,
        "display_size": float(screen_in),
        "Touch_Screen": int(bool(touch)),
        "warrenty": float(warranty),          # spelled as in the dataset
        "res_w": float(res_w),
        "res_h": float(res_h),
        "pixels_mp": res_w * res_h / 1e6,
        "gpu_vram_gb": float(vram),
        "gpu_num": float(gpu_num),
        "gpu_family": gpu_family,
    }


def predict_price(model, meta, features):
    """Model price in rupees."""
    row = pd.DataFrame([features])[meta["feature_order"]]
    return float(np.exp(model.predict(row))[0])


def fair_price_check(model, meta, features, asking_price):
    """Compare an asking price with the model price and a data-driven fair range."""
    pred = predict_price(model, meta, features)
    fp = meta["fair_price"]
    tier = int(np.searchsorted(fp["tier_edges"], pred))
    low, high = pred * fp["ratio_p10"][tier], pred * fp["ratio_p90"][tier]

    if asking_price < low:
        verdict = "underpriced"
    elif asking_price > high:
        verdict = "overpriced"
    else:
        verdict = "fair"

    return {
        "predicted": round(pred),
        "fair_low": round(low),
        "fair_high": round(high),
        "verdict": verdict,
        "diff_pct": (asking_price / pred - 1) * 100,
        "tier": tier,
    }
