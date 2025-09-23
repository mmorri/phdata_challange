"""Generate comparison plots from model/model_info.json."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODEL_INFO_PATH = ROOT / "model" / "model_info.json"
REPORTS_DIR = ROOT / "reports"
REPORTS_DIR.mkdir(exist_ok=True)

COLOR_MAP = {
    "gradient_boosting": "#1f77b4",
    "hist_gradient_boosting": "#ff7f0e",
    "random_forest": "#2ca02c",
    "knn": "#d62728",
}


def load_info() -> dict:
    with MODEL_INFO_PATH.open("r", encoding="utf-8") as file:
        return json.load(file)


def get_colors(names):
    return [COLOR_MAP.get(name, "#7f7f7f") for name in names]


def plot_validation_mae(candidates):
    names = [c["display_name"] for c in candidates]
    values = [c["validation_mae"] for c in candidates]
    colors = get_colors([c["name"] for c in candidates])

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(names, values, color=colors)
    ax.set_ylabel("Validation MAE ($)")
    ax.set_title("Model Comparison – Validation MAE")
    ax.bar_label(bars, fmt="%.0f", padding=3)
    plt.tight_layout()
    output = REPORTS_DIR / "validation_mae.png"
    fig.savefig(output, dpi=150)
    plt.close(fig)
    return output


def plot_r2_scores(candidates):
    names = [c["display_name"] for c in candidates]
    values = [c["test_score"] for c in candidates]
    colors = get_colors([c["name"] for c in candidates])

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(names, values, color=colors)
    ax.set_ylabel("Test R²")
    ax.set_ylim(0.7, 0.95)
    ax.set_title("Model Comparison – Test R²")
    ax.bar_label(bars, fmt="%.3f", padding=3)
    plt.tight_layout()
    output = REPORTS_DIR / "test_r2.png"
    fig.savefig(output, dpi=150)
    plt.close(fig)
    return output


def plot_mae_delta(baseline, champion):
    labels = ["Baseline KNN", "Champion"]
    values = [baseline["validation_mae"], champion["validation_mae"]]
    colors = get_colors([baseline["name"], champion.get("name", "gradient_boosting")])

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(labels, values, color=colors)
    ax.set_ylabel("Validation MAE ($)")
    ax.set_title("Champion vs. KNN Baseline")
    ax.bar_label(bars, fmt="%.0f", padding=3)
    plt.tight_layout()
    output = REPORTS_DIR / "champion_vs_knn.png"
    fig.savefig(output, dpi=150)
    plt.close(fig)
    return output


def main() -> None:
    info = load_info()
    candidates = info.get("candidates", [])
    if not candidates:
        raise SystemExit("No candidate models recorded in model_info.json")

    # order by validation MAE ascending
    candidates = sorted(candidates, key=lambda c: c["validation_mae"])
    baseline = next((c for c in candidates if c["name"] == "knn"), candidates[-1])

    mae_plot = plot_validation_mae(candidates)
    r2_plot = plot_r2_scores(candidates)
    delta_plot = plot_mae_delta(baseline, {
        "name": info.get("model_type", "champion"),
        "validation_mae": info.get("validation_mae", baseline["validation_mae"]),
    })

    print("Generated plots:")
    print(f" - {mae_plot}")
    print(f" - {r2_plot}")
    print(f" - {delta_plot}")


if __name__ == "__main__":
    main()
