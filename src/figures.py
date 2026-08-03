from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from core import ENDPOINT_THRESHOLD, curve_probes, endpoint_probes, support_spill_metrics

def crop_window(image, point, radius=18):
    row, col = point
    row0 = max(int(row) - radius, 0)
    row1 = min(int(row) + radius + 1, image.shape[0])
    col0 = max(int(col) - radius, 0)
    col1 = min(int(col) + radius + 1, image.shape[1])
    return image[row0:row1, col0:col1], row0, col0

def main():
    root = Path(__file__).parent.parent
    output = root / "figures"
    output.mkdir(parents=True, exist_ok=True)
    data = np.load(root / "data" / "retina.npz")
    recon = np.load(root / "results" / "reconstructions.npz")
    truth = data["truth"]
    weighted = recon["weighted_fbp"]
    iterative = recon["iterative"]
    per_crop = pd.read_csv(root / "results" / "per_crop.csv")
    summary = json.loads((root / "results" / "benchmark.json").read_text(encoding="utf-8"))

    fig, axes = plt.subplots(4, 5, figsize=(13, 10.5))
    columns = ("truth", "weighted FBP", "iterative", "|truth-FBP|", "|truth-iterative|")
    for row in range(4):
        images = (truth[row], weighted[row], iterative[row], np.abs(truth[row] - weighted[row]), np.abs(truth[row] - iterative[row]))
        for col, image in enumerate(images):
            axes[row, col].imshow(image, cmap="gray", vmin=0, vmax=0.5 if col >= 3 else 1)
            axes[row, col].axis("off")
            if row == 0:
                axes[row, col].set_title(columns[col])
    fig.suptitle("18-view reconstruction")
    fig.tight_layout()
    fig.savefig(output / "reconstruction.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(4, 5, figsize=(13, 10.5))
    columns = ("truth support", "FBP support", "FBP spill", "iterative support", "iterative spill")
    for row in range(4):
        first = support_spill_metrics(truth[row], weighted[row])
        second = support_spill_metrics(truth[row], iterative[row])
        images = (first["truth_mask"], first["reconstruction_mask"], first["spill_mask"], second["reconstruction_mask"], second["spill_mask"])
        for col, image in enumerate(images):
            axes[row, col].imshow(image, cmap="gray", vmin=0, vmax=1)
            axes[row, col].axis("off")
            if row == 0:
                axes[row, col].set_title(columns[col])
    fig.suptitle("support outside a two-pixel truth tolerance")
    fig.tight_layout()
    fig.savefig(output / "support.png", dpi=180)
    plt.close(fig)

    endpoint_records = []
    for crop_id in range(1, 5):
        first = endpoint_probes(truth[crop_id - 1], weighted[crop_id - 1], crop_id)
        second = endpoint_probes(truth[crop_id - 1], iterative[crop_id - 1], crop_id)
        for probe in first:
            match = min(second, key=lambda item: np.linalg.norm(item["endpoint"] - probe["endpoint"]))
            endpoint_records.append((crop_id, probe, match))
    endpoint_records.sort(key=lambda item: item[1]["excess_continuation_mass"], reverse=True)
    endpoint_records = endpoint_records[:8]

    fig, axes = plt.subplots(len(endpoint_records), 4, figsize=(10, 2.5 * len(endpoint_records)))
    for row, (crop_id, first, second) in enumerate(endpoint_records):
        images = (truth[crop_id - 1], weighted[crop_id - 1], iterative[crop_id - 1], np.abs(weighted[crop_id - 1] - iterative[crop_id - 1]))
        titles = ("truth", f"FBP {first['overrun_length_px']:.0f}px", f"iterative {second['overrun_length_px']:.0f}px", "difference")
        for col, image in enumerate(images):
            patch, row0, col0 = crop_window(image, first["endpoint"])
            axes[row, col].imshow(patch, cmap="gray", vmin=0, vmax=1)
            start = np.array((first["endpoint"][1] - col0, first["endpoint"][0] - row0))
            end = start + np.array((first["outward"][1], first["outward"][0])) * 14
            axes[row, col].plot((start[0], end[0]), (start[1], end[1]), color="white", linewidth=1.2)
            axes[row, col].scatter((start[0],), (start[1],), s=12, facecolors="none", edgecolors="white")
            axes[row, col].axis("off")
            if row == 0:
                axes[row, col].set_title(titles[col])
    fig.suptitle("clean endpoint probes")
    fig.tight_layout()
    fig.savefig(output / "endpoints.png", dpi=180)
    plt.close(fig)

    distances = endpoint_records[0][1]["distances"]
    truth_profiles = np.stack([item[1]["truth_profile"] for item in endpoint_records])
    weighted_profiles = np.stack([item[1]["profile"] for item in endpoint_records])
    iterative_profiles = np.stack([item[2]["profile"] for item in endpoint_records])
    fig, axis = plt.subplots(figsize=(8, 4.8))
    axis.plot(distances, truth_profiles.mean(axis=0), label="truth")
    axis.plot(distances, weighted_profiles.mean(axis=0), label="weighted FBP")
    axis.plot(distances, iterative_profiles.mean(axis=0), label="iterative")
    axis.axhline(ENDPOINT_THRESHOLD, linestyle="--", label="threshold")
    axis.set_xlabel("pixels beyond endpoint")
    axis.set_ylabel("cross-sectional signal")
    axis.legend()
    axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output / "endpoint_signal.png", dpi=180)
    plt.close(fig)

    smooth = lambda matrix: np.stack([np.convolve(row, np.ones(2) / 2, mode="same") for row in matrix])
    fig, axis = plt.subplots(figsize=(8, 4.8))
    axis.step(distances, np.mean(smooth(truth_profiles) > ENDPOINT_THRESHOLD, axis=0), where="mid", label="truth")
    axis.step(distances, np.mean(smooth(weighted_profiles) > ENDPOINT_THRESHOLD, axis=0), where="mid", label="weighted FBP")
    axis.step(distances, np.mean(smooth(iterative_profiles) > ENDPOINT_THRESHOLD, axis=0), where="mid", label="iterative")
    axis.set_xlabel("pixels beyond endpoint")
    axis.set_ylabel("fraction above threshold")
    axis.set_ylim(-0.03, 1.03)
    axis.legend()
    axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output / "endpoint_frequency.png", dpi=180)
    plt.close(fig)

    curve_records = []
    for crop_id in range(1, 5):
        first = {item["branch_id"]: item for item in curve_probes(truth[crop_id - 1], weighted[crop_id - 1], crop_id)}
        second = {item["branch_id"]: item for item in curve_probes(truth[crop_id - 1], iterative[crop_id - 1], crop_id)}
        for branch_id in sorted(set(first) & set(second)):
            curve_records.append((crop_id, first[branch_id], second[branch_id]))
    curve_records.sort(key=lambda item: item[1]["total_turn_deg"], reverse=True)
    selected = curve_records[:8]
    fig, axes = plt.subplots(len(selected), 3, figsize=(9, 2.8 * len(selected)))
    for row, (crop_id, first, second) in enumerate(selected):
        images = (truth[crop_id - 1], weighted[crop_id - 1], iterative[crop_id - 1])
        ridges = (None, first["ridge_points"], second["ridge_points"])
        for col, (image, ridge) in enumerate(zip(images, ridges)):
            points = first["points"]
            minimum = np.maximum(np.floor(points.min(axis=0) - 10).astype(int), 0)
            maximum = np.minimum(np.ceil(points.max(axis=0) + 10).astype(int), np.array(image.shape) - 1)
            patch = image[minimum[0]:maximum[0] + 1, minimum[1]:maximum[1] + 1]
            axes[row, col].imshow(patch, cmap="gray", vmin=0, vmax=1)
            axes[row, col].plot(points[:, 1] - minimum[1], points[:, 0] - minimum[0], color="white", linewidth=1.4)
            if ridge is not None:
                axes[row, col].scatter(ridge[::2, 1] - minimum[1], ridge[::2, 0] - minimum[0], s=5, color="black")
                axes[row, col].scatter(ridge[::2, 1] - minimum[1], ridge[::2, 0] - minimum[0], s=1, color="white")
            axes[row, col].axis("off")
            if row == 0:
                axes[row, col].set_title(("truth", "weighted FBP", "iterative")[col])
    fig.suptitle("curved ridge tracking")
    fig.tight_layout()
    fig.savefig(output / "curves.png", dpi=180)
    plt.close(fig)

    first_curve = pd.read_csv(root / "results" / "curves.csv")
    first_curve = first_curve[first_curve["method"] == "Weighted FBP"].reset_index(drop=True)
    second_curve = pd.read_csv(root / "results" / "curves.csv")
    second_curve = second_curve[second_curve["method"] == "Iterative curve-endpoint"].reset_index(drop=True)
    fig, axis = plt.subplots(figsize=(5.8, 5.2))
    axis.scatter(first_curve["bend_correlation"], second_curve["bend_correlation"])
    low = float(np.nanmin(np.r_[first_curve["bend_correlation"], second_curve["bend_correlation"]]))
    high = float(np.nanmax(np.r_[first_curve["bend_correlation"], second_curve["bend_correlation"]]))
    axis.plot((low, high), (low, high), linestyle="--")
    axis.set_xlabel("weighted-FBP bend correlation")
    axis.set_ylabel("iterative bend correlation")
    axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output / "bend_correlation.png", dpi=180)
    plt.close(fig)

    rows = summary["methods"]
    values = [
        [row["method"], f"{row['mean_psnr_db']:.2f}", f"{row['mean_ssim']:.3f}", f"{row['mean_false_skeleton_length_px']:.1f}", f"{row['mean_endpoint_overrun_px']:.2f}", f"{row['weighted_curve_localization_rmse_px']:.2f}", f"{row['weighted_tangent_mae_deg']:.1f}", f"{row['median_bend_correlation']:.3f}"]
        for row in rows
    ]
    fig, axis = plt.subplots(figsize=(11, 2.3))
    axis.axis("off")
    table = axis.table(cellText=values, colLabels=("method", "PSNR", "SSIM", "spill length", "endpoint", "curve RMSE", "tangent", "bend corr."), loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.7)
    fig.tight_layout()
    fig.savefig(output / "metrics.png", dpi=180)
    plt.close(fig)

if __name__ == "__main__":
    main()
