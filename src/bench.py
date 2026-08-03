from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from core import ANGLES, GAUSSIAN_SIGMA, SART_RELAXATION, SART_SWEEPS, curve_probes, endpoint_probes, image_metrics, iterative_reconstruction, support_spill_metrics, weighted_fbp

METHODS = ("Weighted FBP", "Iterative curve-endpoint")

def summary_for(method, reconstructions, truth, scalar, endpoints, curves):
    scalar_method = scalar[scalar["method"] == method]
    endpoint_method = endpoints[endpoints["method"] == method]
    curve_method = curves[curves["method"] == method]
    pooled_rmse = float(np.sqrt(np.mean(np.concatenate([((truth[index] - reconstructions[index]) ** 2).ravel() for index in range(len(truth))]))))
    return {
        "method": method,
        "mean_psnr_db": float(scalar_method["psnr_db"].mean()),
        "mean_ssim": float(scalar_method["ssim"].mean()),
        "pooled_rmse": pooled_rmse,
        "mean_false_support_area_px": float(scalar_method["spill_area_px"].mean()),
        "mean_false_skeleton_length_px": float(scalar_method["spill_skeleton_length_px"].mean()),
        "endpoint_probe_count": int(len(endpoint_method)),
        "mean_endpoint_overrun_px": float(endpoint_method["overrun_length_px"].mean()),
        "median_endpoint_overrun_px": float(endpoint_method["overrun_length_px"].median()),
        "mean_endpoint_excess_mass": float(endpoint_method["excess_continuation_mass"].mean()),
        "curved_branch_probe_count": int(len(curve_method)),
        "weighted_curve_localization_rmse_px": float(np.average(curve_method["localization_rmse_px"], weights=curve_method["length_samples"])),
        "weighted_tangent_mae_deg": float(np.average(curve_method["tangent_mae_deg"], weights=curve_method["length_samples"])),
        "median_bend_correlation": float(np.nanmedian(curve_method["bend_correlation"])),
        "weighted_bend_nrmse": float(np.average(curve_method["bend_nrmse"], weights=curve_method["length_samples"])),
    }

def main():
    root = Path(__file__).parent.parent
    results = root / "results"
    results.mkdir(parents=True, exist_ok=True)
    arrays = np.load(root / "data" / "retina.npz")
    truth = arrays["truth"]
    noisy = arrays["noisy_sinogram"]
    weighted = np.stack([weighted_fbp(sinogram, ANGLES) for sinogram in noisy])
    iterative = np.stack([iterative_reconstruction(sinogram, ANGLES) for sinogram in noisy])
    reconstructions = {
        "Weighted FBP": weighted,
        "Iterative curve-endpoint": iterative,
    }
    scalar_rows = []
    endpoint_rows = []
    curve_rows = []
    for method, images in reconstructions.items():
        for crop_id, (reference, image) in enumerate(zip(truth, images), start=1):
            metric = image_metrics(reference, image)
            spill = support_spill_metrics(reference, image)
            endpoints = endpoint_probes(reference, image, crop_id)
            curves = curve_probes(reference, image, crop_id)
            for probe_id, probe in enumerate(endpoints):
                endpoint_rows.append({
                    "method": method,
                    "crop_id": crop_id,
                    "probe_id": probe_id,
                    "overrun_length_px": probe["overrun_length_px"],
                    "excess_continuation_mass": probe["excess_continuation_mass"],
                })
            for probe_id, probe in enumerate(curves):
                curve_rows.append({
                    "method": method,
                    "crop_id": crop_id,
                    "probe_id": probe_id,
                    "localization_rmse_px": probe["localization_rmse_px"],
                    "tangent_mae_deg": probe["tangent_mae_deg"],
                    "bend_nrmse": probe["bend_nrmse"],
                    "bend_correlation": probe["bend_correlation"],
                    "total_turn_deg": probe["total_turn_deg"],
                    "length_samples": probe["length_samples"],
                })
            scalar_rows.append({
                "method": method,
                "crop_id": crop_id,
                **metric,
                "spill_area_px": spill["spill_area_px"],
                "spill_skeleton_length_px": spill["spill_skeleton_length_px"],
                "clean_endpoint_count": len(endpoints),
                "mean_endpoint_overrun_px": float(np.mean([probe["overrun_length_px"] for probe in endpoints])) if endpoints else np.nan,
                "mean_endpoint_excess_mass": float(np.mean([probe["excess_continuation_mass"] for probe in endpoints])) if endpoints else np.nan,
                "curved_branch_count": len(curves),
                "weighted_curve_localization_rmse_px": float(np.average([probe["localization_rmse_px"] for probe in curves], weights=[probe["length_samples"] for probe in curves])) if curves else np.nan,
                "weighted_tangent_mae_deg": float(np.average([probe["tangent_mae_deg"] for probe in curves], weights=[probe["length_samples"] for probe in curves])) if curves else np.nan,
                "median_bend_correlation": float(np.nanmedian([probe["bend_correlation"] for probe in curves])) if curves else np.nan,
            })
    scalar = pd.DataFrame(scalar_rows)
    endpoints = pd.DataFrame(endpoint_rows)
    curves = pd.DataFrame(curve_rows)
    scalar.to_csv(results / "per_crop.csv", index=False)
    endpoints.to_csv(results / "endpoints.csv", index=False)
    curves.to_csv(results / "curves.csv", index=False)
    summaries = [
        summary_for("Weighted FBP", weighted, truth, scalar, endpoints, curves),
        summary_for("Iterative curve-endpoint", iterative, truth, scalar, endpoints, curves),
    ]
    report = {
        "target_psnr_db": 16.01,
        "seed": 20260802,
        "angles_deg": ANGLES.astype(int).tolist(),
        "sart_sweeps": SART_SWEEPS,
        "sart_relaxation": SART_RELAXATION,
        "gaussian_sigma": GAUSSIAN_SIGMA,
        "methods": summaries,
    }
    (results / "benchmark.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    np.savez_compressed(results / "reconstructions.npz", weighted_fbp=weighted, iterative=iterative)
    for row in summaries:
        print(row["method"], round(row["mean_psnr_db"], 4), round(row["mean_ssim"], 4), round(row["mean_endpoint_overrun_px"], 4), round(row["weighted_curve_localization_rmse_px"], 4))

if __name__ == "__main__":
    main()
