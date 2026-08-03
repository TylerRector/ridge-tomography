from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi
from scipy.ndimage import map_coordinates
from scipy.signal import savgol_filter
from skimage import filters, metrics, morphology
from skimage.transform import iradon, iradon_sart, resize

SIZE = 128
CROP_SIDE = 250
CROP_CENTERS = ((370, 390), (410, 880), (860, 430), (920, 910))
ANGLES = np.array((2, 7, 11, 44, 46, 50, 61, 64, 84, 85, 90, 101, 108, 113, 128, 134, 157, 163), dtype=float)
SEED = 20260802
SART_SWEEPS = 200
SART_RELAXATION = 0.15
GAUSSIAN_SIGMA = 0.7
MASK_LOW = 0.12
MASK_HIGH = 0.30
ENDPOINT_THRESHOLD = 0.12
NEIGHBORS = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))

def vessel_patch(green, center):
    row, col = center
    half = CROP_SIDE // 2
    patch = green[row-half:row+half, col-half:col+half]
    patch = resize(patch, (SIZE, SIZE), anti_aliasing=True, preserve_range=True)
    background = filters.gaussian(patch, sigma=6, preserve_range=True)
    vessel = np.clip(background - patch, 0, None)
    low, high = np.percentile(vessel, (2, 99.5))
    vessel = np.clip((vessel - low) / max(high - low, 1e-8), 0, 1)
    yy, xx = np.ogrid[:SIZE, :SIZE]
    mask = (yy - (SIZE - 1) / 2) ** 2 + (xx - (SIZE - 1) / 2) ** 2 <= (SIZE / 2 - 3) ** 2
    return (vessel * mask).astype(np.float64)

def circular_voronoi_weights(theta_deg):
    theta = np.mod(np.deg2rad(theta_deg), np.pi)
    order = np.argsort(theta)
    sorted_theta = theta[order]
    gaps = np.diff(np.r_[sorted_theta, sorted_theta[0] + np.pi])
    previous = np.r_[gaps[-1], gaps[:-1]]
    sorted_weights = 0.5 * (previous + gaps)
    weights = np.empty_like(sorted_weights)
    weights[order] = sorted_weights
    return weights

def weighted_fbp(sinogram, theta_deg=ANGLES, out_size=SIZE):
    weights = circular_voronoi_weights(theta_deg)
    nominal = np.pi / len(theta_deg)
    weighted = sinogram * (weights / nominal)[None, :]
    reconstruction = iradon(weighted, theta=theta_deg, output_size=out_size, filter_name="ramp", interpolation="linear", circle=True, preserve_range=True)
    return np.clip(reconstruction, 0, 1)

def iterative_reconstruction(sinogram, theta_deg=ANGLES):
    image = np.zeros((SIZE, SIZE), dtype=np.float64)
    for _ in range(SART_SWEEPS):
        image = iradon_sart(sinogram, theta=theta_deg, image=image, relaxation=SART_RELAXATION)
        image = np.clip(image, 0, 1)
    image = filters.gaussian(image, sigma=GAUSSIAN_SIGMA, preserve_range=True)
    return np.clip(image, 0, 1)

def image_metrics(truth, reconstruction):
    return {
        "psnr_db": float(metrics.peak_signal_noise_ratio(truth, reconstruction, data_range=1)),
        "ssim": float(metrics.structural_similarity(truth, reconstruction, data_range=1)),
        "rmse": float(np.sqrt(np.mean((truth - reconstruction) ** 2))),
    }

def vessel_mask(image):
    mask = filters.apply_hysteresis_threshold(image, MASK_LOW, MASK_HIGH)
    mask = morphology.remove_small_objects(mask, max_size=11)
    return morphology.closing(mask, morphology.disk(1))

def skeleton_bundle(image):
    mask = vessel_mask(image)
    skeleton = morphology.skeletonize(mask)
    degree = ndi.convolve(skeleton.astype(np.int16), np.ones((3, 3), dtype=np.int16), mode="constant") - skeleton.astype(np.int16)
    endpoints = np.argwhere(skeleton & (degree == 1))
    junctions = np.argwhere(skeleton & (degree >= 3))
    return mask, skeleton, endpoints, junctions

def skeleton_neighbors(skeleton, point):
    row, col = point
    result = []
    for dr, dc in NEIGHBORS:
        rr, cc = row + dr, col + dc
        if 0 <= rr < skeleton.shape[0] and 0 <= cc < skeleton.shape[1] and skeleton[rr, cc]:
            result.append((rr, cc))
    return result

def trace_endpoint_inward(skeleton, endpoint, max_length=20):
    path = [tuple(map(int, endpoint))]
    previous = None
    current = path[0]
    for _ in range(max_length - 1):
        candidates = [point for point in skeleton_neighbors(skeleton, current) if point != previous]
        if len(candidates) != 1:
            break
        nxt = candidates[0]
        path.append(nxt)
        previous, current = current, nxt
        if len(skeleton_neighbors(skeleton, current)) != 2:
            break
    return np.asarray(path, dtype=np.float64)

def sample_cross_sections(image, endpoint, outward, distances, half_width=2.5, samples_across=7):
    normal = np.array((-outward[1], outward[0]), dtype=np.float64)
    offsets = np.linspace(-half_width, half_width, samples_across)
    values = []
    for distance in distances:
        points = endpoint[None, :] + distance * outward[None, :] + offsets[:, None] * normal[None, :]
        sampled = map_coordinates(image, (points[:, 0], points[:, 1]), order=1, mode="constant", cval=0)
        values.append(float(np.mean(sampled)))
    return np.asarray(values)

def endpoint_probes(truth, reconstruction, crop_id, maximum_distance=18):
    _, skeleton, endpoints, _ = skeleton_bundle(truth)
    center = np.array(((truth.shape[0] - 1) / 2, (truth.shape[1] - 1) / 2))
    probes = []
    for endpoint in endpoints:
        border_distance = min(endpoint[0], endpoint[1], truth.shape[0] - 1 - endpoint[0], truth.shape[1] - 1 - endpoint[1])
        if border_distance < 12 or np.linalg.norm(endpoint - center) > 50:
            continue
        inward_path = trace_endpoint_inward(skeleton, endpoint, max_length=12)
        if len(inward_path) < 7:
            continue
        inward_reference = np.mean(inward_path[-3:], axis=0)
        outward = inward_path[0] - inward_reference
        norm = np.linalg.norm(outward)
        if norm < 1:
            continue
        outward /= norm
        distances = np.arange(1, maximum_distance + 1, dtype=np.float64)
        truth_profile = sample_cross_sections(truth, inward_path[0], outward, distances)
        if np.max(truth_profile[3:]) > 0.16 or np.mean(truth_profile[5:]) > 0.08:
            continue
        profile = sample_cross_sections(reconstruction, inward_path[0], outward, distances)
        smoothed = np.convolve(profile, np.ones(2) / 2, mode="same")
        above = np.flatnonzero(smoothed > ENDPOINT_THRESHOLD)
        overrun = float(distances[above[-1]]) if len(above) else 0.0
        excess = float(np.sum(np.maximum(profile - truth_profile, 0)))
        probes.append({
            "crop_id": crop_id,
            "endpoint": inward_path[0],
            "outward": outward,
            "distances": distances,
            "truth_profile": truth_profile,
            "profile": profile,
            "overrun_length_px": overrun,
            "excess_continuation_mass": excess,
        })
    return probes

def extract_branches(skeleton):
    coordinates = set(map(tuple, np.argwhere(skeleton)))
    degree = {point: len(skeleton_neighbors(skeleton, point)) for point in coordinates}
    nodes = {point for point, value in degree.items() if value != 2}
    visited = set()
    branches = []
    def key(a, b):
        return tuple(sorted((a, b)))
    for node in nodes:
        for neighbor in skeleton_neighbors(skeleton, node):
            edge = key(node, neighbor)
            if edge in visited:
                continue
            visited.add(edge)
            path = [node, neighbor]
            previous, current = node, neighbor
            while current not in nodes:
                candidates = [point for point in skeleton_neighbors(skeleton, current) if point != previous]
                if len(candidates) != 1:
                    break
                nxt = candidates[0]
                edge = key(current, nxt)
                if edge in visited:
                    break
                visited.add(edge)
                path.append(nxt)
                previous, current = current, nxt
            branches.append(np.asarray(path, dtype=np.float64))
    return branches

def smooth_resample_path(path):
    if len(path) < 7:
        return path
    increments = np.sqrt(np.sum(np.diff(path, axis=0) ** 2, axis=1))
    arc = np.r_[0.0, np.cumsum(increments)]
    uniform = np.arange(0, arc[-1] + 1e-9, 1.0)
    rows = np.interp(uniform, arc, path[:, 0])
    cols = np.interp(uniform, arc, path[:, 1])
    window = min((len(uniform) // 2) * 2 - 1, 9)
    if window >= 5:
        rows = savgol_filter(rows, window, 2, mode="interp")
        cols = savgol_filter(cols, window, 2, mode="interp")
    return np.c_[rows, cols]

def path_geometry(path):
    points = smooth_resample_path(path)
    if len(points) < 7:
        return None
    drow = np.gradient(points[:, 0])
    dcol = np.gradient(points[:, 1])
    speed = np.hypot(drow, dcol) + 1e-10
    tangents = np.c_[drow / speed, dcol / speed]
    angles = np.unwrap(np.arctan2(tangents[:, 0], tangents[:, 1]))
    curvature = np.gradient(angles) / speed
    return points, tangents, curvature

def trace_reconstruction_ridge(reconstruction, points, tangents, half_width=5.0, samples_across=41):
    offsets = np.linspace(-half_width, half_width, samples_across)
    ridge_points = []
    peaks = []
    for point, tangent in zip(points, tangents):
        normal = np.array((-tangent[1], tangent[0]), dtype=np.float64)
        coordinates = point[None, :] + offsets[:, None] * normal[None, :]
        values = map_coordinates(reconstruction, (coordinates[:, 0], coordinates[:, 1]), order=1, mode="constant", cval=0)
        maximum = int(np.argmax(values))
        offset = offsets[maximum]
        if 0 < maximum < samples_across - 1:
            y0, y1, y2 = values[maximum - 1:maximum + 2]
            denominator = y0 - 2 * y1 + y2
            if abs(denominator) > 1e-10:
                offset += 0.5 * (y0 - y2) / denominator * (offsets[1] - offsets[0])
        ridge_points.append(point + offset * normal)
        peaks.append(float(values[maximum]))
    ridge_points = np.asarray(ridge_points)
    if len(ridge_points) >= 9:
        window = min((len(ridge_points) // 2) * 2 - 1, 9)
        ridge_points[:, 0] = savgol_filter(ridge_points[:, 0], window, 2, mode="interp")
        ridge_points[:, 1] = savgol_filter(ridge_points[:, 1], window, 2, mode="interp")
    drow = np.gradient(ridge_points[:, 0])
    dcol = np.gradient(ridge_points[:, 1])
    speed = np.hypot(drow, dcol) + 1e-10
    ridge_tangents = np.c_[drow / speed, dcol / speed]
    return ridge_points, ridge_tangents, np.asarray(peaks)

def tangent_error_degrees(first, second):
    dot = np.abs(np.sum(first * second, axis=1))
    return np.rad2deg(np.arccos(np.clip(dot, -1, 1)))

def bend_shape_metrics(truth_points, ridge_points):
    start, end = truth_points[0], truth_points[-1]
    chord = end - start
    length = np.linalg.norm(chord)
    if length < 1:
        return np.nan, np.nan
    tangent = chord / length
    normal = np.array((-tangent[1], tangent[0]))
    parameter = np.linspace(0, 1, len(truth_points))
    truth_chord = start[None, :] + parameter[:, None] * chord[None, :]
    truth_bend = (truth_points - truth_chord) @ normal
    ridge_start, ridge_end = ridge_points[0], ridge_points[-1]
    ridge_chord = ridge_start[None, :] + parameter[:, None] * (ridge_end - ridge_start)[None, :]
    ridge_bend = (ridge_points - ridge_chord) @ normal
    denominator = np.sqrt(np.mean(truth_bend ** 2)) + 1e-10
    nrmse = float(np.sqrt(np.mean((ridge_bend - truth_bend) ** 2)) / denominator)
    correlation = float(np.corrcoef(truth_bend, ridge_bend)[0, 1]) if np.std(truth_bend) > 1e-7 and np.std(ridge_bend) > 1e-7 else np.nan
    return nrmse, correlation

def curve_probes(truth, reconstruction, crop_id):
    _, skeleton, _, _ = skeleton_bundle(truth)
    center = np.array(((truth.shape[0] - 1) / 2, (truth.shape[1] - 1) / 2))
    probes = []
    for branch_id, branch in enumerate(extract_branches(skeleton)):
        if len(branch) < 14:
            continue
        geometry = path_geometry(branch)
        if geometry is None:
            continue
        points, tangents, curvature = geometry
        if len(points) < 12 or np.max(np.linalg.norm(points - center, axis=1)) > 55:
            continue
        total_turn = float(np.rad2deg(np.sum(np.abs(curvature))))
        if total_turn < 12:
            continue
        ridge_points, ridge_tangents, peak_values = trace_reconstruction_ridge(reconstruction, points, tangents)
        localization = float(np.sqrt(np.mean(np.sum((ridge_points - points) ** 2, axis=1))))
        tangent_mae = float(np.mean(tangent_error_degrees(tangents, ridge_tangents)))
        bend_nrmse, bend_correlation = bend_shape_metrics(points, ridge_points)
        probes.append({
            "crop_id": crop_id,
            "branch_id": branch_id,
            "points": points,
            "ridge_points": ridge_points,
            "localization_rmse_px": localization,
            "tangent_mae_deg": tangent_mae,
            "bend_nrmse": bend_nrmse,
            "bend_correlation": bend_correlation,
            "total_turn_deg": total_turn,
            "mean_ridge_peak": float(np.mean(peak_values)),
            "length_samples": int(len(points)),
        })
    return probes

def support_spill_metrics(truth, reconstruction):
    truth_mask = vessel_mask(truth)
    reconstruction_mask = vessel_mask(reconstruction)
    tolerance = morphology.dilation(truth_mask, morphology.disk(2))
    spill = reconstruction_mask & ~tolerance
    spill_skeleton = morphology.skeletonize(spill)
    return {
        "truth_mask": truth_mask,
        "reconstruction_mask": reconstruction_mask,
        "spill_mask": spill,
        "spill_area_px": int(np.count_nonzero(spill)),
        "spill_skeleton_length_px": int(np.count_nonzero(spill_skeleton)),
    }
