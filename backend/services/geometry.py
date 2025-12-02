# backend/services/geometry.py
import cv2
import numpy as np
from utils.debug import print_debug
from typing import List, Tuple


def compute_centers(contour: np.ndarray) -> dict:
    print_debug("Computing centers for main contour...")
    pts = contour.reshape(-1, 2).astype(float)

    # Mass center (moments)
    M = cv2.moments(contour)
    if M.get("m00", 0) != 0:
        cx_mass = float(M["m10"] / M["m00"])
        cy_mass = float(M["m01"] / M["m00"])
        center_mass = (cx_mass, cy_mass)
    else:
        center_mass = (float(pts[:, 0].mean()), float(pts[:, 1].mean()))

    # Mean of points
    center_mean = (float(pts[:, 0].mean()), float(pts[:, 1].mean()))

    # PCA center == mean of points (we keep separately)
    center_pca = center_mean

    # minEnclosingCircle (OpenCV)
    (ex, ey), R_cv = cv2.minEnclosingCircle(contour)
    center_min_enclosing = (float(ex), float(ey))
    R_cv = float(R_cv)

    # Manual enclosing circle (max distance from mass center)
    dx = pts[:, 0] - center_mass[0]
    dy = pts[:, 1] - center_mass[1]
    dist = np.sqrt(dx * dx + dy * dy)
    R_manual_mass = float(dist.max())

    return {
        "mass": center_mass,
        "mean": center_mean,
        "pca": center_pca,
        "min_enclosing_center": center_min_enclosing,
        "min_enclosing_radius": R_cv,
        "manual_center_used": center_mass,
        "manual_radius_from_mass": R_manual_mass,
    }

def compute_pca_axes(contour: np.ndarray):
    print_debug("Computing PCA axes...")
    pts = contour.reshape(-1, 2).astype(np.float32)
    mean = np.mean(pts, axis=0)
    pts_centered = pts - mean
    cov = np.cov(pts_centered.T)
    eigvals, eigvecs = np.linalg.eig(cov)
    idx = np.argsort(eigvals)[::-1]
    eigvecs = eigvecs[:, idx]
    # Normalize vectors
    v1 = eigvecs[:, 0] / (np.linalg.norm(eigvecs[:, 0]) + 1e-12)
    v2 = eigvecs[:, 1] / (np.linalg.norm(eigvecs[:, 1]) + 1e-12)
    return {
        "center": (float(mean[0]), float(mean[1])),
        "axis1": (float(v1[0]), float(v1[1])),
        "axis2": (float(v2[0]), float(v2[1])),
        "eigvals": eigvals.tolist()
    }

def compute_radial_data(contour: np.ndarray, center: tuple):
    print_debug("Computing radial distances and angles...")
    cx, cy = float(center[0]), float(center[1])
    pts = contour.reshape(-1, 2).astype(float)
    dx = pts[:, 0] - cx
    dy = pts[:, 1] - cy
    radii = np.sqrt(dx**2 + dy**2)
    angles = np.arctan2(dy, dx)  # -pi..pi
    # Normalize angles to 0..2pi for sorting
    angles2 = (angles + 2 * np.pi) % (2 * np.pi)
    return radii, angles2

def find_peaks(radii: np.ndarray, angles: np.ndarray, contour: np.ndarray, threshold_ratio: float = 0.95):
    print_debug("Finding peaks in contour radii...")
    radii = np.asarray(radii)
    max_r = radii.max()
    if max_r == 0:
        return []

    # Indices with radii above threshold
    inds = np.where(radii > max_r * threshold_ratio)[0]
    # Normalize to python list of ints
    if np.isscalar(inds):
        peak_indices = [int(inds)]
    else:
        peak_indices = [int(i) for i in np.atleast_1d(inds)]

    # If no indices by threshold, fallback to local maxima approach
    if len(peak_indices) == 0:
        # simple local maxima on circular signal
        N = len(radii)
        peak_indices = []
        for i in range(N):
            prev_i = (i - 1) % N
            next_i = (i + 1) % N
            if radii[i] >= radii[prev_i] and radii[i] >= radii[next_i]:
                peak_indices.append(i)

    # Build list of peak points with coords, radius and angle
    peaks = []
    pts = contour.reshape(-1, 2)
    for i in peak_indices:
        # guard index
        idx = int(i) % len(pts)
        x, y = float(pts[idx][0]), float(pts[idx][1])
        peaks.append({"index": idx, "point": (x, y), "radius": float(radii[idx]), "angle": float(angles[idx])})
    return peaks

def find_radial_peaks(radii: np.ndarray, angles: np.ndarray, contour: np.ndarray, num_rays: int = 6):
    """
    Finds one peak per radial sector around the center.
    radii: Distances from center to contour points.
    angles: Angles from center to contour points (0 to 2*pi).
    contour: Original contour points [[x, y], ...].
    num_rays: Number of sectors to divide the circle into.
    """
    peaks = []
    angle_step = 2 * np.pi / num_rays

    base_idx = int(np.argmax(radii))
    base_angle = angles[base_idx] + np.pi / 6
    print(base_angle * 180/np.pi)

    for i in range(num_rays):
        start_angle = base_angle + i * angle_step
        end_angle = (i + 1) * angle_step

        # Найти индексы точек в секторе
        # angles >= start_angle & angles < end_angle
        # или angles >= start_angle & angles <= end_angle (для последнего сектора)
        # или использовать (angles - start_angle + 2*np.pi) % (2*np.pi) < angle_step
        # (последний вариант учитывает переход через 0)
        sector_angles = (angles - start_angle + 2 * np.pi) % (2 * np.pi)
        inds_in_sector = np.where(sector_angles < angle_step)[0]

        if len(inds_in_sector) == 0:
            # Если в секторе нет точек, пропустить
            continue

        # Найти индекс точки с максимальным радиусом в секторе
        sector_radii = radii[inds_in_sector]
        local_max_idx_in_sector = np.argmax(sector_radii)
        global_idx = inds_in_sector[local_max_idx_in_sector]
        
        # Собрать информацию о пике
        peak_info = {
            "index": int(global_idx),
            "point": (float(contour[global_idx][0][0]), float(contour[global_idx][0][1])),
            "radius": float(radii[global_idx]),
            "angle": float(angles[global_idx]),
            "sector": i, # <-- Опционально: к какому сектору относится
        }
        peaks.append(peak_info)

    return peaks

def compute_circumcircles(contour: np.ndarray, mass_center: tuple):
    print_debug("Computing circumcircles...")
    pts = contour.reshape(-1, 2).astype(float)
    # OpenCV minEnclosingCircle
    (ex, ey), R_cv = cv2.minEnclosingCircle(contour)
    # Manual circle w.r.t. mass center
    cx, cy = float(mass_center[0]), float(mass_center[1])
    dist = np.sqrt((pts[:, 0] - cx)**2 + (pts[:, 1] - cy)**2)
    R_manual = float(dist.max())
    return {
        "opencv": {"center": (float(ex), float(ey)), "radius": float(R_cv)},
        "manual_from_mass": {"center": (cx, cy), "radius": R_manual},
    }

def analyze_main_contour(contour: np.ndarray) -> dict:
    """
    Pure analysis function. Returns dictionary with EVERY computed value.
    Drawing is separated (see analysis_service.draw_contour_analysis).
    """
    print_debug("Starting main contour analysis...")
    contour = contour.reshape(-1, 1, 2).astype(np.int32) if contour.ndim == 2 else contour.astype(np.int32)

    centers = compute_centers(contour)
    pca = compute_pca_axes(contour)

    # choose a center for radial representation: use mass center (we keep all centers in dict)
    center_for_radial = centers["min_enclosing_center"] #mass
    radii, angles = compute_radial_data(contour, center_for_radial)

    peaks = find_radial_peaks(radii, angles, contour)
    circles = compute_circumcircles(contour, center_for_radial)

    analysis = {
        "centers": centers,
        "pca": pca,
        "radii": radii.tolist(),
        "angles": angles.tolist(),
        "peaks": peaks,
        "circles": circles,
    }
    print_debug("Main contour analysis finished.")
    return analysis





