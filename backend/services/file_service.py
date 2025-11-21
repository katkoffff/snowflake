# backend/services/file_service.py
"""
Сервис работы с файлами.
НЕ зависит от сессий.
"""

from pathlib import Path
import shutil
import cv2
import numpy as np
from core.config import RESULTS_DIR, STAGE2_DIR
from utils.debug import print_debug
from services.image_service import overlay_contour


def ensure_snowflake_dir(session_id: str, orig_name: str) -> Path:
    """
    Создаёт папку по session_id + orig_name.
    НЕ использует сессию.
    """
    base_name = orig_name or f"snowflake_{session_id[:8]}"
    dir_path = RESULTS_DIR / base_name
    dir_path.mkdir(parents=True, exist_ok=True)
    print_debug(f"Created snowflake dir: {dir_path}")
    return dir_path

def ensure_stage2_dir(session_id: str, orig_name: str) -> Path:
    """
    Creates a folder within STAGE2_DIR based on session_id and orig_name.
    DOES NOT depend on session state for location.
    """
    base_name = orig_name or f"stage2_{session_id[:8]}"
    dir_path = STAGE2_DIR / base_name
    dir_path.mkdir(parents=True, exist_ok=True)
    print_debug(f"Created stage2 dir: {dir_path}")
    return dir_path

def save_mask_pack(dir_path: Path, prefix: str, image_np: np.ndarray, contour, mask_bin: np.ndarray, logits: np.ndarray):
    cv2.imwrite(str(dir_path / f"{prefix}.jpg"), cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))
    overlay = overlay_contour(image_np, contour, (255, 0, 0))
    cv2.imwrite(str(dir_path / f"{prefix}_contour.jpg"), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    np.save(str(dir_path / f"{prefix}_mask.npy"), mask_bin)
    np.save(str(dir_path / f"{prefix}_logits.npy"), logits)


def copy_to_stage2(folder_name: str) -> str:
    src_dir = RESULTS_DIR / folder_name
    dst_dir = STAGE2_DIR / folder_name

    if not src_dir.exists():
        raise ValueError(f"Folder not found: {folder_name}")

    required = ["final.jpg", "all_contours.npy"]
    missing = [f for f in required if not (src_dir / f).exists()]
    if missing:
        raise ValueError(f"Missing files in {folder_name}: {missing}")

    if dst_dir.exists():
        shutil.rmtree(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy2(src_dir / "final.jpg", dst_dir / "final.jpg")
    shutil.copy2(src_dir / "all_contours.npy", dst_dir / "all_contours.npy")
    print_debug(f"Copied {folder_name} to stage2")
    return str(dst_dir)

def load_contours_and_image_from_folder(folder_name: str) -> tuple[list, np.ndarray]:
    """
    Loads all_contours.npy and final.jpg from a folder named 'folder_name'.
    First checks in STAGE2_DIR, then in RESULTS_DIR.
    Returns (all_contours_info, main_image_rgb).
    """
    print_debug(f"[DEBUG] Looking for folder '{folder_name}' in STAGE2_DIR and RESULTS_DIR")

    # Попробуем сначала STAGE2_DIR (так как stage2 использует его)
    folder_path = STAGE2_DIR / folder_name
    if not folder_path.exists() or not folder_path.is_dir():
        print_debug(f"[DEBUG] Folder '{folder_name}' not found in STAGE2_DIR, checking RESULTS_DIR")
        # Если не найдено в STAGE2_DIR, проверим RESULTS_DIR
        folder_path = RESULTS_DIR / folder_name
        if not folder_path.exists() or not folder_path.is_dir():
            raise FileNotFoundError(f"Session folder '{folder_name}' not found in STAGE2_DIR ({STAGE2_DIR}) or RESULTS_DIR ({RESULTS_DIR}).")

    print_debug(f"[DEBUG] Found folder at: {folder_path}")

    all_contours_path = folder_path / "all_contours.npy"
    if not all_contours_path.exists():
        raise FileNotFoundError(f"File 'all_contours.npy' not found in '{folder_name}'.")

    main_image_path = folder_path / "final.jpg"
    if not main_image_path.exists():
        raise FileNotFoundError(f"File 'final.jpg' not found in '{folder_name}'.")

    # Загрузка контуров
    all_contours_info = np.load(str(all_contours_path), allow_pickle=True)

    # Загрузка изображения
    main_image_bgr = cv2.imread(str(main_image_path))
    if main_image_bgr is None:
        raise ValueError(f"Could not load 'final.jpg' from '{folder_name}'.")
    main_image_rgb = cv2.cvtColor(main_image_bgr, cv2.COLOR_BGR2RGB)

    return all_contours_info, main_image_rgb

