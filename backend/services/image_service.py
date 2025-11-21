# backend/services/image_service.py
"""
Сервис обработки изображений.
Содержит:
- preprocess_image_cv: полная предобработка (медиана, контраст, резкость, CLAHE)
- image_to_base64_png: конвертация np.ndarray → base64 PNG
"""

from PIL import Image, ImageEnhance
import cv2
import numpy as np
import base64
from skimage.morphology import skeletonize
from typing import List, Tuple
from services.geometry import count_neighbors


def preprocess_image_cv(image: np.ndarray, config: dict | None = None) -> np.ndarray:
    """
    image: RGB numpy (H,W,3) uint8 — из PIL
    config: dict с параметрами (или None → дефолты)
    return: обработанное RGB изображение
    """
    if config is None:
        config = {}

    # median blur
    ksize = config.get("median_ksize", 5)
    if ksize % 2 == 0:
        ksize += 1
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image_smoothed = cv2.medianBlur(image_bgr, ksize=ksize)

    # contrast via PIL
    contrast_factor = config.get("contrast_factor", 1.5)
    pil_img = Image.fromarray(cv2.cvtColor(image_smoothed, cv2.COLOR_BGR2RGB))
    enhancer = ImageEnhance.Contrast(pil_img)
    pil_img = enhancer.enhance(contrast_factor)
    image_contrast = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    # sharpness via PIL
    sharpness_factor = config.get("sharpness_factor", 2.0)
    pil_img = Image.fromarray(cv2.cvtColor(image_contrast, cv2.COLOR_BGR2RGB))
    enhancer = ImageEnhance.Sharpness(pil_img)
    pil_img = enhancer.enhance(sharpness_factor)
    image_sharp = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    # CLAHE on L channel
    clahe_clip = config.get("clahe_clip_limit", 1.5)
    tile_grid = config.get("clahe_tile_grid", (8, 8))
    lab = cv2.cvtColor(image_sharp, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=tile_grid)
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    image_clahe = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    # final smoothing
    image_processed_bgr = cv2.medianBlur(image_clahe, ksize=3)
    image_processed_rgb = cv2.cvtColor(image_processed_bgr, cv2.COLOR_BGR2RGB)
    return image_processed_rgb


def image_to_base64_png(image_rgb: np.ndarray) -> str:
    """RGB numpy → base64 PNG (без data:image/png;base64,)"""
    _, png = cv2.imencode(".png", cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
    return base64.b64encode(png.tobytes()).decode("utf-8")

def overlay_contour(image_rgb: np.ndarray, contour, color) -> np.ndarray:
    """
    Накладывает контур на изображение.
    Используется в save_mask_pack, /segment, /save_all.
    """
    overlay = image_rgb.copy()
    if contour is not None:
        pts = np.array(contour, dtype=np.int32)
        cv2.polylines(overlay, [pts], isClosed=True, color=color, thickness=2)
    return overlay

def create_image_with_contours(main_image_rgb: np.ndarray, all_contours_info: list) -> np.ndarray:
    """
    Creates a new image based on main_image_rgb and draws all contours from all_contours_info on it.
    main: red (255, 0, 0), slave: yellow (0, 255, 255).
    """
    height, width = main_image_rgb.shape[:2]
    # Создаём пустое изображение (например, белый фон)
    img_with_contours = np.full((height, width, 3), 255, dtype=np.uint8)

    for info in all_contours_info:
        contour_type = info.get("type", "")
        contour_pts = np.array(info.get("contour", []), dtype=np.int32)
        if contour_pts.size == 0:
            continue

        if contour_type == "main":
            color = (255, 0, 0) # Красный для main
            thickness = 2
        elif contour_type.startswith("slave_"):
            color = (0, 255, 255) # Жёлтый для slave
            thickness = 2
        else:
            # Игнорируем неизвестные типы или рисуем серым?
            color = (128, 128, 128) # Серый для неизвестных
            thickness = 1

        cv2.polylines(img_with_contours, [contour_pts], isClosed=True, color=color, thickness=thickness)

    return img_with_contours


def contour_to_mask(contour: np.ndarray, img_shape: Tuple[int, int]) -> np.ndarray:
    """Контур → бинарная маска"""
    h, w = img_shape
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [contour], 255)
    return mask

def thin_skeleton(mask: np.ndarray) -> np.ndarray:
    """Скелетонизация: предпочтительно cv2.ximgproc, fallback — skimage"""
    try:
        return cv2.ximgproc.thinning(mask)
    except:
        # fallback
        return (skeletonize(mask > 0).astype(np.uint8) * 255)

def prune_skeleton(skeleton: np.ndarray, min_length: int = 5) -> np.ndarray:
    """Удаляет короткие "усики" — прунинг скелета"""
    pruned = skeleton.copy()
    h, w = skeleton.shape
    visited = np.zeros_like(skeleton, dtype=bool)

    def dfs_prune(x, y, length):
        stack = [(x, y, length)]
        path = []
        while stack:
            cx, cy, clen = stack[-1]
            if visited[cy, cx]:
                stack.pop()
                continue
            visited[cy, cx] = True
            path.append((cx, cy))

            neighbors = [(cx+dx, cy+dy) for dx in [-1,0,1] for dy in [-1,0,1]
                        if not (dx == 0 and dy == 0)
                        and 0 <= cx+dx < w and 0 <= cy+dy < h
                        and pruned[cy+dy, cx+dx] > 0 and not visited[cy+dy, cx+dx]]

            if not neighbors:
                stack.pop()
                if clen < min_length and len(path) > 1:
                    for px, py in path:
                        pruned[py, px] = 0
                path = []
            else:
                nx, ny = neighbors[0]
                stack.append((nx, ny, clen + 1))

    for y in range(h):
        for x in range(w):
            if pruned[y, x] > 0 and not visited[y, x] and count_neighbors(pruned, x, y) == 1:
                dfs_prune(x, y, 0)
    return pruned


