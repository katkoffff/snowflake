# backend/services/analysis_service.py
"""
Сервис анализа результатов.
Полностью соответствует main.py:
- /analyze_results → анализ main/slave контуров
- /analysis_results → чтение JSON
- /analysis_save_chart → миниатюры, линии, OffsetImage
"""

from pathlib import Path
import numpy as np
import json
import cv2
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from datetime import datetime
from typing import List, Dict, Any, Tuple
from core.config import RESULTS_DIR, GRAPH_DIR
from utils.debug import print_debug
import traceback
from services.geometry import bresenham_line, count_neighbors, point_to_tuple
from services.image_service import contour_to_mask, thin_skeleton, prune_skeleton
import math
from scipy.signal import find_peaks, find_peaks_cwt

def analyze_all_sessions() -> List[Dict[str, Any]]:
    """   
    - main.jpg
    - all_contours.npy с type: main / slave_*
    - main_enclosing_radius
    - normalized_perimeter / normalized_area
    - сохраняет в RESULTS_DIR/analysis_results.json
    """
    print_debug("/analyze_results called")
    analysis_data = []

    for session_dir_path in RESULTS_DIR.iterdir():
        if not session_dir_path.is_dir():
            continue

        all_contours_path = session_dir_path / "all_contours.npy"
        if not all_contours_path.exists():
            print_debug(f"Skipping {session_dir_path.name}, no all_contours.npy")
            continue

        main_image_path = session_dir_path / "main.jpg"
        if not main_image_path.exists():
            print_debug(f"Skipping {session_dir_path.name}, no main.jpg")
            continue

        try:
            all_contours_info = np.load(str(all_contours_path), allow_pickle=True)

            main_contour = None
            slave_contours = []
            for info in all_contours_info:
                contour_type = info.get("type", "")
                contour_pts = np.array(info.get("contour", []), dtype=np.int32)
                if contour_type == "main":
                    main_contour = contour_pts
                elif contour_type.startswith("slave_") and contour_pts.size > 0:
                    slave_contours.append(contour_pts)

            if main_contour is None:
                print_debug(f"Skipping {session_dir_path.name}, no 'main' contour")
                continue

            main_perimeter = cv2.arcLength(main_contour, closed=True)
            main_area = cv2.contourArea(main_contour)
            _, main_enclosing_radius = cv2.minEnclosingCircle(main_contour)

            total_slave_perimeter = sum(cv2.arcLength(sc, closed=True) for sc in slave_contours)
            total_slave_area = sum(cv2.contourArea(sc) for sc in slave_contours)

            total_perimeter = main_perimeter + total_slave_perimeter
            total_area = main_area - total_slave_area

            if main_enclosing_radius > 0:
                normalized_perimeter = total_perimeter / (2 * np.pi * main_enclosing_radius)
                normalized_area = total_area / (np.pi * main_enclosing_radius**2)
            else:
                print_debug(f"[WARN] Radius 0 for {session_dir_path.name}")
                normalized_perimeter = normalized_area = 0.0

            analysis_data.append({
                "session_folder": session_dir_path.name,
                "normalized_perimeter": normalized_perimeter,
                "normalized_area": normalized_area,
                "main_image_path": str(main_image_path.relative_to(RESULTS_DIR))
            })

        except Exception as e:
            print_debug(f"[ERROR] Analyze {session_dir_path.name}: {e}")
            traceback.print_exc()
            continue

    # Сохранение
    results_file_path = RESULTS_DIR / "analysis_results.json"
    with open(results_file_path, 'w', encoding='utf-8') as f:
        json.dump(analysis_data, f, indent=2, ensure_ascii=False)

    print_debug(f"Analysis saved to {results_file_path}")
    return {"results_path": results_file_path} # Относительно корня проекта


def read_analysis_results() -> List[Dict[str, Any]]:
    """
    Читает RESULTS_DIR/analysis_results.json
    """
    results_file_path = RESULTS_DIR / "analysis_results.json"
    if not results_file_path.exists():
        raise FileNotFoundError("Analysis results file not found")

    try:
        with open(results_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except json.JSONDecodeError as e:
        print_debug(f"[ERROR] JSON decode: {e}")
        raise
    except Exception as e:
        print_debug(f"[ERROR] Read file: {e}")
        raise


def save_chart(
    points: List[Dict[str, Any]],
    axes: Dict[str, Any],
    miniatures: List[Dict[str, Any]],
    viewport_size: dict
) -> Path:
    """
    Сохраняет график с миниатюрами и линиями.
    Вход: Pydantic модели (как в SaveChartRequest)
    """
    try:
        fig, ax = plt.subplots(figsize=(16, 12))

        # Точки
        x_coords = [p["x"] for p in points]
        y_coords = [p["y"] for p in points]
        colors = [p["color"] for p in points]
        ax.scatter(x_coords, y_coords, c=colors, s=50, alpha=0.7, edgecolors='black', linewidth=0.5)

        # Оси
        ax.set_xlabel(axes["x_label"], fontsize=12)
        ax.set_ylabel(axes["y_label"], fontsize=12)
        ax.set_xlim(axes["x_range"])
        ax.set_ylim(axes["y_range"])
        ax.grid(True, alpha=0.3)

        # Миниатюры
        for miniature in miniatures:
            try:
                img_path = RESULTS_DIR / miniature["image_path"] / miniature["image_file"]
                if not img_path.exists():
                    continue
                img = plt.imread(img_path)
                height, width = img.shape[:2]
                scale = 48.0 / max(height, width)
                imagebox = OffsetImage(img, zoom=scale)
                ab = AnnotationBbox(
                    imagebox,
                    (miniature["svg_x"], miniature["svg_y"]),
                    frameon=True,
                    pad=0.1,
                    bboxprops=dict(edgecolor='gray', linewidth=1)
                )
                ax.add_artist(ab)

                # Линия
                ax.plot(
                    [miniature["dot_x"], miniature["svg_x"]],
                    [miniature["dot_y"], miniature["svg_y"]],
                    'gray', linestyle='-', linewidth=1, alpha=0.7
                )
            except Exception as e:
                print_debug(f"Miniature error: {e}")
                continue

        # Сохранение
        save_dir = GRAPH_DIR / "saved_charts"
        save_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"analysis_chart_{timestamp}.jpg"
        file_path = save_dir / filename
        plt.savefig(file_path, dpi=150, bbox_inches='tight')
        plt.close()

        print_debug(f"Chart saved: {file_path}")
        return file_path

    except Exception as e:
        print_debug(f"Chart save error: {e}")
        traceback.print_exc()
        raise

DEFAULT_COLORS = {
    "contour": (255, 0, 0),              # главный контур — КРАСНЫЙ (яркий, читаемый)
    
    "min_enclosing": (0, 255, 255),      # циан
    "manual_from_mass": (0, 255, 0),     # зелёный
    "pca_circle": (255, 0, 0),           # синий
    
    "pca_axis_major": (128, 0, 255),     # фиолетовый
    "pca_axis_minor": (255, 128, 0),     # оранжевый
    
    "peaks": (0, 165, 255),              # янтарный (лучше видно на фоне снега)
    "radial_lines": (0, 128, 255),       # тёмно-оранжево-синий
    
    "center_marker": (0, 0, 0),          # черный
    "label_text": (0, 0, 0),             # черный текст
}


DEFAULT_SETTINGS = {
    "draw_main_contour": True,
    "draw_centers": True,
    "draw_pca_axes": True,
    "draw_min_enclosing_circle": True,
    "draw_custom_enclosing_circle": True,
    "draw_radial_peaks": True,
    "draw_radial_lines": True,
    "draw_labels": False,
    "circle_line_thickness": 2,
    "contour_thickness": 2,
    # radius limit for axes drawing (if None -> use manual radius)
    "pca_axis_length": None,
}

def _safe_int_tuple(pt):
    return (int(round(pt[0])), int(round(pt[1])))

def draw_contour_analysis(image: np.ndarray, contour: np.ndarray, analysis: dict, settings: dict = None) -> np.ndarray:
    """
    Draws analysis on a copy of image and returns annotated image (RGB/BGR uint8).
    settings controls what to draw; analysis must contain full info from geometry.analyze_main_contour.
    """
    print_debug("Drawing contour analysis on image...")
    if settings is None:
        settings = DEFAULT_SETTINGS
    else:
        s = DEFAULT_SETTINGS.copy()
        s.update(settings)
        settings = s

    img = image.copy()
    # ensure BGR uint8
    if img.dtype != np.uint8:
        img = (np.clip(img, 0, 255)).astype(np.uint8)

    # Draw main contour
    if settings["draw_main_contour"]:
        cv2.polylines(img, [contour], True, DEFAULT_COLORS["contour"], settings["contour_thickness"])

    centers = analysis.get("centers", {})
    circles = analysis.get("circles", {})
    pca = analysis.get("pca", {})
    
    # Draw circles (minEnclosing and manual_from_mass)
    if settings["draw_min_enclosing_circle"] and "opencv" in circles:
        c = circles["opencv"]
        center = _safe_int_tuple(c["center"])
        radius = int(round(c["radius"]))
        cv2.circle(img, center, radius, DEFAULT_COLORS["min_enclosing"], settings["circle_line_thickness"])
        if settings["draw_labels"]:
            cv2.putText(img, "minCircle", (center[0]+6, center[1]-6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, DEFAULT_COLORS["min_enclosing"], 1, cv2.LINE_AA)
        
    if settings["draw_custom_enclosing_circle"] and "manual_from_mass" in circles:
        c = circles["manual_from_mass"]
        center = _safe_int_tuple(c["center"])
        radius = int(round(c["radius"]))
        cv2.circle(img, center, radius, DEFAULT_COLORS["manual_from_mass"], settings["circle_line_thickness"])
        if settings["draw_labels"]:
            cv2.putText(img, "manualCircle", (center[0]+6, center[1]-6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, DEFAULT_COLORS["manual_from_mass"], 1, cv2.LINE_AA)

    # Draw centers (mass, mean, pca, min_enclosing_center)
    if settings["draw_centers"]:
        c_map = {
            "mass": ("mass", DEFAULT_COLORS["manual_from_mass"]),
            "mean": ("mean", DEFAULT_COLORS["pca_circle"]),
            "pca": ("pca", DEFAULT_COLORS["pca_circle"]),
            "min_enclosing_center": ("min_enclosing_center", DEFAULT_COLORS["min_enclosing"]),
            "manual_center_used": ("manual_center_used", DEFAULT_COLORS["manual_from_mass"]),
        }
        for label, (key, color) in c_map.items():
            if key in centers:
                pt = _safe_int_tuple(centers[key])
                cv2.circle(img, pt, 5, color, -1)
                if settings["draw_labels"]:
                    txt = "C_" + key.replace("_", "")
                    cv2.putText(img, txt, (pt[0]+6, pt[1]+6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

    # Draw PCA axes limited by circle/pca_axis_length
    if settings["draw_pca_axes"] and "axis1" in pca:
        center = _safe_int_tuple(pca["center"])
        v1 = np.array(pca["axis1"], dtype=float)
        v2 = np.array(pca["axis2"], dtype=float)
        # axis length: prefer setting value -> else use manual radius
        length = settings["pca_axis_length"] or int(round(analysis.get("circles", {}).get("manual_from_mass", {}).get("radius", 0)))
        if length == 0:
            length = 100
        end1 = (int(center[0] + v1[0] * length), int(center[1] + v1[1] * length))
        end1b = (int(center[0] - v1[0] * length), int(center[1] - v1[1] * length))
        end2 = (int(center[0] + v2[0] * length), int(center[1] + v2[1] * length))
        end2b = (int(center[0] - v2[0] * length), int(center[1] - v2[1] * length))

        cv2.line(img, center, end1, DEFAULT_COLORS["pca_axis_major"], 2)
        cv2.line(img, center, end1b, DEFAULT_COLORS["pca_axis_major"], 2)
        cv2.line(img, center, end2, DEFAULT_COLORS["pca_axis_minor"], 2)
        cv2.line(img, center, end2b, DEFAULT_COLORS["pca_axis_minor"], 2)
        if settings["draw_labels"]:
            cv2.putText(img, "PCA_center", (center[0]+6, center[1]-12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, DEFAULT_COLORS["pca_circle"], 1, cv2.LINE_AA)

    # Draw peaks and optionally radial lines
    if settings["draw_radial_peaks"]:
        peaks = analysis.get("peaks", [])
        sectors_data = []  # Список для хранения данных секторов
        
        # Получаем данные центра и радиуса
        mc = _safe_int_tuple(analysis.get("centers", {}).get("min_enclosing_center", (0,0)))
        radius = int(analysis.get("circles", {}).get("opencv", {}).get("radius", 0))
        
        for p in peaks:
            px, py = int(round(p["point"][0])), int(round(p["point"][1]))
            
            # Рисуем основную точку пика
            cv2.circle(img, (px, py), 5, DEFAULT_COLORS["peaks"], -1)
            
            if settings["draw_labels"]:
                cv2.putText(img, f"peak#{p['index']}", (px+6, py-6), cv2.FONT_HERSHEY_SIMPLEX, 0.4, DEFAULT_COLORS["peaks"], 1, cv2.LINE_AA)
            
            if settings["draw_radial_lines"]:
                # Рисуем линию от масс центра к пику
                cv2.line(img, mc, (px, py), DEFAULT_COLORS["radial_lines"], 1)
                
                # Вычисляем угол радиуса
                dx, dy = px - mc[0], py - mc[1]
                current_angle = math.atan2(dy, dx)
                
                # Углы для поворота ±π/6
                angle_offset = math.pi / 6
                angle_plus = current_angle + angle_offset
                angle_minus = current_angle - angle_offset
               
                # Вычисляем координаты для повернутых линий
                px_plus = int(mc[0] + radius * math.cos(angle_plus))
                py_plus = int(mc[1] + radius * math.sin(angle_plus))                
                px_minus = int(mc[0] + radius * math.cos(angle_minus))
                py_minus = int(mc[1] + radius * math.sin(angle_minus))
                
                # Рисуем пунктирные линии
                dash_length = 5
                gap_length = 3
                
                # Функция для рисования пунктирной линии
                def draw_dashed_line(img, start, end, color, dash_length=5):
                    x1, y1 = start
                    x2, y2 = end
                    dx = x2 - x1
                    dy = y2 - y1
                    distance = math.sqrt(dx*dx + dy*dy)                    
                    dx_normalized = dx / distance
                    dy_normalized = dy / distance
                    
                    for i in range(0, int(distance), dash_length + gap_length):
                        start_i = (int(x1 + dx_normalized * i), int(y1 + dy_normalized * i))
                        end_i = (int(x1 + dx_normalized * (i + dash_length)), int(y1 + dy_normalized * (i + dash_length)))
                        if i + dash_length <= distance:
                            cv2.line(img, start_i, end_i, color, 1, cv2.LINE_AA)
                
                # Рисуем обе пунктирные линии красным цветом
                draw_dashed_line(img, mc, (px_plus, py_plus), (0, 0, 255), dash_length)
                draw_dashed_line(img, mc, (px_minus, py_minus), (0, 0, 255), dash_length)
                # Формируем данные сектора
                sector_data = {
                    "sector_index": p["sector"],
                    "center": mc,
                    "radius": radius,
                    "peak_point": (px, py),
                    "angle_plus_point": (px_plus, py_plus),
                    "angle_minus_point": (px_minus, py_minus),
                    "base_angle": current_angle,
                    "angle_offset": angle_offset
                }
                sectors_data.append(sector_data)
        
        print_debug("Drawing finished.")
        return img, sectors_data

def find_snowflake_center(skeleton: np.ndarray, contour: np.ndarray) -> Tuple[int, int]:
    """Центр по моменту контура (надёжнее, чем пересечение)"""
    M = cv2.moments(contour)
    if M["m00"] == 0:
        return None
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    return (cx, cy)

def find_main_tips(skeleton: np.ndarray, center: Tuple[int, int], min_dist: int = 40) -> List[Tuple[int, int]]:
    """Находит главные кончики: листья скелета, далеко от центра"""
    tips = []
    h, w = skeleton.shape
    cx, cy = center
    for y in range(h):
        for x in range(w):
            if skeleton[y, x] == 0:
                continue
            if count_neighbors(skeleton, x, y) != 1:
                continue
            dist = np.linalg.norm(np.array([x, y]) - np.array([cx, cy]))
            if dist > min_dist:
                tips.append((x, y))
    # Сортируем по углу → 6 штук
    angles = [np.arctan2(y-cy, x-cx) for x, y in tips]
    tips = [t for _, t in sorted(zip(angles, tips))]
    return tips[:6]  # ожидаем 6

def dfs_to_tip(skeleton: np.ndarray, start_x: int, start_y: int,
               radius_line: List[Tuple[int, int]], visited: set,
               min_branch_len: int) -> Tuple[int, int]:
    """DFS от точки ветвления → до кончика"""
    h, w = skeleton.shape
    stack = [(start_x, start_y, 0)]  # (x, y, len)
    path = []

    while stack:
        x, y, length = stack.pop()
        if (x, y) in visited:
            continue
        visited.add((x, y))
        path.append((x, y))

        if count_neighbors(skeleton, x, y) == 1 and length >= min_branch_len:
            return (x, y)

        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                nx, ny = x + dx, y + dy
                if (0 <= nx < w and 0 <= ny < h and
                    skeleton[ny, nx] > 0 and
                    (nx, ny) not in visited and
                    (nx, ny) not in radius_line):
                    stack.append((nx, ny, length + 1))
    return None

def find_side_branches_along_radius(skeleton: np.ndarray, radius_line: List[Tuple[int, int]],
                                   center: Tuple[int, int], main_tip: Tuple[int, int],
                                   min_branch_len: int = 8) -> Dict:
    vec_rad = np.array(main_tip) - np.array(center)
    perp = np.array([-vec_rad[1], vec_rad[0]])  # поворот на 90°

    radius_set = set(radius_line)
    visited = set()
    branch_points = set()
    side_tips_left = []
    side_tips_right = []

    for px, py in radius_line:
        for dy in range(-2, 3):
            for dx in range(-2, 3):
                nx, ny = px + dx, py + dy
                if not (0 <= nx < skeleton.shape[1] and 0 <= ny < skeleton.shape[0]):
                    continue
                if skeleton[ny, nx] == 0 or (nx, ny) in radius_set or (nx, ny) in visited:
                    continue

                tip = dfs_to_tip(skeleton, nx, ny, radius_line, visited, min_branch_len)
                if tip:
                    branch_points.add((px, py))
                    vec_b = np.array(tip) - np.array([px, py])
                    side = np.sign(np.dot(vec_b, perp))
                    if side > 0:
                        side_tips_left.append(tip)
                    else:
                        side_tips_right.append(tip)

    return {
        "main_tip": main_tip,
        "branch_points": list(branch_points),
        "side_tips_left": side_tips_left,
        "side_tips_right": side_tips_right
    }

def analyze_branches_with_skeleton(contour: np.ndarray, img_shape: Tuple[int, int]) -> Dict:
    mask = contour_to_mask(contour, img_shape)
    skeleton = thin_skeleton(mask)
    skeleton = prune_skeleton(skeleton, min_length=5)

    center = find_snowflake_center(skeleton, contour)
    if center is None:
        raise ValueError("Center not found")

    main_tips = find_main_tips(skeleton, center, min_dist=30)
    if len(main_tips) == 0:
        raise ValueError("No main tips found")

    radii_lines = [bresenham_line(center, tip) for tip in main_tips]

    branches = []
    for tip, line in zip(main_tips, radii_lines):
        branch = find_side_branches_along_radius(skeleton, line, center, tip)
        branches.append(branch)

    return {
        "center": center,
        "main_tips": main_tips,
        "radii": radii_lines,
        "branches": branches,
        "skeleton": skeleton
    }

def draw_branch_analysis(img: np.ndarray, contour: np.ndarray, analysis: Dict, settings: Dict = None) -> np.ndarray:
    settings = settings or {}
    out = img.copy()

    colors = {
        "contour": settings.get("contour_color", (255, 255, 255)),
        "center": settings.get("center_color", (0, 255, 0)),
        "main_tip": settings.get("main_tip_color", (255, 0, 0)),
        "radius": settings.get("radius_color", (100, 100, 100)),
        "branch_point": settings.get("branch_point_color", (0, 255, 255)),
        "side_left": settings.get("side_left_color", (255, 100, 100)),
        "side_right": settings.get("side_right_color", (100, 100, 255)),
    }

    # Контур
    cv2.polylines(out, [contour], True, colors["contour"], 1)

    # Центр
    cx, cy = analysis["center"]
    cv2.circle(out, (cx, cy), 6, colors["center"], -1)
    cv2.circle(out, (cx, cy), 8, (255,255,255), 1)

    # Главные кончики
    for tx, ty in analysis["main_tips"]:
        cv2.circle(out, (tx, ty), 5, colors["main_tip"], -1)

    # Радиусы
    for line in analysis["radii"]:
        for x, y in line:
            cv2.circle(out, (x, y), 1, colors["radius"], -1)

    # Ветвления
    for branch in analysis["branches"]:
        for bx, by in branch["branch_points"]:
            cv2.circle(out, (bx, by), 4, colors["branch_point"], -1)
        for sx, sy in branch["side_tips_left"]:
            cv2.circle(out, (sx, sy), 3, colors["side_left"], -1)
        for sx, sy in branch["side_tips_right"]:
            cv2.circle(out, (sx, sy), 3, colors["side_right"], -1)

    return out
#+-------------------------------------------------------------------------------------+
#FIND BRANCHES
#+-------------------------------------------------------------------------------------+
def find_side_branches(image, contour, sectors_data, step_ratio=100, method="distance_based"):
    """
    Находит боковые ветви для каждого сектора
    """
    img_out = image.copy()
    updated_sectors = []
    print_debug("start find branches")
    for sector in sectors_data:
        # Распаковываем данные сектора (все углы уже вычислены)
        center = sector["center"]
        radius = sector["radius"] 
        peak_point = sector["peak_point"]
        base_angle = sector["base_angle"]
        angle_plus_point = sector["angle_plus_point"]  # правая граница сектора
        angle_minus_point = sector["angle_minus_point"]  # левая граница сектора
        
        # Вычисляем шаг
        step = radius / step_ratio
        
        # Ищем боковые ветви для левой и правой стороны
        print_debug("start find branches for side")
        left_branches = find_branches_for_side(contour, center, peak_point, base_angle, 
                                              angle_minus_point, radius, step, side="left", method=method)
        right_branches = find_branches_for_side(contour, center, peak_point, base_angle,
                                               angle_plus_point, radius, step, side="right", method=method)
        
        # Обновляем сектор
        updated_sector = sector.copy()
        updated_sector["left_branches"] = left_branches
        updated_sector["right_branches"] = right_branches
        updated_sectors.append(updated_sector)
        
        # Рисуем найденные ветви
        print_debug("start fdraw branches")
        draw_branches(img_out, left_branches, color=(0, 0, 255))  # зеленый для левых
        draw_branches(img_out, right_branches, color=(0, 0, 255)) # синий для правых
    
    return img_out, updated_sectors

def draw_branches(image, branches, color):
    """
    Рисует линии от точки на радиусе к боковым ветвям и точки на концах
    """
    for branch in branches:
        base_point = branch["base_point"]  # точка на основном радиусе
        branch_point = branch["branch_point"]  # найденный побочный пик
        
        # Рисуем линию от точки на радиусе к побочному пику
        cv2.line(image, 
                (int(base_point[0]), int(base_point[1])), 
                (int(branch_point[0]), int(branch_point[1])), 
                color, 1)
        
        # Рисуем жирную точку на конце ветви
        cv2.circle(image, 
                  (int(branch_point[0]), int(branch_point[1])), 
                  3, color, -1)
        
def find_branches_for_side(contour, center, peak_point, base_angle, boundary_point, radius, step, side, method="distance_based"):
    """
    Находит боковые ветви для одной стороны сектора
    """
    all_candidates = []
    # Сохраняем sector_indices для всей стороны сектора
    first_sector_indices = None

    # Начинаем от пика (peak_point), идем к центру до 2/3 радиуса
    current_distance = radius * 0.9 # начинаем от пика (полный радиус)
    min_distance = radius * 0.3 # останавливаемся на 2/3 радиуса
    
    while current_distance > min_distance:
        # Точка на основном радиусе (движемся от пика к центру)
        point_on_radius = (
            center[0] + current_distance * math.cos(base_angle),
            center[1] + current_distance * math.sin(base_angle)
        )
        
        # Вектор радиуса в этой точке (направление от центра к точке)
        radius_direction = (math.cos(base_angle), math.sin(base_angle))
        
        # Поворачиваем вектор радиуса на ±60° для получения направления сечения
        # Угол поворота: +60° для правой стороны, -60° для левой
        rotation_angle = math.pi/3 if side == "right" else -math.pi/3
        
        # Новое направление сечения (повернутый вектор радиуса)
        section_direction = (
            radius_direction[0] * math.cos(rotation_angle) - radius_direction[1] * math.sin(rotation_angle),
            radius_direction[0] * math.sin(rotation_angle) + radius_direction[1] * math.cos(rotation_angle)
        )
        
        # Вычисляем максимальную длину сечения до границ
        print_debug("start calculate_section_length")
        max_section_length = calculate_section_length(center, point_on_radius, section_direction, boundary_point, radius, side)
        
        if max_section_length > 0:
            # Находим пересечения с контуром вдоль сечения
            print_debug("start find_contour_intersections")
            # Вычисляем угол boundary_point относительно центра
            dx_boundary = boundary_point[0] - center[0]
            dy_boundary = boundary_point[1] - center[1]
            boundary_angle = math.atan2(dy_boundary, dx_boundary)

            # Определяем границы сектора в зависимости от стороны
            if side == "right":
                # Для правой стороны: boundary_point это angle_plus_point (+30°)
                sector_start_angle = base_angle  # основной радиус
                sector_end_angle = boundary_angle  # +30° граница
            else:
                # Для левой стороны: boundary_point это angle_minus_point (-30°)
                sector_start_angle = boundary_angle  # -30° граница  
                sector_end_angle = base_angle  # основной радиус

            intersections, sector_indices = find_contour_intersections(
                contour=contour, 
                start_point=point_on_radius, 
                direction=section_direction, 
                max_length=max_section_length,
                center=center,
                sector_start_angle=sector_start_angle,
                sector_end_angle=sector_end_angle
            )

            if first_sector_indices is None:
                first_sector_indices = sector_indices
            
            if intersections:
                # Находим самую удаленную точку пересечения
                print_debug("start find_farthest_intersection")
                farthest_point = find_farthest_intersection(point_on_radius, intersections)
                
                # Собираем всех кандидатов без фильтрации
                all_candidates.append({
                    "base_point": point_on_radius,
                    "branch_point": farthest_point,
                    "distance_from_center": current_distance,
                    "branch_length": math.dist(point_on_radius, farthest_point)
                })
        
        current_distance -= step    
    
    filtered_branches = filter_branches_global(all_candidates, method=method, contour=contour, sector_indices=first_sector_indices)
    
    return filtered_branches

def calculate_section_length(center, start_point, section_direction, boundary_point, outer_radius, side):
    """
    Вычисляет длину сечения до ближайшей границы (окружности или границы сектора)
    
    Args:
        center: центр снежинки
        start_point: начальная точка на радиусе
        section_direction: направляющий вектор сечения (tuple dx, dy)
        boundary_point: точка на границе сектора (angle_plus_point или angle_minus_point)
        outer_radius: радиус внешней окружности
        side: 'left' или 'right'
    
    Returns:
        float: длина сечения до ближайшей границы
    """
    lengths = []
    
    # 1. Длина до внешней окружности
    length_to_circle = distance_to_circle(center, start_point, section_direction, outer_radius)
    if length_to_circle is not None:
        lengths.append(length_to_circle)
    
    # 2. Длина до границы сектора
    length_to_boundary = distance_to_boundary(center, start_point, section_direction, boundary_point, side)
    if length_to_boundary is not None:
        lengths.append(length_to_boundary)
    
    # Возвращаем минимальную длину (ближайшую границу)
    return min(lengths) if lengths else 0.0

def distance_to_circle(center, start_point, direction, radius):
    """
    Вычисляет расстояние от start_point вдоль direction до пересечения с окружностью
    """
    # Параметрическое уравнение луча: P(t) = start_point + t * direction
    # Уравнение окружности: (x - cx)^2 + (y - cy)^2 = r^2
    
    cx, cy = center
    sx, sy = start_point
    dx, dy = direction
    
    # Коэффициенты квадратного уравнения
    a = dx**2 + dy**2
    b = 2 * (dx * (sx - cx) + dy * (sy - cy))
    c = (sx - cx)**2 + (sy - cy)**2 - radius**2
    
    # Дискриминант
    discriminant = b**2 - 4 * a * c
    
    if discriminant < 0:
        return None  # Нет пересечения
    
    t1 = (-b + math.sqrt(discriminant)) / (2 * a)
    t2 = (-b - math.sqrt(discriminant)) / (2 * a)
    
    # Нас интересует положительное t (движение вперед по лучу)
    positive_ts = [t for t in [t1, t2] if t > 0]
    
    return min(positive_ts) if positive_ts else None

def distance_to_boundary(center, start_point, direction, boundary_point, side):
    """
    Вычисляет расстояние от start_point вдоль direction до пересечения с границей сектора
    """
    # Граница сектора - это луч из центра через boundary_point
    bx, by = boundary_point
    cx, cy = center
    
    # Направляющий вектор границы
    boundary_direction = (bx - cx, by - cy)
    boundary_length = math.sqrt(boundary_direction[0]**2 + boundary_direction[1]**2)
    boundary_direction = (boundary_direction[0] / boundary_length, boundary_direction[1] / boundary_length)
    
    # Решаем систему: start_point + t * direction = center + s * boundary_direction
    sx, sy = start_point
    dx, dy = direction
    bdx, bdy = boundary_direction
    
    # Матричное уравнение: [dx -bdx; dy -bdy] * [t; s] = [cx - sx; cy - sy]
    determinant = dx * (-bdy) - (-bdx) * dy
    
    if abs(determinant) < 1e-10:
        return None  # Лучи параллельны
    
    # Решаем систему методом Крамера
    det_t = (cx - sx) * (-bdy) - (-bdx) * (cy - sy)
    det_s = dx * (cy - sy) - (cx - sx) * dy
    
    t = det_t / determinant
    s = det_s / determinant
    
    # Нас интересует t > 0 (движение вперед) и s >= 0 (точка на луче от центра)
    if t > 0 and s >= 0:
        return t
    return None

def find_contour_intersections(contour, start_point, direction, max_length, center, sector_start_angle, sector_end_angle):
    """
    Находит пересечения контура с лучом, но только для отрезков в заданном секторе
    """
    intersections = []
    
    # Нормализуем направляющий вектор
    dx, dy = direction
    length_dir = math.sqrt(dx**2 + dy**2)
    if length_dir == 0:
        return intersections
        
    dx_norm, dy_norm = dx / length_dir, dy / length_dir
    
    # Конечная точка луча
    end_point = (
        start_point[0] + dx_norm * max_length,
        start_point[1] + dy_norm * max_length
    )
    
    # 1. Преобразуем контур в массив точек и вычисляем углы
    points = contour.reshape(-1, 2)  # [N, 2]
    
    # Векторы от центра к точкам контура
    vectors = points - np.array(center)
    
    # Вычисляем углы всех точек
    angles = np.arctan2(vectors[:, 1], vectors[:, 0])
    
    # 2. Находим точки в секторе
    angles_normalized = np.mod(angles, 2 * math.pi)
    sector_start_normalized = np.mod(sector_start_angle, 2 * math.pi)
    sector_end_normalized = np.mod(sector_end_angle, 2 * math.pi)
    
    # Маска точек в секторе
    if sector_start_normalized <= sector_end_normalized:
        in_sector_mask = (angles_normalized >= sector_start_normalized) & (angles_normalized <= sector_end_normalized)
    else:
        # Сектор пересекает 0°
        in_sector_mask = (angles_normalized >= sector_start_normalized) | (angles_normalized <= sector_end_normalized)
    
    # 3. Получаем индексы точек в секторе
    sector_indices = np.where(in_sector_mask)[0]
    
    if len(sector_indices) == 0:
        return intersections
    
    # 4. Собираем отрезки из соседних точек в секторе
    valid_segments = []
    for i in range(len(sector_indices)):
        idx1 = sector_indices[i]
        idx2 = (idx1 + 1) % len(points)  # следующая точка в контуре
        
        # Если следующая точка тоже в секторе, создаем отрезок
        if in_sector_mask[idx2]:
            pt1 = (float(points[idx1][0]), float(points[idx1][1]))
            pt2 = (float(points[idx2][0]), float(points[idx2][1]))
            valid_segments.append((pt1, pt2))
    
    # 5. Ищем пересечения только с валидными отрезками
    for pt1, pt2 in valid_segments:
        #print_debug("start line_segment_intersection")
        intersection = line_segment_intersection(start_point, end_point, pt1, pt2)
        if intersection:
            intersections.append(intersection)
    
    return intersections, sector_indices

def line_segment_intersection(p1, p2, p3, p4):
    """
    Находит пересечение двух отрезков (p1-p2) и (p3-p4)
    Возвращает точку пересечения или None
    """
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4
    
    # Вычисляем определители
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denom) < 1e-10:
        return None  # Отрезки параллельны
    
    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
    u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / denom
    
    # Проверяем, что пересечение в пределах обоих отрезков
    if 0 <= t <= 1 and 0 <= u <= 1:
        x = x1 + t * (x2 - x1)
        y = y1 + t * (y2 - y1)
        return (x, y)
    
    return None

def find_farthest_intersection(start_point, intersections):
    """
    Находит самую удаленную точку пересечения от start_point
    
    Args:
        start_point: начальная точка
        intersections: список точек пересечения
    
    Returns:
        tuple: самая удаленная точка (x, y)
    """
    if not intersections:
        return None
    
    farthest_point = None
    max_distance = 0
    
    for point in intersections:
        distance = math.dist(start_point, point)
        if distance > max_distance:
            max_distance = distance
            farthest_point = point
    
    return farthest_point

def filter_branches_global(candidates, method="distance_based", contour=None, sector_indices=None, **kwargs):
    """
    Глобальная фильтрация ветвей разными методами
    
    Args:
        candidates: список кандидатов [{"branch_point": (x,y), "base_point": (x,y), ...}]
        method: "distance_based", "simple_clustering", "peak_detection"
    
    Returns:
        list: отфильтрованные ветви
    """
    if method == "distance_based":
        return filter_by_distance(candidates, **kwargs)
    elif method == "simple_clustering":
        return filter_by_simple_clustering(candidates, **kwargs)
    elif method == "peak_detection":
        return filter_by_peak_detection(candidates, **kwargs)
    elif method == "dbscan":
        return filter_by_dbscan(candidates, **kwargs)
    elif method == "curvature":
        return filter_by_curvature(candidates, contour, sector_indices, **kwargs)
    elif method == "combined":
        return filter_combined_method(candidates, contour, sector_indices, **kwargs)
    elif method == "peak_detection_curvature":
        return filter_peack_curvature_method(candidates, contour, sector_indices, **kwargs)
    elif method == "peak_detection_scipy":
        return filter_by_scipy_peaks(candidates, contour, sector_indices, **kwargs)
    else:
        return candidates
    
def filter_by_scipy_peaks(candidates, distance=3, prominence=8, height=10):
    """
    distance=3: минимум 3 измерения между пиками (шаг по радиусу)
    prominence=8: пик должен выступать на 8px над окружающими значениями  
    height=10: минимальная длина ветви для пика
    """
    if not candidates:
        return []
    
    # Сортируем по расстоянию от центра
    candidates_sorted = sorted(candidates, key=lambda x: x["distance_from_center"])    
    lengths = [c["branch_length"] for c in candidates_sorted]
    
    # Ищем пики
    #peaks, properties = find_peaks(
    #    lengths,
    #    height=height,
    #    distance=distance
    #    prominence=prominence
    #)
    """
    widths = np.linspace(2, 25, 10)

    peaks = find_peaks_cwt(
        lengths,
        widths=widths,
        max_distances=np.maximum(2, (widths/3).astype(int)),
        gap_thresh=2,
        min_length=int(0.35 * len(widths)),
        min_snr=2.5,
        noise_perc=10,
        window_size=max(8, len(lengths)//25)
    )
    """
    peaks = find_peaks_cwt(lengths, np.arange(1,7)) #, min_snr=1, min_length=2
    peaks_filtered = [candidates_sorted[i] for i in peaks]
    # Затем кластеризуем оставшиеся
    final_filtered = filter_by_dbscan(
        peaks_filtered, 
        eps=15, 
        min_samples=1, 
        min_branch_length=10
    )
    return final_filtered

def filter_by_distance(candidates, min_branch_length=5, min_group_distance=20):
    """
    Простая фильтрация по расстоянию между ветвями
    """
    if not candidates:
        return []
    
    # Сортируем по расстоянию от центра (от дальних к ближним)
    candidates_sorted = sorted(candidates, key=lambda x: x["distance_from_center"], reverse=True)
    
    filtered = []
    
    for candidate in candidates_sorted:
        # Проверяем минимальную длину ветви
        if candidate["branch_length"] < min_branch_length:
            continue
            
        # Проверяем расстояние до уже отобранных ветвей
        too_close = False
        for existing in filtered:
            distance = math.dist(candidate["branch_point"], existing["branch_point"])
            if distance < min_group_distance:
                too_close = True
                # Если новая ветвь длиннее, заменяем
                if candidate["branch_length"] > existing["branch_length"]:
                    filtered.remove(existing)
                    filtered.append(candidate)
                break
        
        if not too_close:
            filtered.append(candidate)
    
    return filtered

def filter_by_simple_clustering(candidates, cluster_radius=15, min_branch_length=5):
    """
    Простая кластеризация по расстоянию между точками
    """
    if not candidates:
        return []
    
    # Фильтруем по минимальной длине
    candidates = [c for c in candidates if c["branch_length"] >= min_branch_length]
    
    if not candidates:
        return []
    
    clusters = []
    
    for candidate in candidates:
        found_cluster = False
        for cluster in clusters:
            # Проверяем расстояние до центроида кластера
            centroid = cluster["centroid"]
            distance = math.dist(candidate["branch_point"], centroid)
            if distance < cluster_radius:
                # Добавляем в кластер
                cluster["points"].append(candidate)
                # Обновляем центроид
                points = [c["branch_point"] for c in cluster["points"]]
                cluster["centroid"] = (
                    sum(p[0] for p in points) / len(points),
                    sum(p[1] for p in points) / len(points)
                )
                found_cluster = True
                break
        
        if not found_cluster:
            # Создаем новый кластер
            clusters.append({
                "points": [candidate],
                "centroid": candidate["branch_point"]
            })
    
    # В каждом кластере выбираем ветвь с максимальной длиной
    result = []
    for cluster in clusters:
        best_branch = max(cluster["points"], key=lambda x: x["branch_length"])
        result.append(best_branch)
    
    return result

def filter_by_peak_detection(candidates, min_branch_length=5, window_size=10):
    """
    Обнаружение пиков в последовательности ветвей вдоль радиуса
    """
    if not candidates:
        return []
    
    # Сортируем по расстоянию от центра
    candidates_sorted = sorted(candidates, key=lambda x: x["distance_from_center"])
    
    # Фильтруем по минимальной длине
    candidates_filtered = [c for c in candidates_sorted if c["branch_length"] >= min_branch_length]
    
    if len(candidates_filtered) < window_size:
        return candidates_filtered
    
    peaks = []
    
    for i in range(len(candidates_filtered)):
        # Проверяем окно вокруг текущей точки
        start = max(0, i - window_size)
        end = min(len(candidates_filtered), i + window_size + 1)
        window = candidates_filtered[start:end]
        
        # Текущая точка - локальный максимум по длине ветви
        current_length = candidates_filtered[i]["branch_length"]
        if current_length == max(c["branch_length"] for c in window):
            peaks.append(candidates_filtered[i])
    
    return peaks

def filter_by_dbscan(candidates, eps=15, min_samples=1, min_branch_length=5):
    """
    Кластеризация DBSCAN для группировки близких ветвей
    """
    from sklearn.cluster import DBSCAN
    
    if not candidates:
        return []
    
    # Фильтруем по минимальной длине
    candidates = [c for c in candidates if c["branch_length"] >= min_branch_length]
    
    if len(candidates) < 2:
        return candidates
    
    # Подготавливаем данные для кластеризации
    points = np.array([c["branch_point"] for c in candidates])
    
    # Кластеризация DBSCAN
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(points)
    labels = clustering.labels_
    
    # Группируем по кластерам
    clusters = {}
    for i, label in enumerate(labels):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(candidates[i])
    
    # В каждом кластере выбираем ветвь с максимальной длиной
    result = []
    for label, cluster_candidates in clusters.items():
        if label == -1:  # Выбросы - оставляем все
            result.extend(cluster_candidates)
        else:
            best_branch = max(cluster_candidates, key=lambda x: x["branch_length"])
            result.append(best_branch)
    
    return result

def filter_by_curvature(candidates, contour, sector_indices, min_curvature=0.1, min_branch_length=5):
    """
    Фильтрация по кривизне контура в точках ветвей (только для сектора)
    """
    if not candidates:
        return []
    
    candidates = [c for c in candidates if c["branch_length"] >= min_branch_length]
    if not candidates:
        return []
    
    # Вычисляем кривизну только для точек в секторе
    curvature_map = compute_contour_curvature(contour, sector_indices)
    
    filtered = []
    for candidate in candidates:
        branch_point = candidate["branch_point"]
        closest_idx = find_closest_contour_point_in_sector(contour, branch_point, sector_indices)
        curvature = curvature_map.get(closest_idx, 0)
        
        if curvature > min_curvature:
            candidate["curvature"] = curvature
            filtered.append(candidate)
    
    return filtered

def filter_combined_method(candidates, contour, sector_indices=None, min_branch_length=5, curvature_threshold=0.05, distance_threshold=15):
    """
    Комбинированный метод: DBSCAN + кривизна
    """
    # Сначала фильтруем по кривизне
    curvature_filtered = filter_by_curvature(
        candidates, 
        contour, 
        sector_indices=sector_indices,
        min_curvature=curvature_threshold, 
        min_branch_length=min_branch_length
    )
    
    # Затем кластеризуем оставшиеся
    final_filtered = filter_by_dbscan(
        curvature_filtered, 
        eps=distance_threshold, 
        min_samples=1, 
        min_branch_length=0
    )
    
    return final_filtered

def filter_peack_curvature_method(candidates, contour, sector_indices=None, min_branch_length=5, curvature_threshold=0.1, window_size=10):
    """
    Комбинированный метод: DBSCAN + кривизна
    """

    # Затем кластеризуем оставшиеся
    final_filtered = filter_by_peak_detection(candidates, min_branch_length=5, window_size=window_size)

    # Сначала фильтруем по кривизне
    curvature_filtered = filter_by_curvature(
        final_filtered, 
        contour, 
        sector_indices=sector_indices,
        min_curvature=curvature_threshold, 
        min_branch_length=min_branch_length
    )

    return curvature_filtered

# Вспомогательные функции для кривизны
def compute_contour_curvature(contour, sector_indices, window=10):
    """
    Вычисляет кривизну только для точек в секторе
    """
    points = contour.reshape(-1, 2)
    curvature = {}
    
    for i in sector_indices:
        # Берем окно точек вокруг текущей (только из сектора)
        window_indices = []
        for j in range(-window, window + 1):
            idx = (i + j) % len(points)
            if idx in sector_indices:  # Только точки из сектора
                window_indices.append(idx)
        
        if len(window_indices) >= 3:
            window_points = points[window_indices]
            dx = np.gradient(window_points[:, 0])
            dy = np.gradient(window_points[:, 1])
            d2x = np.gradient(dx)
            d2y = np.gradient(dy)
            
            curv = np.abs(dx * d2y - dy * d2x) / (dx**2 + dy**2)**1.5
            curvature[i] = np.mean(curv)
    
    return curvature

def find_closest_contour_point_in_sector(contour, point, sector_indices):
    """
    Находит ближайшую точку контура в секторе
    """
    points = contour.reshape(-1, 2)
    sector_points = points[sector_indices]
    distances = np.linalg.norm(sector_points - np.array(point), axis=1)
    closest_sector_idx = np.argmin(distances)
    return sector_indices[closest_sector_idx]
