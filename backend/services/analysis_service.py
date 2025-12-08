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
from typing import List, Dict, Any, Tuple, Optional
from core.config import RESULTS_DIR, GRAPH_DIR
from utils.debug import print_debug
import traceback
import math
from scipy.signal import find_peaks, find_peaks_cwt
from scipy.optimize import curve_fit
from matplotlib.patches import Circle
from matplotlib.table import Table
import base64
from io import BytesIO


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
            
            radius_on_lenght = main_perimeter / (2 * np.pi)
            radius_on_area = np.sqrt(main_area / np.pi)

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

            log_perimetr = normalized_perimeter ** 2 #np.log(total_perimeter)
            log_area = normalized_area #np.log(total_area)

            fractal_dimension = round(log_area / (log_perimetr), 2)

            analysis_data.append({
                "session_folder": session_dir_path.name,
                "normalized_perimeter": normalized_perimeter,
                "normalized_area": normalized_area,
                "main_enclosing_radius": main_enclosing_radius,
                "log_perimetr": log_perimetr,
                "log_area": log_area,
                "fractal_dimension": fractal_dimension,
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
    viewport_size: dict,
    chart_type: str
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
        filename = f"{chart_type}_chart_{timestamp}.jpg"
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
    "draw_pca_axes": False,
    "draw_min_enclosing_circle": True,
    "draw_custom_enclosing_circle": False,
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
        mc = _safe_int_tuple(analysis.get("circles", {}).get("opencv", {}).get("center", (0,0)))
        radius = int(analysis.get("circles", {}).get("opencv", {}).get("radius", 0))
        
        # Преобразуем контур в список точек для анализа
        contour_points = contour.reshape(-1, 2)  # [N, 2]
        
        # Вычисляем углы всех точек контура относительно центра
        contour_angles = []
        for point in contour_points:
            dx, dy = point[0] - mc[0], point[1] - mc[1]
            angle = math.atan2(dy, dx)
            contour_angles.append(angle)

        # Находим индекс точки пика в контуре
        def _find_contour_point_index(contour_points, target_point):
            """Находит индекс точки в контуре, ближайшей к target_point"""
            min_dist = float('inf')
            min_index = -1
            for i, point in enumerate(contour_points):
                dist = math.sqrt((point[0]-target_point[0])**2 + (point[1]-target_point[1])**2)
                if dist < min_dist:
                    min_dist = dist
                    min_index = i
            return min_index

        for ic, p in enumerate(peaks):
            px, py = int(round(p["point"][0])), int(round(p["point"][1]))
            
            # Рисуем основную точку пика
            cv2.circle(img, (px, py), 5, DEFAULT_COLORS["peaks"], -1)
            
            if settings["draw_labels"]:
                cv2.putText(img, f"peak#{p['index']}", (px+6, py-6), cv2.FONT_HERSHEY_SIMPLEX, 0.4, DEFAULT_COLORS["peaks"], 1, cv2.LINE_AA)
            
            if settings["draw_radial_lines"]:

                def _normalize_angle(a: float) -> float:
                    """Нормализует угол в диапазон [-pi, pi]."""
                    return (a + math.pi) % (2 * math.pi) - math.pi


                def _find_boundary_by_walking(contour_points, start_index, center, base_angle, side, max_angle_deg=40):
                    
                    max_angle_rad = math.radians(max_angle_deg)
                    
                    # clockwise контур → i+1 — вправо, i-1 — влево
                    step = -1 if side == "right" else 1
                    
                    # Сохраняем все локальные максимумы
                    local_maxima = []
                    current_max_angle = 0
                    current_min_angle = np.pi
                    current_max_index = start_index
                    current_max_point = []
                    
                    
                    # Начинаем от пика и идём по контуру
                    n = len(contour_points)
                    i = start_index
                    iterations = 0
                    max_iterations = n // 2  # Чтобы не ходить по кругу
                    point = contour_points[i]
                    dx, dy = point[0] - center[0], point[1] - center[1]
                    current_max_radius = math.sqrt(dx*dx + dy*dy)

                    while iterations < max_iterations:
                        point = contour_points[i]
                        dx, dy = point[0] - center[0], point[1] - center[1]
                        angle = math.atan2(dy, dx)
                        
                        # Вычисляем отклонение от базового угла
                        angle_diff = _normalize_angle(angle - base_angle)
                        #angle_diff = _normalize_angle(angle_diff)  # нормализуем к [-π, π]
                        
                        # Абсолютный лимит: если вышли за max_angle_deg - стоп
                        if abs(angle_diff) > max_angle_rad:
                            break

                        dx, dy = point[0] - center[0], point[1] - center[1]
                        current_radius = math.sqrt(dx*dx + dy*dy)

                        # Обновляем текущий максимум
                        if abs(angle_diff) > abs(current_max_angle):
                            current_max_angle = angle_diff
                            current_max_index = i
                            current_max_point = point
                            dx, dy = point[0] - center[0], point[1] - center[1]
                            current_max_radius = math.sqrt(dx*dx + dy*dy)

                        # Если текущая точка - локальный максимум
                        if abs(current_max_angle) > abs(angle_diff) + math.radians(2) and current_max_radius >  current_radius:
                            local_maxima.append((current_max_angle, current_max_index, current_max_point))  
                            current_max_angle = 0                           
                            #cv2.circle(img, (current_max_point[0], current_max_point[1]), 5, DEFAULT_COLORS["peaks"], -1)

                        # Переходим к следующей точке
                        i = (i + step) % n
                        iterations += 1
                        
                        # Если вернулись к стартовой точке - выходим
                        if i == start_index:
                            break
                    #cv2.circle(img, (current_max_point[0], current_max_point[1]), 5, DEFAULT_COLORS["peaks"], -1)
                    # Теперь выбираем граничный угол, максимально близко к твоей исходной логике.
                    # local_maxima содержит кортежи (angle_diff, index, point), где angle_diff — подписанная разность
                    if len(local_maxima) >= 2:
                        candidates = local_maxima[0:len(local_maxima) - 1]

                        # Фильтрация по знаку, чтобы предпочесть пики в направлении side
                        if side == 'right':
                            signed_candidates = [c for c in candidates if c[0] > 0]
                        else:
                            signed_candidates = [c for c in candidates if c[0] < 0]

                        if signed_candidates:
                            boundary_angle_diff = max(signed_candidates, key=lambda x: abs(x[0]))[0]
                        else:
                            # fallback: берем максимальный по модулю среди кандидатов
                            boundary_angle_diff = max(candidates, key=lambda x: abs(x[0]))[0]

                        # Возвращаем прямо base_angle + signed_diff (без нормализации/abs)
                        return base_angle + boundary_angle_diff + (math.radians(0.5) if boundary_angle_diff > 0 else -math.radians(0.5))

                    elif local_maxima:
                        # Если только один максимум - берём его
                        boundary_angle_diff = local_maxima[0][0]
                        return base_angle + boundary_angle_diff

                    else:
                        # Если максимумов нет - берём текущий накопленный максимум/значение
                        if current_max_angle is None:
                            return None
                        return base_angle + boundary_angle_diff
                    
                # Рисуем линию от масс центра к пику
                cv2.line(img, mc, (px, py), DEFAULT_COLORS["radial_lines"], 1)
                
                # Вычисляем угол радиуса
                dx, dy = px - mc[0], py - mc[1]
                current_angle = math.atan2(dy, dx)
                
                # Углы для поворота ±π/6
                angle_offset = math.pi / 6
                angle_plus = current_angle + angle_offset
                angle_minus = current_angle - angle_offset

                # Находим индекс точки пика в контуре
                #"""
                peak_index = _find_contour_point_index(contour_points, (px, py))
                #print_debug(f"peak_index: {peak_index}, current_angle: {current_angle}")
                if peak_index != -1:
                    # Ищем границы обходом контура
                    angle_minus_m = _find_boundary_by_walking(
                        contour_points, peak_index, mc, current_angle, "left", max_angle_deg=40                        
                    )
                    
                    angle_plus_m = _find_boundary_by_walking(
                        contour_points, peak_index, mc, current_angle, "right", max_angle_deg=40
                    )
                #"""     
                #print_debug(f"angle_minus: {angle_minus}, angle_plus: {angle_plus}") 
                #print_debug(f"angle_minus_m: {angle_minus_m}, angle_plus_m: {angle_plus_m}")                 
                # Вычисляем координаты для повернутых линий
                mode = 'advanced' #'standart' 'advanced'
                if mode == 'standart':
                    px_plus = int(mc[0] + radius * math.cos(angle_plus))
                    py_plus = int(mc[1] + radius * math.sin(angle_plus))                
                    px_minus = int(mc[0] + radius * math.cos(angle_minus))
                    py_minus = int(mc[1] + radius * math.sin(angle_minus))
                elif mode == 'advanced':
                    px_plus = int(mc[0] + radius * math.cos(angle_plus_m))
                    py_plus = int(mc[1] + radius * math.sin(angle_plus_m))                
                    px_minus = int(mc[0] + radius * math.cos(angle_minus_m))
                    py_minus = int(mc[1] + radius * math.sin(angle_minus_m))

                #print_debug(f"px_plus: {px_plus}, py_plus: {py_plus}, px_minus: {px_minus}, py_minus: {py_minus}") 
                #print_debug(f"px_plus_m: {px_plus_m}, py_plus_m: {py_plus_m}, px_minus_m: {px_minus_m}, py_minus_m: {py_minus_m}")

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
                colors = {0: (255, 0, 0), 1: (0, 255, 0), 2: (0, 0, 255), 3: (255, 255, 0), 4: (255, 0, 255), 5: (0, 255, 255)}
                # Рисуем обе пунктирные линии красным цветом
                draw_dashed_line(img, mc, (px_plus, py_plus), colors[ic], dash_length)
                draw_dashed_line(img, mc, (px_minus, py_minus), colors[ic], dash_length)
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
    #print_debug("start find branches")
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
        #print_debug("start find branches for side")
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
        #print_debug("start fdraw branches")
        draw_branches(img_out, left_branches, color=(0, 0, 255))  # зеленый для левых
        draw_branches(img_out, right_branches, color=(0, 0, 255)) # синий для правых
    return img_out, updated_sectors

def get_envelops(image, contour, sectors_data):
    img_out = image.copy()    
    #print_debug("start find branches")
    # Списки для сбора данных графиков
    all_contours = []      # Список контуров в безразмерных координатах
    all_curves = []        # Список кривых в безразмерных координатах
    all_params = []        # Список параметров
    all_left_branches = [] # Левые веточки
    all_right_branches = [] # Правые веточки
    for updated_sector in sectors_data:
        x_combined, z_combined, x_all, z_all, x_bp, z_bp, growth_axis_unit, perp_axis = prepare_combined_fitting_data(
                                                                                contour=contour,
                                                                                snowflake_center=updated_sector["center"],
                                                                                dendrite_tip=updated_sector["peak_point"],
                                                                                left_branches=updated_sector["left_branches"],
                                                                                right_branches=updated_sector["right_branches"],
                                                                                angle_minus_point=updated_sector["angle_minus_point"],
                                                                                angle_plus_point=updated_sector["angle_plus_point"]
                                                                            )
        #print_debug(f"x_combined: {x_combined} z_combined: {z_combined} growth_axis_unit: {growth_axis_unit} perp_axis: {perp_axis}")
        
        # Совместный фит
        #params = fit_combined_model(x_combined, z_combined)
        params = fit_alpha_rho(x_combined, z_combined, cutoff_ratio=0.95, radius=updated_sector["radius"]*0.05)

        print_debug(f"Результат: α = {params['alpha']:.3f}, ρ = {params['rho']:.2f} пикселей")
        
        gen_params = fit_gen_model(x_bp, z_bp, params['alpha'], params['rho'], initial_guess=[1.0, 1.5], bounds=([0.1, 0.5], [10.0, 5.0]))

        print_debug(f"Результат: lambda = {gen_params['lambda']:.3f}, n = {gen_params['n']:.2f}")

        # Визуализация - теперь все параметры есть!
        #fit_points = visualize_fit(
        #    x_all, z_all, params, 
        #    updated_sector["peak_point"], growth_axis_unit, perp_axis
        #)
        # Получаем точки для отрисовки и безразмерные координаты
        fit_points, contour_dimless, curve_dimless, left_branches_dimless, right_branches_dimless = visualize_fit_all(
            x_all, z_all,  
            gen_params['lambda'], gen_params['n'], 
            params['alpha'], params['rho'], 
            updated_sector["peak_point"], 
            growth_axis_unit, perp_axis,
            left_branches=updated_sector["left_branches"],
            right_branches=updated_sector["right_branches"],
            num_points=100
        )
        
        # Сохраняем данные для графиков
        all_contours.append(contour_dimless)
        all_curves.append(curve_dimless)
        all_params.append({
            'alpha': params['alpha'],
            'rho': params['rho'],
            'lambda': gen_params['lambda'],
            'n': gen_params['n']
        })
        all_left_branches.append(left_branches_dimless)
        all_right_branches.append(right_branches_dimless)

        # Рисуем на изображении
        cv2.polylines(img_out, [fit_points], isClosed=False, color=(0, 0, 0), thickness=1)

    # Создаем сводный график если есть данные
    if all_contours:
        # Создаем фигуру
        fig = create_envelops_summary_plot(
            contours=all_contours,
            curves=all_curves,
            params=all_params,
            left_branches_list=all_left_branches,
            right_branches_list=all_right_branches,
            num_sectors=6,
            isCircle=False
        )
        
        # Конвертируем в base64
        # Вариант 1: Прямо в base64
        # base64_str = matplotlib_figure_to_base64(fig, dpi=150)
        
        # Вариант 2: Через numpy (как у тебя)
        image_rgb = matplotlib_figure_to_image_rgb(fig, dpi=150)
        #base64_str = image_to_base64_png(image_rgb)
        
        return image_rgb
    else:
        # Если нет данных для графиков
        return img_out

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

def prepare_combined_fitting_data(contour, snowflake_center, dendrite_tip, left_branches, right_branches, 
                                 angle_minus_point, angle_plus_point):
    """
    Подготавливает объединенные данные используя угловые границы сектора
    """
    def get_branches_points(branches_point_left, branches_point_right):
        out = []
        branches_point_left + branches_point_right
        for bp in branches_point_left:
            point = bp["branch_point"]
            vector = np.array(point) - np.array(dendrite_tip)
            z = np.dot(vector, growth_axis_unit)
            x = np.dot(vector, perp_axis)
            out.append((x, z))
        for bp in branches_point_right:
            point = bp["branch_point"]
            vector = np.array(point) - np.array(dendrite_tip)
            z = np.dot(vector, growth_axis_unit)
            x = np.dot(vector, perp_axis)
            out.append((x, z))    
        return out    

    # 1. Определяем систему координат
    growth_axis = np.array(dendrite_tip) - np.array(snowflake_center)
    growth_axis_unit = growth_axis / np.linalg.norm(growth_axis)
    perp_axis = np.array([-growth_axis_unit[1], growth_axis_unit[0]])
    
    # 2. Вычисляем угловые границы сектора
    vector_minus = np.array(angle_minus_point) - np.array(snowflake_center)
    vector_plus = np.array(angle_plus_point) - np.array(snowflake_center)
    
    angle_minus = np.arctan2(vector_minus[1], vector_minus[0])
    angle_plus = np.arctan2(vector_plus[1], vector_plus[0])
    
    # 3. Находим точки контура в этом угловом диапазоне
    contour_points = contour.reshape(-1, 2)
    vectors_to_contour = contour_points - np.array(snowflake_center)
    angles_to_contour = np.arctan2(vectors_to_contour[:, 1], vectors_to_contour[:, 0])
    
    # Маска точек в секторе
    if angle_minus <= angle_plus:
        in_sector_mask = (angles_to_contour >= angle_minus) & (angles_to_contour <= angle_plus)
    else:
        in_sector_mask = (angles_to_contour >= angle_minus) | (angles_to_contour <= angle_plus)
    
    sector_contour_points = contour_points[in_sector_mask]
    
    # 4. Находим границу по ветвям
    z_boundary_left = find_boundary_from_branches(left_branches, dendrite_tip, growth_axis_unit, sector_contour_points)
    z_boundary_right = find_boundary_from_branches(right_branches, dendrite_tip, growth_axis_unit, sector_contour_points)
    z_boundary = min(z_boundary_left, z_boundary_right)
    
    # 5. Преобразуем только точки из сектора
    all_points = []
    first_branches_point = []
    for point in sector_contour_points:
        vector = np.array(point) - np.array(dendrite_tip)
        z = np.dot(vector, growth_axis_unit)
        x = np.dot(vector, perp_axis)
        all_points.append((x, z))
        if -z_boundary <= z <= 0:
            first_branches_point.append((x, z))
    
    if not first_branches_point:
        raise ValueError("Не найдено точек для фиттинга")
    
    x_bp, z_bp = zip(*first_branches_point)
    x_all, z_all = zip(*all_points)
    fit_bp = get_branches_points(left_branches, right_branches)
    x_fit_bp, z_fit_bp = zip(*fit_bp)
    return np.array(x_bp), np.array(z_bp), np.array(x_all), np.array(z_all), np.array(x_fit_bp), np.array(z_fit_bp), growth_axis_unit, perp_axis

def fit_alpha_rho(x, z, cutoff_ratio=0.85, radius=None):
    """
    Фит α и ρ через curve_fit: сначала ρ по окружности, потом α по всей модели
    """
    # 1. Функция отбора точек кончика для фита окружности
    def get_tip_points(x, z, cutoff_ratio):
        z_min = np.min(z)
        cutoff_z = z_min * cutoff_ratio
        
        shallow_mask = z > cutoff_z
        x_shallow = x[shallow_mask]
        z_shallow = z[shallow_mask]
        
        tip_mask = np.abs(x_shallow) > np.abs(z_shallow) 
        dop_tip_mask = tip_mask & (np.abs(x_shallow) < np.mean(np.abs(x_shallow)) + np.std(np.abs(x_shallow)))
        return x_shallow[dop_tip_mask], z_shallow[dop_tip_mask]
    
    # 2. Фит ρ по окружности
    x_tip, z_tip = get_tip_points(x, z, cutoff_ratio)
    print_debug(f"x_tip: {x_tip}, z_tip: {z_tip}")
    if radius is None:
        if len(x_tip) < 2:
            return None
        
        def circle_model(x, rho):
            return -rho + np.sqrt(rho**2 - x**2 + 1e-10)
        
        z_abs = np.abs(z_tip)
        lower_bound = np.max(np.abs(x_tip)) * 1.001
        rho_guess = np.median((x_tip**2 + z_tip**2) / (2 * z_abs + 1e-10))
        if rho_guess < lower_bound:
            rho_guess = lower_bound
        
        print_debug(f"rho_guess: {rho_guess}, lower_bound: {lower_bound}")
        try:
            rho_opt, _ = curve_fit(
                circle_model,
                x_tip,
                z_tip,
                p0=[rho_guess],
                bounds=([lower_bound], [1000.0])
            )
            rho = rho_opt[0]
        except Exception as e:
            print_debug(f"error: {e}")
            rho = None
    else:
        rho = radius        
    print_debug(f"rho: {rho}")

    # 3. Фит α через curve_fit на всей модели с фиксированным ρ
    def full_model(x, alpha):
        x_norm = 2 * x / rho
        z_ag = z_AG_2d(x_norm)
        return alpha * z_ag
    
    # Начальное приближение для α
    alpha_guess = 1.0
    
    # Фит α
    try:
        alpha_opt, _ = curve_fit(
            full_model,
            x_tip,
            2 * z_tip / rho,
            p0=[alpha_guess],
            bounds=([0.1], [10.0])
        )
        alpha = alpha_opt[0]
    except:
        # Если не удалось, используем аналитическое решение
        #x_norm = 2 * x / rho
        #z_AG_vals = z_AG_2d(x_norm)
        #z_model_scaled = z_AG_vals * rho / 2
        #alpha = np.sum(z * z_model_scaled) / np.sum(z_model_scaled**2)
        return None
    print_debug(f"alpha: {alpha}")
    return {"alpha": alpha, "rho": rho}

def fit_combined_model(x, z, max_iterations=3):
    """
    Совместный фит модели с параметрами alpha и rho
    """
    def model(x, alpha, rho):
        """Полная модель: z = alpha * z_AG(2*x/rho) * rho/2"""
        x_norm = 2 * x / rho
        z_ag = z_AG_2d(x_norm)
        return alpha * z_ag * rho / 2
    
    # Начальные guess - проверяем и корректируем
    alpha_guess = 1.0
    rho_guess = estimate_initial_rho(x, z)
    
    # Убедимся, что rho_guess в разумных пределах
    rho_guess = max(1.0, min(rho_guess, 200.0))
    
    print(f"Начальное приближение: ρ = {rho_guess:.2f}")
    
    # Границы
    lower_bounds = [0.1, max(0.5, rho_guess * 0.5)]
    upper_bounds = [10.0, min(1000.0, rho_guess * 10)]
    
    # Убедимся, что начальное приближение внутри границ
    if not (lower_bounds[0] <= alpha_guess <= upper_bounds[0] and 
            lower_bounds[1] <= rho_guess <= upper_bounds[1]):
        rho_guess = (lower_bounds[1] + upper_bounds[1]) / 2
    
    print(f"Границы: alpha [{lower_bounds[0]}, {upper_bounds[0]}], rho [{lower_bounds[1]:.1f}, {upper_bounds[1]:.1f}]")
    
    # Итеративный фит с очисткой выбросов
    x_current, z_current = x.copy(), z.copy()
    best_params = None
    best_error = float('inf')
    
    for iteration in range(max_iterations):
        print(f"Итерация {iteration + 1}, точек: {len(x_current)}")
        
        try:
            # Фиттинг с двумя параметрами
            params_opt, pcov = curve_fit(
                model, x_current, z_current, 
                p0=[alpha_guess, rho_guess],
                bounds=(lower_bounds, upper_bounds)
            )
            
            alpha_opt, rho_opt = params_opt
            
            # Вычисляем ошибку
            z_pred = model(x_current, alpha_opt, rho_opt)
            current_error = np.mean((z_current - z_pred)**2)
            
            print(f"  α = {alpha_opt:.3f}, ρ = {rho_opt:.2f}, ошибка = {current_error:.6f}")
            
            # Сохраняем лучшие параметры
            if current_error < best_error:
                best_error = current_error
                best_params = {
                    'alpha': alpha_opt,
                    'rho': rho_opt,
                    'error': current_error
                }
            
            # Очистка выбросов для следующей итерации
            if iteration < max_iterations - 1:
                residuals = np.abs(z_current - z_pred)
                threshold = 2 * np.std(residuals)
                keep_mask = residuals < threshold
                
                x_current = x_current[keep_mask]
                z_current = z_current[keep_mask]
                
                print(f"  Удалено выбросов: {np.sum(~keep_mask)}")
                
        except Exception as e:
            print(f"Ошибка на итерации {iteration}: {e}")
            # Если фит не удался - ПРОСТО ПРОПУСКАЕМ ИТЕРАЦИЮ
            continue
    
    if best_params is None:
        print("Фит не удался на всех итерациях!")
        return None
    
    return best_params

def fit_gen_model(x_data, z_data, alpha, rho, initial_guess=[1.0, 1.5], bounds=([0.1, 0.5], [10.0, 5.0])):
    """
    Фиттинг обобщенной модели с параметрами lambda и n
    """
    def model_gen(x, lambda_param, n):
        """Полная GEN модель: z = z_GEN(2*x/rho)"""        
        z_gen_norm = z_GEN_2d(x, lambda_param, n, alpha)
        return z_gen_norm
    
    x_norm = 2 * x_data / rho
    z_norm = 2 * z_data / rho

    try:
        params_opt, pcov = curve_fit(
                            model_gen,  # передаём функцию напрямую
                            x_norm, z_norm,
                            p0=initial_guess,
                            bounds=bounds
                        )
        
        lambda_opt, n_opt = params_opt
        return {'lambda': lambda_opt, 'n': n_opt}
        
    except Exception as e:
        print(f"Ошибка фиттинга GEN модели: {e}")
        return None

def estimate_initial_rho(x, z):
    """Простая оценка начального ρ"""
    # Используем точки вблизи кончика (первые 20% по |z|)
    z_threshold = 0.2 * np.max(np.abs(z))
    near_mask = np.abs(z) < z_threshold
    
    if np.sum(near_mask) > 5:
        try:
            # z ≈ -x²/(2ρ) => ρ ≈ -x²/(2z)
            valid_mask = near_mask & (z != 0)
            if np.sum(valid_mask) > 3:
                rho_estimates = -x[valid_mask]**2 / (2 * z[valid_mask])
                return np.median(rho_estimates[rho_estimates > 0])
        except:
            pass
    
    return 50  # консервативное значение по умолчанию

def visualize_fit(x_data, z_data, params, dendrite_tip, growth_axis_unit, perp_axis, num_points=100):
    """Визуализация фита из готовых координат"""
    # Ограничиваем диапазон x разумными пределами модели
    # z_AG_2d имеет экспоненциальное убывание, но для визуализации ограничим
    x_max = min(np.max(np.abs(x_data)) * 1.5, params['rho'] * 3)  # не больше 3ρ
    x_range = np.linspace(-x_max, x_max, num_points)
    
    # Предсказание модели
    z_pred = params['alpha'] * z_AG_2d(2 * x_range / params['rho']) * params['rho'] / 2
    
    # Дополнительно: обрезаем точки где z_pred слишком большой (далеко от кончика)
    max_z_threshold = np.max(np.abs(z_data)) * 2
    valid_mask = np.abs(z_pred) < max_z_threshold
    x_range = x_range[valid_mask]
    z_pred = z_pred[valid_mask]
    
    # Преобразование обратно в координаты изображения
    points_image = []
    for x, z in zip(x_range, z_pred):
        vector = z * growth_axis_unit + x * perp_axis
        point_image = np.array(dendrite_tip) + vector
        points_image.append(point_image.astype(int))
    
    return np.array(points_image)

def visualize_fit_all(x_data, z_data, lambda_param, n, alpha, rho, dendrite_tip, growth_axis_unit, perp_axis, 
                      left_branches=None, right_branches=None, num_points=100):
    """
    Возвращает:
    1. Точки в пикселях для отрисовки кривой
    2. Безразмерные координаты предсказанной кривой
    3. Безразмерные координаты исходных данных контура
    4. Безразмерные координаты веточек (левый и правый)
    """
    x_min = np.min(x_data)
    x_max = np.max(x_data)
    x_range = np.linspace(x_min, x_max, num_points)

    # Предсказанная кривая в безразмерных координатах
    X_curve_dimless = 2 * x_range / rho
    Z_curve_pred = z_GEN_2d(X_curve_dimless, lambda_param, n, alpha, m=3)
    curve_dimless = np.column_stack((X_curve_dimless, Z_curve_pred))
    
    # Исходные данные контура в безразмерных координатах
    X_contour_dimless = 2 * x_data / rho
    Z_contour_dimless = 2 * z_data / rho
    contour_dimless = np.column_stack((X_contour_dimless, Z_contour_dimless))
    
    # Преобразование веточек в безразмерные координаты
    left_branches_dimless = []
    right_branches_dimless = []
    
    if left_branches is not None:
        for branch in left_branches:
            # Преобразуем точки веточки в безразмерные
            base_relative = np.array(branch["base_point"]) - np.array(dendrite_tip)
            branch_relative = np.array(branch["branch_point"]) - np.array(dendrite_tip)
            
            # Проекции на оси
            base_x = np.dot(base_relative, perp_axis)
            base_z = np.dot(base_relative, growth_axis_unit)
            branch_x = np.dot(branch_relative, perp_axis)
            branch_z = np.dot(branch_relative, growth_axis_unit)
            
            # Безразмерные координаты
            base_dimless = np.array([2 * base_x / rho, 2 * base_z / rho])
            branch_dimless = np.array([2 * branch_x / rho, 2 * branch_z / rho])
            
            left_branches_dimless.append({
                "base_point": base_dimless,
                "branch_point": branch_dimless
            })
    
    if right_branches is not None:
        for branch in right_branches:
            # Преобразуем точки веточки в безразмерные
            base_relative = np.array(branch["base_point"]) - np.array(dendrite_tip)
            branch_relative = np.array(branch["branch_point"]) - np.array(dendrite_tip)
            
            # Проекции на оси
            base_x = np.dot(base_relative, perp_axis)
            base_z = np.dot(base_relative, growth_axis_unit)
            branch_x = np.dot(branch_relative, perp_axis)
            branch_z = np.dot(branch_relative, growth_axis_unit)
            
            # Безразмерные координаты
            base_dimless = np.array([2 * base_x / rho, 2 * base_z / rho])
            branch_dimless = np.array([2 * branch_x / rho, 2 * branch_z / rho])
            
            right_branches_dimless.append({
                "base_point": base_dimless,
                "branch_point": branch_dimless
            })
    
    # Преобразование обратно в координаты изображения (для отрисовки кривой)
    points_image = []
    for x, z in zip(x_range, Z_curve_pred * rho / 2):
        vector = z * growth_axis_unit + x * perp_axis
        point_image = np.array(dendrite_tip) + vector
        points_image.append(point_image.astype(int))
    
    return (
        np.array(points_image), 
        curve_dimless, 
        contour_dimless,
        left_branches_dimless,
        right_branches_dimless
    )

def z_AG_2d(x, k=3):
    """Теоретическая форма кончика для 2D случая"""
    x_abs = np.abs(x)
    b_S = np.exp(-1 / (x_abs**(2*k)))
    b_L = np.exp(-(x_abs**(2*k)))
    
    numerator = - (b_S * x**2 + b_L * x_abs**(5/3))
    denominator = (b_S * x_abs**(1/3) + b_L * x_abs**(-1/3))
    
    z_ag_value = numerator / denominator
    z_ag_value = np.where(x_abs < 1e-10, 0, z_ag_value)
    return z_ag_value

def z_GEN_2d(x, lambda_param, n, alpha, m=3):
    """Обобщенная функция формы для 2D случая в безразмерных координатах"""
    x_abs = np.abs(x)
    
    # Stitching functions
    a_S = np.exp(-1 / (x_abs**(2*m)))
    a_L = np.exp(-(x_abs**(2*m)))
    
    # z_TIP функция (уже в безразмерных координатах)
    z_tip = alpha * z_AG_2d(x)  # x уже нормализован как 2*x/rho
    
    # Вычисляем z_GEN
    numerator = a_L * x_abs**n - a_S * z_tip
    denominator = lambda_param * a_S * x_abs**(-n) * z_tip + a_L * x_abs**n / z_tip
    
    z_gen = numerator / denominator
    return z_gen

def find_boundary_from_branches(branches_data, dendrite_tip, growth_axis_unit, contour):
    """Находит границу по первой боковой ветви"""
    min_distance = float('inf')
    
    for branch in branches_data:
        base_point = branch["branch_point"]
        vector = np.array(base_point) - np.array(dendrite_tip)
        z_distance = np.dot(vector, growth_axis_unit)
        
        if 0 < z_distance < min_distance:
            min_distance = z_distance
    
    # Если ветвей нет, используем разумную долю от длины главной ветви
    if min_distance == float('inf'):
        # Оцениваем длину главной ветви по самой дальней точке контура вдоль оси роста
        max_z = 0
        for point in contour:
            vector = np.array(point) - np.array(dendrite_tip)
            z = np.dot(vector, growth_axis_unit)
            if z < max_z:  # z отрицательные, так что ищем минимальное
                max_z = z
        min_distance = abs(max_z) * 0.3  # используем 30% от длины ветви
    
    return min_distance
        
def find_branches_for_side(contour, center, peak_point, base_angle, boundary_point, radius, step, side, method="distance_based"):
    """
    Находит боковые ветви для одной стороны сектора
    """
    all_candidates = []
    # Сохраняем sector_indices для всей стороны сектора
    first_sector_indices = None

    # Начинаем от пика (peak_point), идем к центру до 2/3 радиуса
    current_distance = radius # начинаем от пика (полный радиус)
    min_distance = radius * 0 # останавливаемся на 2/3 радиуса
    
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
        #print_debug("start calculate_section_length")
        max_section_length = calculate_section_length(center, point_on_radius, section_direction, boundary_point, radius, side)
        
        if max_section_length > 0:
            # Находим пересечения с контуром вдоль сечения
            #print_debug("start find_contour_intersections")
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
                #print_debug("start find_farthest_intersection")
                farthest_point = find_farthest_intersection(point_on_radius, intersections)
                
                # Собираем всех кандидатов без фильтрации
                all_candidates.append({
                    "base_point": point_on_radius,
                    "branch_point": farthest_point,
                    "distance_from_center": current_distance,
                    "branch_length": math.dist(point_on_radius, farthest_point)
                })
        
        current_distance -= step    
    
    filtered_branches = filter_branches_global(all_candidates, method=method, 
                                               contour=contour, sector_indices=first_sector_indices,
                                               check_symmetry=False,
                                               symmetry_tolerance=0.4,
                                               offset=2                                               
                                               )
    
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

def filter_branches_global(candidates, method="distance_based",
                           contour=None, sector_indices=None,
                           check_symmetry=False,
                           symmetry_tolerance=0.4,
                           offset=2,                          
                           **kwargs):
    """
    Глобальная фильтрация ветвей + геометрическая проверка настоящих кончиков.
    """

    filtered = candidates  # стартовое значение

    # ----- 1. Основные методы отбора -----

    if method == "distance_based":
        filtered = filter_by_distance(candidates, **kwargs)

    elif method == "simple_clustering":
        filtered = filter_by_simple_clustering(candidates, **kwargs)

    elif method == "peak_detection":
        filtered = filter_by_peak_detection(
            candidates,
            min_branch_length=5,
            window_size=3,
            **kwargs
        )

    elif method == "dbscan":
        filtered = filter_by_dbscan(
            candidates,
            eps=5,
            min_samples=1,
            min_branch_length=5,
            **kwargs
        )

    elif method == "curvature":
        filtered = compute_local_profile_curvature(candidates, **kwargs)

    elif method == "combined":
        filtered = filter_combined_method(candidates, contour, sector_indices, **kwargs)

    elif method == "peak_detection_curvature":
        filtered = filter_peack_curvature_method(candidates, contour, sector_indices, **kwargs)

    elif method == "peak_detection_scipy":
        filtered = filter_by_scipy_peaks(candidates, contour, sector_indices, **kwargs)
        filtered = filter_by_dbscan(
            filtered,
            eps=15,
            min_samples=1,
            min_branch_length=5,
            **kwargs
        )
    else:
        filtered = candidates


    # ----- 2. Дополнительный геометрический фильтр всех кандидатов -----

    filtered = filter_real_tips(
        filtered,
        contour,
        sector_indices,
        offset=offset,        
        check_symmetry=check_symmetry,
        symmetry_tol=symmetry_tolerance        
    )

    return filtered

def filter_real_tips(
        candidates,
        contour,
        sector_indices,
        offset=12,
        check_symmetry=False,
        symmetry_tol=0.4
    ):

    if not candidates:
        return []

    if sector_indices is None or len(sector_indices) < 2:
        return candidates

    # --------------------------  
    # 1) Строим сегменты КОНТУРА строго внутри сектора  
    # --------------------------
    valid_segments = [] 
    n = len(contour) 
    for i in range(len(sector_indices)): 
        idx1 = sector_indices[i] 
        idx2 = (idx1 + 1) % n 
        # следующий индекс должен также быть в секторе 
        if idx2 not in sector_indices: 
            continue 
        p1 = contour[idx1] 
        p2 = contour[idx2] 
        valid_segments.append((p1.astype(float), p2.astype(float)))

    if not valid_segments:
        return candidates

    # --------------------------
    # Пересечение отрезков
    # --------------------------
    def seg_intersection(p, q, a, b):
        px, py = p
        qx, qy = q
        ax, ay = a
        bx, by = b

        s1_x, s1_y = qx - px, qy - py
        s2_x, s2_y = bx - ax, by - ay

        denom = (-s2_x * s1_y + s1_x * s2_y)
        if abs(denom) < 1e-9:
            return None

        s = (-s1_y * (px - ax) + s1_x * (py - ay)) / denom
        t = ( s2_x * (py - ay) - s2_y * (px - ax)) / denom

        if 0 <= s <= 1 and 0 <= t <= 1:
            return (px + t * s1_x, py + t * s1_y)
        return None

    # --------------------------
    # Ищем ближайшее пересечение на луче
    # --------------------------
    def nearest_outward_intersection(mid_point, direction, base, bp):
        mp = np.array(mid_point, float)
        dir_norm = np.array(direction, float)

        best_t = None
        best_inter = None
        best_seg = None

        # ищем первое пересечение вдоль луча
        for a, b in valid_segments:
            inter = seg_intersection(mp, mp + dir_norm * 1e6, a, b)
            
            if inter is None:
                continue

            vec = np.array(inter) - mp
            t = np.dot(vec, dir_norm)

            if t <= 0:
                continue  # позади точки — не подходит

            # выбираем пересечение с минимальным t (первое по лучу)
            if best_t is None or t < best_t:
                best_t = t
                best_inter = inter
                best_seg = (a, b)
                

        if best_inter is None:
            return None  # пересечений нет напрямую

        # -----------------------------
        # проверяем направление прохода: inside→outside или outside→inside
        # -----------------------------
        a, b = best_seg
        seg = b - a
        
        # нормаль наружу (если контур CCW)
        n = np.array([seg[1], -seg[0]])
        """
        # знак dot(direction, n) показывает тип перехода
        if np.dot(dir_norm, n) < 0:
            # inside → outside (правильное)
            #print_debug(f"mid: {mid_point}; best_seg: {best_seg}; base {base}; bp: {bp}")
            return best_inter
        else:
            # outside → inside — сразу отбрасываем
            return None
        """
        if np.dot(dir_norm, n) < 0:
            # inside → outside (правильное)

            # ДОБАВЛЯЕМ ЭТО:
            axis = np.array(bp) - np.array(base)
            v = np.array(best_inter) - np.array(base)

            # 1) Пересечение должно быть в сторону bp
            if np.dot(axis, v) <= 0:
                return None

            # 2) Проекция должна быть близка к истинному кончику
            proj = np.dot(axis, v) / np.linalg.norm(axis)

            if proj < 0.8 * np.linalg.norm(axis):
                return None

            return best_inter
        else:
            return None


    # --------------------------
    # Основной цикл
    # --------------------------
    real_tips = []

    for cand in candidates:
        bp = np.array(cand["branch_point"], float)
        base = np.array(cand["base_point"], float)

        v = base - bp
        norm = np.linalg.norm(v)
        if norm < 1e-6:
            continue

        v_unit = v / norm

        # точка для построения перпендикуляров
        mid_point = bp + v_unit * offset

        # перп
        perp = np.array([-v_unit[1], v_unit[0]])

        # два неограниченных луча
        inter_left  = nearest_outward_intersection(mid_point,  perp, base, bp)
        inter_right = nearest_outward_intersection(mid_point, -perp, base, bp)

        # нужны оба пересечения
        if inter_left is None or inter_right is None:
            continue

        # симметрия (опционально)
        if check_symmetry:
            d1 = np.linalg.norm(np.array(inter_left)  - bp)
            d2 = np.linalg.norm(np.array(inter_right) - bp)
            d1, d2 = sorted([d1, d2])
            if d1 == 0:
                continue
            ratio = d2 / d1
            if ratio > (1 + symmetry_tol):
                continue

        real_tips.append(cand)

    return real_tips

    
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
    #lengths = [c["branch_length"] for c in candidates_sorted]
    #candidates_sorted = compute_local_profile_curvature(candidates)
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
    """
    # Затем кластеризуем оставшиеся
    final_filtered = filter_by_dbscan(
        peaks_filtered, 
        eps=15, 
        min_samples=1, 
        min_branch_length=10
    )
    """
    return peaks_filtered

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

def filter_by_curvature(candidates, contour, sector_indices, min_curvature=0.2, min_branch_length=5):
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

def compute_local_profile_curvature(candidates):
    """
    Вычисляет "кривизну" профиля отклонения от радиуса: |d''(s)|

    Args:
        candidates: список кандидатов (для присвоения результата)
        radial_distances: массив s_i — расстояния от центра до точек Q_i на радиусе (в пикселях)
        deviations: массив d_i — расстояния от Q_i до контура по секущей (в пикселях)

    Returns:
        list: candidates с добавленным "local_curvature"
    """
    filtered = []
    candidates_sorted = sorted(candidates, key=lambda x: x["distance_from_center"])    
    radial_distances = [c["branch_length"] for c in candidates_sorted]
    deviations = [c["distance_from_center"] for c in candidates_sorted]

    if len(radial_distances) < 3 or len(deviations) != len(radial_distances):
        # Недостаточно точек для второй производной        
        return filtered

    # Первая производная: dd/ds
    dd_ds = np.gradient(deviations, radial_distances)
    # Вторая производная: d²d/ds²
    d2d_ds2 = np.gradient(dd_ds, radial_distances)
    # "Кривизна" — абсолютное значение второй производной
    curvature_profile = np.abs(d2d_ds2)

    # Теперь нужно сопоставить точки профиля с кандидатами
    # Предположим, что каждый кандидат имеет "distance_from_center",
    # и он совпадает с одним из значений в radial_distances    

    for c, cf in zip(candidates_sorted, curvature_profile):
        if cf > 0.1:
            filtered.append(c)
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

def create_envelops_summary_plot(
    contours: List[np.ndarray],
    curves: List[np.ndarray],
    params: List[Dict],
    left_branches_list: List[List[Dict]],
    right_branches_list: List[List[Dict]],
    num_sectors: int = 6,
    isCircle=False
) -> plt.Figure:
    """
    Создает сводный график для конвертов.
    """
    fig, axes = plt.subplots(
        2, 4, 
        figsize=(20, 11),
        gridspec_kw={
            'width_ratios': [1, 1, 1, 0.9],
            'height_ratios': [1, 1],
            'wspace': 0.25,
            'hspace': 0.3
        }
    )
    
    fig.patch.set_facecolor('white')
    
    # Графики в ячейках
    for i in range(num_sectors):
        row = i // 3
        col = i % 3
        
        ax = axes[row, col]
        
        if i < len(contours):
            draw_single_envelop_plot(
                ax=ax,
                contour=contours[i],
                curve=curves[i],
                params=params[i],
                sector_idx=i+1,
                left_branches=left_branches_list[i] if i < len(left_branches_list) else None,
                right_branches=right_branches_list[i] if i < len(right_branches_list) else None,
                isCircle=isCircle
            )
        else:
            draw_empty_plot(ax=ax, sector_idx=i+1)
    
    # Таблица
    fig.delaxes(axes[1, 3])
    draw_parameters_table(axes[0, 3], params, num_sectors, fig)
    
    fig.suptitle("Envelope Analysis - All Sectors", fontsize=18, y=0.98, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 0.95, 0.96])
    
    return fig

def draw_single_envelop_plot(
    ax: plt.Axes,
    contour: np.ndarray,
    curve: np.ndarray,
    params: Dict,
    sector_idx: int,
    left_branches: Optional[List[Dict]] = None,
    right_branches: Optional[List[Dict]] = None,
    isCircle=False
) -> None:
    """
    Рисует один график конверта.
    """
    # Извлекаем параметры
    alpha = params.get('alpha', 0)
    rho = params.get('rho', 0)
    lambda_val = params.get('lambda', 0)
    n_val = params.get('n', 0)
    
    # Определяем границы для осей ИЗ ДАННЫХ ЭТОГО ГРАФИКА
    x_contour = contour[:, 0]
    z_contour = contour[:, 1]
    
    # Вычисляем диапазоны ТОЛЬКО из данных этого графика
    x_min = np.min(x_contour)
    x_max = np.max(x_contour)
    z_min = np.min(z_contour)
    z_max = np.max(z_contour)
    
    # Добавляем 15% запаса от диапазона ЭТОГО графика
    x_range = x_max - x_min
    z_range = z_max - z_min
    
    # Не делаем padding если диапазон нулевой (например, одна точка)
    if x_range > 0:
        x_padding = x_range * 0.15
    else:
        x_padding = 0.1  # минимальный padding
    
    if z_range > 0:
        z_padding = z_range * 0.15
    else:
        z_padding = 0.1
    
    x_min -= x_padding
    x_max += x_padding
    z_min -= z_padding
    z_max += z_padding
    
    # Рисуем контур БЕЗ СОРТИРОВКИ!
    ax.plot(
        x_contour, z_contour,
        '-', linewidth=1.0, alpha=0.8, color='gray',
        label='Contour' if sector_idx == 1 else None,
        zorder=1
    )
    
    # Рисуем кривую аппроксимации
    if curve is not None and len(curve) > 0:
        ax.plot(
            curve[:, 0], curve[:, 1],
            'r-', linewidth=2.0,
            label='GEN fit' if sector_idx == 1 else None,
            zorder=4
        )
    if isCircle:
        # Рисуем единичную окружность (ρ=1) - центр смещен вниз на 1
        circle_center = (0, -1)
        circle = Circle(
            circle_center, 1,
            fill=False, 
            linestyle='-',
            linewidth=1.0,
            edgecolor='black',
            alpha=0.8,
            label='ρ=1' if sector_idx == 1 else None,
            zorder=2
        )
        ax.add_patch(circle)
    
    # Рисуем веточки
    all_branches = []
    if left_branches:
        all_branches.extend(left_branches)
    if right_branches:
        all_branches.extend(right_branches)
    
    for i, branch in enumerate(all_branches):
        base_point = branch.get("base_point")
        branch_point = branch.get("branch_point")
        
        if base_point is not None and branch_point is not None:
            ax.plot(
                [base_point[0], branch_point[0]],
                [base_point[1], branch_point[1]],
                '--', linewidth=1.0, color='#666666', alpha=0.6,
                label='Sidebranches' if sector_idx == 1 and i == 0 else None,
                zorder=2
            )
            
            ax.scatter(
                branch_point[0], branch_point[1],
                s=30, color='#666666', alpha=0.8,
                marker='o', edgecolors='white', linewidth=0.5,
                zorder=3,
                label='Branch tips' if sector_idx == 1 and i == 0 else None
            )
    
    # Устанавливаем границы для ЭТОГО графика
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(z_min, z_max)
    
    # Сетка
    ax.grid(True, alpha=0.6, linestyle='-', linewidth=0.2)
    
    # Подписи осей
    ax.set_xlabel('X (dimensionless)', fontsize=9)
    ax.set_ylabel('Z (dimensionless)', fontsize=9)
    
    # Заголовок
    title_text = f'Sector {sector_idx}\n'
    title_text += f'α={alpha:.3f}, ρ={rho:.1f}\n'
    title_text += f'λ={lambda_val:.3f}, n={n_val:.2f}'
    ax.set_title(title_text, fontsize=10, pad=10, fontweight='semibold')
    
    # Легенда
    if sector_idx == 1:
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(
            by_label.values(), 
            by_label.keys(), 
            loc='upper right', 
            fontsize=7, 
            framealpha=0.9,
            frameon=True
        )
    
    # Убираем aspect='equal' - пусть график растягивается
    # ax.set_aspect('equal', adjustable='datalim')  # КОММЕНТИРУЕМ!
    
    # Линии осей
    ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5, alpha=0.3, zorder=0)
    ax.axvline(x=0, color='k', linestyle='-', linewidth=0.5, alpha=0.3, zorder=0)
    
    # Кончик
    ax.scatter(
        0, 0, 
        s=30, color='black', marker='o', 
        edgecolors='white', linewidth=0.8,
        zorder=5,
        label='Tip' if sector_idx == 1 else None
    )


def draw_empty_plot(ax: plt.Axes, sector_idx: int) -> None:
    """
    Рисует пустой график с ТАКИМИ ЖЕ ГРАНИЦАМИ.
    """
    # Устанавливаем стандартные границы для пустых графиков
    x_min, x_max = -1.5, 1.5
    z_min, z_max = -1.5, 1.5  # Симметричные границы
    
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(z_min, z_max)
    
    # Сетка
    ax.grid(True, alpha=0.15, linestyle='-', linewidth=0.2)
    
    # Линии осей
    ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5, alpha=0.3, zorder=0)
    ax.axvline(x=0, color='k', linestyle='-', linewidth=0.5, alpha=0.3, zorder=0)
    
    # Сообщение
    ax.text(
        0, 0, 'NO DATA',
        ha='center', va='center',
        fontsize=12, fontweight='bold',
        color='gray', alpha=0.4
    )
    
    # Заголовок
    ax.set_title(f'Sector {sector_idx}', fontsize=10, pad=10, color='gray')
    
    # Подписи осей
    ax.set_xlabel('X (dimensionless)', fontsize=9, color='gray')
    ax.set_ylabel('Z (dimensionless)', fontsize=9, color='gray')
    
    # Равный масштаб
    ax.set_aspect('equal', adjustable='datalim')

def draw_parameters_table(
    ax: plt.Axes,
    params: List[Dict],
    num_sectors: int,
    fig: plt.Figure
) -> None:
    """
    Рисует таблицу параметров.
    """
    # Скрываем оси
    ax.axis('off')
    
    # Создаем данные для таблицы
    table_data = []
    headers = ['Sector', 'α', 'ρ', 'λ', 'n']
    table_data.append(headers)
    
    for i in range(num_sectors):
        if i < len(params):
            p = params[i]
            row = [
                f'{i+1}',
                f'{p.get("alpha", 0):.3f}',
                f'{p.get("rho", 0):.1f}',
                f'{p.get("lambda", 0):.3f}',
                f'{p.get("n", 0):.2f}'
            ]
        else:
            row = [f'{i+1}', '-', '-', '-', '-']
        table_data.append(row)
    
    # Создаем таблицу с увеличенными колонками
    table = ax.table(
        cellText=table_data,
        cellLoc='center',
        loc='center',
        colWidths=[0.12, 0.22, 0.22, 0.22, 0.22]  # Увеличил ширину колонок
    )
    
    # Настройка стиля таблицы
    table.auto_set_font_size(False)
    table.set_fontsize(10)  # Увеличил шрифт
    table.scale(1.3, 2.0)  # Увеличил масштаб
    
    # Стиль заголовка
    for j in range(len(headers)):
        table[(0, j)].set_facecolor('#4a7cb8')
        table[(0, j)].set_text_props(weight='bold', color='white', fontsize=11)  # Увеличил
        table[(0, j)].set_edgecolor('white')
        table[(0, j)].set_height(0.12)  # Увеличил высоту заголовка
    
    # Стиль ячеек данных
    for i in range(1, len(table_data)):
        for j in range(len(headers)):
            cell = table[(i, j)]
            cell.set_edgecolor('#dddddd')
            cell.set_height(0.1)  # Увеличил высоту строк
            
            # Чередование цветов строк
            if i % 2 == 0:
                cell.set_facecolor('#f8f8f8')
            else:
                cell.set_facecolor('#ffffff')
    
    # Заголовок таблицы
    ax.set_title('Parameters Summary', fontsize=12, pad=15, fontweight='bold')

def matplotlib_figure_to_base64(fig: plt.Figure, dpi: int = 150) -> str:
    """
    Конвертирует matplotlib figure в base64 PNG.
    
    Returns:
        base64 строка (без префикса 'data:image/png;base64,')
    """
    buf = BytesIO()
    fig.savefig(
        buf, 
        format='png', 
        dpi=dpi, 
        bbox_inches='tight',
        facecolor='white',
        edgecolor='none'
    )
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()
    
    # Закрываем фигуру чтобы освободить память
    plt.close(fig)
    
    return img_str

def matplotlib_figure_to_image_rgb(fig: plt.Figure, dpi: int = 150) -> np.ndarray:
    """
    Конвертирует matplotlib figure в numpy array (RGB).
    
    Returns:
        numpy array в формате RGB (H, W, 3)
    """
    buf = BytesIO()
    fig.savefig(
        buf, 
        format='png', 
        dpi=dpi, 
        bbox_inches='tight',
        facecolor='white',
        edgecolor='none'
    )
    buf.seek(0)
    
    # Читаем PNG через OpenCV
    file_bytes = np.asarray(bytearray(buf.read()), dtype=np.uint8)
    img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    buf.close()
    plt.close(fig)
    
    return img_rgb


