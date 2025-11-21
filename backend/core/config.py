# backend/core/config.py
"""
Центральный конфигурационный модуль.
Содержит:
- Пути к данным и моделям
- Устройство (GPU/CPU)
- Настройки по умолчанию для:
    • Препроцессинга изображения
    • Автоматической генерации масок (SAM2)
- Валидацию конфигов
"""

from pathlib import Path
import torch

# ==================================================================
# 1. БАЗОВЫЕ ПУТИ
# ==================================================================
BASE_DIR = Path(__file__).parent.parent

# Папки для результатов
RESULTS_DIR = BASE_DIR / "results"          # ← сессии, маски, final.jpg
GRAPH_DIR = BASE_DIR / "graph"              # ← графики анализа
STAGE2_DIR = BASE_DIR / "results_stage2"    # ← отобранные снежинки

# Папки создаются при старте
RESULTS_DIR.mkdir(exist_ok=True)
GRAPH_DIR.mkdir(exist_ok=True)
STAGE2_DIR.mkdir(exist_ok=True)

# ==================================================================
# 2. МОДЕЛЬ SAM2
# ==================================================================
# Путь к весам
CHECKPOINT_PATH = "models/sam2.1_hiera_large.pt"

# Конфиг модели (внутри библиотеки sam2, не копируем)
MODEL_CFG = "configs/sam2.1/sam2.1_hiera_l.yaml"

# ==================================================================
# 3. УСТРОЙСТВО (GPU / CPU)
# ==================================================================
device = "cuda" if torch.cuda.is_available() else "cpu"
autocast_dtype = torch.bfloat16 if device == "cuda" else None
autocast_device_type = device if device == "cuda" else "cpu"

# ==================================================================
# 4. ПРЕДВАРИТЕЛЬНАЯ ОБРАБОТКА ИЗОБРАЖЕНИЯ
# Используется в: /init, /update_settings
# ==================================================================
DEFAULT_PREPROCESS_CFG = {
    "median_ksize": 5,              # Размер медианного фильтра
    "contrast_factor": 1.5,         # Усиление контраста (PIL)
    "sharpness_factor": 2.0,        # Усиление резкости (PIL)
    "clahe_clip_limit": 1.5,        # CLAHE clip limit
    "clahe_tile_grid": (8, 8),      # Размер сетки CLAHE
}

ALLOWED_PREPROCESS_KEYS = set(DEFAULT_PREPROCESS_CFG.keys())

# ==================================================================
# 5. АВТОМАТИЧЕСКАЯ ГЕНЕРАЦИЯ МАСОК (SAM2)
# Используется в: /init_autogen, /update_autogen
# ==================================================================
DEFAULT_AUTOGEN_CFG = {
    "points_per_side": 16,                    # Плотность точек на сторону
    "points_per_batch": 32,                   # Пакетная обработка
    "pred_iou_thresh": 0.7,                   # Порог IoU
    "stability_score_thresh": 0.9,            # Порог стабильности
    "stability_score_offset": 0.7,            # Смещение стабильности
    "crop_n_layers": 1,                       # Слои кропа
    "box_nms_thresh": 0.7,                    # NMS для боксов
    "crop_n_points_downscale_factor": 2,      # Даунскейл точек в кропе
    "min_mask_region_area": 50,               # Мин. площадь маски
    "use_m2m": False,                         # Mask-to-mask предсказание
}

ALLOWED_AUTOGEN_KEYS = set(DEFAULT_AUTOGEN_CFG.keys())

# ==================================================================
# 6. ВАЛИДАЦИЯ КОНФИГОВ (защита от фронта)
# ==================================================================
def validate_preprocess_cfg(cfg: dict) -> dict:
    """Возвращает валидный конфиг с дефолтами"""
    return {
        k: cfg.get(k, v)
        for k, v in DEFAULT_PREPROCESS_CFG.items()
        if k in ALLOWED_PREPROCESS_KEYS
    }

def validate_autogen_cfg(cfg: dict) -> dict:
    """Возвращает валидный конфиг автогенерации"""
    return {
        k: cfg.get(k, v)
        for k, v in DEFAULT_AUTOGEN_CFG.items()
        if k in ALLOWED_AUTOGEN_KEYS
    }

# ==================================================================
# 7. ЛОГИРОВАНИЕ ПРИ СТАРТЕ
# ==================================================================
print(f"[CONFIG] Device: {device}")
print(f"[CONFIG] Autocast: dtype={autocast_dtype}, device_type={autocast_device_type}")
print(f"[CONFIG] Results: {RESULTS_DIR}")
print(f"[CONFIG] Stage 2: {STAGE2_DIR}")
print(f"[CONFIG] Graph: {GRAPH_DIR}")
print(f"[CONFIG] Model: {CHECKPOINT_PATH}")
