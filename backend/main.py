# C:\snowflakes\backend\main.py
from fastapi import FastAPI, File, Form, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.responses import StreamingResponse
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor 
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
import numpy as np
import torch
import cv2
import base64
import os
import json
import time
import uuid
from io import BytesIO
from PIL import Image, ImageEnhance
from pathlib import Path
from skimage import measure
import traceback # <-- Для отладки
import shutil

app = FastAPI(title="Snowflakes SAM iterative API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

CHECKPOINT_PATH = "models/sam2.1_hiera_large.pt" # Пример для SAM 2.1 large
MODEL_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml" # Соответствующий конфиг

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[DEBUG] Using device: {device}") # <-- Отладка

# Загрузка модели SAM 2 Image Predictor
print("[DEBUG] Starting to load SAM2ImagePredictor...") # <-- Отладка
sam2 = build_sam2(MODEL_CONFIG, CHECKPOINT_PATH, device=device, apply_postprocessing=False)
predictor = SAM2ImagePredictor(sam2)
print("[DEBUG] SAM2ImagePredictor loaded successfully.") # <-- Отладка

mask_generator = SAM2AutomaticMaskGenerator(sam2)
print("[DEBUG] SAM2AutomaticMaskGenerator loaded successfully.") # <-- Отладка

# Установка dtype для autocast, если используется CUDA
autocast_dtype = torch.bfloat16 if device == "cuda" else None
autocast_device = device if device == "cuda" else "cpu"
print(f"[DEBUG] Autocast dtype: {autocast_dtype}, device: {autocast_device}") # <-- Отладка

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

# In-memory sessions store (simple)
sessions: dict = {}  # session_id -> { "image": np.ndarray, "logits": np.ndarray|None, "orig_name": str, "points": list, "is_first_click": bool }

def ensure_snowflake_dir(sess):
    """Создаёт папку снежинки при первом сохранении."""
    if "snowflake_dir" in sess:
        return Path(sess["snowflake_dir"])

    base_name = sess.get("orig_name", f"snowflake_{uuid.uuid4().hex[:8]}")
    dir_path = RESULTS_DIR / base_name
    dir_path.mkdir(parents=True, exist_ok=True)

    sess["snowflake_dir"] = str(dir_path)
    return dir_path


def save_mask_pack(dir_path: Path, prefix: str, image_np, contour, mask_bin, logits):
    """Сохраняет единый комплект файлов маски."""
    # base image (RGB)
    cv2.imwrite(str(dir_path / f"{prefix}.jpg"),
                cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))

    # contour preview
    overlay = image_np.copy()
    if contour:
        pts = np.array(contour, dtype=np.int32)
        cv2.polylines(overlay, [pts], True, (255,0,0), 2)
    cv2.imwrite(str(dir_path / f"{prefix}_contour.jpg"),
                cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    # mask npy
    np.save(str(dir_path / f"{prefix}_mask.npy"), mask_bin)
    np.save(str(dir_path / f"{prefix}_logits.npy"), logits)

# -------------------------
# Helper: preprocessing (based on user's function, toned)
# -------------------------
def preprocess_image_cv(image: np.ndarray, config: dict | None = None) -> np.ndarray:
    """
    image: BGR (OpenCV) or RGB? we'll expect RGB numpy from PIL; convert accordingly.
    We'll operate assuming image is RGB (H,W,3) as np.uint8.
    Return processed RGB uint8 array.
    """
    if config is None:
        config = {}
    # median blur
    ksize = config.get("median_ksize", 5)
    if ksize % 2 == 0:
        ksize = ksize + 1
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image_smoothed = cv2.medianBlur(image_bgr, ksize=ksize)

    # contrast via PIL
    contrast_factor = config.get("contrast_factor", 1.5)
    pil_img = Image.fromarray(cv2.cvtColor(image_smoothed, cv2.COLOR_BGR2RGB))
    enhancer = ImageEnhance.Contrast(pil_img)
    pil_img = enhancer.enhance(contrast_factor)
    image_contrast = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    # sharpness via PIL (less aggressive)
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

# -------------------------
# Utilities
# -------------------------
def image_to_base64_png(image_rgb: np.ndarray) -> str:
    """Return base64 string (no  prefix) of PNG image from RGB numpy."""
    _, png = cv2.imencode(".png", cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
    return base64.b64encode(png.tobytes()).decode("utf-8")


def get_unique_filename(base_name: str, ext: str) -> str:
    candidate = RESULTS_DIR / f"{base_name}{ext}"
    counter = 1
    while candidate.exists():
        candidate = RESULTS_DIR / f"{base_name}_{counter}{ext}"
        counter += 1
    return str(candidate)


# -------------------------
# API: initialize session (upload image -> preprocess -> set_image)
# -------------------------
@app.post("/init")
async def init_session(
    file: UploadFile = File(...),
    config: str | None = Form(None),
):
    """
    Initialize a segmentation session with uploaded image.
    Applies default preprocessing if no config provided.
    """
    print(f"[DEBUG] /init called for file: {file.filename}")
    try:
        raw = await file.read()
        pil = Image.open(BytesIO(raw)).convert("RGB")
        image_np = np.array(pil)  # RGB

        # === 1. Определяем конфиг ===
        default_cfg = {
            "median_ksize": 5,
            "contrast_factor": 1.5,
            "sharpness_factor": 2.0,
            "clahe_clip_limit": 1.5,
            "clahe_tile_grid": (8, 8),
        }

        # если фронт прислал кастомный — парсим
        if config:
            try:
                cfg = json.loads(config)
                print(f"[DEBUG] Custom preprocessing config received: {cfg}")
            except Exception as e:
                print(f"[WARN] Failed to parse config JSON: {e}")
                cfg = default_cfg.copy()
        else:
            cfg = default_cfg.copy()

        # === 2. Применяем препроцесс ===
        processed = preprocess_image_cv(image_np, cfg)
        print("[DEBUG] Image preprocessed")

        # === 3. Подготавливаем SAM ===
        with torch.inference_mode(), torch.autocast(autocast_device, dtype=autocast_dtype):
            predictor.set_image(processed)

        # === 4. Создаём сессию ===
        session_id = str(uuid.uuid4())
        sessions[session_id] = {
            "image": processed,
            "original_processed_image": image_np, # <-- НОВОЕ: Сохраняем изначально обработанное изображение
            "logits": None,
            "orig_name": Path(file.filename).stem if file.filename else f"image_{int(time.time())}",
            "points": [],
            "is_first_click": True,
            "config": cfg,  # <--- сохраняем текущие настройки            
        }

        # === 5. Возвращаем предпросмотр и реальные параметры ===
        preview_b64 = image_to_base64_png(processed)
        return {
            "session_id": session_id,
            "preview_b64": preview_b64,
            "used_config": cfg,  # <--- фронт покажет эти значения на слайдерах
        }

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error during initialization: {str(e)}")

# --- НОВЫЙ ЭНДПОИНТ /init_autogen ---
@app.post("/init_autogen")
async def init_autogen_session(
    file: UploadFile = File(...),
    config: str | None = Form(None), # config как JSON строка
):
    """
    Initialize a segmentation session for AUTOMATIC generation.
    Applies preprocessing, runs SAM2AutomaticMaskGenerator.generate(),
    stores results, and returns session_id, preview, and auto_masks.
    """
    print(f"[DEBUG] /init_autogen called for file: {file.filename}")
    try:
        raw = await file.read()
        pil = Image.open(BytesIO(raw)).convert("RGB")
        image_np = np.array(pil)  # RGB

        # === 1. Определяем конфиг (используем дефолтные или кастомные) ===
        default_cfg = {
            "median_ksize": 5,
            "contrast_factor": 1.5,
            "sharpness_factor": 2.0,
            "clahe_clip_limit": 1.5,
            "clahe_tile_grid": (8, 8),
        }
        # Настройки для autogen
        default_autogen_cfg = {
            "points_per_side": 16,
            "points_per_batch": 32,
            "pred_iou_thresh": 0.7,
            "stability_score_thresh": 0.9,
            "stability_score_offset": 0.7,
            "crop_n_layers": 1,
            "box_nms_thresh": 0.7,
            "crop_n_points_downscale_factor": 2,
            "min_mask_region_area": 50,
            "use_m2m": False,
        }

        if config:
            try:
                cfg = json.loads(config)
                print(f"[DEBUG] Custom config received: {cfg}")
                # Обновляем default_autogen_cfg кастомными значениями
                for key in default_autogen_cfg:
                    if key in cfg:
                        default_autogen_cfg[key] = cfg[key]
            except Exception as e:
                print(f"[WARN] Failed to parse config JSON: {e}")
                # Используем default_autogen_cfg как есть
        else:
            cfg = default_autogen_cfg.copy()

        # === 2. Применяем ПРЕПРОЦЕССИНГ (используем функцию из /init) ===
        # Нужно немного модифицировать, чтобы она принимала cfg
        # Пока встроим сюда, позже можно вынести
        processed = preprocess_image_cv(image_np, {
            "median_ksize": cfg.get("median_ksize", 5),
            "contrast_factor": cfg.get("contrast_factor", 1.5),
            "sharpness_factor": cfg.get("sharpness_factor", 2.0),
            "clahe_clip_limit": cfg.get("clahe_clip_limit", 1.5),
            "clahe_tile_grid": cfg.get("clahe_tile_grid", (8, 8)),
        })
        print("[DEBUG] Image preprocessed for autogen")

        # === 3. Подготавливаем SAM для ИНТЕРАКТИВНОГО режима (если понадобится позже) ===
        with torch.inference_mode(), torch.autocast(autocast_device, dtype=autocast_dtype):
            predictor.set_image(processed)

        # === 4. ЗАПУСКАЕМ АВТОГЕНЕРАЦИЮ ===
        # Применяем настройки autogen к mask_generator
        # Создаём временный экземпляр с новыми параметрами, чтобы не изменять глобальный
        temp_mask_generator = SAM2AutomaticMaskGenerator(
            model=sam2,
            points_per_side=cfg.get("points_per_side", 32),
            points_per_batch=cfg.get("points_per_batch", 64),
            pred_iou_thresh=cfg.get("pred_iou_thresh", 0.8),
            stability_score_thresh=cfg.get("stability_score_thresh", 0.9),
            stability_score_offset=cfg.get("stability_score_offset", 1.0),
            crop_n_layers=cfg.get("crop_n_layers", 0),
            box_nms_thresh=cfg.get("box_nms_thresh", 0.7),
            crop_n_points_downscale_factor=cfg.get("crop_n_points_downscale_factor", 1),
            min_mask_region_area=cfg.get("min_mask_region_area", 100),
            use_m2m=cfg.get("use_m2m", False),
        )
        print(f"[DEBUG] Running autogen with config: {cfg}")
        auto_gen_masks = temp_mask_generator.generate(processed)
        print(f"[DEBUG] Autogen returned {len(auto_gen_masks)} masks")

        # === 5. ПОДГОТАВЛИВАЕМ РЕЗУЛЬТАТЫ ДЛЯ ОТПРАВКИ ===
        # Преобразуем маски в нужный формат (id, segmentation, area, bbox, predicted_iou, stability_score)
        formatted_masks = []
        for i, mask_dict in enumerate(auto_gen_masks):
            # segmentation - это бинарная маска (H, W) bool или uint8
            # Нам нужно получить контуры
            mask_bin = mask_dict['segmentation'].astype(np.uint8)
            mask_for_contours = (mask_bin * 255).astype(np.uint8)
            contours_cv2, _ = cv2.findContours(mask_for_contours, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # Берём внешний контур (или все, если их несколько?)
            # Пока возьмём самый большой по площади
            if contours_cv2:
                largest_contour = max(contours_cv2, key=cv2.contourArea)
                contour_points = [[[int(pt[0][0]), int(pt[0][1])] for pt in largest_contour]]
            else:
                contour_points = [] # или пропустить эту маску?

            formatted_masks.append({
                "id": f"mask_{i}",
                "segmentation": contour_points, # Отправляем контур, а не бинарную маску
                "area": float(mask_dict['area']),
                "bbox": [float(x) for x in mask_dict['bbox']], # [x, y, width, height]
                "predicted_iou": float(mask_dict['predicted_iou']),
                "stability_score": float(mask_dict['stability_score']),
                # "crop_box": mask_dict.get('crop_box') # При необходимости
            })

        # === 6. СОЗДАЁМ СЕССИЮ ===
        session_id = str(uuid.uuid4())
        sessions[session_id] = {
            "image": processed, # Для интерактивного режима
            "logits": None, # Для интерактивного режима
            "orig_name": Path(file.filename).stem if file.filename else f"image_{int(time.time())}",
            "points": [], # Для интерактивного режима
            "is_first_click": True, # Для интерактивного режима
            "config": { # Сохраняем конфиг препроцессинга
                "median_ksize": cfg.get("median_ksize", 5),
                "contrast_factor": cfg.get("contrast_factor", 1.5),
                "sharpness_factor": cfg.get("sharpness_factor", 2.0),
                "clahe_clip_limit": cfg.get("clahe_clip_limit", 1.5),
                "clahe_tile_grid": cfg.get("clahe_tile_grid", (8, 8)),
            },
            "auto_masks": formatted_masks, # <-- Сохраняем результаты автогенерации
            "original_processed_image": processed, # <-- Сохраняем для перегенерации
            # Добавим поля для нового режима
            "confirmed_masks": [],
            "final_masks": [],
            "selected_mask_ids": [],
            "main_mask_id": None,
            "selection_confirmed": False,
            "refinement_completed": False,
        }

        # === 7. ВОЗВРАЩАЕМ РЕЗУЛЬТАТ ===
        preview_b64 = image_to_base64_png(processed)
        return {
            "session_id": session_id,
            "preview_b64": preview_b64,
            "used_config": cfg, # Возвращаем использованный конфиг (включая autogen)
            "auto_masks": formatted_masks # Возвращаем маски
        }

    except Exception as e:
        print(f"[ERROR] Exception in /init_autogen: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error during autogen initialization: {str(e)}")


# --- НОВЫЙ ЭНДПОИНТ /update_autogen ---
@app.post("/update_autogen")
async def update_autogen_session(
    session_id: str = Form(...),
    config: str = Form(...) # config как JSON строка
):
    """
    Update an autogen session by regenerating masks with new settings.
    """
    print(f"[DEBUG] /update_autogen called for session: {session_id}")
    if session_id not in sessions:
        raise HTTPException(status_code=400, detail="Invalid session_id")

    sess = sessions[session_id]

    # Проверяем, есть ли оригинальное изображение для перегенерации
    original_image = sess.get("original_processed_image")
    if original_image is None:
        raise HTTPException(status_code=400, detail="Original processed image not found in session. Cannot regenerate.")

    try:
        # === 1. РАЗБИРАЕМ НОВЫЙ КОНФИГ ===
        cfg = json.loads(config)
        print(f"[DEBUG] New autogen config for update: {cfg}")

        # === 2. ЗАПУСКАЕМ АВТОГЕНЕРАЦИЮ С НОВЫМИ НАСТРОЙКАМИ ===
        temp_mask_generator = SAM2AutomaticMaskGenerator(
            model=sam2,
            points_per_side=cfg.get("points_per_side", 32),
            points_per_batch=cfg.get("points_per_batch", 64),
            pred_iou_thresh=cfg.get("pred_iou_thresh", 0.8),
            stability_score_thresh=cfg.get("stability_score_thresh", 0.9),
            stability_score_offset=cfg.get("stability_score_offset", 1.0),
            crop_n_layers=cfg.get("crop_n_layers", 0),
            box_nms_thresh=cfg.get("box_nms_thresh", 0.7),
            crop_n_points_downscale_factor=cfg.get("crop_n_points_downscale_factor", 1),
            min_mask_region_area=cfg.get("min_mask_region_area", 100),
            use_m2m=cfg.get("use_m2m", False),
        )
        print(f"[DEBUG] Running autogen update with config: {cfg}")
        auto_gen_masks = temp_mask_generator.generate(original_image)
        print(f"[DEBUG] Autogen update returned {len(auto_gen_masks)} masks")

        # === 3. ПОДГОТАВЛИВАЕМ НОВЫЕ РЕЗУЛЬТАТЫ ===
        formatted_masks = []
        for i, mask_dict in enumerate(auto_gen_masks):
            mask_bin = mask_dict['segmentation'].astype(np.uint8)
            mask_for_contours = (mask_bin * 255).astype(np.uint8)
            contours_cv2, _ = cv2.findContours(mask_for_contours, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours_cv2:
                largest_contour = max(contours_cv2, key=cv2.contourArea)
                contour_points = [[[int(pt[0][0]), int(pt[0][1])] for pt in largest_contour]]
            else:
                contour_points = []

            formatted_masks.append({
                "id": f"mask_{i}",
                "segmentation": contour_points,
                "area": float(mask_dict['area']),
                "bbox": [float(x) for x in mask_dict['bbox']],
                "predicted_iou": float(mask_dict['predicted_iou']),
                "stability_score": float(mask_dict['stability_score']),
            })

        # === 4. ОБНОВЛЯЕМ СЕССИЮ ===
        # Сбрасываем связанные с выбором/подтверждением состояния
        sess["auto_masks"] = formatted_masks
        sess["selected_mask_ids"] = []
        sess["confirmed_masks"] = []
        sess["main_mask_id"] = None
        sess["selection_confirmed"] = False
        sess["refinement_completed"] = False

        # === 5. ВОЗВРАЩАЕМ НОВЫЕ МАСКИ ===
        return {
            "auto_masks": formatted_masks
        }

    except Exception as e:
        print(f"[ERROR] Exception in /update_autogen: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error during autogen update: {str(e)}")

@app.post("/update_settings")
async def update_settings(
    session_id: str = Form(...),
    median_ksize: int = Form(5),
    contrast_factor: float = Form(1.5),
    sharpness_factor: float = Form(2.0),
    clahe_clip_limit: float = Form(1.5),
    clahe_tile_grid: str = Form("8,8"),
):
    if session_id not in sessions:
        raise HTTPException(status_code=400, detail="Invalid session_id")

    sess = sessions[session_id]
    image_np = sess["original_processed_image"]

    grid_tuple = tuple(map(int, clahe_tile_grid.split(",")))

    config = {
        "median_ksize": median_ksize,
        "contrast_factor": contrast_factor,
        "sharpness_factor": sharpness_factor,
        "clahe_clip_limit": clahe_clip_limit,
        "clahe_tile_grid": grid_tuple,
    }

    processed = preprocess_image_cv(image_np, config)
    with torch.inference_mode(), torch.autocast(autocast_device, dtype=autocast_dtype):
        predictor.set_image(processed)

    sess["image"] = processed
    preview_b64 = image_to_base64_png(processed)
    return {"preview_b64": preview_b64}


# -------------------------
# API: iterative segmentation (фикс - сохраняем 3 файла, без заливки)
# -------------------------
@app.post("/segment")
async def segment_session(
    session_id: str = Form(...),
    x: int | None = Form(None),
    y: int | None = Form(None),
    label: int | None = Form(1),
    save: bool | None = Form(False),
):
    if session_id not in sessions:
        raise HTTPException(status_code=400, detail="Invalid session_id")

    sess = sessions[session_id]
    image_np = sess["image"]

    if x is not None and y is not None:
        sess["points"].append([int(x), int(y), int(label)])

    if not sess["points"]:
        raise HTTPException(status_code=400, detail="No points provided")
    
    #print(f"[DEBUG] session parametrs: {sess}")

    all_coords = [[p[0], p[1]] for p in sess["points"]]
    all_labels = [p[2] for p in sess["points"]]
    input_point = [all_coords]
    input_label = [all_labels]
    is_first_click = sess.get("is_first_click", True)
    mask_input = sess.get("logits", None)
    multimask_output = is_first_click

    with torch.inference_mode(), torch.autocast(autocast_device, dtype=autocast_dtype):
        predict_kwargs = {
            "point_coords": input_point,
            "point_labels": input_label,
            "multimask_output": multimask_output,
        }
        if mask_input is not None:
            predict_kwargs["mask_input"] = mask_input[None, :, :]

        masks, scores, logits = predictor.predict(**predict_kwargs)

    if multimask_output:
        best_idx = int(np.argmax(scores))
        mask = masks[best_idx]
        sess["logits"] = logits[best_idx]
        sess["is_first_click"] = False
    else:
        mask = masks[0]
        sess["logits"] = logits[0]

    mask_bin = (mask > 0.0).astype(np.uint8)
    sess["last_mask"] = mask_bin

    # Контур без заливки
    mask_for_contours = (mask_bin * 255).astype(np.uint8)
    contours_cv2, _ = cv2.findContours(mask_for_contours, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [[[int(pt[0][0]), int(pt[0][1])] for pt in c] for c in contours_cv2 if len(c) > 2]

    # Рисуем только контур
    overlay = image_np.copy()
    for c in contours:
        pts_arr = np.array(c, dtype=np.int32)
        cv2.polylines(overlay, [pts_arr], isClosed=True, color=(255, 0, 0), thickness=2)

    overlay_b64 = image_to_base64_png(overlay)

    saved = {}
    if save:
        base_name = sess.get("orig_name", f"image_{int(time.time())}")
        #jpg_path = get_unique_filename(base_name, ".jpg")
        jpg_path = RESULTS_DIR / f"{base_name}.jpg"
        #segmented_path = jpg_path.replace(".jpg", "_segmented.jpg")
        segmented_path = RESULTS_DIR / f"{base_name}_segmented.jpg"
        #npy_path = jpg_path.replace(".jpg", ".npy")
        npy_path = RESULTS_DIR / f"{base_name}.npy"
        logits_path = RESULTS_DIR / f"{base_name}_logits.npy"

        # сохраняем предобработанную
        cv2.imwrite(str(jpg_path), cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))
        # сохраняем с контуром
        cv2.imwrite(str(segmented_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        # сохраняем маску
        np.save(str(npy_path), mask_bin)

        np.save(str(logits_path), sess["logits"])
        saved = {"base": jpg_path, "segmented": segmented_path, "mask": npy_path, "logits": logits_path}        

    return {"session_id": session_id, "contours": contours, "overlay_b64": overlay_b64, "saved": saved}

@app.post("/segment_main")
async def segment_main(
    session_id: str = Form(...),
    x: int | None = Form(None),
    y: int | None = Form(None),
    label: int = Form(1),
    save: bool = Form(False),
):
    if session_id not in sessions:
        raise HTTPException(status_code=400, detail="Invalid session_id")

    sess = sessions[session_id]
    image_np = sess["image"]

    # accumulate points
    if x is not None and y is not None:
        sess.setdefault("main_points", [])
        sess["main_points"].append([int(x), int(y), int(label)])

    if not sess.get("main_points"):
        raise HTTPException(status_code=400, detail="No points provided")

    pts = sess["main_points"]
    coords = [[p[0], p[1]] for p in pts]
    labels = [p[2] for p in pts]

    mask_input = sess.get("main_logits")
    multimask_output = mask_input is None  # first click → True

    if multimask_output:
        sess["last_main_mask_bin"] = None
        sess["last_main_contours"] = []
    
    # --- ЛОГИКА ОТЛИЧАЕТСЯ ПРИ save=True ---
    if save:
        # Используем ПОСЛЕДНИЕ сохранённые результаты, НЕ пересчитываем
        mask_bin = sess.get("last_main_mask_bin", None)
        contours = sess.get("last_main_contours", [])        
        logits_to_save = sess.get("main_logits")
        if mask_bin is None or not contours:
            raise HTTPException(400, "No previous segmentation results to save. Run segmentation first.")
    else:
        # --- ПЕРЕСЧЁТ ---
        with torch.inference_mode(), torch.autocast(autocast_device, dtype=autocast_dtype):
            kwargs = {
                "point_coords": [coords],
                "point_labels": [labels],
                "multimask_output": multimask_output,
            }
            if mask_input is not None:
                kwargs["mask_input"] = mask_input[None, :, :]

            masks, scores, logits = predictor.predict(**kwargs)

        if multimask_output:
            idx = int(np.argmax(scores))
            mask = masks[idx]
            sess["main_logits"] = logits[idx]
        else:
            mask = masks[0]
            sess["main_logits"] = logits[0]

        mask_bin = (mask > 0).astype(np.uint8)
        sess["main_mask"] = mask_bin
        logits_to_save = sess["main_logits"] # Обновляем для следующего возможного пересчёта или сохранения
        # contours
        cnt_img = (mask_bin * 255).astype(np.uint8)
        cnts, _ = cv2.findContours(cnt_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = [[[int(p[0][0]), int(p[0][1])] for p in c] for c in cnts if len(c) > 2]

        # Сохраняем результаты в сессии
        sess["last_main_mask_bin"] = mask_bin
        sess["last_main_contours"] = contours
            
    overlay = image_np.copy()
    for c in contours:
        cv2.polylines(overlay, [np.array(c)], True, (255, 0, 0), 2)

    overlay_b64 = image_to_base64_png(overlay)

    # saving
    saved = {}
    if save:
        contours_sorted_by_length_asc = sorted(contours, key=lambda c: len(c), reverse=True)
        dir_path = ensure_snowflake_dir(sess)
        save_mask_pack(dir_path, "main", image_np, contours_sorted_by_length_asc[0] if contours_sorted_by_length_asc else None,
                       mask_bin, logits_to_save)
        saved = {"dir": str(dir_path)}

    return {"session_id": session_id, "contours": contours, "overlay_b64": overlay_b64, "saved": saved}

@app.post("/start_inner_box")
async def start_inner_box(
    session_id: str = Form(...),
    x1: int = Form(...),
    y1: int = Form(...),
    x2: int = Form(...),
    y2: int = Form(...),
):
    if session_id not in sessions:
        raise HTTPException(status_code=400, detail="Invalid session_id")

    sess = sessions[session_id]
    sess["inner_box"] = [x1, y1, x2, y2]
    sess["inner_points"] = []
    sess["inner_logits"] = None
    sess["last_inner_mask_bin"] = None, # Для хранения последней mask_bin
    sess["last_inner_contours"] = [],   # Для хранения последних contours

    return {"status": "ok", "box": sess["inner_box"]}

# --- В /segment_inner ---
@app.post("/segment_inner")
async def segment_inner(
    session_id: str = Form(...),
    x: int | None = Form(None),
    y: int | None = Form(None),
    label: int = Form(1),
    save: bool = Form(False),
    selected: str | None = Form(None),
):
    if session_id not in sessions:
        raise HTTPException(400, "Invalid session")

    sess = sessions[session_id]
    image_np = sess["image"]

    if "inner_box" not in sess:
        raise HTTPException(400, "inner box not set")

    x1, y1, x2, y2 = sess["inner_box"]

    # collect points
    if x is not None and y is not None:
        sess["inner_points"].append([x, y, label])

    pts = sess["inner_points"]
    if not pts:
        raise HTTPException(400, "No points for inner segmentation")

    coords = [[p[0], p[1]] for p in pts]
    labels = [p[2] for p in pts]

    mask_input = sess.get("inner_logits")
    multimask_output = mask_input is None

    box_xyxy = np.array([x1, y1, x2, y2])

    # --- ЛОГИКА ОТЛИЧАЕТСЯ ПРИ save=True ---
    if save and selected:
        # Используем ПОСЛЕДНИЕ сохранённые результаты, НЕ пересчитываем
        mask_bin = sess.get("last_inner_mask_bin")
        contours = sess.get("last_inner_contours", [])
        # logits для сохранения можно брать как last_inner_logits или inner_logits, в зависимости от задачи
        # Пусть для сохранения используется last_inner_logits (т.е. логитс от момента, когда был последний пересчёт)
        # или inner_logits, если мы хотим использовать логитс от последнего вызова predict (даже если save=False)
        # Выберем inner_logits, так как он обновляется при predict
        logits_to_save = sess.get("inner_logits")

        if mask_bin is None or not contours:
            raise HTTPException(400, "No previous segmentation results to save. Run segmentation first.")
    else:
        # --- ПЕРЕСЧЁТ ---
        with torch.inference_mode(), torch.autocast(autocast_device, dtype=autocast_dtype):
            kwargs = {
                "point_coords": [coords],
                "point_labels": [labels],
                "box": box_xyxy[None, :],
                "multimask_output": multimask_output,
            }
            if mask_input is not None:
                kwargs["mask_input"] = mask_input[None, :, :]

            masks, scores, logits = predictor.predict(**kwargs)

        if multimask_output:
            idx = int(np.argmax(scores))
            mask = masks[idx]
            sess["inner_logits"] = logits[idx]
        else:
            mask = masks[0]
            sess["inner_logits"] = logits[0]

        mask_bin = (mask > 0).astype(np.uint8)
        logits_to_save = sess["inner_logits"] # Обновляем для следующего возможного пересчёта или сохранения

        # --- СОХРАНЕНИЕ ПОСЛЕДНИХ РЕЗУЛЬТАТОВ ---
        cnt_img = (mask_bin * 255).astype(np.uint8)
        cnts, _ = cv2.findContours(cnt_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = []
        for c in cnts:
            if len(c) >= 1: # Проверка >= 3 была, можно оставить >= 1, как у тебя
                cc = [[int(pt[0][0]), int(pt[0][1])] for pt in c]
                contours.append(cc)

        # Сохраняем результаты в сессии
        sess["last_inner_mask_bin"] = mask_bin
        sess["last_inner_contours"] = contours
        # sess["last_inner_logits"] = sess["inner_logits"] # Если хочешь хранить отдельно, можно раскомментировать
        # logits_to_save уже равен sess["inner_logits"]


    # overlay (только если не save, чтобы не отправлять пустой overlay при сохранении?)
    # Нет, overlay нужен всегда, чтобы показывать текущее состояние
    overlay = image_np.copy()
    for c in contours:
        cv2.polylines(overlay, [np.array(c)], True, (0, 255, 255), 2)

    overlay_b64 = image_to_base64_png(overlay)

    # -------------------------
    # SAVE SELECTED CONTOURS (только если save=True и selected есть)
    # -------------------------
    saved = {}
    if save and selected:
        sel = [int(s) for s in selected.split(",") if s.strip().isdigit()]
        dir_path = ensure_snowflake_dir(sess)
        print(f"[DEBUG] Total contours available: {len(contours)}, Selected indices: {sel}") # Отладка

        # find current index
        current = len([f for f in os.listdir(dir_path) if f.startswith("slave_")]) // 4

        for i, idx in enumerate(sel):
            if idx < 0 or idx >= len(contours): # Правильная проверка: < len, а не <= len - 1
                print(f"[WARN] Index {idx} is out of bounds for contours list of length {len(contours)}. Skipping.")
                continue

            # --- НОВАЯ ЛОГИКА: Создание маски только для конкретного контура ---
            contour_to_save = contours[idx]
            # Создаём пустую маску того же размера
            single_contour_mask = np.zeros_like(mask_bin, dtype=np.uint8)
            # Заполняем только этот один контур
            pts = np.array(contour_to_save, dtype=np.int32)
            cv2.fillPoly(single_contour_mask, [pts], 1) # Заполняем 1

            prefix = f"slave_{current + i + 1}"
            # Передаём single_contour_mask вместо mask_bin
            save_mask_pack(
                dir_path,
                prefix,
                image_np,
                contour_to_save, # <-- контур для отображения
                single_contour_mask, # <-- маска ТОЛЬКО для этого контура
                logits_to_save, # <-- logits, соответствующие моменту последнего пересчёта
            )

        # after save → reset inner (только если save был успешен?)
        #sess["inner_points"] = []
        #sess.pop("inner_logits", None)
        #sess.pop("inner_box", None)
        # Опционально: очистить last_inner_... после сохранения
        # sess.pop("last_inner_mask_bin", None)
        # sess.pop("last_inner_contours", None)

        saved = {"count": len(sel)}

    return {
        "contours": contours,
        "overlay_b64": overlay_b64,
        "saved": saved,
    }

@app.post("/save_all")
async def save_all(session_id: str = Form(...)):
    if session_id not in sessions:
        raise HTTPException(400, "Invalid session")

    sess = sessions[session_id]
    dir_path = ensure_snowflake_dir(sess)

    img = sess["image"].copy()

    # draw main
    """
    if "main_mask" in sess:
        cnt_img = (sess["main_mask"] * 255).astype(np.uint8)
        cnts, _ = cv2.findContours(cnt_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            pts = np.array([[int(p[0][0]), int(p[0][1])] for p in c], dtype=np.int32)
            cv2.polylines(img, [pts], True, (255,0,0), 2)
    """        
    if (dir_path / "main_mask.npy").exists():
        main_mask = np.load(str(dir_path / "main_mask.npy"))
        cnt_img = (main_mask * 255).astype(np.uint8)
        cnts, _ = cv2.findContours(cnt_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            pts = np.array([[int(p[0][0]), int(p[0][1])] for p in c], dtype=np.int32)
            cv2.polylines(img, [pts], True, (255, 0, 0), 2) # Красный для main        

    # draw inner slaves
    slave_files = sorted([f for f in os.listdir(dir_path) if f.endswith("_mask.npy") and f.startswith("slave_")])
    for sf in slave_files:
        mask = np.load(str(dir_path / sf))
        cnt_img = (mask * 255).astype(np.uint8)
        cnts,_ = cv2.findContours(cnt_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            pts = np.array([[int(pt[0][0]), int(pt[0][1])] for pt in c], dtype=np.int32)
            cv2.polylines(img, [pts], True, (255,0,0), 2)

    out = dir_path / "final.jpg"
    cv2.imwrite(str(out), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    return {"final": str(out)}

# -------------------------
# Reset API (новый, для кнопки Reset)
# -------------------------
@app.post("/reset")
async def reset_session(session_id: str = Form(...), submode: str = Form(...)):
    print(f"[DEBUG] /reset called for session: {session_id}") # <-- Отладка
    if session_id not in sessions:
        print(f"[ERROR] Session {session_id} not found in /reset") # <-- Отладка
        raise HTTPException(status_code=400, detail="Invalid session_id")
    sess = sessions[session_id]
    sess["points"] = []
    sess["logits"] = None # <-- Сбрасываем logits
    sess["is_first_click"] = True # <-- Сбрасываем флаг
    sess["last_mask"] = None # <-- Можно сбросить, если нужно
    if submode == 'main':
        #main reject
        sess["main_points"] = []
        sess["main_logits"] = None    
        sess["main_mask"] = None
        sess["last_main_mask_bin"] = None
        sess["last_main_contours"] = []
    elif submode == 'inner':
        #inner reject
        sess["inner_points"] = []
        sess["inner_logits"] = None    
        sess["inner_mask"] = None
        sess["last_inner_mask_bin"] = None
        sess["last_inner_contours"] = []

    print("[DEBUG] /reset completed successfully") # <-- Отладка
    return {"status": "ok"}

@app.post("/clear_session_dir")
async def clear_session_dir(session_id: str = Form(...)):
    """
    Clears all files in the session's directory.
    Does NOT remove the directory itself.
    """
    if session_id not in sessions:
        raise HTTPException(status_code=400, detail="Invalid session_id")

    sess = sessions[session_id]
    dir_path = ensure_snowflake_dir(sess)

    if not dir_path.exists():
        # Директория не существует, значит, и нечего удалять
        return {"status": "success", "message": "Directory does not exist."}

    try:
        # Очищаем содержимое директории
        for item in dir_path.iterdir():
            if item.is_file():
                item.unlink() # Удаляем файл
            elif item.is_dir():
                shutil.rmtree(item) # Удаляем поддиректорию рекурсивно
        print(f"[DEBUG] Cleared directory: {dir_path}")
        # Удалим запись из sessions, если нужно
        del sessions[session_id] # <- Опционально, если хочешь полностью сбросить сессию на бэкенде
        return {"status": "success", "message": f"Directory {dir_path} cleared."}
    except Exception as e:
        print(f"[ERROR] Failed to clear directory {dir_path}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to clear directory: {str(e)}")
    
# -------------------------
# Results listing + image serving (paginated)
# -------------------------
@app.get("/results/list")
def list_results(page: int = Query(1, ge=1), per_page: int = Query(12, ge=1, le=50)):
    """Показываем только сегментированные снежинки (_segmented.jpg), без ошибок если пусто"""
    all_files = sorted(
        RESULTS_DIR.glob("*_segmented.jpg"),
        key=os.path.getmtime,
        reverse=True
    )
    total = len(all_files)
    start = (page - 1) * per_page
    end = start + per_page
    selected = all_files[start:end]
    files = []
    for f in selected:
        stat = f.stat()
        files.append({
            "name": f.name,
            "size_kb": round(stat.st_size / 1024, 1),
            "mtime": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(stat.st_mtime))
        })
    return {
        "page": page,
        "per_page": per_page,
        "total": total,
        "results": files
    }



@app.get("/results/image")
def get_result_image(name: str):
    path = RESULTS_DIR / name
    if not path.exists():
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(path)


@app.get("/load_result")
def load_result(name: str):
    """
    Загружаем ранее сохранённую снежинку (_segmented.jpg).
    Возвращаем preview с НАРИСОВАННЫМ красным контуром, контурные точки и новую session_id.
    """
    print(f"[DEBUG] /load_result called for: {name}")

    # вычистим базовое имя (поддерживаем как "foo_segmented.jpg", так и "foo.jpg")
    base_name = Path(name).stem.replace("_segmented", "")
    jpg_path = RESULTS_DIR / f"{base_name}.jpg"
    npy_path = RESULTS_DIR / f"{base_name}.npy"
    logits_path = RESULTS_DIR / f"{base_name}_logits.npy"

    if not jpg_path.exists() or not npy_path.exists():
        print("[WARN] Missing files for load_result:", jpg_path, npy_path)
        return {
            "error": "Missing files",
            "session_id": None,
            "preview_b64": None,
            "contours": [],
        }

    # загружаем предобработанное изображение (чистое, без контуров)
    image = Image.open(jpg_path).convert("RGB")
    image_np = np.array(image)

    # загружаем маску из .npy (логика: mask saved as binary 0/1)
    mask = np.load(npy_path, allow_pickle=True)
    mask_bin = (mask > 0).astype(np.uint8)
    mask_uint8 = (mask_bin * 255).astype(np.uint8)

    # находим контуры (cv2 ожидает uint8)
    contours_cv2, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [
        [[int(pt[0][0]), int(pt[0][1])] for pt in c] for c in contours_cv2 if len(c) > 2
    ]

    # нарисуем **красный** контур поверх исходного изображения (без заливки)
    overlay = image_np.copy()
    for c in contours:
        if len(c) < 2:
            continue
        pts_arr = np.array(c, dtype=np.int32)
        # цвет (255,0,0) — красный; толщина 2
        cv2.polylines(overlay, [pts_arr], isClosed=True, color=(255, 0, 0), thickness=2)

    preview_b64 = image_to_base64_png(overlay)

    # создаём новую сессию и ставим изображение в predictor (без переконфигурации)
    session_id = str(uuid.uuid4())
    with torch.inference_mode(), torch.autocast(autocast_device, dtype=autocast_dtype):
        predictor.set_image(image_np)

    # если логиты есть — восстанавливаем полностью
    if logits_path.exists():
        logits = np.load(logits_path, allow_pickle=True)
        is_first_click = False
    else:
        logits = None
        is_first_click = True

    sessions[session_id] = {
        "image": image_np,
        "logits": logits,
        "orig_name": base_name,
        "points": [],
        "is_first_click": is_first_click,
    }    

    print(f"[DEBUG] /load_result ready: {session_id}, contours: {len(contours)}")
    return {
        "session_id": session_id,
        "preview_b64": preview_b64,
        "contours": contours,
    }


