# C:\snowflakes\backend\main.py
from fastapi import FastAPI, File, Form, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.responses import StreamingResponse
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor # <-- Возвращаемся к ImagePredictor
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
sam2 = build_sam2(MODEL_CONFIG, CHECKPOINT_PATH, device=device)
predictor = SAM2ImagePredictor(sam2)
print("[DEBUG] SAM2ImagePredictor loaded successfully.") # <-- Отладка

# Установка dtype для autocast, если используется CUDA
autocast_dtype = torch.bfloat16 if device == "cuda" else None
autocast_device = device if device == "cuda" else "cpu"
print(f"[DEBUG] Autocast dtype: {autocast_dtype}, device: {autocast_device}") # <-- Отладка

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

# In-memory sessions store (simple)
sessions: dict = {}  # session_id -> { "image": np.ndarray, "logits": np.ndarray|None, "orig_name": str, "points": list, "is_first_click": bool }

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
    image_np = sess["image"]

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



# -------------------------
# Reset API (новый, для кнопки Reset)
# -------------------------
@app.post("/reset")
async def reset_session(session_id: str = Form(...)):
    print(f"[DEBUG] /reset called for session: {session_id}") # <-- Отладка
    if session_id not in sessions:
        print(f"[ERROR] Session {session_id} not found in /reset") # <-- Отладка
        raise HTTPException(status_code=400, detail="Invalid session_id")
    sess = sessions[session_id]
    sess["points"] = []
    sess["logits"] = None # <-- Сбрасываем logits
    sess["is_first_click"] = True # <-- Сбрасываем флаг
    # sess["last_mask"] = None # <-- Можно сбросить, если нужно
    print("[DEBUG] /reset completed successfully") # <-- Отладка
    return {"status": "ok"}


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


