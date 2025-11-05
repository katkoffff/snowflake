# C:\snowflakes\backend\main.py
from fastapi import FastAPI, File, Form, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
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

# Загрузка модели SAM 2
sam2 = build_sam2(MODEL_CONFIG, CHECKPOINT_PATH, device=device)
predictor = SAM2ImagePredictor(sam2)

# Установка dtype для autocast, если используется CUDA
autocast_dtype = torch.bfloat16 if device == "cuda" else None
autocast_device = device if device == "cuda" else "cpu"

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

# In-memory sessions store (simple)
sessions: dict = {}  # session_id -> { "image": np.ndarray, "logits": np.ndarray|None, "orig_name": str, "points": list }

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
    """Return base64 string (no data: prefix) of PNG image from RGB numpy."""
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
async def init_session(file: UploadFile = File(...)):
    """
    Initialize a segmentation session with uploaded image.
    Returns session_id and processed preview (base64 PNG).
    """
    raw = await file.read()
    pil = Image.open(BytesIO(raw)).convert("RGB")
    image_np = np.array(pil)  # RGB

    # preprocess
    processed = preprocess_image_cv(image_np)

    # set model image
    with torch.inference_mode(), torch.autocast(autocast_device, dtype=autocast_dtype):
        predictor.set_image(processed)

    # create session
    session_id = str(uuid.uuid4())
    sessions[session_id] = {
        "image": processed,
        "logits": None, # <-- Сохраняем logits
        "orig_name": Path(file.filename).stem if file.filename else f"image_{int(time.time())}",
        "points": [], # <-- Инициализируем points
        "is_first_click": True # <-- Флаг для определения первого клика
    }

    preview_b64 = image_to_base64_png(processed)
    return {"session_id": session_id, "preview_b64": preview_b64}

# -------------------------
# API: iterative segmentation (фикс - с использованием logits)
# -------------------------
@app.post("/segment")
async def segment_session(
    session_id: str = Form(...),
    x: int | None = Form(None),
    y: int | None = Form(None),
    label: int | None = Form(1), # label по умолчанию 1 (объект)
    save: bool | None = Form(False),
):
    if session_id not in sessions:
        raise HTTPException(status_code=400, detail="Invalid session_id")

    sess = sessions[session_id]
    image_np = sess["image"]

    # добавляем новую точку в память
    if x is not None and y is not None:
        sess["points"].append([int(x), int(y), int(label or 1)])

    if not sess["points"]:
        raise HTTPException(status_code=400, detail="No points provided")

    # --- Подготовка точек для SAM 2 ---
    all_coords = [[p[0], p[1]] for p in sess["points"]]
    all_labels = [p[2] for p in sess["points"]]
    input_point = [all_coords]
    input_label = [all_labels]

    # --- Определяем параметры для predict ---
    is_first_click = sess.get("is_first_click", True)
    mask_input = sess.get("logits", None)

    multimask_output = is_first_click # Первый клик - multimask, далее - нет
    # mask_input может быть None на первом клике, что нормально

    # --- Вызов predict внутри контекстного менеджера ---
    with torch.inference_mode(), torch.autocast(autocast_device, dtype=autocast_dtype):
        # Подготовим аргументы
        predict_kwargs = {
            "point_coords": input_point,
            "point_labels": input_label,
            "multimask_output": multimask_output,
        }
        if mask_input is not None:
             # Убедимся, что mask_input имеет правильную форму [1, C, H, W]
             # logits от SAM 2 имеют форму [C, H, W], нужно добавить размерность батча
            predict_kwargs["mask_input"] = mask_input[None, :, :]

        masks, scores, logits = predictor.predict(**predict_kwargs)

    # --- Обработка результата ---
    if multimask_output:
        # Выбираем лучшую маску из 3
        best_idx = int(np.argmax(scores))
        mask = masks[best_idx]
        # Сохраняем соответствующие logits для уточнения
        sess["logits"] = logits[best_idx]
        # Сбрасываем флаг первого клика
        sess["is_first_click"] = False
    else:
        # multimask_output=False, возвращается одна маска
        mask = masks[0]
        # Сохраняем соответствующие logits для дальнейшего уточнения
        sess["logits"] = logits[0]
        # Флаг остаётся False, так как это уже не первый клик

    # mask_bin уже бинарная маска от SAM 2, но убедимся, что она uint8
    mask_bin = (mask > 0.0).astype(np.uint8)
    sess["last_mask"] = mask_bin

    # контур через cv2 (как в примере show_mask)
    # Работаем с булевой/float маской напрямую
    # cv2.findContours требует uint8, внутренне он использует порог
    # для определения границы (обычно 127 для uint8, что соответствует 0.5 для нормализованной [0,1])
    # маска для cv2 должна быть 0 или 255, но SAM возвращает bool/float [0, 1]
    # преобразуем маску к uint8 [0, 255]
    mask_for_contours = (mask_bin * 255).astype(np.uint8)
    contours_cv2, _ = cv2.findContours(mask_for_contours, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Фильтруем короткие контуры
    contours = [[[int(pt[0][0]), int(pt[0][1])] for pt in c] for c in contours_cv2 if len(c) > 2]

    # полупрозрачный оверлей
    overlay = image_np.copy()
    red = np.zeros_like(image_np)
    red[mask_bin > 0] = (255, 0, 0)
    overlay = cv2.addWeighted(overlay, 1.0, red, 0.25, 0)
    for c in contours:
        pts_arr = np.array(c, dtype=np.int32)
        cv2.polylines(overlay, [pts_arr], isClosed=True, color=(255, 0, 0), thickness=2)

    overlay_b64 = image_to_base64_png(overlay)

    saved = {}
    if save:
        base_name = sess.get("orig_name", f"image_{int(time.time())}")
        jpg_path = get_unique_filename(f"{base_name}_segmented", ".jpg")
        npy_path = jpg_path.replace(".jpg", ".npy")
        cv2.imwrite(jpg_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        np.save(npy_path, mask_bin)
        saved = {"jpg": jpg_path, "npy": npy_path}

    return {"session_id": session_id, "contours": contours, "overlay_b64": overlay_b64, "saved": saved}


# -------------------------
# Reset API (новый, для кнопки Reset)
# -------------------------
@app.post("/reset")
async def reset_session(session_id: str = Form(...)):
    if session_id not in sessions:
        raise HTTPException(status_code=400, detail="Invalid session_id")
    sess = sessions[session_id]
    sess["points"] = []
    sess["logits"] = None # <-- Сбрасываем logits
    sess["is_first_click"] = True # <-- Сбрасываем флаг
    # sess["last_mask"] = None # <-- Можно сбросить, если нужно
    return {"status": "ok"}



# -------------------------
# Results listing + image serving (paginated)
# -------------------------
@app.get("/results/list")
def list_results(page: int = Query(1, ge=1), per_page: int = Query(12, ge=1, le=50)):
    all_files = sorted(RESULTS_DIR.glob("*.jpg"), key=os.path.getmtime, reverse=True)
    total = len(all_files)
    start = (page - 1) * per_page
    end = start + per_page
    selected = all_files[start:end]
    files = []
    for f in selected:
        stat = f.stat()
        files.append({"name": f.name, "size_kb": round(stat.st_size / 1024, 1), "mtime": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(stat.st_mtime))})
    return {"page": page, "per_page": per_page, "total": total, "results": files}


@app.get("/results/image")
def get_result_image(name: str):
    path = RESULTS_DIR / name
    if not path.exists():
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(path)