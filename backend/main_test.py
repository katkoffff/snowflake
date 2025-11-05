# C:\snowflakes\backend\main.py
from fastapi import FastAPI, File, Form, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from sam2.build_sam import build_sam2_video_predictor # <-- Изменён импорт
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

CHECKPOINT_PATH = "models/sam2.1_hiera_large.pt"
MODEL_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"

device = "cuda" if torch.cuda.is_available() else "cpu"

# Загрузка модели SAM 2 Video Predictor
# --- ПРАВКА: Теперь используем build_sam2_video_predictor ---
predictor = build_sam2_video_predictor(MODEL_CONFIG, CHECKPOINT_PATH, device=device)

# Установка dtype для autocast, если используется CUDA
autocast_dtype = torch.bfloat16 if device == "cuda" else None
autocast_device = device if device == "cuda" else "cpu"

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

# In-memory sessions store (simple)
# --- ПРАВКА: Изменена структура сессии ---
sessions: dict = {}  # session_id -> { "inference_state": dict, "obj_id": int, "points": list, "is_first_click": bool, "last_mask": np.ndarray|None, "orig_name": str }

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
# API: initialize session (upload image -> preprocess -> init_state)
# -------------------------
@app.post("/init")
async def init_session(file: UploadFile = File(...)):
    """
    Initialize a segmentation session with uploaded image.
    Treats image as a video with one frame.
    Returns session_id and processed preview (base64 PNG).
    """
    raw = await file.read()
    pil = Image.open(BytesIO(raw)).convert("RGB")
    image_np = np.array(pil)  # RGB

    # preprocess
    processed = preprocess_image_cv(image_np)

    # Create a temporary image file for init_state (SAM2VideoPredictor expects a path)
    # или передать напрямую numpy массив, если SAM2VideoPredictor поддерживает (проверить документацию)
    # В документации и примерах обычно используется путь к файлу или путь к директории с кадрами.
    # Для одного кадра можно создать временную директорию.
    temp_dir = RESULTS_DIR / f"temp_{uuid.uuid4()}"
    temp_dir.mkdir(exist_ok=True)
    temp_img_path = temp_dir / "frame.jpg"
    cv2.imwrite(str(temp_img_path), cv2.cvtColor(processed, cv2.COLOR_RGB2BGR))

    # --- ПРАВКА: init_state теперь внутри контекстного менеджера ---
    with torch.inference_mode(), torch.autocast(autocast_device, dtype=autocast_dtype):
        inference_state = predictor.init_state(
            video_path=str(temp_dir), # <-- Передаём путь к директории с кадром
            # offload_video_to_cpu=False # <-- Можно настроить
        )
        # inference_state теперь содержит состояние для "видео" (одного кадра)

    # create session
    session_id = str(uuid.uuid4())
    sessions[session_id] = {
        "inference_state": inference_state,
        "obj_id": 1, # <-- Простой ID объекта
        "points": [], # <-- Инициализируем points
        "is_first_click": True, # <-- Флаг для определения первого клика
        "last_mask": None, # <-- Для сохранения последней маски
        "orig_name": Path(file.filename).stem if file.filename else f"image_{int(time.time())}",
        "temp_dir": temp_dir, # <-- Сохраняем путь к временной папке для очистки
    }

    preview_b64 = image_to_base64_png(processed)
    return {"session_id": session_id, "preview_b64": preview_b64}

# -------------------------
# API: iterative segmentation (фикс - с использованием add_new_points_or_box)
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
    inference_state = sess["inference_state"]
    obj_id = sess["obj_id"]

    # добавляем новую точку в память
    if x is not None and y is not None:
        sess["points"].append([int(x), int(y), int(label or 1)])

    if not sess["points"]:
        raise HTTPException(status_code=400, detail="No points provided")

    # --- Подготовка точек для add_new_points_or_box ---
    all_coords = [[p[0], p[1]] for p in sess["points"]]
    all_labels = [p[2] for p in sess["points"]]

    # --- Определяем параметры для add_new_points_or_box ---
    is_first_click = sess.get("is_first_click", True)
    # clear_old_points = is_first_click # <-- Сбрасываем старые точки только при первом клике?
    # НЕТ, мы накапливаем точки. SAM2VideoPredictor сам обрабатывает внутреннее состояние.
    # Веб-демо: добавляет точку к существующим. Это соответствует clear_old_points=False.
    # Мы передаём *все* накопленные точки каждый раз, но SAM2VideoPredictor "помнит" предыдущее состояние.
    # Правильный способ: передаём только *новую* точку и используем clear_old_points=False (или True для первой).
    # Или передаём *все* точки и clear_old_points=True (тогда модель пересчитывает всё заново).
    # Второй вариант (передать все) более похож на логику веб-демо, где ты кликаешь, и маска пересчитывается с учётом всех кликов.
    # Попробуем передавать *все* точки с clear_old_points=True, чтобы пересчитывать всё.
    # Или передавать *все* точки с clear_old_points=False, но тогда нужно быть осторожным, чтобы не дублировать.
    # Лучше передавать *все* точки с clear_old_points=True, чтобы гарантировать пересчёт.
    # Нет, это неэффективно. Лучше передавать только *новую* точку и clear_old_points=False (если не первая).
    # Для первой точки: передаём её и clear_old_points=True.
    # Для последующих: передаём *новую* точку и clear_old_points=False.

    # Итак, нужно различать: первая точка или добавление к существующей маске.
    # В веб-демо: первая точка -> multimask, выбор. Последующие -> уточнение.
    # В SAM2VideoPredictor: add_new_points_or_box с clear_old_points=True -> сбрасывает предыдущие промты для obj_id на этом кадре и добавляет новые.
    # add_new_points_or_box с clear_old_points=False -> добавляет новые промты к существующим для obj_id на этом кадре.
    # Это *очень* похоже.
    # Для первого клика: clear_old_points = True
    # Для последующих: clear_old_points = False, передаём только *новую* точку.
    # Но мы хотим передавать *все*, как в веб-демо.
    # В веб-демо коде: clear_old_points передаётся как параметр.
    # Допустим, мы передаём все точки каждый раз.
    # Тогда при первом вызове clear_old_points=True, при последующих False.
    # Но если мы передаём все точки, и они не меняются, зачем clear_old_points=False?
    # SAM2VideoPredictor *внутри* может использовать предыдущее состояние (logits, mask) для уточнения.
    # Если передать все точки с clear_old_points=True, это как бы "перезапуск" с этими точками.
    # Если передать *новую* точку с clear_old_points=False, это "уточнение".
    # Нам нужно: передать *все* накопленные точки и заставить модель "пересчитать" маску, учитывая *все* точки.
    # Это делается через clear_old_points=True, передавая *все* точки.
    # Или передать *новую* точку и clear_old_points=False.
    # Второй вариант (новая точка + False) может быть нестабильным, если модель "забудет" контекст.
    # Первый вариант (все точки + True) гарантирует, что модель видит *всё*.
    # Попробуем: при is_first_click передаём все точки с clear_old_points=True. При последующих - *все* точки с clear_old_points=True.
    # Нет, это бессмысленно, будет пересчёт каждый раз.
    # Попробуем: при is_first_click передаём все точки с clear_old_points=True. При последующих - *новую* точку с clear_old_points=False.
    # НО: как передать *новую* точку, если у нас есть список всех?
    # Нужно сохранить предыдущее количество точек.
    # Или передавать *все* точки, но использовать clear_old_points=False *после* первого раза.
    # Это сбивает с толку. Давайте посмотрим ещё раз на веб-демо код.
    # add_new_points_or_box(
    #     inference_state=inference_state,
    #     frame_idx=frame_idx,
    #     obj_id=obj_id,
    #     points=points,  # <-- Все новые точки
    #     labels=labels,  # <-- Все новые метки
    #     clear_old_points=clear_old_points, # <-- Сбросить предыдущие?
    #     ...
    # )
    # В inference.py веб-демо: clear_old_points передаётся как параметр в AddPointsRequest.
    # В Stage.tsx (фронтенд) я не видел, чтобы он передавал разные точки или управлял clear_old_points.
    # Значит, фронтенд, вероятно, передаёт *все* клики, и бэкенд решает, как обрабатывать.
    # В inference.py: points=points, labels=labels, где points/labels - это *все* переданные точки.
    # Используется ли clear_old_points? Да, передаётся.
    # Как его определяет фронтенд? Неясно из App.tsx. Но в `runMultiMaskModel` и `runModel` используется разная логика.
    # `runMultiMaskModel` - первый клик. `runModel` - уточнение.
    # В `runModel` `last_pred_mask` передаётся. Это соответствует *уточнению*.
    # В `add_new_points_or_box` SAM2VideoPredictor, `clear_old_points` контролирует, заменить ли предыдущие промты.
    # Логика: Первая точка -> clear_old_points=True (создаёт начальное состояние). Последующие -> clear_old_points=False (уточняет).
    # Но если мы передаём *все* точки, это как бы "переопределение".
    # Правильная логика: Первая точка: clear_old_points=True. Последующие: *новая* точка, clear_old_points=False.
    # Чтобы это реализовать, нужно хранить предыдущее количество точек или саму последнюю точку.
    # Попробуем: при /segment, если is_first_click, передаём все точки с clear_old_points=True и сбрасываем флаг.
    # Если не is_first_click, передаём *все* точки (включая новую) с clear_old_points=True. Это как "переоценка".
    # Или: при /segment, если is_first_click, передаём все точки с clear_old_points=True и сбрасываем флаг.
    # Если не is_first_click, передаём *только последнюю (новую)* точку с clear_old_points=False.

    # Выберем: 1. Первая точка: clear_old_points=True. 2. Последующие: новая точка, clear_old_points=False.
    # Для этого нужно хранить предыдущее количество точек.
    # Или: 1. Первая точка: clear_old_points=True. 2. Последующие: *все* точки, clear_old_points=True.
    # Вариант 2 проще, но менее эффективен.
    # Попробуем Вариант 1, как в "правильной" логике.

    new_point_coords = np.array([all_coords[-1]]) # <-- Только новая точка
    new_point_labels = np.array([all_labels[-1]]) # <-- Только новая метка
    clear_old_points = is_first_click # <-- Только при первом клике

    # --- Вызов add_new_points_or_box внутри контекстного менеджера ---
    with torch.inference_mode(), torch.autocast(autocast_device, dtype=autocast_dtype):
        # Правильный вызов: передаём только новую точку, если не первая
        frame_idx, object_ids, masks = predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=0, # <-- Кадр 0, так как у нас изображение
            obj_id=obj_id,
            points=new_point_coords,
            labels=new_point_labels,
            clear_old_points=clear_old_points,
            normalize_coords=False, # <-- SAM2VideoPredictor не нормализует координаты по умолчанию
        )

    # --- Обработка результата ---
    # masks.shape: [num_objects, 1, H, W]
    # object_ids: [obj_id]
    # Берём маску для нашего объекта
    if obj_id in object_ids:
        obj_idx = object_ids.index(obj_id)
        mask = masks[obj_idx, 0] # <-- Убираем размерность батча/объекта [H, W]
    else:
        # Это ошибка, объект не найден в результатах
        raise HTTPException(status_code=500, detail="Object mask not returned from predictor")

    # mask от SAM2VideoPredictor - torch.Tensor
    mask_np = mask.cpu().numpy() # <-- Преобразуем в numpy
    # mask_bin уже бинарная маска от SAM2VideoPredictor? Нет, это float [0, 1]
    mask_bin = (mask_np > 0.0).astype(np.uint8) # <-- Бинаризуем
    sess["last_mask"] = mask_bin
    # Сбрасываем флаг первого клика после первого вызова predict
    if is_first_click:
        sess["is_first_click"] = False

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

    # Получаем оригинальное изображение для оверлея
    # Оно лежит в временной папке
    temp_img_path = sess["temp_dir"] / "frame.jpg"
    image_bgr = cv2.imread(str(temp_img_path))
    image_np = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

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

    # Сбрасываем состояние в SAM2VideoPredictor
    # predictor.reset_state(sess["inference_state"]) # <-- Сбрасывает всё состояние (все объекты, все промты)
    # Или просто сбрасываем точки и флаги, но оставляем inference_state?
    # Веб-демо: "Reset" сбрасывает *все* промты. Это соответствует reset_state.
    predictor.reset_state(sess["inference_state"])

    # Пересоздаём inference_state с тем же изображением
    temp_dir = sess["temp_dir"]
    with torch.inference_mode(), torch.autocast(autocast_device, dtype=autocast_dtype):
        new_inference_state = predictor.init_state(
            video_path=str(temp_dir),
        )

    # Обновляем сессию
    sess["inference_state"] = new_inference_state
    sess["points"] = []
    sess["is_first_click"] = True # <-- Сбрасываем флаг
    sess["last_mask"] = None
    # temp_dir остаётся тем же
    return {"status": "ok"}


# -------------------------
# Cleanup: Удаляем временные файлы при завершении работы
# -------------------------
import atexit
import shutil

def cleanup_temp_dirs():
    for session_data in sessions.values():
        temp_dir = session_data.get("temp_dir")
        if temp_dir and temp_dir.exists():
            try:
                shutil.rmtree(temp_dir)
                print(f"Removed temporary directory: {temp_dir}")
            except Exception as e:
                print(f"Could not remove temporary directory {temp_dir}: {e}")

atexit.register(cleanup_temp_dirs)

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