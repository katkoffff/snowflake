# backend/routers/init.py
"""
Инициализация сессии:
- /init
- /init_autogen
- /update_autogen
- /update_settings
"""

from fastapi import APIRouter, File, Form, UploadFile, HTTPException, Request
from services.session_service import create_session, get_session, update_session
from services.image_service import preprocess_image_cv, image_to_base64_png, create_image_with_contours
from services.sam2_service import set_image, generate_automasks
from services.file_service import ensure_snowflake_dir, load_contours_and_image_from_folder
from core.config import (
    DEFAULT_PREPROCESS_CFG, validate_preprocess_cfg,
    DEFAULT_AUTOGEN_CFG, validate_autogen_cfg,
    STAGE2_DIR
)
from PIL import Image
from io import BytesIO
import json
import numpy as np
from pathlib import Path
from utils.debug import print_debug
import traceback

router = APIRouter()


@router.post("/init")
async def init_session(
    request: Request,
    file: UploadFile = File(...),
    config: str | None = Form(None)
):
    raw = await file.read()
    pil = Image.open(BytesIO(raw)).convert("RGB")
    image_np = np.array(pil)

    cfg = validate_preprocess_cfg(json.loads(config) if config else {})
    processed = preprocess_image_cv(image_np, cfg)
    set_image(request.app.state.sam2_objects['predictor'], processed)

    session_id = create_session(
        image_np=image_np,
        orig_name=Path(file.filename).stem,
        processed_image=processed,
        config=cfg,
        auto_masks=None
    )

    return {
        "session_id": session_id,
        "preview_b64": image_to_base64_png(processed),
        "used_config": cfg,
    }


@router.post("/init_autogen")
async def init_autogen_session(
    request: Request,
    file: UploadFile = File(...),
    config: str | None = Form(None)
):
    raw = await file.read()
    pil = Image.open(BytesIO(raw)).convert("RGB")
    image_np = np.array(pil)

    cfg = validate_preprocess_cfg(json.loads(config) if config else {})
    processed = preprocess_image_cv(image_np, cfg)
    set_image(processed)

    autogen_cfg = validate_autogen_cfg(json.loads(config) if config else {})
    auto_masks = generate_automasks(request.app.state.sam2_objects['sam2_model'], processed, autogen_cfg)

    session_id = create_session(
        image_np=image_np,
        orig_name=Path(file.filename).stem,
        processed_image=processed,
        config=cfg,
        auto_masks=auto_masks
    )

    return {
        "session_id": session_id,
        "preview_b64": image_to_base64_png(processed),
        "auto_masks": auto_masks,
        "used_config": cfg,
        "autogen_cfg": autogen_cfg,
    }


@router.post("/update_autogen")
async def update_autogen(request: Request, session_id: str, config: str = Form(...)):
    sess = get_session(session_id)
    autogen_cfg = validate_autogen_cfg(json.loads(config))
    auto_masks = generate_automasks(request.app.state.sam2_objects['sam2_model'], sess["image"], autogen_cfg)

    update_session(session_id, auto_masks=auto_masks)
    return {"auto_masks": auto_masks, "autogen_cfg": autogen_cfg}


@router.post("/update_settings")
async def update_settings(
    request: Request,
    session_id: str = Form(...),
    median_ksize: int = Form(5),
    contrast_factor: float = Form(1.5),
    sharpness_factor: float = Form(2.0),
    clahe_clip_limit: float = Form(1.5),
    clahe_tile_grid: str = Form("8,8"),
):
    
    try:
        sess = get_session(session_id)
    except KeyError:
        raise HTTPException(status_code=400, detail="Invalid session_id")

    try:
        grid_tuple = tuple(map(int, clahe_tile_grid.split(",")))
        if len(grid_tuple) != 2:
            raise ValueError
    except:
        raise HTTPException(status_code=400, detail="Invalid clahe_tile_grid format. Use '8,8'")

    config = {
        "median_ksize": median_ksize,
        "contrast_factor": contrast_factor,
        "sharpness_factor": sharpness_factor,
        "clahe_clip_limit": clahe_clip_limit,
        "clahe_tile_grid": grid_tuple,
    }

    # Валидация через core.config
    config = validate_preprocess_cfg(config)

    # Препроцессинг
    processed = preprocess_image_cv(sess["original_processed_image"], config)

    # Обновление SAM2
    set_image(request.app.state.sam2_objects['predictor'], processed)

    # Сохранение в сессию
    update_session(session_id, image=processed, config=config)

    return {
        "preview_b64": image_to_base64_png(processed),
        "used_config": config,
    }

# backend/routers/init.py
# ... (все импорты, включая load_contours_and_image_from_folder из services.file_service) ...
# ... (остальные эндпоинты) ...

@router.post("/init_stage2") # <-- Изменённый эндпоинт
async def init_stage2(
    request: Request,
    folder_name: str = Form(...) # Имя папки сессии, например, "photo_1_2025-04-12_12-16-42"
):
    """
    Initializes stage 2 from a saved session directory.
    Loads all_contours.npy, creates an image with drawn contours, returns session_id and preview.
    """
    print_debug(f"[DEBUG] /init_stage2 called for folder: {folder_name}")
    try:
        # --- ИСПОЛЬЗУЕМ СЕРВИС (теперь ищет в STAGE2_DIR или RESULTS_DIR) ---
        all_contours_info, main_image_rgb = load_contours_and_image_from_folder(folder_name)
        # --- /ИСПОЛЬЗУЕМ СЕРВИС ---
        
        # --- ИСПОЛЬЗУЕМ СЕРВИС ---
        img_with_contours = create_image_with_contours(main_image_rgb, all_contours_info)
        # --- /ИСПОЛЬЗУЕМ СЕРВИС ---

        # --- ИСПОЛЬЗУЕМ СЕРВИС ---
        preview_b64 = image_to_base64_png(img_with_contours)
        # --- /ИСПОЛЬЗУЕМ СЕРВИС ---

        # --- ИСПОЛЬЗУЕМ СЕРВИС ---
        stage2_session_id = create_session(
            image_np=img_with_contours, # <-- Изображение с контурами как "исходное"
            orig_name=f"stage2_{folder_name}", # <-- Уникальное имя сессии stage2
            processed_image=img_with_contours, # <-- processed_image тоже с контурами
            config={}, # <-- Можно передать config, если нужно, или пустой, если stage2 не использует препроцессинг
            # --- Поля для stage2 ---
            stage2_source_folder=folder_name, # <-- Указываем исходную папку
            stage2_all_contours_info=all_contours_info, # <-- Сохраняем контуры
            # --- /Поля для stage2 ---
        )
        # --- /ИСПОЛЬЗУЕМ СЕРВИС ---

        print_debug(f"[DEBUG] Stage 2 initialized for folder {folder_name}, session ID: {stage2_session_id}")
        return {
            "session_id": stage2_session_id,
            "preview_b64": preview_b64,
            # Опционально: возвратить список имён контуров
            # "contour_names": [info.get("type", "unknown") for info in all_contours_info]
        }

    except FileNotFoundError as e:
        print_debug(f"[ERROR] /init_stage2: {e}")
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        print_debug(f"[ERROR] /init_stage2: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        print_debug(f"[ERROR] /init_stage2: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error during stage 2 initialization: {str(e)}")


