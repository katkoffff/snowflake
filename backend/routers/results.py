# backend/routers/results.py
"""
Результаты:
- /results/list — пагинация, final.jpg, mtime
- /results/list_images_in_dir — пагинация .jpg
- /results/image_in_dir — FileResponse
- /save_to_stage2
"""

from fastapi import APIRouter, Query, HTTPException, Form
from fastapi.responses import FileResponse
from schemas.request import SaveToStage2Request
from services.file_service import copy_to_stage2
from core.config import RESULTS_DIR
from pathlib import Path
import os
import time
from services.file_service import ensure_snowflake_dir
from services.session_service import get_session
import cv2
import numpy as np

router = APIRouter()


@router.get("/results/list")
def list_results_dirs(page: int = Query(1, ge=1), per_page: int = Query(12, ge=1, le=200)):
    """
    Показывает список папок сессий.
    """
    all_dirs = sorted([d for d in RESULTS_DIR.iterdir() if d.is_dir()], key=os.path.getmtime, reverse=True)
    total = len(all_dirs)
    start = (page - 1) * per_page
    end = start + per_page
    selected_dirs = all_dirs[start:end]

    dirs_info = []
    for d in selected_dirs:
        # Ищем файл final.jpg в папке
        final_img_path = d / "final.jpg"
        preview_img = None
        if final_img_path.exists():
            preview_img = final_img_path.name # Имя файла, которое можно использовать как "превью"
        # Можешь включить и другие файлы, если хочешь, но preview_img покажет, есть ли финальный результат
        dirs_info.append({
            "name": d.name, # Имя папки
            "preview_image": preview_img, # Имя финального изображения (если есть)
            "mtime": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(d.stat().st_mtime)),
            "path": str(d.relative_to(RESULTS_DIR)) # Относительный путь к папке, если понадобится
        })

    return {
        "page": page,
        "per_page": per_page,
        "total": total,
        "results": dirs_info # <-- Теперь список папок, а не файлов
    }


@router.get("/results/list_images_in_dir")
def list_images_in_dir(
    dir_name: str = Query(...), # Имя папки сессии
    page: int = Query(1, ge=1),
    per_page: int = Query(50, ge=1, le=200) # Много изображений может быть
    ):
    """
    Возвращает список изображений (.jpg) из указанной папки сессии.
    """
    dir_path = RESULTS_DIR / dir_name
    if not dir_path.exists() or not dir_path.is_dir():
        raise HTTPException(status_code=404, detail="Directory not found")

    all_files = sorted([f for f in dir_path.iterdir() if f.is_file() and f.suffix.lower() == ".jpg"],
                       key=os.path.getmtime, reverse=True) # Сортируем по времени, можно по имени
    total = len(all_files)
    start = (page - 1) * per_page
    end = start + per_page
    selected_files = all_files[start:end]

    files_info = []
    for f in selected_files:
        files_info.append({
            "name": f.name,
            "size_kb": round(f.stat().st_size / 1024, 1),
            "mtime": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(f.stat().st_mtime)),
        })

    return {
        "dir_name": dir_name,
        "page": page,
        "per_page": per_page,
        "total": total,
        "images": files_info
    }


@router.get("/results/image_in_dir")
def get_image_in_dir(
    dir_name: str = Query(...), # Имя папки сессии
    file_name: str = Query(...)  # Имя файла изображения
    ):
    """
    Возвращает изображение из указанной папки сессии.
    """
    img_path = RESULTS_DIR / dir_name / file_name
    if not img_path.exists():
        raise HTTPException(status_code=404, detail="Image not found in directory")

    return FileResponse(img_path)


@router.post("/results/save_to_stage2")
async def save_to_stage2(request: SaveToStage2Request):
    """
    Сохраняет снежинку в STAGE2.
    Использует file_service.copy_to_stage2.
    """
    folder_name = request.folder_name.strip()
    if not folder_name:
        raise HTTPException(status_code=400, detail="folder_name is required")

    try:
        path = copy_to_stage2(folder_name)
        return {
            "message": f"Снежинка '{folder_name}' сохранена в stage2",
            "path": path,
            "copied_files": ["final.jpg", "all_contours.npy"]
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to copy files: {str(e)}")
    
@router.post("/save_all")
async def save_all(session_id: str = Form(...)):
    try:
        sess = get_session(session_id)
    except KeyError:
        raise HTTPException(status_code=400, detail="Invalid session_id")
    dir_path = ensure_snowflake_dir(session_id, sess["orig_name"])

    img = sess["image"].copy()

    all_contours_with_info = [] # <-- НОВОЕ: Список для хранения всех контуров и их типа

    # draw main
    main_mask_path = dir_path / "main_mask.npy" # <-- Явно указываем путь
    if main_mask_path.exists():
        main_mask = np.load(str(main_mask_path))
        cnt_img = (main_mask * 255).astype(np.uint8)
        cnts, _ = cv2.findContours(cnt_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            pts = np.array([[int(p[0][0]), int(p[0][1])] for p in c], dtype=np.int32)
            cv2.polylines(img, [pts], True, (255, 0, 0), 2) # Красный для main
            # --- НОВОЕ: Сохраняем контур ---
            all_contours_with_info.append({
                "type": "main",
                "contour": pts.tolist() # .tolist() для сериализации в JSON-compatible формат, если будем использовать json.dump потом
            })
            # --- /НОВОЕ ---

    # draw inner slaves
    slave_files = sorted([f for f in os.listdir(dir_path) if f.endswith("_mask.npy") and f.startswith("slave_")])
    for sf in slave_files:
        mask_path = dir_path / sf # <-- Явно указываем путь к файлу маски
        mask = np.load(str(mask_path))
        cnt_img = (mask * 255).astype(np.uint8)
        cnts, _ = cv2.findContours(cnt_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            pts = np.array([[int(pt[0][0]), int(pt[0][1])] for pt in c], dtype=np.int32)
            cv2.polylines(img, [pts], True, (255, 0, 0), 2) # Жёлтый для slave
            # --- НОВОЕ: Сохраняем контур ---
            # Извлекаем номер слейва из имени файла
            # sf = "slave_1_mask.npy" -> prefix = "slave_1"
            prefix = sf.replace("_mask.npy", "")
            all_contours_with_info.append({
                "type": prefix, # e.g., "slave_1", "slave_2", ...
                "contour": pts.tolist()
            })
            # --- /НОВОЕ ---

    out_jpg = dir_path / "final.jpg"
    cv2.imwrite(str(out_jpg), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    # --- НОВОЕ: Сохраняем все контуры в один .npy файл ---
    out_contours_npy = dir_path / "all_contours.npy"
    # all_contours_with_info - это список словарей, где каждый словарь содержит "type" и "contour"
    # np.save сохраняет Python-объекты, если они совместимы (списки, словари, числа, numpy массивы)
    # contours (pts) уже конвертированы в list через .tolist()
    np.save(str(out_contours_npy), all_contours_with_info)
    # --- /НОВОЕ ---

    return {"final": str(out_jpg)} # <-- Возвращаем путь к новому файлу