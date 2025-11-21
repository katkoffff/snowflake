# backend/routers/segment.py
"""
Сегментация:
- /segment
- /segment_main
- /segment_inner
- /start_inner_box
"""

from fastapi import APIRouter, HTTPException, Form, Request
from services.session_service import get_session, update_session, reset_interactive, reset_main_refinement, reset_inner_refinement
from services.sam2_service import predict_with_points, predict_with_box
from services.image_service import overlay_contour, image_to_base64_png
from services.file_service import ensure_snowflake_dir, save_mask_pack
from utils.debug import print_debug
import numpy as np
import cv2
import os
from core.config import RESULTS_DIR
import time

router = APIRouter()


@router.post("/segment")
async def segment_session(
    request: Request,
    session_id: str = Form(...),
    x: int | None = Form(None),
    y: int | None = Form(None),
    label: int | None = Form(1),
    save: bool | None = Form(False),
):
    try:
        sess = get_session(session_id)
    except KeyError:
        raise HTTPException(status_code=400, detail="Invalid session_id")
    
    image_np = sess["image"]

    if x is not None and y is not None:
        sess["points"].append([int(x), int(y), int(label)])

    if not sess["points"]:
        raise HTTPException(status_code=400, detail="No points provided")
    
    all_coords = [[p[0], p[1]] for p in sess["points"]]
    all_labels = [p[2] for p in sess["points"]]
    
    is_first_click = sess.get("is_first_click", True)
    mask_input = sess.get("logits", None)
    multimask_output = is_first_click

    masks, scores, logits = predict_with_points(
            predictor=request.app.state.sam2_objects['predictor'],
            points=all_coords,
            labels=all_labels,
            multimask_output=multimask_output,
            mask_input=mask_input        )

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
        jpg_path = RESULTS_DIR / f"{base_name}.jpg"        
        segmented_path = RESULTS_DIR / f"{base_name}_segmented.jpg"        
        npy_path = RESULTS_DIR / f"{base_name}.npy"
        logits_path = RESULTS_DIR / f"{base_name}_logits.npy"        
        cv2.imwrite(str(jpg_path), cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))        
        cv2.imwrite(str(segmented_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))        
        np.save(str(npy_path), mask_bin)
        np.save(str(logits_path), sess["logits"])
        saved = {"base": jpg_path, "segmented": segmented_path, "mask": npy_path, "logits": logits_path}        

    return {"session_id": session_id, "contours": contours, "overlay_b64": overlay_b64, "saved": saved}


@router.post("/segment_main")
async def segment_main(
    request: Request,
    session_id: str = Form(...),
    x: int | None = Form(None),
    y: int | None = Form(None),
    label: int = Form(1),
    save: bool = Form(False),
):    
    try:
        sess = get_session(session_id)
    except KeyError:
        raise HTTPException(status_code=400, detail="Invalid session_id")

    image_np = sess["image"]

    # === НАКОПЛЕНИЕ ТОЧЕК ===
    if x is not None and y is not None:
        sess.setdefault("main_points", [])
        sess["main_points"].append([int(x), int(y), int(label)])

    if not sess.get("main_points"):
        raise HTTPException(status_code=400, detail="No points provided")

    pts = sess["main_points"]
    coords = [[p[0], p[1]] for p in pts]
    labels = [p[2] for p in pts]

    mask_input = sess.get("main_logits")
    multimask_output = mask_input is None

    if multimask_output:
        sess["last_main_mask_bin"] = None
        sess["last_main_contours"] = []

    # === ПЕРЕСЧЁТ ИЛИ SAVE ===
    if save:
        mask_bin = sess.get("last_main_mask_bin")
        contours = sess.get("last_main_contours", [])
        logits_to_save = sess.get("main_logits")
        if mask_bin is None or not contours:
            raise HTTPException(status_code=400, detail="No previous segmentation results to save.")
    else:
        # --- ПРЕДСКАЗАНИЕ ---
        masks, scores, logits = predict_with_points(
            predictor=request.app.state.sam2_objects['predictor'],
            points=coords,
            labels=labels,
            multimask_output=multimask_output,
            mask_input=mask_input
        )

        if multimask_output:
            idx = int(np.argmax(scores))
            mask = masks[idx]
            sess["main_logits"] = logits[idx]
        else:
            mask = masks[0]
            sess["main_logits"] = logits[0]

        mask_bin = (mask > 0).astype(np.uint8)
        sess["main_mask"] = mask_bin
        logits_to_save = sess["main_logits"]

        # --- КОНТУРЫ ---
        cnt_img = (mask_bin * 255).astype(np.uint8)
        cnts, _ = cv2.findContours(cnt_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = [[[int(p[0][0]), int(p[0][1])] for p in c] for c in cnts if len(c) > 2]

        sess["last_main_mask_bin"] = mask_bin
        sess["last_main_contours"] = contours

    # === ОВЕРЛЕЙ ===
    overlay = image_np.copy()
    for c in contours:
        cv2.polylines(overlay, [np.array(c)], True, (255, 0, 0), 2)
    overlay_b64 = image_to_base64_png(overlay)

    # === СОХРАНЕНИЕ ===
    saved = {}
    if save:        
        contours_sorted = sorted(contours, key=lambda c: len(c), reverse=True)
        dir_path = ensure_snowflake_dir(session_id, sess["orig_name"])
        save_mask_pack(
            dir_path=dir_path,
            prefix="main",
            image_np=image_np,
            contour=contours_sorted[0] if contours_sorted else None,
            mask_bin=mask_bin,
            logits=logits_to_save
        )
        saved = {"dir": str(dir_path)}

    # === ОБНОВЛЕНИЕ СЕССИИ ===
    update_session(session_id, **sess)

    return {
        "session_id": session_id,
        "contours": contours,
        "overlay_b64": overlay_b64,
        "saved": saved
    }


@router.post("/segment_inner")
async def segment_inner(
    request: Request, 
    session_id: str = Form(...),
    x: int | None = Form(None),
    y: int | None = Form(None),
    label: int = Form(1),
    save: bool = Form(False),
    selected: str | None = Form(None),
):
    try:
        sess = get_session(session_id)
    except KeyError:
        raise HTTPException(status_code=400, detail="Invalid session_id")
    
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
    
    if save and selected:       
        mask_bin = sess.get("last_inner_mask_bin")
        contours = sess.get("last_inner_contours", [])        
        logits_to_save = sess.get("inner_logits")
        if mask_bin is None or not contours:
            raise HTTPException(400, "No previous segmentation results to save. Run segmentation first.")
    else:
        masks, scores, logits = predict_with_box(
                                    predictor=request.app.state.sam2_objects['predictor'],
                                    points=coords,
                                    labels=labels,
                                    box_xyxy=box_xyxy,
                                    multimask_output=multimask_output,
                                    mask_input=mask_input
                                )

        if multimask_output:
            idx = int(np.argmax(scores))
            mask = masks[idx]
            sess["inner_logits"] = logits[idx]
        else:
            mask = masks[0]
            sess["inner_logits"] = logits[0]
        mask_bin = (mask > 0).astype(np.uint8)
        logits_to_save = sess["inner_logits"]         
        cnt_img = (mask_bin * 255).astype(np.uint8)
        cnts, _ = cv2.findContours(cnt_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = []
        for c in cnts:
            if len(c) >= 1: 
                cc = [[int(pt[0][0]), int(pt[0][1])] for pt in c]
                contours.append(cc)
        sess["last_inner_mask_bin"] = mask_bin
        sess["last_inner_contours"] = contours    
    overlay = image_np.copy()
    for c in contours:
        cv2.polylines(overlay, [np.array(c)], True, (0, 255, 255), 2)
    overlay_b64 = image_to_base64_png(overlay)
    saved = {}
    if save and selected:
        sel = [int(s) for s in selected.split(",") if s.strip().isdigit()]
        dir_path = ensure_snowflake_dir(session_id, sess["orig_name"])
        print_debug(f"Total contours available: {len(contours)}, Selected indices: {sel}")        
        current = len([f for f in os.listdir(dir_path) if f.startswith("slave_")]) // 4
        for i, idx in enumerate(sel):
            if idx < 0 or idx >= len(contours): 
                print_debug(f"Index {idx} is out of bounds for contours list of length {len(contours)}. Skipping.")
                continue            
            contour_to_save = contours[idx]            
            single_contour_mask = np.zeros_like(mask_bin, dtype=np.uint8)            
            pts = np.array(contour_to_save, dtype=np.int32)
            cv2.fillPoly(single_contour_mask, [pts], 1) 
            prefix = f"slave_{current + i + 1}"            
            save_mask_pack(
                dir_path,
                prefix,
                image_np,
                contour_to_save, 
                single_contour_mask, 
                logits_to_save, 
            )
        saved = {"count": len(sel)}
    # === ОБНОВЛЕНИЕ СЕССИИ ===
    update_session(session_id, **sess)    
    return {
        "contours": contours,
        "overlay_b64": overlay_b64,
        "saved": saved,
    }

@router.post("/start_inner_box")
async def start_inner_box(
    session_id: str = Form(...),
    x1: int = Form(...),
    y1: int = Form(...),
    x2: int = Form(...),
    y2: int = Form(...),
):
    try:
        sess = get_session(session_id)
    except KeyError:
        raise HTTPException(status_code=400, detail="Invalid session_id")
    
    sess["inner_box"] = [x1, y1, x2, y2]
    sess["inner_points"] = []
    sess["inner_logits"] = None
    sess["last_inner_mask_bin"] = None, # Для хранения последней mask_bin
    sess["last_inner_contours"] = [],   # Для хранения последних contours

    update_session(session_id, **sess) 

    return {"status": "ok", "box": sess["inner_box"]}