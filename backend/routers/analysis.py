# backend/routers/analysis.py
"""
Анализ:
- /analyze_results → анализ main/slave, сохраняет в RESULTS_DIR/analysis_results.json
- /analysis_results → возвращает JSON
- /analysis_save_chart → миниатюры, линии, OffsetImage
"""

from fastapi import APIRouter, Form, HTTPException
from services.analysis_service import (
    analyze_all_sessions, read_analysis_results, save_chart, draw_contour_analysis, find_side_branches, get_envelops
)
from schemas.request import SaveChartRequest
from core.config import RESULTS_DIR
from pathlib import Path
import json
from services.geometry import analyze_main_contour
from utils.debug import print_debug
from services.session_service import get_session, update_session
from services.image_service import image_to_base64_png
import numpy as np
from services.image_service import image_to_base64_png
from scipy.signal import savgol_filter
from scipy.interpolate import splprep, splev

router = APIRouter()


@router.post("/analyze_results")
async def analyze_results():
    """
    Анализирует все сессии, сохраняет в analysis_results.json
    """
    data = analyze_all_sessions()

    return data


@router.get("/analysis_results")
async def get_analysis_results():
    """
    Возвращает содержимое analysis_results.json
    """
    try:
        data = read_analysis_results()
        return data
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Analysis results file not found")
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="Failed to read analysis results file")
    except Exception:
        raise HTTPException(status_code=500, detail="Failed to read analysis results file")


@router.post("/analysis_save_chart")
async def analysis_save_chart(request: SaveChartRequest):
    """
    Сохраняет график с миниатюрами.
    """
    try:
        path = save_chart(
            points=[p.model_dump() for p in request.points],
            axes=request.axes.model_dump(),
            miniatures=[m.model_dump() for m in request.miniatures],
            viewport_size=request.viewport_size
        )
        return {"status": "success"}
    except Exception as e:
        return {"status": "error", "message": str(e)}    

@router.post("/find_centroid")
async def find_centroid(session_id: str = Form(...), settings_json: str = Form(None)):
    """
    Performs full contour analysis for the MAIN contour and returns preview_b64 and analysis dict.
    Optional `settings_json` can be passed (stringified JSON with draw settings).
    """
    print_debug(f"[DEBUG] /find_centroid called, session_id={session_id}")
    session = get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    if "stage2_all_contours_info" not in session:
        raise HTTPException(status_code=400, detail="This session has no contours (not a stage2 session).")

    all_contours_info = session["stage2_all_contours_info"]
    image_np = session.get("image")
    if image_np is None:
        raise HTTPException(status_code=500, detail="Session has no image")

    # find main contour
    main = None
    for info in all_contours_info:
        if info.get("type") == "main":
            main = np.array(info["contour"], dtype=np.int32)
            break
    if main is None:
        raise HTTPException(status_code=404, detail="Main contour not found")
    
    # =======================
    # APPLY CONTOUR SMOOTHING
    # =======================

    # Укажи метод: "savgol" или "spline" или None
    contour_smooth_method = "" #"savgol", "spline"

    if contour_smooth_method == "savgol":
        # параметры можно вынести в настройки
        window = 11  # должно быть нечетным
        poly = 3

        x = main[:, 0].astype(float)
        y = main[:, 1].astype(float)

        x_smooth = savgol_filter(x, window_length=window, polyorder=poly)
        y_smooth = savgol_filter(y, window_length=window, polyorder=poly)

        main = np.stack([x_smooth, y_smooth], axis=1).astype(np.int32)

    elif contour_smooth_method == "spline":
        s = 0  
        x = main[:, 0].astype(float)
        y = main[:, 1].astype(float)
        
        try:
            tck, u = splprep([x, y], s=s)
            
            # РАВНОМЕРНАЯ перевыборка
            # Выбираем количество точек (можно сделать равным исходному или больше)
            num_points = len(x) * 1  # или len(x) * 2 для большей детализации
            u_new = np.linspace(0, 1, num_points)
            x_smooth, y_smooth = splev(u_new, tck)
            
            main = np.stack([x_smooth, y_smooth], axis=1).astype(np.int32)
        except Exception as e:
            print(f"Spline smoothing failed: {e}")

    # parse settings if provided
    settings = None
    if settings_json:
        try:
            settings = json.loads(settings_json)
        except Exception as e:
            print_debug(f"Failed to parse settings_json: {e}")
            settings = None

    # analysis
    try:
        analysis = analyze_main_contour(main)
    except Exception as e:
        print_debug(f"Error during analyze_main_contour: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis error: {e}")

    # draw with settings
    try:
        img_out, sectors = draw_contour_analysis(image_np, main, analysis, settings=settings)
        analysis["sectors"] = sectors
        # Глобальная фильтрация всех кандидатов
        # "distance_based"
        # "simple_clustering"
        # "peak_detection"
        # "dbscan"
        # "curvature"
        # "combined"
        # "peak_detection_curvature"
        # "peak_detection_scipy"
        img_out, updated_sectors = find_side_branches(img_out, main, sectors, step_ratio=100, method="peak_detection_scipy")
        analysis["sectors"] = updated_sectors
        analysis["main_contour"] = main
    except Exception as e:
        print_debug(f"Error during draw_contour_analysis: {e}")
        raise HTTPException(status_code=500, detail=f"Drawing error: {e}")

    # save updated image and analysis in session
    update_session(session_id, **{"image": img_out, "stage2_last_analysis": analysis})

    preview_b64 = image_to_base64_png(img_out)
    return {"preview_b64": preview_b64, "session_id": session_id}

@router.post("/calculate_envelop")
async def calculate_envelop(session_id: str = Form(...), settings_json: str = Form(None)):
    """
    Performs full contour analysis for the MAIN contour and returns preview_b64 and analysis dict.
    Optional `settings_json` can be passed (stringified JSON with draw settings).
    """
    print_debug(f"[DEBUG] /calculate_envelop called, session_id={session_id}")
    session = get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    if "stage2_last_analysis" not in session:
        raise HTTPException(status_code=400, detail="This session has no analysis.")
    
    image_np = session.get("image")
    analysis = session.get("stage2_last_analysis")

    # draw with settings
    try:
        
        img_out = get_envelops(image_np, analysis.get("main_contour"), analysis.get("sectors"))
        
    except Exception as e:
        print_debug(f"Error during calculate envelop: {e}")
        raise HTTPException(status_code=500, detail=f"Drawing error: {e}")

    # save updated image and analysis in session
    update_session(session_id, **{"image": img_out})

    preview_b64 = image_to_base64_png(img_out)
    return {"preview_b64": preview_b64, "session_id": session_id}