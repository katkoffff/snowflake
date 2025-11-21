# backend/routers/analysis.py
"""
Анализ:
- /analyze_results → анализ main/slave, сохраняет в RESULTS_DIR/analysis_results.json
- /analysis_results → возвращает JSON
- /analysis_save_chart → миниатюры, линии, OffsetImage
"""

from fastapi import APIRouter, Form, HTTPException
from services.analysis_service import (
    analyze_all_sessions, read_analysis_results, save_chart, draw_contour_analysis, find_side_branches
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
from services.analysis_service import analyze_branches_with_skeleton, draw_branch_analysis
from services.image_service import image_to_base64_png

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
    except Exception as e:
        print_debug(f"Error during draw_contour_analysis: {e}")
        raise HTTPException(status_code=500, detail=f"Drawing error: {e}")

    # save updated image and analysis in session
    update_session(session_id, **{"image": img_out, "stage2_last_analysis": analysis})

    preview_b64 = image_to_base64_png(img_out)
    return {"preview_b64": preview_b64, "session_id": session_id}


@router.post("/find_branches")
async def find_branches(session_id: str = Form(...), settings_json: str = Form(None)):
    session = get_session(session_id)
    if not session or "stage2_all_contours_info" not in session:
        raise HTTPException(400, "Invalid session")

    image_np = session.get("image")
    if image_np is None:
        raise HTTPException(500, "No image")

    main = next((np.array(info["contour"], dtype=np.int32)
                 for info in session["stage2_all_contours_info"] if info.get("type") == "main"), None)
    if main is None:
        raise HTTPException(404, "Main contour not found")

    settings = json.loads(settings_json) if settings_json else {}

    analysis = analyze_branches_with_skeleton(main, image_np.shape[:2])
    img_out = draw_branch_analysis(image_np, main, analysis, settings)

    update_session(session_id, **{
        "image": img_out,
        "stage2_branch_analysis": analysis
    })

    return {
        "preview_b64": image_to_base64_png(img_out),
        "session_id": session_id
    }