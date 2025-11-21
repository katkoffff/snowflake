# backend/routers/control.py
"""
Контроль:
- /reset — Form(...), submode, reset через сервисы
- /clear_session_dir — Form(...), вызов clear_session_dir_content()
"""

from fastapi import APIRouter, Form, HTTPException
from services.session_service import (
    get_session,
    reset_interactive, reset_main_refinement, reset_inner_refinement,
    clear_session_dir_content
)
from utils.debug import print_debug


router = APIRouter()


@router.post("/reset")
async def reset_session(session_id: str = Form(...), submode: str = Form(...)):
    print_debug(f"/reset called for session: {session_id}")

    try:
        get_session(session_id)
    except KeyError:
        print_debug(f"[ERROR] Session {session_id} not found")
        raise HTTPException(status_code=400, detail="Invalid session_id")

    reset_interactive(session_id)

    if submode == "main":
        reset_main_refinement(session_id)
    elif submode == "inner":
        reset_inner_refinement(session_id)
    else:
        reset_main_refinement(session_id)
        reset_inner_refinement(session_id)

    print_debug("/reset completed successfully")
    return {"status": "ok"}


@router.post("/clear_session_dir")
async def clear_session_dir(session_id: str = Form(...)):
    """
    Очищает содержимое папки сессии.
    НЕ удаляет саму папку.
    Удаляет сессию из памяти.
    """
    try:
        result = clear_session_dir_content(session_id)
        return result
    except Exception:
        raise HTTPException(status_code=500, detail="Failed to clear directory")