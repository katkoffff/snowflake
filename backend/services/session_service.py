# backend/services/session_service.py
"""
Сервис управления сессиями.
"""

import uuid
from typing import Dict, Any, List, Optional
import numpy as np
from utils.debug import print_debug
import shutil
from services.file_service import ensure_snowflake_dir


_sessions: Dict[str, Dict[str, Any]] = {}


def create_session(
    image_np: np.ndarray,
    orig_name: str,
    processed_image: np.ndarray,
    config: dict,
    auto_masks: Optional[List[dict]] = None,
    stage2_source_folder: Optional[str] = None,
    stage2_all_contours_info: Optional[list] = None,
) -> str:
    session_id = str(uuid.uuid4())
    _sessions[session_id] = {
        "image": processed_image,
        "original_processed_image": image_np,
        "orig_name": orig_name,
        "config": config,
        "auto_masks": auto_masks or [],

        "points": [],
        "logits": None,
        "is_first_click": True,
        "last_mask": None,

        "confirmed_masks": [],
        "final_masks": [],
        "selected_mask_ids": [],
        "main_mask_id": None,
        "selection_confirmed": False,
        "refinement_completed": False,

        "main_points": [],
        "main_logits": None,
        "main_mask": None,
        "last_main_mask_bin": None,
        "last_main_contours": [],

        "inner_box": None,
        "inner_points": [],
        "inner_logits": None,
        "inner_mask": None,
        "last_inner_mask_bin": None,
        "last_inner_contours": [],

        "snowflake_dir": None,

        "stage2_source_folder": stage2_source_folder,
        "stage2_all_contours_info": stage2_all_contours_info,
        "stage2_last_analysis": {}
    }
    print_debug(f"Session created: {session_id} | {orig_name}")
    return session_id


def get_session(session_id: str) -> Dict[str, Any]:
    if session_id not in _sessions:
        raise KeyError(f"Session {session_id} not found")
    return _sessions[session_id]


def update_session(session_id: str, **kwargs):
    sess = get_session(session_id)
    sess.update(kwargs)
    print_debug(f"Session {session_id} updated: {list(kwargs.keys())}")


def reset_interactive(session_id: str):
    sess = get_session(session_id)
    sess.update({
        "points": [],
        "logits": None,
        "is_first_click": True,
        "last_mask": None,
    })
    print_debug(f"Session {session_id} interactive reset")


def reset_main_refinement(session_id: str):
    sess = get_session(session_id)
    sess.update({
        "main_points": [],
        "main_logits": None,
        "main_mask": None,
        "last_main_mask_bin": None,
        "last_main_contours": [],
    })
    print_debug(f"Session {session_id} main refinement reset")


def reset_inner_refinement(session_id: str):
    sess = get_session(session_id)
    sess.update({
        "inner_box": None,
        "inner_points": [],
        "inner_logits": None,
        "inner_mask": None,
        "last_inner_mask_bin": None,
        "last_inner_contours": [],
    })
    print_debug(f"Session {session_id} inner refinement reset")


def reset_selection(session_id: str):
    sess = get_session(session_id)
    sess.update({
        "confirmed_masks": [],
        "final_masks": [],
        "selected_mask_ids": [],
        "main_mask_id": None,
        "selection_confirmed": False,
        "refinement_completed": False,
    })
    print_debug(f"Session {session_id} selection reset")


def reset_all(session_id: str):
    sess = get_session(session_id)
    sess.update({
        "points": [], "logits": None, "is_first_click": True, "last_mask": None,
        "confirmed_masks": [], "final_masks": [], "selected_mask_ids": [],
        "main_mask_id": None, "selection_confirmed": False, "refinement_completed": False,
        "main_points": [], "main_logits": None, "main_mask": None,
        "last_main_mask_bin": None, "last_main_contours": [],
        "inner_box": None, "inner_points": [], "inner_logits": None,
        "inner_mask": None, "last_inner_mask_bin": None, "last_inner_contours": [],
    })
    print_debug(f"Session {session_id} fully reset (except base data)")


def delete_session(session_id: str):
    if session_id in _sessions:
        sess = _sessions[session_id]
        if sess.get("snowflake_dir"):
            try:
                shutil.rmtree(sess["snowflake_dir"])
                print_debug(f"Deleted directory: {sess['snowflake_dir']}")
            except Exception as e:
                print_debug(f"Failed to delete dir: {e}")
        del _sessions[session_id]
        print_debug(f"Session {session_id} deleted")


def clear_session_dir_content(session_id: str):
    """
    Очищает содержимое папки сессии (НЕ удаляет папку).
    Удаляет сессию из памяти.
    """
    sess = _sessions[session_id]
    dir_path = ensure_snowflake_dir(session_id, sess.get("orig_name"))

    if not dir_path.exists():
        delete_session(session_id)
        return {"status": "success", "message": "Directory does not exist."}

    try:
        for item in dir_path.iterdir():
            if item.is_file():
                item.unlink()
            elif item.is_dir():
                shutil.rmtree(item)
        print_debug(f"Cleared directory: {dir_path}")

        delete_session(session_id)
        return {
            "status": "success",
            "message": f"Directory {dir_path} cleared."
        }
    except Exception as e:
        print_debug(f"[ERROR] Failed to clear {dir_path}: {e}")
        raise