# backend/services/sam2_service.py
"""
Сервис работы с SAM2.
Содержит:
- Глобальные модели (один раз при старте)
- set_image() — для интерактивного режима
- predict() — по точкам (обычный, main, inner)
- generate_automasks() — с кастомными параметрами
- predict_with_box() — для inner refinement
"""

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from core.config import (
    CHECKPOINT_PATH, MODEL_CFG, device,
    autocast_dtype, autocast_device_type,
    DEFAULT_AUTOGEN_CFG, validate_autogen_cfg
)
from utils.debug import print_debug
import torch
import numpy as np
from typing import List, Tuple, Dict, Any


# ==================================================================
# 1. ГЛОБАЛЬНЫЕ МОДЕЛИ — ОДИН РАЗ ПРИ СТАРТЕ
# ==================================================================

def initialize_sam2():
    """Явно инициализирует модели SAM2 и возвращает объекты"""
    print_debug("Loading SAM2 model...")
    sam2_model = build_sam2(MODEL_CFG, CHECKPOINT_PATH, device=device, apply_postprocessing=False)
    predictor = SAM2ImagePredictor(sam2_model)
    
    print_debug("SAM2 model loaded successfully.")
    
    return {        
        'sam2_model': sam2_model,
        'predictor': predictor,         
    }

# ==================================================================
# 2. УСТАНОВКА ИЗОБРАЖЕНИЯ (для интерактивного режима)
# ==================================================================
def set_image(predictor: SAM2ImagePredictor, image_np: np.ndarray):
    """
    Устанавливает изображение в predictor.
    Используется в /init, /update_settings.
    """
    with torch.inference_mode(), torch.autocast(autocast_device_type, dtype=autocast_dtype):
        predictor.set_image(image_np)
    print_debug("Image set in predictor")


# ==================================================================
# 3. ПРЕДСКАЗАНИЕ ПО ТОЧКАМ (обычный / main / inner)
# ==================================================================
def predict_with_points(
    predictor: SAM2ImagePredictor,    
    points: List[List[float]],
    labels: List[int],
    multimask_output: bool = False,
    mask_input: np.ndarray | None = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    kwargs = {
        "point_coords": np.array([points]),
        "point_labels": np.array([labels]),
        "multimask_output": multimask_output,
    }
    if mask_input is not None:
        kwargs["mask_input"] = mask_input[None, :, :]

    with torch.inference_mode(), torch.autocast(autocast_device_type, dtype=autocast_dtype):
        masks, scores, logits = predictor.predict(**kwargs)

    return masks, scores, logits


# ==================================================================
# 4. ПРЕДСКАЗАНИЕ ПО БОКСУ (для inner refinement)
# ==================================================================
def predict_with_box(
    predictor: SAM2ImagePredictor,
    points: List[List[float]],
    labels: List[int],    
    box_xyxy: List[float],  # [x1, y1, x2, y2]
    multimask_output: bool = False,
    mask_input: np.ndarray | None = None
) -> Tuple[np.ndarray, float, np.ndarray]:    
    kwargs = {
                "point_coords": [points],
                "point_labels": [labels],
                "box": box_xyxy[None, :],
                "multimask_output": multimask_output,
            }
    if mask_input is not None:
        kwargs["mask_input"] = mask_input[None, :, :]
    
    with torch.inference_mode(), torch.autocast(autocast_device_type, dtype=autocast_dtype):
        masks, scores, logits = predictor.predict(**kwargs)

    return masks, scores, logits


# ==================================================================
# 5. АВТОМАТИЧЕСКАЯ ГЕНЕРАЦИЯ МАСОК
# ==================================================================
def generate_automasks(
    sam2_model,
    image_np: np.ndarray,
    cfg: Dict[str, Any] | None = None
) -> List[Dict[str, Any]]:
    """
    Генерирует маски с кастомными параметрами.
    Всегда создаёт новый генератор с нужными настройками.
    """
    cfg = validate_autogen_cfg(cfg or {})
    print_debug(f"Generating automasks with config: {cfg}")

    # Создаём генератор с кастомными параметрами
    generator = SAM2AutomaticMaskGenerator(
        model=sam2_model,
        points_per_side=cfg["points_per_side"],
        points_per_batch=cfg["points_per_batch"],
        pred_iou_thresh=cfg["pred_iou_thresh"],
        stability_score_thresh=cfg["stability_score_thresh"],
        stability_score_offset=cfg["stability_score_offset"],
        crop_n_layers=cfg["crop_n_layers"],
        box_nms_thresh=cfg["box_nms_thresh"],
        crop_n_points_downscale_factor=cfg["crop_n_points_downscale_factor"],
        min_mask_region_area=cfg["min_mask_region_area"],
        use_m2m=cfg["use_m2m"],
    )

    with torch.inference_mode(), torch.autocast(autocast_device_type, dtype=autocast_dtype):
        masks = generator.generate(image_np)

    print_debug(f"Generated {len(masks)} automasks")
    return masks