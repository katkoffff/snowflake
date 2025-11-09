// Тип для одной маски из автогенерации
export interface AutoGenMask {
  id: string; // Уникальный ID для идентификации
  segmentation: [number, number][][]; // Контур маски, как возвращается из backend
  area: number;
  bbox: [number, number, number, number]; // [x, y, width, height]
  predicted_iou: number;
  stability_score: number;
  // Можно добавить и другие поля из backend, если понадобятся
}

// Тип для настроек автогенерации
export interface AutoGenConfig {
  points_per_side: number;
  points_per_batch: number;
  pred_iou_thresh: number;
  stability_score_thresh: number;
  stability_score_offset: number;
  crop_n_layers: number;
  box_nms_thresh: number;
  crop_n_points_downscale_factor: number;
  min_mask_region_area: number;
  use_m2m: boolean;
  // Добавь другие настройки, которые планируешь использовать
}

// Тип для подтверждённой/финальной маски (возможно, с меткой)
export interface ConfirmedMask extends AutoGenMask {
  type: 'main' | 'slave'; // или используй ID main-маски, если slave привязан к main
}