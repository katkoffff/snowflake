// frontend\src\components\Sidebar.tsx
import React, { useRef, useEffect } from "react";
import { api } from "../api/client";
import { useApp } from "../hooks/useAppContext"; // Убедись, что путь к хуку правильный
// --- ИМПОРТ ТИПА ---
import type { AutoGenConfig } from "../types/autogen";
import "../css/sidebar.css";

export default function Sidebar() {
  const fileInput = useRef<HTMLInputElement | null>(null);
  const {
    // --- Существующие ---
    sessionId, setSessionId,
    setPreview, setOverlay, setContours,
    setLoading, setRefreshGallery,
    medianKsize, setMedianKsize,
    contrastFactor, setContrastFactor,
    sharpnessFactor, setSharpnessFactor,
    claheClipLimit, setClaheClipLimit,
    claheTileGrid, setClaheTileGrid,
    setPoints,

    // --- Новые для автогенерации ---
    mode, setMode,
    autoGenConfig, setAutoGenConfig,
    setAutoMasks, setSelectedMaskIds, setConfirmedMasks, setMainMaskId,
    setSelectionConfirmed, setRefinementCompleted,
    // УБРАНО: autoMasks // <-- Не деструктурируем autoMasks здесь, если не используем напрямую
  } = useApp();

  const isInteractive = mode === 'interactive';
  const isAuto = mode === 'auto';

  // --- Upload ---
  const handleFileSelect = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const f = e.target.files?.[0];
    if (!f) return;
    const fd = new FormData();
    fd.append("file", f);
    setLoading(true);
    try {
      // Выбираем эндпоинт в зависимости от режима
      const endpoint = mode === 'auto' ? '/init_autogen' : '/init';
      const res = await api.post(endpoint, fd);
      console.log(res)
      setSessionId(res.data.session_id);
      setPreview(`data:image/png;base64,${res.data.preview_b64}`);
      setOverlay(null);
      setContours([]);
      // Обработка used_config (если пришёл)
      if (res.data.used_config) {
        const cfg = res.data.used_config;
        setMedianKsize(cfg.median_ksize ?? 5);
        setContrastFactor(cfg.contrast_factor ?? 1.5);
        setSharpnessFactor(cfg.sharpness_factor ?? 2.0);
        setClaheClipLimit(cfg.clahe_clip_limit ?? 1.5);
        const tileGrid = Array.isArray(cfg.clahe_tile_grid) ? cfg.clahe_tile_grid : [8, 8];
        setClaheTileGrid(tileGrid);
      }

      // Если режим 'auto', сохраняем результаты автогенерации
      if (mode === 'auto' && res.data.auto_masks) {
        setAutoMasks(res.data.auto_masks);
        // Сбрасываем связанные с выбором/подтверждением состояния
        setSelectedMaskIds([]);
        setConfirmedMasks([]);
        setMainMaskId(null);
        setSelectionConfirmed(false);
        setRefinementCompleted(false);
      }

    } catch {
      alert("Init failed");
    } finally {
      setLoading(false);
    }
  };

  // --- Save ---
  const handleSave = async () => {
    if (!sessionId) return alert("No session");
    setLoading(true);
    try {
      const form = new FormData();
      form.append("session_id", sessionId);
      form.append("save", "true");
      const res = await api.post("/segment", form); // Пока используем старый endpoint
      if (res.data.saved?.jpg) {
        alert(`Saved:\n${res.data.saved.jpg}`);
        setRefreshGallery(true);
      }
    } catch {
      alert("Save failed");
    } finally {
      setLoading(false);
    }
  };

  // --- Reset ---
  const handleReset = async () => {
    if (!sessionId) return;
    const form = new FormData();
    form.append("session_id", sessionId);
    await api.post("/reset", form);
    setOverlay(null);
    setContours([]);
    setPoints([]); // <-- очищаем точки тоже
    // Сбрасываем состояния автогенерации
    setAutoMasks(null);
    setSelectedMaskIds([]);
    setConfirmedMasks([]);
    setMainMaskId(null);
    setSelectionConfirmed(false);
    setRefinementCompleted(false);
  };

  // --- Update Settings (для препроцессинга) ---
  const handlePreprocessingSettingChange = async (key: string, value: number) => {
    // локально обновляем значение
    switch (key) {
      case "medianKsize":
        setMedianKsize(value);
        break;
      case "contrastFactor":
        setContrastFactor(value);
        break;
      case "sharpnessFactor":
        setSharpnessFactor(value);
        break;
      case "claheClipLimit":
        setClaheClipLimit(value);
        break;
      case "claheTileGrid":
        setClaheTileGrid([value, value]);
        break;
    }

    // отправляем все значения на сервер (только в интерактивном режиме?)
    // Пока оставим как было, но можно добавить проверку mode
    if (sessionId && mode === 'interactive') { // Пример проверки
      setLoading(true);
      try {
        const form = new FormData();
        form.append("session_id", sessionId);
        form.append("median_ksize", String(value === medianKsize ? value : medianKsize));
        form.append("contrast_factor", String(value === contrastFactor ? value : contrastFactor));
        form.append("sharpness_factor", String(value === sharpnessFactor ? value : sharpnessFactor));
        form.append("clahe_clip_limit", String(value === claheClipLimit ? value : claheClipLimit));
        form.append("clahe_tile_grid", claheTileGrid.join(","));

        const res = await api.post("/update_settings", form);
        if (res.data.preview_b64) {
          setPreview(`image/png;base64,${res.data.preview_b64}`);
          setOverlay(null);
        }
      } catch (err) {
        console.error(err);
        alert("Settings update failed");
      } finally {
        setLoading(false);
      }
    }
  };

  // --- Update AutoGen Settings ---
  //const handleAutoGenSettingChange = (key: keyof AutoGenConfig, value: any) => { // <-- Тип 'any' для value можно уточнить позже, если нужно
  //  setAutoGenConfig(prev => ({ // <-- Тип 'prev' теперь выводится из типа setAutoGenConfig
  //    ...prev,
  //    [key]: value
  //  }));
  //};

  const handleAutoGenSettingChange = <K extends keyof AutoGenConfig>(
    key: K,
    value: AutoGenConfig[K]
  ) => {
    setAutoGenConfig((prev: AutoGenConfig) => ({
      ...prev,
      [key]: value
    }));
  };

  // --- Перегенерация при изменении настроек (в режиме 'auto') ---
  useEffect(() => {
    if (mode === 'auto' && sessionId && autoGenConfig) {
      const regenerate = async () => {
        setLoading(true);
        try {
          const form = new FormData();
          form.append("session_id", sessionId);
          // Сериализуем config в JSON строку
          form.append("config", JSON.stringify(autoGenConfig));

          const res = await api.post("/update_autogen", form);
          if (res.data.auto_masks) {
            setAutoMasks(res.data.auto_masks);
            // Сбрасываем выбор, если маски изменились
            setSelectedMaskIds([]);
            setConfirmedMasks([]);
            setMainMaskId(null);
            setSelectionConfirmed(false);
            setRefinementCompleted(false);
          }
        } catch (err) {
          console.error("Autogen update failed:", err);
          // alert("Autogen update failed"); // Опционально
        } finally {
          setLoading(false);
        }
      };

      // Используем setTimeout, чтобы дать время другим обновлениям состояния завершиться
      const timer = setTimeout(regenerate, 0);
      return () => clearTimeout(timer);
    }
    // УБРАНО: autoMasks из зависимостей, чтобы избежать цикла
    // Зависимости: autoGenConfig (или конкретные поля, если не нужно при каждом изменении)
    // Пока реагируем на любое изменение autoGenConfig
  }, [autoGenConfig, mode, sessionId, setLoading, setAutoMasks, setSelectedMaskIds, setConfirmedMasks, setMainMaskId, setSelectionConfirmed, setRefinementCompleted]);

  return (
    <div className="sidebar flex flex-col gap-3 p-4">
      <input type="file" ref={fileInput} className="hidden" onChange={handleFileSelect} />

      {/* Переключатель режима */}
      <div className="mode-selector mb-2">
        <label className="block text-sm font-medium text-gray-700">Mode</label>
        <select
          value={mode}
          onChange={(e) => setMode(e.target.value as 'interactive' | 'auto')}
          className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-indigo-500 focus:ring-indigo-500 sm:text-sm"
        >
          <option value="interactive">Interactive</option>
          <option value="auto">Autogenerate</option>
        </select>
      </div>

      <button className="btn-primary" onClick={() => fileInput.current?.click()}>
        Upload
      </button>
      <button className="btn-secondary" onClick={handleSave}>
        Save
      </button>
      <button className="btn-secondary" onClick={() => setRefreshGallery(true)}>
        Results
      </button>
      <button className="btn-danger" onClick={handleReset}>
        Reset
      </button>

      {/* Условный рендеринг блока настроек */}
      {mode === 'interactive' ? (
        <>
          {/* Старые настройки препроцессинга */}
          <div className="settings mt-3 space-y-2">
            <div className="setting">
              <label>Median ksize: {medianKsize}</label>
              <input type="range" min="1" max="11" step="2"
                     value={medianKsize}
                     onChange={(e) => handlePreprocessingSettingChange("medianKsize", parseInt(e.target.value))}
                     disabled={!!sessionId && isAuto} // <-- Сравнение теперь корректно
                     />
            </div>
            <div className="setting">
              <label>Contrast: {contrastFactor.toFixed(1)}</label>
              <input type="range" min="0.5" max="3" step="0.1"
                     value={contrastFactor}
                     onChange={(e) => handlePreprocessingSettingChange("contrastFactor", parseFloat(e.target.value))}
                     disabled={!!sessionId && isAuto} // <-- Сравнение теперь корректно
                     />
            </div>
            <div className="setting">
              <label>Sharpness: {sharpnessFactor.toFixed(1)}</label>
              <input type="range" min="0.5" max="5" step="0.1"
                     value={sharpnessFactor}
                     onChange={(e) => handlePreprocessingSettingChange("sharpnessFactor", parseFloat(e.target.value))}
                     disabled={!!sessionId && isAuto} // <-- Сравнение теперь корректно
                     />
            </div>
            <div className="setting">
              <label>CLAHE Clip Limit: {claheClipLimit.toFixed(1)}</label>
              <input type="range" min="1" max="10" step="0.1"
                     value={claheClipLimit}
                     onChange={(e) => handlePreprocessingSettingChange("claheClipLimit", parseFloat(e.target.value))}
                     disabled={!!sessionId && isAuto} // <-- Сравнение теперь корректно
                     />
            </div>
            <div className="setting">
              <label>CLAHE Grid: {claheTileGrid[0]}</label>
              <input type="range" min="4" max="16" step="1"
                     value={claheTileGrid[0]}
                     onChange={(e) => handlePreprocessingSettingChange("claheTileGrid", parseInt(e.target.value))}
                     disabled={!!sessionId && isAuto} // <-- Сравнение теперь корректно
                     />
            </div>
          </div>
        </>
      ) : (
        <>
          {/* Новые настройки автогенерации */}
          <div className="autogen-settings mt-3 space-y-2">
            <div className="setting">
              <label>Points per side: {autoGenConfig.points_per_side}</label>
              <input type="range" min="16" max="64" step="16"
                     value={autoGenConfig.points_per_side}
                     onChange={(e) => handleAutoGenSettingChange("points_per_side", parseInt(e.target.value))}
                     disabled={!!sessionId && isInteractive} // <-- Сравнение теперь корректно
                     />
            </div>
            <div className="setting">
              <label>Pred IOU thresh: {autoGenConfig.pred_iou_thresh.toFixed(2)}</label>
              <input type="range" min="0.5" max="1.0" step="0.05"
                     value={autoGenConfig.pred_iou_thresh}
                     onChange={(e) => handleAutoGenSettingChange("pred_iou_thresh", parseFloat(e.target.value))}
                     disabled={!!sessionId && isInteractive} // <-- Сравнение теперь корректно
                     />
            </div>
            <div className="setting">
              <label>Stability score thresh: {autoGenConfig.stability_score_thresh.toFixed(2)}</label>
              <input type="range" min="0.5" max="1.0" step="0.05"
                     value={autoGenConfig.stability_score_thresh}
                     onChange={(e) => handleAutoGenSettingChange("stability_score_thresh", parseFloat(e.target.value))}
                     disabled={!!sessionId && isInteractive} // <-- Сравнение теперь корректно
                     />
            </div>
            <div className="setting">
              <label>Min area: {autoGenConfig.min_mask_region_area}</label>
              <input type="range" min="10" max="500" step="10"
                     value={autoGenConfig.min_mask_region_area}
                     onChange={(e) => handleAutoGenSettingChange("min_mask_region_area", parseFloat(e.target.value))}
                     disabled={!!sessionId && isInteractive} // <-- Сравнение теперь корректно
                     />
            </div>
            <div className="setting">
              <label>
                <input
                  type="checkbox"
                  checked={autoGenConfig.use_m2m}
                  onChange={(e) => handleAutoGenSettingChange("use_m2m", e.target.checked)}
                  disabled={!!sessionId && isInteractive} // <-- Сравнение теперь корректно
                /> Use M2M
              </label>
            </div>
            {/* Добавь другие настройки по аналогии */}
          </div>
        </>
      )}
    </div>
  );
}