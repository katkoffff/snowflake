// frontend\src\components\Sidebar.tsx
import React, { useRef, useEffect } from "react";
import { api } from "../api/client";
import { useApp } from "../hooks/useAppContext"; // Убедись, что путь к хуку правильный
// --- ИМПОРТ ТИПА ---
import type { AutoGenConfig } from "../types/autogen";
import AnalysisModal from "../modals/AnalysisModal";
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
    interactiveSubMode,
    setInnerContours,
    setInnerBox,
    setHoverContour,
    selectedInnerContours, setSelectedInnerContours,
    isUpdatingSettings, setIsUpdatingSettings,
    // --- НОВОЕ: Переменные для AnalysisModal ---
    //isAnalysisModalOpen,
     setIsAnalysisModalOpen,    
    // --- /НОВОЕ ---
     currentStage, setCurrentStage,
  } = useApp();

  const isInteractive = mode === 'interactive';
  const isAuto = mode === 'auto';

  // --- Определяем состояние кнопок ---
  const finalizeBtnDisabled = mode !== 'interactive' || !sessionId;
  const clearDirBtnDisabled = !sessionId;

  // --- Upload ---
  const handleFileSelect = async (e: React.ChangeEvent<HTMLInputElement>) => {
    console.log("load")
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
      setPoints([]); // <-- очищаем точки тоже
      if (mode === 'interactive') {    
        if (interactiveSubMode === 'main') {
          setContours([]);          
        };
        if (interactiveSubMode === 'inner') {
          setInnerBox(null);
          setInnerContours([]);
          setSelectedInnerContours([]);
        };
      };
    
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
        setContours([]);
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

    if (selectedInnerContours.length > 0) {
      form.append("save", "true");
      form.append("selected", selectedInnerContours.join(","));
      const res = await api.post("/segment_inner", form);
      alert(`Saved ${res.data.saved.count} contours`);
      setRefreshGallery(true);
      return;
    }

    // fallback main
    form.append("save", "true");
    const r = await api.post("/segment_main", form);
    alert(`Saved Main on path ${r.data.saved}`);
    setRefreshGallery(true);

  } finally {
    setLoading(false);
  }
};


  // --- Reset ---
  const handleReset = async () => {
    if (!sessionId) return;
    const form = new FormData();
    form.append("session_id", sessionId);
    form.append("submode", interactiveSubMode);
    await api.post("/reset", form);
    setOverlay(null);    
    setPoints([]); // <-- очищаем точки тоже
    if (mode === 'interactive') {    
      if (interactiveSubMode === 'main') {
        setContours([]);
      };
      if (interactiveSubMode === 'inner') {
        setInnerContours([]);
        setSelectedInnerContours([]);
      };
    };
    // Сбрасываем состояния автогенерации
    if (mode === 'auto') {
      setContours([]);
      setAutoMasks(null);
      setSelectedMaskIds([]);
      setConfirmedMasks([]);
      setMainMaskId(null);
      setSelectionConfirmed(false);
      setRefinementCompleted(false);
    };  
  };

  // --- Finalize ---
  const handleFinalize = async () => {
    if (!sessionId) return alert("No session to finalize.");
    setLoading(true);
    try {
      const form = new FormData();
      form.append("session_id", sessionId);
      const res = await api.post("/save_all", form);
      alert(`Final image saved: ${res.data.final}`);
      // Опционально: обновить галерею, если финальное изображение туда попадает
      setRefreshGallery(true);
    } catch (err) {
      console.error("Finalize failed:", err);
      alert("Finalize failed");
    } finally {
      setLoading(false);
    }
  };
  
  // --- Clear Session Directory ---
  const handleClearDir = async () => {
    if (!sessionId) return alert("No session to clear.");
    const confirmed = window.confirm("Are you sure you want to delete all files in the session directory?");
    if (!confirmed) return;

    setLoading(true);
    try {
      const form = new FormData();
      form.append("session_id", sessionId);
      // --- ПОЛУЧАЕМ ОТВЕТ ---
      const res = await api.post("/clear_session_dir", form);
      console.log("[DEBUG] Clear dir response:", res.data); // <-- Для отладки
      alert("Session directory cleared.");

      // --- СБРОС ВСЕХ СОСТОЯНИЙ ПОСЛЕ УСПЕШНОГО ОТВЕТА ---
      setPreview(null);
      setOverlay(null);
      setContours([]);
      setPoints([]);
      // Сброс состояний автогенерации
      setAutoMasks(null);
      setSelectedMaskIds([]);
      setConfirmedMasks([]);
      setMainMaskId(null);
      setSelectionConfirmed(false);
      setRefinementCompleted(false);
      // Сброс состояний inner
      setInnerBox(null);
      setInnerContours([]);
      setSelectedInnerContours([]);
      setHoverContour(null);
      // Сброс других, если есть
      setSessionId(null); // <-- Сбрасываем session_id ПОСЛЕ УСПЕШНОГО ОТВЕТА
      setRefreshGallery(true);

    } catch (err) {
      console.error("Clear directory failed:", err);
      // --- СБРОС ВСЕХ СОСТОЯНИЙ ТАКЖЕ И В СЛУЧАЕ ОШИБКИ ---
      // Если бэкенд удалил сессию (или файлы), но что-то пошло не так,
      // лучше сбросить фронтенд, чтобы избежать несогласованности.
      setPreview(null);
      setOverlay(null);
      setContours([]);
      setPoints([]);
      setAutoMasks(null);
      setSelectedMaskIds([]);
      setConfirmedMasks([]);
      setMainMaskId(null);
      setSelectionConfirmed(false);
      setRefinementCompleted(false);
      setInnerBox(null);
      setInnerContours([]);
      setSelectedInnerContours([]);
      setHoverContour(null);
      setSessionId(null); // <-- Сбрасываем session_id даже при ошибке
      setRefreshGallery(true);
      // --- /Сброс ---
      alert("Clear directory failed. Session state reset locally.");
    } finally {
      setLoading(false);
    }
  };



  // --- Update Settings (для препроцессинга) ---
  const handlePreprocessingSettingChange = async (key: string, value: number) => {
    // Проверяем, идёт ли сейчас обновление
    if (isUpdatingSettings) {
      // Игнорируем, если обновление уже в процессе
      return;
    }
    // Обновляем состояние в AppContext
    setIsUpdatingSettings(true);
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
          setPreview(`data:image/png;base64,${res.data.preview_b64}`);
          setOverlay(null);
        }
      } catch (err) {
        console.error(err);
        alert("Settings update failed");
      } finally {
        setLoading(false);
        // ВСЕГДА сбрасываем флаг в AppContext
        setIsUpdatingSettings(false);
      }
    } else {
        // Если sessionId нет или режим не interactive, просто сбрасываем флаг в AppContext
        setIsUpdatingSettings(false);
      }

  };

  const handleAutoGenSettingChange = <K extends keyof AutoGenConfig>(
    key: K,
    value: AutoGenConfig[K]
  ) => {
    setAutoGenConfig((prev: AutoGenConfig) => ({
      ...prev,
      [key]: value
    }));
  };

  // --- Handle Analyze Results ---
  const handleAnalyzeResults = async () => {
    //if (!sessionId) {
    //  alert("No session to analyze. Upload an image first.");
    //  return;
    //}

    setLoading(true);
    try {
      // Вызываем эндпоинт анализа
      const res = await api.post("/analyze_results", new FormData()); // formData пустая, т.к. endpoint принимает только session_id
      console.log("[DEBUG] Analyze results response:", res.data);
      
      if (res.data.results_path) {        
        // Открываем модальное окно анализа
        setIsAnalysisModalOpen(true);
      } else {
        alert("Analyze failed: No results path returned.");
      }
    } catch (err) {
      console.error("Analyze failed:", err);
      alert("Analyze failed");
    } finally {
      setLoading(false);
    }
  };

  const goToStage2 = () => {
    setCurrentStage("stage2"); // Переключаемся на stage2
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
      <input key={`file-input-${sessionId || 'cleared'}`} type="file" ref={fileInput} className="hidden" onChange={handleFileSelect}/>

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

      <button
        className="btn-secondary" // <-- Убираем условный класс btn-disabled, он теперь определяется :disabled
        onClick={handleFinalize}
        disabled={finalizeBtnDisabled} // <-- Это важно для :disabled
      >
        Finalize
      </button>

      <button
        className="btn-secondary" // <-- Убираем условный класс btn-disabled
        onClick={handleClearDir}
        disabled={clearDirBtnDisabled} // <-- Это важно для :disabled
      >
        Clear Dir
      </button>

      {/* --- НОВАЯ КНОПКА: Analyze Results --- */}
      <button
        className="btn-secondary"
        onClick={handleAnalyzeResults}
        //disabled={!sessionId} // Отключена, если нет сессии
      >
        Analyze Results
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
                     disabled={!!sessionId && isAuto || isUpdatingSettings} // <-- Сравнение теперь корректно
                     />
            </div>
            <div className="setting">
              <label>Contrast: {contrastFactor.toFixed(1)}</label>
              <input type="range" min="0.5" max="3" step="0.1"
                     value={contrastFactor}
                     onChange={(e) => handlePreprocessingSettingChange("contrastFactor", parseFloat(e.target.value))}
                     disabled={!!sessionId && isAuto || isUpdatingSettings} // <-- Сравнение теперь корректно
                     />
            </div>
            <div className="setting">
              <label>Sharpness: {sharpnessFactor.toFixed(1)}</label>
              <input type="range" min="0.5" max="5" step="0.1"
                     value={sharpnessFactor}
                     onChange={(e) => handlePreprocessingSettingChange("sharpnessFactor", parseFloat(e.target.value))}
                     disabled={!!sessionId && isAuto || isUpdatingSettings} // <-- Сравнение теперь корректно
                     />
            </div>
            <div className="setting">
              <label>CLAHE Clip Limit: {claheClipLimit.toFixed(1)}</label>
              <input type="range" min="1" max="10" step="0.1"
                     value={claheClipLimit}
                     onChange={(e) => handlePreprocessingSettingChange("claheClipLimit", parseFloat(e.target.value))}
                     disabled={!!sessionId && isAuto || isUpdatingSettings} // <-- Сравнение теперь корректно
                     />
            </div>
            <div className="setting">
              <label>CLAHE Grid: {claheTileGrid[0]}</label>
              <input type="range" min="4" max="16" step="1"
                     value={claheTileGrid[0]}
                     onChange={(e) => handlePreprocessingSettingChange("claheTileGrid", parseInt(e.target.value))}
                     disabled={!!sessionId && isAuto || isUpdatingSettings} // <-- Сравнение теперь корректно
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
      {/* --- НОВАЯ КНОПКА: Перейти к Этапу 2 --- */}
      <button className="btn-secondary" onClick={goToStage2}>
        Go to Stage 2
      </button>
      {/* --- /НОВАЯ КНОПКА --- */}
      {/* --- РЕНДЕРИМ AnalysisModal --- */}
      <AnalysisModal />
    </div>
  );
}
