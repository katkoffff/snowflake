// frontend/src/components/Sidebar2.tsx
import React, { useRef, useEffect } from "react";
import { api } from "../api/client";
import { useApp } from "../hooks/useAppContext"; // Убедись, что путь правильный
import "../css/sidebar.css"; // Используем тот же CSS, или создай sidebar2.css

export default function Sidebar2() {
  const fileInput = useRef<HTMLInputElement | null>(null);
  const { sessionId, setSessionId,
          setPreview,
          setPoints,
          setLoading, setOverlay, 
          setCurrentStage 
        } = useApp(); // Получаем setCurrentStage из контекста

  const goToStage1 = () => {
    setCurrentStage("stage1"); // Переключаемся обратно на stage1
  };

  // --- Upload ---
    const handleFileSelect = async (e: React.ChangeEvent<HTMLInputElement>) => {      
      const f = e.target.files?.[0];
      if (!f) return;
      const fullPath = f.webkitRelativePath;
      const folderName = fullPath.split('/')[0];
      
      const fd = new FormData();
      fd.append("folder_name", folderName);
      setLoading(true);
      try {
        const endpoint = "/init_stage2"
        const res = await api.post(endpoint, fd);        
        setSessionId(res.data.session_id);
        setPreview(`data:image/png;base64,${res.data.preview_b64}`);
        setOverlay(null);
        setPoints([]);  
      } catch {
        alert("Init failed");
      } finally {
        setLoading(false);
      }
        
    };

  const handleFindCentroid = async () => {
    if (!sessionId) return alert("No session to find centroid for.");

    setLoading(true);
    try {
      const form = new FormData();
      form.append("session_id", sessionId);

      const res = await api.post("/find_centroid", form); // <-- НОВЫЙ эндпоинт
      // Предполагаем, что эндпоинт возвращает обновлённое изображение с точкой
      if (res.data.preview_b64) {
        setPreview(`data:image/png;base64,${res.data.preview_b64}`);
        setOverlay(null); // Сбрасываем overlay, если используется preview
      }
      // Опционально: если эндпоинт возвращает координаты
      // if (res.data.centroid) {
      //   console.log("Centroid found at:", res.data.centroid);
      //   // Можешь сохранить в AppContext, если нужно для других целей
      // }
    } catch (err) {
      console.error("Find centroid failed:", err);
      alert("Find centroid failed");
    } finally {
      setLoading(false);
    }
  };    

  return (
    <div className="sidebar flex flex-col gap-3 p-4">
      <input key={`file-input-${sessionId || 'cleared'}`} type="file" ref={fileInput} className="hidden" webkitdirectory="" onChange={handleFileSelect}/>
      <button className="btn-primary" onClick={() => fileInput.current?.click()}>
        Upload
      </button>      
      <button className="btn-secondary" onClick={handleFindCentroid} disabled={!sessionId}>
        Find Centroid
      </button>
      <button className="btn-secondary" onClick={goToStage1}>
        Back to Stage 1
      </button>      
    </div>
  );
}