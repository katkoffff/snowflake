// frontend\src\components\EditorPanel2.tsx
import React, { useRef, useEffect } from "react";
import { useApp } from "../hooks/useAppContext"; // Убедись, что путь правильный
import "../css/editor.css"; // Используем тот же CSS, или создай editor2.css

export default function EditorPanel2() {
  const imgRef = useRef<HTMLImageElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  const {
    // --- Используем только нужные для отображения ---
    sessionId,
    preview, // <-- Это preview_b64 от /init_stage2
    overlay, // <-- Пока не используется, но может понадобиться
    // setLoading, // <-- Скорее всего не нужен в EditorPanel2 напрямую
    // points, setPoints, // <-- Не нужны
    // setContours, // <-- Не нужен
    // interactiveSubMode, // <-- Не нужен
    // setInteractiveSubMode, // <-- Не нужен
    // ... (остальные, не связанные с отображением preview)
  } = useApp();

  // --- Синхронизация canvas с изображением (если нужно рисовать поверх) ---
  useEffect(() => {
    const img = imgRef.current;
    const canvas = canvasRef.current;
    if (!img || !canvas) return;

    const onLoad = () => {
      canvas.width = img.naturalWidth;
      canvas.height = img.naturalHeight;
      canvas.style.width = `${img.width}px`;
      canvas.style.height = `${img.height}px`;
      // drawOverlay(); // <-- Не нужно, если preview уже содержит контуры
    };

    if (img.complete) onLoad();
    else img.addEventListener("load", onLoad);

    return () => img.removeEventListener("load", onLoad);
  }, [preview, overlay]); // <-- Зависит от preview (и overlay, если она меняет размер)

  // --- (Опционально) useEffect для отрисовки поверх (например, сносок) ---
  // useEffect(() => {
  //   const canvas = canvasRef.current;
  //   const img = imgRef.current;
  //   if (!canvas || !img) return;
  //   const ctx = canvas.getContext("2d");
  //   if (!ctx) return;
  //
  //   ctx.clearRect(0, 0, canvas.width, canvas.height);
  //   // ... (логика отрисовки поверх preview, например, линии от точек к миниатюрам, если preview не содержит всё)
  // }, [/* зависимости для перерисовки */]);

  return (
    <div className="editor-panel flex flex-col items-center justify-center w-full h-full bg-gray-100 overflow-auto">
      {!preview ? (
        <div className="text-gray-400 text-lg">
          Load a result folder in Stage 2 to start viewing ❄️
        </div>
      ) : (
        <div className="relative inline-block select-none">
          <img
            ref={imgRef}
            src={overlay || preview || undefined} // <-- Используем preview (с контурами)
            alt="Stage 2 preview"
            className="max-w-[80vw] h-auto cursor-default" // cursor-default, так как нет интерактивности
            // onMouseDown={handleMouseDown} // <-- Убрано
            // onContextMenu={(e) => e.preventDefault()} // <-- Можно оставить, если не хочешь контекстное меню
            draggable={false}
          />
          {/* Canvas для *дополнительных* элементов (например, сносок, если нужно рисовать поверх preview) */}
          {/* Если preview уже содержит *все* контуры, этот canvas может быть не нужен */}
          <canvas
            ref={canvasRef}
            className="absolute left-0 top-0 pointer-events-none"
            style={{ left: 0, top: 0 }}
          />
        </div>
      )}
    </div>
  );
}