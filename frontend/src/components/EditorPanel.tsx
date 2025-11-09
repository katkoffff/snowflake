// frontend/src/components/EditorPanel.tsx
import React, { useRef, useEffect, useState } from "react";
import { api } from "../api/client";
import { useApp } from "../hooks/useAppContext";
import "../css/editor.css";

export default function EditorPanel() {
  const {
    sessionId,
    preview,
    overlay,
    setOverlay,
    setContours,
    setLoading,
    points,
    setPoints,
    interactiveSubMode,
    setInteractiveSubMode,
    innerBox,
    setInnerBox,
    innerContours, setInnerContours,
    selectedInnerContours, setSelectedInnerContours,
    hoverContour, setHoverContour,
    isUpdatingSettings
  } = useApp();

  const imgRef = useRef<HTMLImageElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  // --- inner-режим: управление боксом
  const [drawingBox, setDrawingBox] = useState(false);
  const [boxStart, setBoxStart] = useState<{ x: number; y: number } | null>(
    null
  );
  const [boxEnd, setBoxEnd] = useState<{ x: number; y: number } | null>(null);

  // ------------------------------
  // 🔹 РИСОВАНИЕ OVERLAY
  // ------------------------------
  const drawOverlay = () => {
    const canvas = canvasRef.current;
    const img = imgRef.current;
    if (!canvas || !img) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // MAIN mode — точки
    if (interactiveSubMode === "main") {
      points.forEach((p) => {
        ctx.beginPath();
        ctx.arc(p.x, p.y, 5, 0, Math.PI * 2);
        ctx.fillStyle =
          p.label === 1 ? "rgba(0,200,0,0.95)" : "rgba(0,120,255,0.95)";
        ctx.fill();
      });
    }

    // INNER mode — бокс + точки + контуры
    if (interactiveSubMode === "inner") {
      ctx.lineWidth = 2;

      // постоянный бокс
      if (innerBox) {
        const [x1, y1, x2, y2] = innerBox;
        ctx.strokeStyle = "rgba(255,255,0,1)";
        ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
      }

      // временный бокс
      if (drawingBox && boxStart && boxEnd) {
        ctx.setLineDash([4, 4]);
        ctx.strokeStyle = "rgba(255,255,0,0.7)";
        ctx.strokeRect(
          boxStart.x,
          boxStart.y,
          boxEnd.x - boxStart.x,
          boxEnd.y - boxStart.y
        );
        ctx.setLineDash([]);
      }

      // точки
      if (innerBox) {
        points.forEach((p) => {
          ctx.beginPath();
          ctx.arc(p.x, p.y, 5, 0, Math.PI * 2);
          ctx.fillStyle = p.label === 1 ? "yellow" : "cyan";
          ctx.fill();
        });
      }

      // контуры (если есть)
      innerContours.forEach((c, idx) => {
          const isHovered = hoverContour === idx;
          const isSelected = selectedInnerContours.includes(idx);
          ctx.strokeStyle = isHovered
            ? "rgba(255,0,0,1)"
            : isSelected
            ? "rgba(255,255,255,1)"
            : "rgba(0,255,255,0.7)";
          ctx.beginPath();
          c.forEach(([x, y], i) => {
            if (i === 0) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);
          });
          ctx.closePath();
          ctx.stroke();
        });
      
    }
  };

  // --- инициализация canvas
  useEffect(() => {
    const img = imgRef.current;
    const canvas = canvasRef.current;
    if (!img || !canvas) return;

    const onLoad = () => {
      canvas.width = img.naturalWidth;
      canvas.height = img.naturalHeight;
      canvas.style.width = img.width + "px";
      canvas.style.height = img.height + "px";
      drawOverlay();
    };

    if (img.complete) onLoad();
    else img.addEventListener("load", onLoad);

    return () => img.removeEventListener("load", onLoad);
  }, [preview, overlay]);

  useEffect(() => {
    drawOverlay();
  }, [
    points,
    innerBox,
    drawingBox,
    interactiveSubMode,
    overlay,
    hoverContour,
    selectedInnerContours,
  ]);

  // ------------------------------
  // 🔹 ОБРАБОТКА МЫШИ
  // ------------------------------
  const handleMouseDown = async (e: React.MouseEvent<HTMLImageElement>) => {
    if (!sessionId || !imgRef.current) return;
    e.preventDefault();

    const rect = imgRef.current.getBoundingClientRect();
    const scaleX = imgRef.current.naturalWidth / rect.width;
    const scaleY = imgRef.current.naturalHeight / rect.height;
    const x = Math.round((e.clientX - rect.left) * scaleX);
    const y = Math.round((e.clientY - rect.top) * scaleY);

    // MAIN
    if (interactiveSubMode === "main") {
      const label = e.button === 2 ? 0 : 1;
      setPoints([...points, { x, y, label }]);
      setLoading(true);
      try {
        const f = new FormData();
        f.append("session_id", sessionId);
        f.append("x", String(x));
        f.append("y", String(y));
        f.append("label", String(label));
        const r = await api.post("/segment_main", f);
        setContours(r.data.contours || []);
        if (r.data.overlay_b64)
          setOverlay("data:image/png;base64," + r.data.overlay_b64);
      } finally {
        setLoading(false);
      }
      return;
    }

    // INNER: рисуем бокс ЛКМ
    if (interactiveSubMode === "inner" && !innerBox) {
      if (e.button !== 0) return;
      setDrawingBox(true);
      setBoxStart({ x, y });
      setBoxEnd({ x, y });
      return;
    }

    // INNER: внутри бокса — точки
    if (interactiveSubMode === "inner" && innerBox) {
      const [x1, y1, x2, y2] = innerBox;
      if (x < x1 || x > x2 || y < y1 || y > y2) return;

      const label = e.button === 2 ? 0 : 1;
      setPoints([...points, { x, y, label }]);

      setLoading(true);
      try {
        const f = new FormData();
        f.append("session_id", sessionId);
        f.append("x", String(x));
        f.append("y", String(y));
        f.append("label", String(label));

        const r = await api.post("/segment_inner", f);
        if (r.data.contours) {
          setInnerContours(r.data.contours);
          //setSelectedInnerContours(r.data.contours.map((_, i: number) => i)); // все выбраны по умолчанию
        }
        if (r.data.overlay_b64)
          setOverlay("data:image/png;base64," + r.data.overlay_b64);
      } finally {
        setLoading(false);
      }
    }
  };

  // --- глобальные события (движение и отпускание)
  useEffect(() => {
    const onMove = (ev: MouseEvent) => {
      if (!drawingBox || !boxStart || !imgRef.current) return;
      const rect = imgRef.current.getBoundingClientRect();
      const scaleX = imgRef.current.naturalWidth / rect.width;
      const scaleY = imgRef.current.naturalHeight / rect.height;
      setBoxEnd({
        x: Math.round((ev.clientX - rect.left) * scaleX),
        y: Math.round((ev.clientY - rect.top) * scaleY),
      });
    };

    const onUp = () => { //ev: MouseEvent
      if (!drawingBox || !boxStart || !boxEnd || innerBox) {
        setDrawingBox(false);
        return;
      }
      setDrawingBox(false);

      const x1 = Math.min(boxStart.x, boxEnd.x);
      const y1 = Math.min(boxStart.y, boxEnd.y);
      const x2 = Math.max(boxStart.x, boxEnd.x);
      const y2 = Math.max(boxStart.y, boxEnd.y);

      if (Math.abs(x2 - x1) < 8 || Math.abs(y2 - y1) < 8) {
        setBoxStart(null);
        setBoxEnd(null);
        return;
      }

      const f = new FormData();
      f.append("session_id", sessionId!);
      f.append("x1", String(x1));
      f.append("y1", String(y1));
      f.append("x2", String(x2));
      f.append("y2", String(y2));

      api
        .post("/start_inner_box", f)
        .then((r) => {
          setInnerBox(r.data.box);
          setPoints([]);
          setOverlay(null);          
          setInnerContours([]);
          setSelectedInnerContours([]);
        })
        .catch((err) => console.error("start_inner_box failed", err))
        .finally(() => {
          setBoxStart(null);
          setBoxEnd(null);
        });
    };

    window.addEventListener("mousemove", onMove);
    window.addEventListener("mouseup", onUp);
    return () => {
      window.removeEventListener("mousemove", onMove);
      window.removeEventListener("mouseup", onUp);
    };
  }, [drawingBox, boxStart, boxEnd, innerBox, sessionId]);

  // ------------------------------
  // 🔹 КНОПКИ/ОПЕРАЦИИ
  // ------------------------------
  /*
  const handleSaveInner = async () => {
    if (!sessionId || !innerContours.length || !selectedInnerContours.length)
      return alert("Nothing to save");

    setLoading(true);
    try {
      const form = new FormData();
      form.append("session_id", sessionId);
  
      if (selectedInnerContours.length > 0) {
        form.append("save", "true");
        form.append("selected", selectedInnerContours.join(","));
        const res = await api.post("/segment_inner", form);        
        alert(`Saved ${res.data.saved.count} inner masks.`);  
      }          
    } catch (err) {
      console.error("Save inner failed", err);
      alert("Save failed");
    } finally {
      setLoading(false);
    }
  };
  */
  const switchToMain = () => {
    setInteractiveSubMode("main");
    setInnerBox(null);
    setPoints([]);
    setContours([]);
    setOverlay(null);
    setInnerContours([]);
    setSelectedInnerContours([]);
  };

  const switchToInner = () => {
    setInteractiveSubMode("inner");
    setInnerBox(null);
    setPoints([]);
    setContours([]);
    setOverlay(null);
    setInnerContours([]);
    setSelectedInnerContours([]);
  };

  // ------------------------------
  // 🔹 РЕНДЕР
  // ------------------------------
  return (
    <div className="editor-panel flex flex-row w-full h-full">
      <div className="flex flex-col items-center justify-center flex-1">
        {!preview ? (
          <div className="text-gray-400 text-lg">Upload an image to start ❄️</div>
        ) : (
          <>
            <div className="mb-3 flex gap-3">
              <button
                onClick={switchToMain}
                className={`px-4 py-2 rounded ${
                  interactiveSubMode === "main"
                    ? "bg-blue-600 text-white"
                    : "bg-gray-300"
                }`}
              >
                MAIN
              </button>
              <button
                onClick={switchToInner}
                className={`px-4 py-2 rounded ${
                  interactiveSubMode === "inner"
                    ? "bg-yellow-600 text-white"
                    : "bg-gray-300"
                }`}
              >
                INNER
              </button>
            </div>

            <div className="relative inline-block select-none">
              <img
                ref={imgRef}
                src={overlay || preview || undefined}
                alt="preview"
                className="max-w-[80vw] h-auto cursor-crosshair"
                style={{ opacity: isUpdatingSettings ? 0.5 : 1 }} // <-- Опционально: затемнить изображение
                onMouseDown={handleMouseDown}
                onContextMenu={(e) => e.preventDefault()}
                draggable={false}
              />
              <canvas
                ref={canvasRef}
                className="absolute left-0 top-0 pointer-events-none"
              />
              {/* --- СПИННЕР --- */}
              {isUpdatingSettings && (
                <div
                  className="absolute inset-0 flex items-center justify-center bg-black bg-opacity-30 z-10" // z-10 должен быть выше canvas, но ниже других UI элементов
                >
                  {/* Простой CSS-анимированный спиннер */}
                  <div className="w-12 h-12 border-4 border-blue-500 border-t-transparent rounded-full animate-spin"></div>
                </div>
              )}
              {/* --- /СПИННЕР --- */}
            </div>

            {interactiveSubMode === "inner" && (
              <div className="mt-3 flex gap-2">
                <button
                  onClick={() => {
                    setInnerBox(null);
                    setPoints([]);
                    setOverlay(null);
                    setContours([]);                    
                    setInnerContours([]);
                    setSelectedInnerContours([]);
                  }}
                  className={`px-4 py-2 rounded ${
                    innerBox
                      ? "bg-red-600 text-white"
                      : "bg-gray-300 text-gray-500 cursor-not-allowed"
                  }`}
                  disabled={!innerBox}
                >
                  Reset inner box
                </button>
                {/*{innerContours.length > 0 && (
                  <button
                    onClick={handleSaveInner}
                    className="px-4 py-2 bg-green-600 text-white rounded"
                  >
                    Save selected
                  </button>
                )} */}
              </div>
            )}

            <div className="mt-3 text-gray-600 text-sm">
              {interactiveSubMode === "main" &&
                "ЛКМ — объект, ПКМ — фон. Итеративная сегментация."}
              {interactiveSubMode === "inner" &&
                (!innerBox
                  ? "ЛКМ — растянуть бокс."
                  : "ЛКМ — объект, ПКМ — фон внутри бокса.")}
            </div>
          </>
        )}
      </div>

      {/* 🔸 СПРАВА — ПАНЕЛЬ КОНТУРОВ */}
      {interactiveSubMode === "inner" && innerContours.length > 0 && (
      <div className="w-[250px] h-full overflow-y-auto border-l border-gray-300 bg-gray-50 p-2">
        <h3 className="text-gray-700 font-semibold mb-2 text-center">
          Контуры
        </h3>

        {innerContours.map((c, idx) => (
          <div
            key={idx}
            className="flex items-center justify-between px-2 py-1 hover:bg-gray-200 rounded cursor-pointer"
            onMouseEnter={() => setHoverContour(idx)}
            onMouseLeave={() => setHoverContour(null)}
          >
            <span className="text-sm text-gray-800">Контур {idx + 1}</span>
            <input
              type="checkbox"
              checked={selectedInnerContours.includes(idx)}
              onChange={(e) => {
                setSelectedInnerContours((prev) =>
                  e.target.checked
                    ? [...prev, idx]
                    : prev.filter((p) => p !== idx)
                );
              }}
            />
          </div>
        ))}
      </div>
    )}
    </div>
  );
}
