import React, { useRef, useEffect } from "react";
import axios from "axios";
import { useApp } from "../context/AppContext";
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
    setPoints
  } = useApp();

  const imgRef = useRef<HTMLImageElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  // sync canvas size to natural image size
  useEffect(() => {
    const img = imgRef.current;
    const canvas = canvasRef.current;
    if (!img || !canvas) return;
    const onLoad = () => {
      canvas.width = img.naturalWidth;
      canvas.height = img.naturalHeight;
      canvas.style.width = `${img.width}px`;
      canvas.style.height = `${img.height}px`;
      drawOverlay(); // initial draw
    };
    if (img.complete) onLoad();
    else img.addEventListener("load", onLoad);
    return () => img.removeEventListener("load", onLoad);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [preview, overlay]);

  // redraw when contours or points change
  useEffect(() => {
    drawOverlay();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [overlay, points, setContours]);

  // draw contours (filled optional) and points on canvas (natural coords)
  const drawOverlay = () => {
    const canvas = canvasRef.current;
    const img = imgRef.current;
    if (!canvas || !img) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    // clear
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // draw contours (from context.setContours data) — but UI stores contours in context.contours
    // we will draw contours from overlay image only if overlay not null (server already draws polylines in overlay)
    // additionally draw points on top
    // NOTE: overlay image itself is shown beneath; canvas draws points/contours in natural coords

    // draw points
    for (const p of points) {
      const { x, y, label } = p;
      ctx.beginPath();
      ctx.arc(x, y, Math.max(4, Math.round(Math.min(canvas.width, canvas.height) * 0.01)), 0, Math.PI * 2);
      if (label === 1) {
        ctx.fillStyle = "rgba(0,200,0,0.95)"; // green = object
        ctx.strokeStyle = "rgba(0,120,0,1)";
      } else {
        ctx.fillStyle = "rgba(0,120,255,0.95)"; // blue = background
        ctx.strokeStyle = "rgba(0,70,160,1)";
      }
      ctx.fill();
      ctx.lineWidth = 1;
      ctx.stroke();
    }
  };

  // handle click: add point and call backend (iterative)
  const handleMouseDown = async (e: React.MouseEvent<HTMLImageElement>) => {
    e.preventDefault();
    const img = imgRef.current;
    if (!img || !sessionId) return;

    const rect = img.getBoundingClientRect();
    const scaleX = img.naturalWidth / rect.width;
    const scaleY = img.naturalHeight / rect.height;
    const x = Math.round((e.clientX - rect.left) * scaleX);
    const y = Math.round((e.clientY - rect.top) * scaleY);
    const label = e.button === 2 ? 0 : 1;

    // locally append the point so UI is responsive
    setPoints([...points, { x, y, label }]);

    setLoading(true);
    try {
      const form = new FormData();
      form.append("session_id", sessionId);
      form.append("x", x.toString());
      form.append("y", y.toString());
      form.append("label", label.toString());
      const res = await axios.post("http://localhost:8000/segment", form);
      setContours(res.data.contours || []);
      if (res.data.overlay_b64) setOverlay(`data:image/png;base64,${res.data.overlay_b64}`);
      // if server returned contours, we could replace overlay or update points from server if needed
    } catch (err) {
      console.error(err);
      alert("Segment failed");
      // on failure we should remove last point to keep UI consistent
      setPoints(points.slice(0, -1));
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="editor-panel flex flex-col items-center justify-center w-full h-full">
      {!preview ? (
        <div className="text-gray-400 text-lg">Upload an image to start ❄️</div>
      ) : (
        <div style={{ display: "flex", flexDirection: "column", alignItems: "center" }}>
          <div className="relative inline-block">
            <img
              ref={imgRef}
              src={overlay || preview}
              alt="preview"
              className="max-w-[80vw] h-auto cursor-crosshair rounded shadow"
              onMouseDown={handleMouseDown}
              onContextMenu={(e) => e.preventDefault()}
            />
            <canvas
              ref={canvasRef}
              className="absolute left-0 top-0 pointer-events-none"
              style={{ left: 0, top: 0 }}
            />
          </div>

          {/* Legend / hints under image */}
          <div style={{ marginTop: 8, color: "#444", fontSize: 13 }}>
            <span style={{ marginRight: 12 }}>
              <strong>ЛКМ</strong> — объект (зелёная точка)
            </span>
            <span style={{ marginRight: 12 }}>
              <strong>ПКМ</strong> — фон (синяя точка)
            </span>
            <span style={{ marginLeft: 12, color: "#888" }}>
              Точки добавляются итеративно — каждая улучшает контур.
            </span>
          </div>
        </div>
      )}
    </div>
  );
}
