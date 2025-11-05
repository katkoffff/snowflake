import React, { useState, useRef, useEffect } from "react";
import axios from "axios";

type Pt = { x: number; y: number; label: number };

export default function ImageUploader() {
  const [file, setFile] = useState<File | null>(null);
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [contours, setContours] = useState<number[][][]>([]);
  const [overlayB64, setOverlayB64] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState<any[]>([]);
  const [page, setPage] = useState<number>(1);

  const imgRef = useRef<HTMLImageElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  // синхронизация canvas с изображением
  useEffect(() => {
    const img = imgRef.current;
    const canvas = canvasRef.current;
    if (!img || !canvas) return;
    const onLoad = () => {
      canvas.width = img.naturalWidth;
      canvas.height = img.naturalHeight;
      canvas.style.width = `${img.width}px`;
      canvas.style.height = `${img.height}px`;
      drawContours();
    };
    if (img.complete) onLoad();
    else img.addEventListener("load", onLoad);
    return () => img.removeEventListener("load", onLoad);
  }, [preview, overlayB64, contours]);

  // рисуем контуры поверх изображения
  const drawContours = () => {
    const canvas = canvasRef.current;
    const img = imgRef.current;
    if (!canvas || !img) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = "rgba(255,0,0,0.18)";
    ctx.strokeStyle = "rgba(255,0,0,1)";
    ctx.lineWidth = 2;
    contours.forEach((cont) => {
      if (!cont || cont.length < 2) return;
      ctx.beginPath();
      cont.forEach(([x, y], i) => {
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      });
      ctx.closePath();
      ctx.fill();
      ctx.stroke();
    });
  };

  // загрузка файла -> init сессии
  const handleSelect = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const f = e.target.files?.[0];
    if (!f) return;
    setFile(f);
    setOverlayB64(null);
    setContours([]);
    setPreview(null);
    setSessionId(null);

    const fd = new FormData();
    fd.append("file", f);
    setLoading(true);
    try {
      const res = await axios.post("http://localhost:8000/init", fd, {
        headers: { "Content-Type": "multipart/form-data" },
      });
      setSessionId(res.data.session_id);
      setPreview(`data:image/png;base64,${res.data.preview_b64}`);
    } catch (err) {
      console.error(err);
      alert("Init failed");
    } finally {
      setLoading(false);
    }
  };

  // клик по изображению → сегментация (итеративно)
  const handleMouseDown = async (e: React.MouseEvent<HTMLImageElement>) => {
    e.preventDefault();
    const img = imgRef.current;
    if (!img || !sessionId || !file) return;

    const rect = img.getBoundingClientRect();
    const scaleX = img.naturalWidth / rect.width;
    const scaleY = img.naturalHeight / rect.height;
    const x = Math.round((e.clientX - rect.left) * scaleX);
    const y = Math.round((e.clientY - rect.top) * scaleY);
    const label = e.button === 2 ? 0 : 1; // левая = объект, правая = фон

    setLoading(true);
    try {
      const form = new FormData();
      form.append("session_id", sessionId);
      form.append("x", x.toString());
      form.append("y", y.toString());
      form.append("label", label.toString());

      const res = await axios.post("http://localhost:8000/segment", form, {
        headers: { "Content-Type": "multipart/form-data" },
      });

      setContours(res.data.contours || []);
      if (res.data.overlay_b64)
        setOverlayB64(`data:image/png;base64,${res.data.overlay_b64}`);
    } catch (err) {
      console.error(err);
      alert("Segment failed");
    } finally {
      setLoading(false);
    }
  };

  const handleContextMenu = (e: React.MouseEvent) => e.preventDefault();

  // сохранение результата
  const handleSave = async () => {
    if (!sessionId) return;
    setLoading(true);
    try {
      const form = new FormData();
      form.append("session_id", sessionId);
      form.append("save", "true");
      const res = await axios.post("http://localhost:8000/segment", form, {
        headers: { "Content-Type": "multipart/form-data" },
      });
      if (res.data.saved?.jpg) {
        alert(`Saved:\n${res.data.saved.jpg}\n${res.data.saved.npy}`);
      } else {
        alert("Saved");
      }
    } catch (err) {
      console.error(err);
      alert("Save failed");
    } finally {
      setLoading(false);
    }
  };

  // сброс сессии
  const handleReset = async () => {
    if (!sessionId) return;
    try {
      const form = new FormData();
      form.append("session_id", sessionId);
      await axios.post("http://localhost:8000/reset", form);
      setContours([]);
      setOverlayB64(null);
    } catch {
      alert("Reset failed");
    }
  };

  // просмотр результатов
  const perPage = 12;
  const fetchResults = async (p = 1) => {
    try {
      const res = await axios.get(
        `http://localhost:8000/results/list?page=${p}&per_page=${perPage}`
      );
      setResults(res.data.results || []);
      setPage(res.data.page || p);
    } catch {
      alert("Cannot fetch results");
    }
  };

  const getResultImageUrl = (name: string) =>
    `http://localhost:8000/results/image?name=${encodeURIComponent(name)}`;

  return (
    <div style={{ textAlign: "center", marginTop: 12 }}>
      <div style={{ marginBottom: 8 }}>
        <input type="file" accept="image/*" onChange={handleSelect} />
        <button
          onClick={handleSave}
          disabled={!sessionId || loading}
          style={{ marginLeft: 8 }}
        >
          Save
        </button>
        <button
          onClick={() => fetchResults(1)}
          style={{ marginLeft: 8 }}
        >
          Results
        </button>
        <button
          onClick={handleReset}
          disabled={!sessionId || loading}
          style={{ marginLeft: 8 }}
        >
          Reset
        </button>
      </div>

      <div style={{ position: "relative", display: "inline-block" }}>
        {preview && (
          <>
            <img
              ref={imgRef}
              src={overlayB64 || preview}
              alt="preview"
              style={{
                display: "block",
                maxWidth: "80vw",
                height: "auto",
                cursor: "crosshair",
              }}
              onMouseDown={handleMouseDown}
              onContextMenu={handleContextMenu}
            />
            <canvas
              ref={canvasRef}
              style={{
                position: "absolute",
                left: 0,
                top: 0,
                pointerEvents: "none",
              }}
            />
          </>
        )}
      </div>

      {results && results.length > 0 && (
        <div style={{ marginTop: 12, maxWidth: "90vw", textAlign: "left" }}>
          <h3>Saved results (page {page})</h3>
          <div
            style={{
              display: "flex",
              gap: 12,
              flexWrap: "wrap",
              justifyContent: "center",
            }}
          >
            {results.map((r) => (
              <div
                key={r.name}
                style={{
                  width: 160,
                  border: "1px solid #ddd",
                  padding: 6,
                  borderRadius: 8,
                }}
              >
                <div
                  style={{
                    height: 120,
                    overflow: "hidden",
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                  }}
                >
                  <img
                    src={getResultImageUrl(r.name)}
                    alt={r.name}
                    style={{ maxWidth: "100%", maxHeight: "100%" }}
                  />
                </div>
                <div style={{ marginTop: 6, fontSize: 12 }}>
                  <div>{r.name}</div>
                  <div>
                    {r.size_kb} KB • {r.mtime}
                  </div>
                </div>
                <div style={{ marginTop: 6 }}>
                  <a
                    href={getResultImageUrl(r.name)}
                    target="_blank"
                    rel="noreferrer"
                  >
                    Open
                  </a>
                </div>
              </div>
            ))}
          </div>
          <div style={{ marginTop: 8 }}>
            <button
              disabled={page === 1}
              onClick={() => fetchResults(page - 1)}
            >
              Prev
            </button>
            <button
              onClick={() => fetchResults(page + 1)}
              style={{ marginLeft: 8 }}
            >
              Next
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
