import React, { useRef } from "react";
import axios from "axios";
import { useApp } from "../context/AppContext";
import "../css/sidebar.css";

export default function Sidebar() {
  const fileInput = useRef<HTMLInputElement | null>(null);
  const {
    sessionId, setSessionId,
    setPreview, setOverlay, setContours,
    setLoading, setRefreshGallery,
    medianKsize, setMedianKsize,
    contrastFactor, setContrastFactor,
    sharpnessFactor, setSharpnessFactor,
    claheClipLimit, setClaheClipLimit,
    claheTileGrid, setClaheTileGrid,
    setPoints
  } = useApp();

  // --- Upload ---
  const handleFileSelect = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const f = e.target.files?.[0];
    if (!f) return;
    const fd = new FormData();
    fd.append("file", f);
    setLoading(true);
    try {
      const res = await axios.post("http://localhost:8000/init", fd);
      setSessionId(res.data.session_id);
      setPreview(`data:image/png;base64,${res.data.preview_b64}`);
      setOverlay(null);
      setContours([]);
      if (res.data.used_config) {
        const cfg = res.data.used_config;
        setMedianKsize(cfg.median_ksize ?? 5);
        setContrastFactor(cfg.contrast_factor ?? 1.5);
        setSharpnessFactor(cfg.sharpness_factor ?? 2.0);
        setClaheClipLimit(cfg.clahe_clip_limit ?? 1.5);
        const tileGrid = Array.isArray(cfg.clahe_tile_grid) ? cfg.clahe_tile_grid : [8, 8];
        setClaheTileGrid(tileGrid);
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
      const res = await axios.post("http://localhost:8000/segment", form);
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
    await axios.post("http://localhost:8000/reset", form);
    setOverlay(null);
    setContours([]);
    setPoints([]); // <-- очищаем точки тоже
    };

  // --- Update Settings ---
  const handleSettingChange = async (key: string, value: number) => {
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

    // отправляем все значения на сервер
    if (sessionId) {
      setLoading(true);
      try {
        const form = new FormData();
        form.append("session_id", sessionId);
        form.append("median_ksize", String(value === medianKsize ? value : medianKsize));
        form.append("contrast_factor", String(value === contrastFactor ? value : contrastFactor));
        form.append("sharpness_factor", String(value === sharpnessFactor ? value : sharpnessFactor));
        form.append("clahe_clip_limit", String(value === claheClipLimit ? value : claheClipLimit));
        form.append("clahe_tile_grid", claheTileGrid.join(","));

        const res = await axios.post("http://localhost:8000/update_settings", form);
        if (res.data.preview_b64) {
          setPreview(`data:image/png;base64,${res.data.preview_b64}`);
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

  return (
    <div className="sidebar flex flex-col gap-3 p-4">
      <input type="file" ref={fileInput} className="hidden" onChange={handleFileSelect} />
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

      {/* Настройки */}
      <div className="settings mt-3 space-y-2">
        <div className="setting">
          <label>Median ksize: {medianKsize}</label>
          <input type="range" min="1" max="11" step="2"
                 value={medianKsize}
                 onChange={(e) => handleSettingChange("medianKsize", parseInt(e.target.value))}/>
        </div>
        <div className="setting">
          <label>Contrast: {contrastFactor.toFixed(1)}</label>
          <input type="range" min="0.5" max="3" step="0.1"
                 value={contrastFactor}
                 onChange={(e) => handleSettingChange("contrastFactor", parseFloat(e.target.value))}/>
        </div>
        <div className="setting">
          <label>Sharpness: {sharpnessFactor.toFixed(1)}</label>
          <input type="range" min="0.5" max="5" step="0.1"
                 value={sharpnessFactor}
                 onChange={(e) => handleSettingChange("sharpnessFactor", parseFloat(e.target.value))}/>
        </div>
        <div className="setting">
          <label>CLAHE Clip Limit: {claheClipLimit.toFixed(1)}</label>
          <input type="range" min="1" max="10" step="0.1"
                 value={claheClipLimit}
                 onChange={(e) => handleSettingChange("claheClipLimit", parseFloat(e.target.value))}/>
        </div>
        <div className="setting">
          <label>CLAHE Grid: {claheTileGrid[0]}</label>
          <input type="range" min="4" max="16" step="1"
                 value={claheTileGrid[0]}
                 onChange={(e) => handleSettingChange("claheTileGrid", parseInt(e.target.value))}/>
        </div>
      </div>
    </div>
  );
}
