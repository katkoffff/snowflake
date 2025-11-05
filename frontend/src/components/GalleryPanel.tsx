import React, { useEffect, useState } from "react";
import axios from "axios";
import { useApp } from "../context/AppContext";
import "../css/gallery.css";

export default function GalleryPanel() {
  const { setPreview, setOverlay, setSessionId, refreshGallery, setRefreshGallery, setContours } = useApp();
  const [results, setResults] = useState<any[]>([]);
  const [page, setPage] = useState(1);
  const perPage = 12;

  const fetchResults = async (p = 1) => {
    try {
      const res = await axios.get(`http://localhost:8000/results/list?page=${p}&per_page=${perPage}`);
      setResults(res.data.results || []);
      setPage(res.data.page || p);
    } catch {
      alert("Failed to fetch results");
    }
  };

  useEffect(() => {
    fetchResults();
    setRefreshGallery(false);
  }, [refreshGallery]);

  const handleLoad = async (name: string) => {
    try {
      const res = await axios.get(
        `http://localhost:8000/load_result?name=${encodeURIComponent(name)}`
      );
      setPreview(`data:image/png;base64,${res.data.preview_b64}`);
      setOverlay(null);
      setContours(res.data.contours || []);
      setSessionId(res.data.session_id);
    } catch (err) {
      console.error(err);
      alert("Load failed");
    }
  };

  const getResultImageUrl = (name: string) =>
    `http://localhost:8000/results/image?name=${encodeURIComponent(name)}`;

  return (
    <div className="gallery-panel p-4">
      <h2 className="text-lg font-semibold mb-3 text-gray-700">Gallery</h2>
      <div className="grid grid-cols-2 gap-3">
        {results.map((r) => (
          <div
            key={r.name}
            className="border border-gray-200 rounded-lg shadow-sm overflow-hidden"
          >
            <img
              src={getResultImageUrl(r.name)}
              alt={r.name}
              className="w-full h-auto cursor-pointer"
              onClick={() => handleLoad(r.name)}
            />
            <div className="p-2 text-xs text-center text-gray-600">{r.name}</div>
          </div>
        ))}
      </div>
      <div className="flex justify-center mt-3 gap-2">
        <button disabled={page === 1} onClick={() => fetchResults(page - 1)}>
          Prev
        </button>
        <button onClick={() => fetchResults(page + 1)}>Next</button>
      </div>
    </div>
  );
}
