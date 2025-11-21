// frontend\src\components\GalleryPanel.tsx

import React, { useEffect, useState } from "react";
import { api } from "../api/client"; // Убедись, что путь правильный
import { useApp } from "../hooks/useAppContext"; // Убедись, что путь правильный
import "../css/gallery.css"; // <-- Добавь ЭТУ строку
import ResultsModal from "../modals/ResultsModal";

// --- Интерфейс для папки (отличается от GalleryResult) ---
interface FolderResult {
  name: string; // Имя папки
  preview_image: string | null; // Имя файла final.jpg (если есть)
  mtime: string; // Время модификации
  path: string; // Относительный путь
}

export default function GalleryPanel() {
  const {
    refreshGallery, 
    setRefreshGallery, 
    setIsResultsModalOpen,
    setInitialFolderNameForModal,    
  } = useApp();

  // --- Состояние для списка папок ---
  const [folders, setFolders] = useState<FolderResult[]>([]);
  const [page, setPage] = useState(1);
  const perPage = 12;

  const fetchFolders = async (p = 1) => {
    try {
      // Используем обновлённый эндпоинт, который возвращает папки
      const res = await api.get(`/results/list?page=${p}&per_page=${perPage}`);
      setFolders(res.data.results || []);
      setPage(res.data.page || p);
    } catch (err) {
      console.error("Failed to fetch folders:", err);
      alert("Failed to fetch results");
    }
  };

  // --- Загрузка папок при монтировании ---
  useEffect(() => {
    fetchFolders();
  }, []); // Загружаем папки при монтировании компонента

  // --- Обновление папок при изменении refreshGallery ---
  useEffect(() => {
    if (refreshGallery) {
      fetchFolders(); // Загружаем папки при изменении refreshGallery на true
      setRefreshGallery(false); // Сбрасываем флаг после загрузки
    }
  }, [refreshGallery, setRefreshGallery]); // Зависимости: refreshGallery и его setter

  // --- Функция для открытия модального окна ---
  const openResultsModal = (folderName: string) => {
    setInitialFolderNameForModal(folderName);
    setIsResultsModalOpen(true);
  };

  return (
    <div className="gallery-panel">
      <h2>Results Gallery</h2>
      <div className="grid">
        {folders.map((folder) => (
          <div key={folder.name}>
            {folder.preview_image ? (
              <img
                src={`http://localhost:8000/api/results/image_in_dir?dir_name=${encodeURIComponent(folder.name)}&file_name=${folder.preview_image}`}
                alt={`Preview from ${folder.name}`}
                onClick={() => openResultsModal(folder.name)}
              />
            ) : (              
              <div className="placeholder">
                No Preview
              </div>              
            )}
            <div className="truncate">{folder.name}</div>
          </div>
        ))}
      </div>
      <div className="pagination">
        <button disabled={page === 1} onClick={() => fetchFolders(page - 1)}>
          Prev
        </button>
        <button onClick={() => fetchFolders(page + 1)}>Next</button>
      </div>
      <ResultsModal />
    </div>
  );
}