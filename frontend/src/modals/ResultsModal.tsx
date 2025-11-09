// frontend\src\modal\ResultsModal.tsx
import React, { useState, useEffect } from "react";
import { api } from "../api/client"; // Убедись, что путь правильный
import { useApp } from "../hooks/useAppContext"; // Убедись, что путь правильный
import "../css/results_modal.css"; // Создадим файл стилей

interface FolderInfo {
  name: string;
  preview_image: string | null; // или используй другое поле, если оно есть
  mtime: string;
  path: string;
}

const ResultsModal: React.FC = () => {
  const {
    isResultsModalOpen,
    setIsResultsModalOpen,
    initialFolderNameForModal,
  } = useApp();

  const [folders, setFolders] = useState<FolderInfo[]>([]);
  const [currentIndex, setCurrentIndex] = useState<number>(-1); // -1 означает, что индекс не установлен
  const [currentImageSrc, setCurrentImageSrc] = useState<string | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  // Загружаем список папок при открытии модального окна
  useEffect(() => {
    if (isResultsModalOpen && initialFolderNameForModal) {
      const fetchFolders = async () => {
        setLoading(true);
        setError(null);
        try {
          // Загружаем *все* папки (аналогично /results/list)
          const res = await api.get("/results/list?page=1&per_page=200"); // Увеличь per_page, если нужно больше
          const allFolders: FolderInfo[] = res.data.results;

          // Фильтруем, оставляя только те папки, у которых есть final.jpg
          const foldersWithFinal = allFolders.filter(folder => {
            // Проверяем, есть ли preview_image, которое, как мы предполагаем, указывает на final.jpg
            // Если в /results/list поле preview_image указывает на final.jpg, то проверяем его
            // Если нет, то нужно будет по-другому проверить наличие final.jpg
            // Пока будем считать, что preview_image === final.jpg
            // Если preview_image === null, то final.jpg нет
            return folder.preview_image !== null;
          });

          setFolders(foldersWithFinal);

          // Находим индекс initialFolderNameForModal
          const initialIndex = foldersWithFinal.findIndex(f => f.name === initialFolderNameForModal);
          if (initialIndex !== -1) {
            setCurrentIndex(initialIndex);
            // Загружаем изображение для текущего индекса
            loadFinalImage(foldersWithFinal[initialIndex].name);
          } else {
            console.warn(`Initial folder '${initialFolderNameForModal}' not found or has no final.jpg`);
            // Можно выбрать первую папку или показать ошибку
            if (foldersWithFinal.length > 0) {
              setCurrentIndex(0);
              loadFinalImage(foldersWithFinal[0].name);
            } else {
              setError("No folders with final.jpg found.");
              setCurrentImageSrc(null);
            }
          }
        } catch (err) {
          console.error("Failed to fetch folders:", err);
          setError("Failed to load folders.");
          setFolders([]);
          setCurrentIndex(-1);
          setCurrentImageSrc(null);
        } finally {
          setLoading(false);
        }
      };

      fetchFolders();
    } else if (!isResultsModalOpen) {
      // Сбрасываем состояние при закрытии
      setFolders([]);
      setCurrentIndex(-1);
      setCurrentImageSrc(null);
      setError(null);
    }
    // Зависимости: при isResultsModalOpen=false или initialFolderNameForModal=null - сброс
    // при isResultsModalOpen=true и initialFolderNameForModal - загрузка
  }, [isResultsModalOpen, initialFolderNameForModal]);

  const loadFinalImage = async (folderName: string) => {
    try {
      // Используем эндпоинт, который возвращает файл из папки
      // Предполагаем, что /results/image_in_dir принимает dir_name и file_name
      const imageUrl = `http://localhost:8000/results/image_in_dir?dir_name=${encodeURIComponent(folderName)}&file_name=final.jpg`;
      setCurrentImageSrc(imageUrl);
    } catch (err) {
      console.error("Failed to load final image:", err);
      setError("Failed to load image.");
      setCurrentImageSrc(null);
    }
  };

  const goToNext = () => {
    if (currentIndex < folders.length - 1) {
      const newIndex = currentIndex + 1;
      setCurrentIndex(newIndex);
      loadFinalImage(folders[newIndex].name);
    }
  };

  const goToPrev = () => {
    if (currentIndex > 0) {
      const newIndex = currentIndex - 1;
      setCurrentIndex(newIndex);
      loadFinalImage(folders[newIndex].name);
    }
  };

  const closeModal = () => {
    setIsResultsModalOpen(false);
    // Состояния folders, currentIndex, currentImageSrc, error сбросятся в useEffect при isResultsModalOpen=false
  };

  if (!isResultsModalOpen) {
    return null; // Не рендерим, если модальное окно закрыто
  }

  return (
    // Заменяем длинный className на короткое имя, определённое в gallery.css
    <div className="modal-overlay" onClick={closeModal}>
      {/* Заменяем длинный className на короткое имя */}
      <div className="modal-content" onClick={(e) => e.stopPropagation()}>
        {/* Заменяем длинный className на короткое имя */}
        <button className="modal-close-btn" onClick={closeModal}>
          <svg xmlns="http://www.w3.org/2000/svg" className="w-5 h-5 text-gray-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
          </svg>
        </button>

        {/* Индикатор загрузки */}
        {loading && <div className="modal-loading">Loading...</div>}
        {error && <div className="modal-error">{error}</div>}

        {/* Изображение */}
        {!loading && !error && currentImageSrc && (
          // Заменяем className на короткое имя
          <div className="modal-image-container">
            {/* Заменяем className на короткое имя */}
            <img src={currentImageSrc} alt={`Final from ${folders[currentIndex]?.name}`} className="modal-image" />
          </div>
        )}

        {/* Информация о текущей папке */}
        {!loading && !error && currentIndex !== -1 && (
          // Заменяем className на короткое имя
          <div className="modal-info">
            Folder: {folders[currentIndex]?.name}
          </div>
        )}

        {/* Навигация */}
        {/* Заменяем className на короткое имя */}
        <div className="modal-navigation">
          <button className="nav-btn" onClick={goToPrev} disabled={currentIndex <= 0}>
            {'<'}
          </button>
          <span className="nav-info">
            {currentIndex !== -1 ? `${currentIndex + 1} / ${folders.length}` : "No folders"}
          </span>
          <button className="nav-btn" onClick={goToNext} disabled={currentIndex >= folders.length - 1}>
            {'>'}
          </button>
        </div>
      </div>
    </div>
  );
};

export default ResultsModal;