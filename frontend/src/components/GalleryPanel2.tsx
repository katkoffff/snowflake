// frontend/src/components/GalleryPanel2.tsx
import React from "react";
import { useApp } from "../hooks/useAppContext"; // Убедись, что путь правильный
import "../css/gallery.css"; // Используем тот же CSS, или создай gallery2.css

export default function GalleryPanel2() {
  // const { ... } = useApp(); // Используй нужные переменные из контекста stage2, если они будут

  return (
    <div className="gallery-panel flex flex-col items-center justify-center w-full h-full bg-white overflow-y-auto">
      <h2 className="text-xl font-semibold text-gray-800">Stage 2 Gallery Panel</h2>
      <p className="text-gray-600">Content for Stage 2 goes here.</p>
      {/* Добавь содержимое для stage2 по мере необходимости */}
    </div>
  );
}