// frontend\src\App.tsx
import React from "react";
import Header from "./components/Header";
import Sidebar from "./components/Sidebar";
import Sidebar2 from "./components/Sidebar2"; // <-- Убедись, что импортирован
import EditorPanel from "./components/EditorPanel";
import EditorPanel2 from "./components/EditorPanel2"; // <-- Убедись, что импортирован
import GalleryPanel from "./components/GalleryPanel";
import GalleryPanel2 from "./components/GalleryPanel2"; // <-- Убедись, что импортирован
import Footer from "./components/Footer";

import { useApp } from "./hooks/useAppContext"; // <-- Импортируем useApp
import "./app.css"; // <-- Импортируем стили для App

export default function App() {
  const { currentStage } = useApp(); // <-- Получаем currentStage из контекста

  return (    
    <div className="app-container">
      <header className="app-header">
        <Header />
      </header>
      <aside className="app-sidebar">
        {currentStage === "stage1" ? <Sidebar /> : <Sidebar2 />}
      </aside>
      <main className="app-main">
        {currentStage === "stage1" ? <EditorPanel /> : <EditorPanel2 />}
      </main>
      <aside className="app-gallery">
        {currentStage === "stage1" ? <GalleryPanel /> : <GalleryPanel2 />}
      </aside>
      <footer className="app-footer">
        <Footer />
      </footer>
    </div>    
  );
}