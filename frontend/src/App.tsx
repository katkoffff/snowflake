import React from "react";
import Header from "./components/Header";
import Sidebar from "./components/Sidebar";
import EditorPanel from "./components/EditorPanel";
import GalleryPanel from "./components/GalleryPanel";
import Footer from "./components/Footer";
import { AppProvider } from "./context/AppContext";

//flex items-center justify-center bg-gray-100 overflow-y-auto h-0
//flex items-center justify-center bg-gray-100

export default function App() {
  return (
    <AppProvider>
      <div className="grid grid-cols-[220px_1fr_320px] grid-rows-[auto_1fr_auto] h-screen bg-gray-50 text-gray-800">
        <header className="col-span-3 border-b border-gray-200 bg-white shadow-sm">
          <Header />
        </header>
        <aside className="border-r border-gray-200 bg-white overflow-y-auto">
          <Sidebar />
        </aside>
        <main className="bg-gray-100 overflow-hidden">
          <EditorPanel />
        </main>
        <aside className="border-l border-gray-200 bg-white overflow-y-auto">
          <GalleryPanel />
        </aside>
        <footer className="col-span-3 border-t border-gray-200 bg-white text-sm text-gray-500 py-2 text-center">
          <Footer />
        </footer>
      </div>
    </AppProvider>
  );
}
