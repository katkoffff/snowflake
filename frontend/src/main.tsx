// frontend\src\main.tsx
import React from "react";
import ReactDOM from "react-dom/client";
import App from "./App"; // <-- Импортирует App (новый или старый)
import "./index.css";
import { AppProvider } from "./context/AppContext"; // <-- Импортируем AppProvider

// AppProvider ОБЁРТЫВАЕТ App
ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <AppProvider> {/* <-- AppProvider СНАРУЖИ */}
      <App /> {/* <-- AppProvider предоставляет контекст для App и его детей */}
    </AppProvider>
  </React.StrictMode>
);