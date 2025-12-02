// frontend/src/components/EnvelopModal.tsx
import React from "react";
import { useApp } from "../hooks/useAppContext";
import "../css/envelop_modal.css";

const EnvelopModal: React.FC = () => {
  const { 
    isEnvelopModalOpen, 
    setEnvelopModalOpen, 
    envelopGraphData 
  } = useApp();

  const closeModal = () => {
    setEnvelopModalOpen(false);
  };

  if (!isEnvelopModalOpen) {
    return null;
  }

  return (
    <div className="envelop-modal-overlay" onClick={closeModal}>
      <div 
        className="envelop-modal-content" 
        onClick={(e) => e.stopPropagation()}
      >
        {/* Кнопка закрытия */}
        <button className="envelop-modal-close-btn" onClick={closeModal}>
          <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
          </svg>
        </button>

        {/* Заголовок */}
        <div className="envelop-modal-header">
          <h2>Envelope Analysis Results</h2>
        </div>

        {/* Тело с изображением */}
        <div className="envelop-modal-body">
          {envelopGraphData ? (
            <img 
              src={envelopGraphData}
              alt="Envelope analysis results"
              className="envelop-image"
            />
          ) : (
            <div className="envelop-modal-error">
              No image data available
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default EnvelopModal;