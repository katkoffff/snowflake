// frontend\src\context\AppContext.tsx
import React, { createContext, useState } from "react";
import type { AutoGenMask, AutoGenConfig, ConfirmedMask } from "../types/autogen";

// --- Типы ---
export type Pt = { x: number; y: number; label: number }; // label: 1 = object (left), 0 = background (right)
type Stage = "stage1" | "stage2"; // Тип для этапа работы приложения

export type AppContextType = {
  // --- Сессия ---
  sessionId: string | null;
  setSessionId: (id: string | null) => void;

  // --- Изображение и отображение ---
  preview: string | null;
  setPreview: (p: string | null) => void;
  overlay: string | null;
  setOverlay: (p: string | null) => void;
  contours: number[][][]; // list of contours (each is [[x,y],...])
  setContours: (c: number[][][]) => void;

  // --- Состояния загрузки и галереи ---
  loading: boolean;
  setLoading: (v: boolean) => void;
  refreshGallery: boolean;
  setRefreshGallery: (v: boolean) => void;

  // --- Интерактивная сегментация (Main) ---
  points: Pt[];
  setPoints: (pts: Pt[]) => void;

  // --- Настройки препроцессинга (Main) ---
  medianKsize: number;
  setMedianKsize: (v: number) => void;
  contrastFactor: number;
  setContrastFactor: (v: number) => void;
  sharpnessFactor: number;
  setSharpnessFactor: (v: number) => void;
  claheClipLimit: number;
  setClaheClipLimit: (v: number) => void;
  claheTileGrid: [number, number];
  setClaheTileGrid: (v: [number, number]) => void;

  // --- Режимы работы (Main) ---
  mode: 'interactive' | 'auto';
  setMode: React.Dispatch<React.SetStateAction<'interactive' | 'auto'>>;
  interactiveSubMode: "main" | "inner";
  setInteractiveSubMode: (m: "main" | "inner") => void;

  // --- Внутренняя сегментация (Inner Box) ---
  innerBox: [number, number, number, number] | null;
  setInnerBox: (b: [number, number, number, number] | null) => void;
  innerContours: number[][][];          // массив контуров
  setInnerContours: (arr: number[][][]) => void;
  selectedInnerContours: number[];      // индексы выбранных
  setSelectedInnerContours: React.Dispatch<React.SetStateAction<number[]>>;
  hoverContour: number | null;          // какой контур подсвечивается
  setHoverContour: (v: number | null) => void;

  // --- Автогенерация (Auto) ---
  autoGenConfig: AutoGenConfig;
  setAutoGenConfig: React.Dispatch<React.SetStateAction<AutoGenConfig>>;
  autoMasks: AutoGenMask[] | null;
  setAutoMasks: (masks: AutoGenMask[] | null) => void;
  selectedMaskIds: string[];
  setSelectedMaskIds: React.Dispatch<React.SetStateAction<string[]>>;
  confirmedMasks: ConfirmedMask[];
  setConfirmedMasks: (masks: ConfirmedMask[]) => void;
  finalMasks: ConfirmedMask[];
  setFinalMasks: (masks: ConfirmedMask[]) => void;
  mainMaskId: string | null;
  setMainMaskId: (id: string | null) => void;
  selectionConfirmed: boolean;
  setSelectionConfirmed: (confirmed: boolean) => void;
  refinementCompleted: boolean;
  setRefinementCompleted: (completed: boolean) => void;

  // --- Обновление настроек ---
  isUpdatingSettings: boolean;
  setIsUpdatingSettings: (v: boolean) => void;

  // --- Модальные окна ---
  isResultsModalOpen: boolean;
  setIsResultsModalOpen: (isOpen: boolean) => void;
  initialFolderNameForModal: string | null;
  setInitialFolderNameForModal: (folderName: string | null) => void;
  isAnalysisModalOpen: boolean;
  setIsAnalysisModalOpen: (isOpen: boolean) => void;
  isFractalDimensionModalOPen: boolean;
  setIsFractalDimensionModalOPen: (isOpen: boolean) => void;

  isEnvelopModalOpen: boolean;
  setEnvelopModalOpen: (open: boolean) => void;
  envelopGraphData: string | null;
  setEnvelopGraphData: (p: string | null) => void;

  // --- Управление этапами ---
  currentStage: Stage;
  setCurrentStage: (stage: Stage) => void;
};

// --- Создание контекста ---
const AppContext = createContext<AppContextType | null>(null);

// --- Компонент-провайдер ---
export const AppProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  // --- Сессия ---
  const [sessionId, setSessionId] = useState<string | null>(null);

  // --- Изображение и отображение ---
  const [preview, setPreview] = useState<string | null>(null);
  const [overlay, setOverlay] = useState<string | null>(null);
  const [contours, setContours] = useState<number[][][]>([]);

  // --- Состояния загрузки и галереи ---
  const [loading, setLoading] = useState(false);
  const [refreshGallery, setRefreshGallery] = useState(false);

  // --- Интерактивная сегментация (Main) ---
  const [points, setPoints] = useState<Pt[]>([]);

  // --- Настройки препроцессинга (Main) ---
  const [medianKsize, setMedianKsize] = useState<number>(5);
  const [contrastFactor, setContrastFactor] = useState<number>(1.5);
  const [sharpnessFactor, setSharpnessFactor] = useState<number>(2.0);
  const [claheClipLimit, setClaheClipLimit] = useState<number>(1.5);
  const [claheTileGrid, setClaheTileGrid] = useState<[number, number]>([8, 8]);

  // --- Режимы работы (Main) ---
  const [mode, setMode] = useState<'interactive' | 'auto'>('interactive');
  const [interactiveSubMode, setInteractiveSubMode] = useState<"main" | "inner">("main");

  // --- Внутренняя сегментация (Inner Box) ---
  const [innerBox, setInnerBox] = useState<[number, number, number, number] | null>(null);
  const [innerContours, setInnerContours] = useState<number[][][]>([]);
  const [selectedInnerContours, setSelectedInnerContours] = useState<number[]>([]);
  const [hoverContour, setHoverContour] = useState<number | null>(null);

  // --- Автогенерация (Auto) ---
  const [autoGenConfig, setAutoGenConfig] = useState<AutoGenConfig>({
    points_per_side: 16,
    points_per_batch: 32,
    pred_iou_thresh: 0.7,
    stability_score_thresh: 0.9,
    stability_score_offset: 0.7,
    crop_n_layers: 1,
    box_nms_thresh: 0.7,
    crop_n_points_downscale_factor: 2,
    min_mask_region_area: 50,
    use_m2m: false,
  });
  const [autoMasks, setAutoMasks] = useState<AutoGenMask[] | null>(null);
  const [selectedMaskIds, setSelectedMaskIds] = useState<string[]>([]);
  const [confirmedMasks, setConfirmedMasks] = useState<ConfirmedMask[]>([]);
  const [finalMasks, setFinalMasks] = useState<ConfirmedMask[]>([]);
  const [mainMaskId, setMainMaskId] = useState<string | null>(null);
  const [selectionConfirmed, setSelectionConfirmed] = useState<boolean>(false);
  const [refinementCompleted, setRefinementCompleted] = useState<boolean>(false);

  // --- Обновление настроек ---
  const [isUpdatingSettings, setIsUpdatingSettings] = useState(false);

  // --- Модальные окна ---
  const [isResultsModalOpen, setIsResultsModalOpen] = useState<boolean>(false);
  const [initialFolderNameForModal, setInitialFolderNameForModal] = useState<string | null>(null);
  const [isAnalysisModalOpen, setIsAnalysisModalOpen] = useState<boolean>(false);
  const [isFractalDimensionModalOPen, setIsFractalDimensionModalOPen] = useState<boolean>(false);

  const [isEnvelopModalOpen, setEnvelopModalOpen] = useState(false);
  const [envelopGraphData, setEnvelopGraphData] = useState<string | null>(null);

  // --- Управление этапами ---
  const [currentStage, setCurrentStage] = useState<Stage>("stage1");

  return (
    <AppContext.Provider
      value={{
        // --- Сессия ---
        sessionId, setSessionId,

        // --- Изображение и отображение ---
        preview, setPreview, overlay, setOverlay, contours, setContours,

        // --- Состояния загрузки и галереи ---
        loading, setLoading, refreshGallery, setRefreshGallery,

        // --- Интерактивная сегментация (Main) ---
        points, setPoints,

        // --- Настройки препроцессинга (Main) ---
        medianKsize, setMedianKsize,
        contrastFactor, setContrastFactor,
        sharpnessFactor, setSharpnessFactor,
        claheClipLimit, setClaheClipLimit,
        claheTileGrid, setClaheTileGrid,

        // --- Режимы работы (Main) ---
        mode, setMode,
        interactiveSubMode, setInteractiveSubMode,

        // --- Внутренняя сегментация (Inner Box) ---
        innerBox, setInnerBox,
        innerContours, setInnerContours,
        selectedInnerContours, setSelectedInnerContours,
        hoverContour, setHoverContour,

        // --- Автогенерация (Auto) ---
        autoGenConfig, setAutoGenConfig,
        autoMasks, setAutoMasks,
        selectedMaskIds, setSelectedMaskIds,
        confirmedMasks, setConfirmedMasks,
        finalMasks, setFinalMasks,
        mainMaskId, setMainMaskId,
        selectionConfirmed, setSelectionConfirmed,
        refinementCompleted, setRefinementCompleted,

        // --- Обновление настроек ---
        isUpdatingSettings, setIsUpdatingSettings,

        // --- Модальные окна ---
        isResultsModalOpen, setIsResultsModalOpen,
        initialFolderNameForModal, setInitialFolderNameForModal,
        isAnalysisModalOpen, setIsAnalysisModalOpen,
        isFractalDimensionModalOPen, setIsFractalDimensionModalOPen,

        isEnvelopModalOpen, setEnvelopModalOpen,
        envelopGraphData, setEnvelopGraphData,

        // --- Управление этапами ---
        currentStage, setCurrentStage,
      }}
    >
      {children}
    </AppContext.Provider>
  );
};

// --- Экспорт контекста ---
export { AppContext };