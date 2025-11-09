// frontend\src\context\AppContext.tsx
import React, { createContext, useState } from "react";
import type { AutoGenMask, AutoGenConfig, ConfirmedMask } from "../types/autogen";

// --- Типы ---
export type Pt = { x: number; y: number; label: number }; // label: 1 = object (left), 0 = background (right)

export type AppContextType = {
  // --- Существующие поля ---
  sessionId: string | null;
  setSessionId: (id: string | null) => void;
  preview: string | null;
  setPreview: (p: string | null) => void;
  overlay: string | null;
  setOverlay: (p: string | null) => void;
  contours: number[][][]; // list of contours (each is [[x,y],...])
  setContours: (c: number[][][]) => void;
  loading: boolean;
  setLoading: (v: boolean) => void;
  refreshGallery: boolean;
  setRefreshGallery: (v: boolean) => void;

  // точки для интерактивной сегментации
  points: Pt[];
  setPoints: (pts: Pt[]) => void;

  // подробные настройки препроцессинга — по отдельности (чтобы UI мог их менять)
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

    // --- Новый режим работы ---
  mode: 'interactive' | 'auto';
  setMode: React.Dispatch<React.SetStateAction<'interactive' | 'auto'>>;

  interactiveSubMode: "main" | "inner";
  setInteractiveSubMode: (m: "main" | "inner") => void;

  innerBox: [number, number, number, number] | null;
  setInnerBox: (b: [number, number, number, number] | null) => void;    

  // Настройки автогенерации
  autoGenConfig: AutoGenConfig;
  setAutoGenConfig: React.Dispatch<React.SetStateAction<AutoGenConfig>>;

  // Результаты автогенерации
  autoMasks: AutoGenMask[] | null;
  setAutoMasks: (masks: AutoGenMask[] | null) => void;

  // Выбранные маски (ID)
  selectedMaskIds: string[];
  setSelectedMaskIds: React.Dispatch<React.SetStateAction<string[]>>;

  // Подтверждённые маски (после "Подтвердить выбор")
  confirmedMasks: ConfirmedMask[];
  setConfirmedMasks: (masks: ConfirmedMask[]) => void;

  // Финальные маски (после уточнения и сохранения)
  finalMasks: ConfirmedMask[];
  setFinalMasks: (masks: ConfirmedMask[]) => void;

  // ID основной маски (main)
  mainMaskId: string | null;
  setMainMaskId: (id: string | null) => void;

  // Флаг, указывающий, что выбор масок завершён
  selectionConfirmed: boolean;
  setSelectionConfirmed: (confirmed: boolean) => void;

  // Флаг, указывающий, что уточнение масок завершено
  refinementCompleted: boolean;
  setRefinementCompleted: (completed: boolean) => void;

  innerContours: number[][][];          // массив контуров
  setInnerContours: (arr: number[][][]) => void;

  selectedInnerContours: number[];      // индексы выбранных
  setSelectedInnerContours: React.Dispatch<React.SetStateAction<number[]>>;

  hoverContour: number | null;          // какой контур подсвечивается
  setHoverContour: (v: number | null) => void;

  isUpdatingSettings: boolean;
  setIsUpdatingSettings: (v: boolean) => void;

};
  

// --- Создание контекста ---
const AppContext = createContext<AppContextType | null>(null);

// --- Компонент-провайдер ---
export const AppProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  // --- Существующие состояния ---
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [overlay, setOverlay] = useState<string | null>(null);
  const [contours, setContours] = useState<number[][][]>([]);
  const [loading, setLoading] = useState(false);
  const [refreshGallery, setRefreshGallery] = useState(false);

  // точки (итеративные клики)
  const [points, setPoints] = useState<Pt[]>([]);

  // настройки препроцессинга (defaults must match backend defaults)
  const [medianKsize, setMedianKsize] = useState<number>(5);
  const [contrastFactor, setContrastFactor] = useState<number>(1.5);
  const [sharpnessFactor, setSharpnessFactor] = useState<number>(2.0);
  const [claheClipLimit, setClaheClipLimit] = useState<number>(1.5);
  const [claheTileGrid, setClaheTileGrid] = useState<[number, number]>([8, 8]);

    // режим работы: interactive / auto / inner
  const [mode, setMode] = useState<'interactive' | 'auto'>('interactive');

  const [interactiveSubMode, setInteractiveSubMode] =
  useState<"main" | "inner">("main");

  const [innerBox, setInnerBox] =
  useState<[number, number, number, number] | null>(null);
  
  const [innerContours, setInnerContours] = useState<number[][][]>([]);
  const [selectedInnerContours, setSelectedInnerContours] = useState<number[]>([]);
  const [hoverContour, setHoverContour] = useState<number | null>(null);

  const [autoGenConfig, setAutoGenConfig] = useState<AutoGenConfig>({
    // Установим дефолтные значения для автогенерации, как в примере
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

  const [isUpdatingSettings, setIsUpdatingSettings] = useState(false);

  return (
    <AppContext.Provider
      value={{
        // --- Существующие значения ---
        sessionId,
        setSessionId,
        preview,
        setPreview,
        overlay,
        setOverlay,
        contours,
        setContours,
        loading,
        setLoading,
        refreshGallery,
        setRefreshGallery,

        points,
        setPoints,

        medianKsize,
        setMedianKsize,
        contrastFactor,
        setContrastFactor,
        sharpnessFactor,
        setSharpnessFactor,
        claheClipLimit,
        setClaheClipLimit,
        claheTileGrid,
        setClaheTileGrid,

        // --- Новые значения ---
        mode,
        setMode,
        interactiveSubMode,
        setInteractiveSubMode,
        innerBox,
        setInnerBox,

        innerContours,
        setInnerContours,

        selectedInnerContours,
        setSelectedInnerContours,

        hoverContour,
        setHoverContour,

        autoGenConfig,
        setAutoGenConfig,
        autoMasks,
        setAutoMasks,
        selectedMaskIds,
        setSelectedMaskIds,
        confirmedMasks,
        setConfirmedMasks,
        finalMasks,
        setFinalMasks,
        mainMaskId,
        setMainMaskId,
        selectionConfirmed,
        setSelectionConfirmed,
        refinementCompleted,
        setRefinementCompleted,
        isUpdatingSettings,
        setIsUpdatingSettings
      }}
    >
      {children}
    </AppContext.Provider>
  );
};

// --- Экспорт контекста ---
export { AppContext };