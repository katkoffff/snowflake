import React, { createContext, useContext, useState } from "react";

type Pt = { x: number; y: number; label: number }; // label: 1 = object (left), 0 = background (right)

type AppContextType = {
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
};

const AppContext = createContext<AppContextType | null>(null);

export const AppProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
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

  return (
    <AppContext.Provider
      value={{
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
      }}
    >
      {children}
    </AppContext.Provider>
  );
};

export const useApp = () => {
  const ctx = useContext(AppContext);
  if (!ctx) throw new Error("useApp must be inside AppProvider");
  return ctx;
};
