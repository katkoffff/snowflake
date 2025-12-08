// frontend\src\modals\AnalysisModal.tsx
import React, { useState, useEffect, useRef } from "react";
import { api } from "../api/client";
import { useApp } from "../hooks/useAppContext";
import "../css/analysis_modal.css";
import type { AnalysisResult, SaveChartRequest, PointData, AxesData, MiniatureData } from "../types/analysing";
import * as d3 from "d3";

// --- ИНТЕРФЕЙС ДЛЯ МИНИАТЮРЫ ---
interface MiniatureInfo {
  id: number;
  isVisible: boolean;
  x: number;
  y: number;
  dotX: number;
  dotY: number;
  // --- НОВОЕ: Состояние перетаскивания ---
  isDragging?: boolean;
}

const AnalysisModal: React.FC = () => {
  const { isFractalDimensionModalOPen, setIsFractalDimensionModalOPen } = useApp();

  const [results, setResults] = useState<AnalysisResult[]>([]);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  // --- Состояния для миниатюр ---
  const [miniatures, setMiniatures] = useState<Record<number, MiniatureInfo>>({});

  // --- Ref для контейнера графика ---
  const chartRef = useRef<HTMLDivElement>(null);
  const svgRef = useRef<SVGSVGElement | null>(null);
  const gRef = useRef<SVGGElement | null>(null);

  // --- НОВОЕ: Ref для перетаскивания ---
  const dragStartPos = useRef({ x: 0, y: 0 });
  const isDragging = useRef(false);
  const draggingMiniatureId = useRef<number | null>(null); // <-- ДОБАВЛЯЕМ: отслеживаем ID перетаскиваемой миниатюры
  const dragThreshold = 5;

  // --- Загрузка результатов ---
  useEffect(() => {
    if (isFractalDimensionModalOPen) {
      const fetchResults = async () => {
        setLoading(true);
        setError(null);
        try {
          const res = await api.get("/analysis_results");
          const data: AnalysisResult[] = res.data;
          setResults(data);
          setMiniatures({});
        } catch (err) {
          console.error("Failed to fetch analysis results:", err);
          setError(`Failed to load analysis results: ${(err as Error).message}`);
          setResults([]);
          setMiniatures({});
        } finally {
          setLoading(false);
        }
      };

      fetchResults();
    } else if (!isFractalDimensionModalOPen) {
      setResults([]);
      setError(null);
      setMiniatures({});
    }
  }, [isFractalDimensionModalOPen]);

  // --- НОВОЕ: Обработчики перетаскивания ---
  const handleMiniatureMouseDown = (idx: number, e: React.MouseEvent) => {
    e.stopPropagation();
    e.preventDefault();
    
    // Сохраняем начальную позицию и ID миниатюры
    dragStartPos.current = {
        x: e.clientX,
        y: e.clientY
    };
    isDragging.current = false;
    draggingMiniatureId.current = idx; // <-- ЗАПОМИНАЕМ ID

    // Добавляем обработчики на document
    document.addEventListener('mousemove', handleMouseMove);
    document.addEventListener('mouseup', handleMouseUp);

    // Устанавливаем состояние перетаскивания
    setMiniatures(prev => ({
        ...prev,
        [idx]: {
        ...prev[idx],
        isDragging: false
        }
    }));
    };

  const handleMouseMove = (e: MouseEvent) => {
    if (!isDragging.current && draggingMiniatureId.current !== null) {
        // Проверяем, превысили ли порог для начала перетаскивания
        const dx = e.clientX - dragStartPos.current.x;
        const dy = e.clientY - dragStartPos.current.y;
        const distance = Math.sqrt(dx * dx + dy * dy);
        
        if (distance > dragThreshold) {
        isDragging.current = true;
        
        // Используем сохраненный ID вместо поиска
        setMiniatures(prev => ({
            ...prev,
            [draggingMiniatureId.current!]: {
            ...prev[draggingMiniatureId.current!],
            isDragging: true
            }
        }));
        }
    }

    if (isDragging.current && draggingMiniatureId.current !== null) {
        const dx = e.clientX - dragStartPos.current.x;
        const dy = e.clientY - dragStartPos.current.y;
        
        setMiniatures(prev => ({
        ...prev,
        [draggingMiniatureId.current!]: {
            ...prev[draggingMiniatureId.current!],
            x: prev[draggingMiniatureId.current!].x + dx,
            y: prev[draggingMiniatureId.current!].y + dy
        }
        }));

        // Обновляем стартовую позицию для следующего движения
        dragStartPos.current = {
        x: e.clientX,
        y: e.clientY
        };
    }
    };

  const handleMouseUp = (e: MouseEvent) => {
    // Убираем обработчики
    document.removeEventListener('mousemove', handleMouseMove);
    document.removeEventListener('mouseup', handleMouseUp);

    // УБИРАЕМ логику удаления при клике
    if (isDragging.current && draggingMiniatureId.current !== null) {
        // Завершаем перетаскивание
        setMiniatures(prev => ({
        ...prev,
        [draggingMiniatureId.current!]: {
            ...prev[draggingMiniatureId.current!],
            isDragging: undefined
        }
        }));
    }

    // СБРАСЫВАЕМ СОСТОЯНИЕ ПЕРЕТАСКИВАНИЯ
    isDragging.current = false;
    draggingMiniatureId.current = null;
    };

    // Добавляем в компонент (рядом с другими функциями)

const handleSavePlot = async () => {
  if (!results || !svgRef.current || !gRef.current) return;

  try {
    // 1. Собираем точки из результатов
    const pointsData: PointData[] = results.map(result => ({
      x: result.log_perimetr,
      y: result.log_area,
      color: "blue"
    }));    

    // 2. Собираем данные осей    
    const xMin = d3.min(results, (d) => d.log_perimetr) || 0;
    const xMax = d3.max(results, (d) => d.log_perimetr) || 1;
    const yMin = d3.min(results, (d) => d.log_area) || 0;
    const yMax = d3.max(results, (d) => d.log_area) || 1;

    const xRange = xMax - xMin;
    const yRange = yMax - yMin;
    const axesData: AxesData = {
      x_label: "Ln(Perimeter)",
      y_label: "Ln(Area)", 
      x_range: [xMin - xRange * 0.05, xMax + xRange * 0.05] as [number, number],
      y_range: [yMin - yRange * 0, yMax + yRange * 0.25] as [number, number]
    };

    // 3. Создаем шкалы для преобразования координат
    const margin = { top: 20, right: 30, bottom: 40, left: 50 };
    const container = chartRef.current!;
    const width = container.clientWidth - margin.left - margin.right;
    const height = container.clientHeight - margin.top - margin.bottom;

    const xScale = d3.scaleLinear()
      .domain(axesData.x_range)
      .range([0, width]);

    const yScale = d3.scaleLinear()  
      .domain(axesData.y_range)
      .range([height, 0]);

    // 4. Собираем миниатюры с РЕАЛЬНЫМИ координатами
    const miniaturesData = Object.values(miniatures)
      .filter(info => info.isVisible)
      .map(info => {
        const result = results[info.id];
        
        // РЕАЛЬНОЕ ПРЕОБРАЗОВАНИЕ КООРДИНАТ КАК В ЛИНИЯХ
        const miniatureElement = document.querySelector(`[data-miniature-id="${info.id}"]`);
        if (!miniatureElement) return null;

        const miniatureRect = miniatureElement.getBoundingClientRect();
        const miniatureCenterX = miniatureRect.left + miniatureRect.width / 2;
        const miniatureCenterY = miniatureRect.top + miniatureRect.height / 2;

        const svgPoint = svgRef.current!.createSVGPoint();
        svgPoint.x = miniatureCenterX;
        svgPoint.y = miniatureCenterY;

        const screenToSvgMatrix = svgRef.current!.getScreenCTM();
        if (!screenToSvgMatrix) return null;

        const pointInSvg = svgPoint.matrixTransform(screenToSvgMatrix.inverse());

        const gMatrix = gRef.current!.getCTM();
        if (!gMatrix) return null;

        const pointInG = pointInSvg.matrixTransform(gMatrix.inverse());

        // ПРЕОБРАЗУЕМ В СИСТЕМУ ДАННЫХ
        const dataX = xScale.invert(pointInG.x);
        const dataY = yScale.invert(pointInG.y);

        return {
          image_path: result.session_folder,
          image_file: 'main.jpg',
          display_x: info.x,
          display_y: info.y, 
          display_width: 64,
          display_height: 64,
          dot_x: result.log_perimetr,  // координаты точки
          dot_y: result.log_area,       // координаты точки
          svg_x: dataX,  // РЕАЛЬНЫЕ координаты миниатюры в системе данных!
          svg_y: dataY   // РЕАЛЬНЫЕ координаты миниатюры в системе данных!
        };
      }).filter(Boolean) as MiniatureData[];

    // 4. Формируем запрос
    const formData: SaveChartRequest = {
      points: pointsData,
      axes: axesData,
      miniatures: miniaturesData,
      viewport_size: {
        width: window.innerWidth,
        height: window.innerHeight
      },
      chart_type: 'dimension'
    };

    const response = await api.post('/analysis_save_chart', formData);
    
    if (response.data.status === "success") {
      alert('Chart saved successfully!');
    } else {
      alert('Failed to save chart: ' + response.data.message);
    }

  } catch (error) {
    console.error('Failed to save chart:', error);
    alert('Failed to save chart');
  }
};
  // --- Отрисовка графика ---
  useEffect(() => {
    if (!isFractalDimensionModalOPen || !chartRef.current || results.length === 0) return;

    const container = chartRef.current;
    d3.select(container).selectAll("svg").remove();

    const margin = { top: 20, right: 30, bottom: 40, left: 50 };
    const width = container.clientWidth - margin.left - margin.right;
    const height = container.clientHeight - margin.top - margin.bottom;

    const svg = d3
      .select(container)
      .append("svg")
      .attr("width", width + margin.left + margin.right)
      .attr("height", height + margin.top + margin.bottom)
      .attr("viewBox", [0, 0, width + margin.left + margin.right, height + margin.top + margin.bottom])
      .attr("font-family", "sans-serif");

    svgRef.current = svg.node() as SVGSVGElement;

    const g = svg
      .append("g")
      .attr("transform", `translate(${margin.left},${margin.top})`);

    gRef.current = g.node() as SVGGElement;

    // --- Определение шкал ---
    // --- Определение шкал ---
    const xMin = d3.min(results, (d) => d.log_perimetr) || 0;
    const xMax = d3.max(results, (d) => d.log_perimetr) || 1;
    const yMin = d3.min(results, (d) => d.log_area) || 0;
    const yMax = d3.max(results, (d) => d.log_area) || 1;

    // --- ФИКС РАСПРЕДЕЛЕНИЯ ОСЕЙ ---
    // Добавляем небольшой отступ к доменам для лучшего визуального восприятия
    const xRange = xMax - xMin;
    const yRange = yMax - yMin;
    const xDomain = [xMin - xRange * 0.05, xMax + xRange * 0.05];
    const yDomain = [yMin - yRange * 0, yMax + yRange * 0.25];
    // --- /ФИКС РАСПРЕДЕЛЕНИЯ ОСЕЙ ---

    const xScale = d3
        .scaleLinear()
        .domain(xDomain) // <-- Используем исправленные домены
        .range([0, width])
        .nice(); // <-- Добавляем nice для красивых значений на осях

    const yScale = d3
        .scaleLinear()
        .domain(yDomain) // <-- Используем исправленные домены
        .range([height, 0])
        .nice(); // <-- Добавляем nice для красивых значений на осях

    // --- Оси ---
    const xAxis = d3.axisBottom(xScale).ticks(6).tickFormat(d3.format(".2f"));
    const yAxis = d3.axisLeft(yScale).ticks(6).tickFormat(d3.format(".2f"));

    g.append("g")
      .attr("transform", `translate(0,${height})`)
      .call(xAxis)
      .append("text")
      .attr("x", width / 2)
      .attr("y", 30)
      .attr("fill", "currentColor")
      .attr("text-anchor", "middle")
      .text("Ln(Perimeter)");

    g.append("g")
      .call(yAxis)
      .append("text")
      .attr("transform", "rotate(-90)")
      .attr("y", -40)
      .attr("x", -height / 2)
      .attr("fill", "currentColor")
      .attr("text-anchor", "middle")
      .text("Ln(Area)");

    // --- Рисование точек ---
    const dots = g.selectAll(".dot")
      .data(results)
      .enter()
      .append("circle")
      .attr("class", "dot")
      .attr("cx", (d) => xScale(d.log_perimetr))
      .attr("cy", (d) => yScale(d.log_area))
      .attr("r", 5)
      .attr("fill", "blue")
      .attr("stroke", "white")
      .attr("stroke-width", 1)
      .style("cursor", "pointer");

    // --- Обработчик кликов по точкам ---
    // --- Обработчик кликов по точкам ---
    dots.on("click", (event, d) => {
        const index = results.indexOf(d);
        if (index === -1) return;

        setMiniatures(prev => {
            const currentInfo = prev[index];
            const newIsVisible = !currentInfo?.isVisible;

            const dotX = xScale(d.log_perimetr);
            const dotY = yScale(d.log_area);

            const containerRect = container.getBoundingClientRect();
            const offsetX = 10;
            const offsetY = -10;

            let newX = containerRect.left + margin.left + dotX + offsetX;
            let newY = containerRect.top + margin.top + dotY + offsetY;

            // --- ФИКС ГРАНИЦ ОКНА ---
            const viewportWidth = window.innerWidth;
            const viewportHeight = window.innerHeight;
            const miniatureWidth = 64; // w-16 в CSS = 64px
            const miniatureHeight = 64; // h-16 в CSS = 64px

            // Проверяем правую границу
            if (newX + miniatureWidth > viewportWidth) {
            newX = viewportWidth - miniatureWidth - 10; // Отступ от края
            }
            // Проверяем нижнюю границу
            if (newY + miniatureHeight > viewportHeight) {
            newY = viewportHeight - miniatureHeight - 10; // Отступ от края
            }
            // Проверяем левую границу (на всякий случай)
            if (newX < 10) {
            newX = 10;
            }
            // Проверяем верхнюю границу (на всякий случай)
            if (newY < 10) {
            newY = 10;
            }
            // --- /ФИКС ГРАНИЦ ОКНА ---

            return {
            ...prev,
            [index]: {
                id: index,
                isVisible: newIsVisible,
                x: newX,
                y: newY,
                dotX: dotX,
                dotY: dotY
            }
            };
        });
        });

    return () => {
      svgRef.current = null;
      gRef.current = null;
    };

  }, [results, isFractalDimensionModalOPen]);

  // --- Отрисовка сносок ---
  useEffect(() => {
    if (!isFractalDimensionModalOPen || !svgRef.current || !gRef.current || !results) return;

    const svg = d3.select(svgRef.current);
    const g = d3.select(gRef.current);

    g.selectAll(".annotation-line").remove();

    Object.values(miniatures).forEach(info => {
      if (!info.isVisible) return;

      const dotXInG = info.dotX;
      const dotYInG = info.dotY;

      const miniatureElement = document.querySelector(`[data-miniature-id="${info.id}"]`);
      if (!miniatureElement) return;

      const svgRect = svgRef.current!.getBoundingClientRect(); // Исправляем ошибку TS с !
      const miniatureRect = miniatureElement.getBoundingClientRect();

      const miniatureCenterX = miniatureRect.left + miniatureRect.width / 2;
      const miniatureCenterY = miniatureRect.top + miniatureRect.height / 2;

      const svgPoint = svgRef.current!.createSVGPoint(); // Исправляем ошибку TS с !
      svgPoint.x = miniatureCenterX;
      svgPoint.y = miniatureCenterY;

      const screenToSvgMatrix = svgRef.current!.getScreenCTM(); // Исправляем ошибку TS с !
      if (!screenToSvgMatrix) return;

      const pointInSvg = svgPoint.matrixTransform(screenToSvgMatrix.inverse());

      const gMatrix = gRef.current!.getCTM(); // Исправляем ошибку TS с !
      if (!gMatrix) return;

      const pointInG = pointInSvg.matrixTransform(gMatrix.inverse());

      if (!isNaN(dotXInG) && !isNaN(dotYInG) && !isNaN(pointInG.x) && !isNaN(pointInG.y)) {
        g.append("line")
          .attr("class", "annotation-line")
          .attr("x1", dotXInG)
          .attr("y1", dotYInG)
          .attr("x2", pointInG.x)
          .attr("y2", pointInG.y)
          .attr("stroke", "gray")
          .attr("stroke-width", 1)
          .attr("stroke-dasharray", "4 2");
      }
    });

  }, [miniatures, isFractalDimensionModalOPen, results]);

  // --- Убираем старый обработчик клика ---
  // handleMiniatureClick больше не нужен

  const closeModal = () => {
    setIsFractalDimensionModalOPen(false);
  };

  if (!isFractalDimensionModalOPen) {
    return null;
  }

  return (
    <div className="analysis-modal-overlay" onClick={closeModal}>
      <div className="analysis-modal-content" onClick={(e) => e.stopPropagation()}>
        <div className="analysis-modal-header">
          <h2>Fractal Dimension</h2>
        </div>

        <div className="analysis-modal-body">
          {loading && <div className="analysis-modal-loading">Loading analysis data...</div>}
          {error && <div className="analysis-modal-error">{error}</div>}

          {!loading && !error && results.length > 0 && (
            <div className="analysis-chart-container" ref={chartRef}>
              {Object.entries(miniatures).map(([idxStr, info]) => {
                if (!info.isVisible) return null;

                const idx = parseInt(idxStr);
                const result = results[idx];
                if (!result) return null;

                const imgSrc = `http://localhost:8000/api/results/image_in_dir?dir_name=${encodeURIComponent(result.session_folder)}&file_name=main.jpg`;

                return (
                  <div
                    key={`miniature-${idx}`}
                    className={`analysis-miniature ${info.isDragging ? 'analysis-miniature-dragging' : ''}`}
                    style={{ left: `${info.x}px`, top: `${info.y}px` }}
                    onMouseDown={(e) => handleMiniatureMouseDown(idx, e)}
                    data-miniature-id={idx}
                >
                    {/* КНОПКА ЗАКРЫТИЯ */}
                    <button 
                    className="analysis-miniature-close"
                    onClick={(e) => {
                        e.stopPropagation();
                        setMiniatures(prev => ({
                        ...prev,
                        [idx]: {
                            ...prev[idx],
                            isVisible: false
                        }
                        }));
                    }}
                    >
                    ×
                    </button>
                    {/* 🔥 НАЗВАНИЕ МИНИАТЮРЫ (SESSION FOLDER) */}
                    <div className="analysis-miniature-label">
                      {result.session_folder}
                    </div>
                    {/* 🔥 Фрактальная размерность (fractal_dimension) */}
                    <div className="analysis-miniature-data">
                      {`Dimension: ${result.fractal_dimension}`}
                    </div>
                    <img
                    src={imgSrc}
                    alt={`Preview from ${result.session_folder}`}
                    className="analysis-miniature-img"
                    />
                  </div>
                );
              })}
            </div>
          )}

          {!loading && !error && results.length === 0 && (
            <div className="analysis-modal-error">No analysis results found.</div>
          )}
        </div>

        <div className="analysis-modal-footer">
          <button onClick={handleSavePlot} disabled={loading || results.length === 0}>
            Save Plot
          </button>
          <button onClick={closeModal}>
            Close
          </button>
        </div>
      </div>
    </div>
  );
};

export default AnalysisModal;