const handleSavePlot = async () => {
  if (!results || !svgRef.current || !gRef.current) return;

  try {
    // 1. Собираем точки из результатов
    const pointsData: PointData[] = results.map(result => ({
      x: result.normalized_perimeter,
      y: result.normalized_area,
      color: "blue"
    }));

    // 2. Собираем данные осей    
    const xMin = d3.min(results, (d) => d.normalized_perimeter) || 0;
    const xMax = d3.max(results, (d) => d.normalized_perimeter) || 1;
    const yMin = d3.min(results, (d) => d.normalized_area) || 0;
    const yMax = d3.max(results, (d) => d.normalized_area) || 1;

    const xRange = xMax - xMin;
    const yRange = yMax - yMin;
    const axesData: AxesData = {
      x_label: "Normalized Perimeter (L/Lc)",
      y_label: "Normalized Area (S/Sc)", 
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
          dot_x: result.normalized_perimeter,  // координаты точки
          dot_y: result.normalized_area,       // координаты точки
          svg_x: dataX,  // РЕАЛЬНЫЕ координаты миниатюры в системе данных!
          svg_y: dataY   // РЕАЛЬНЫЕ координаты миниатюры в системе данных!
        };
      }).filter(Boolean) as MiniatureData[];

    // 5. Формируем запрос
    const formData: SaveChartRequest = {
      points: pointsData,
      axes: axesData,
      miniatures: miniaturesData,
      viewport_size: {
        width: window.innerWidth,
        height: window.innerHeight
      }
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