// --- Тип для результатов анализа ---
export interface AnalysisResult {
  session_folder: string;
  normalized_perimeter: number;
  normalized_area: number;
  main_enclosing_radius: number;
  log_perimetr: number;
  log_area: number;
  fractal_dimension: number;
  main_image_path: string; // Относительный путь
}

export interface PointData {
  x: number;
  y: number;
  color?: string;
}

export interface AxesData {
  x_label: string;
  y_label: string;
  x_range: [number, number];
  y_range: [number, number];
}

export interface MiniatureData {
  image_path: string;
  image_file: string;
  display_x: number;
  display_y: number;
  display_width: number;
  display_height: number;
  dot_x: number;
  dot_y: number;
  svg_x: number;
  svg_y: number;
}

export interface SaveChartRequest {
  points: PointData[];
  axes: AxesData;
  miniatures: MiniatureData[];
  viewport_size: { 
    width: number;
    height: number;
  };
  chart_type: string;
}

