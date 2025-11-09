// Тип для одного элемента результата из /results/list
export interface GalleryResult {
  name: string;      // Имя файла (например, "image_1_segmented.jpg")
  size_kb: number;   // Размер в килобайтах (например, 123.4)
  mtime: string;     // Время модификации (например, "2025-04-12 12:16:42")
}

// Тип для ответа от /results/list
export interface GalleryListResponse {
  page: number;          // Номер текущей страницы
  per_page: number;      // Количество элементов на странице
  total: number;         // Общее количество элементов
  results: GalleryResult[]; // Массив результатов
}

// Тип для ответа от /load_result
export interface LoadResultResponse {
  session_id: string;  // ID новой сессии
  preview_b64: string; // base64 строка PNG-изображения с контуром
  contours: [number, number][][]; // Массив контуров, каждый контур - массив точек [x, y]
  // error?: string;   // Опционально: поле error, если оно возвращается в случае ошибки
}