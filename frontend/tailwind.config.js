/** @type {import('tailwindcss').Config} */
export default {
  // Убираем content, пусть vite + postcss определяют автоматически
  // content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'], // Не нужно при postcss + vite
  theme: {
    extend: {
      // Тут можно расширять тему, если нужно
    },
  },
  plugins: [
    // Тут можно подключать плагины, если нужно
  ],
  // Опционально: отключение префиксов, сброс стилей и т.д.
  // corePlugins: {
  //   preflight: false, // отключить базовые стили (reset)
  // }
}