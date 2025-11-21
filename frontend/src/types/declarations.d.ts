// Расширяем интерфейс для HTML-атрибутов input
declare namespace React {
  interface InputHTMLAttributes<T> extends DOMAttributes<T> {
    // Добавляем webkitdirectory как строковое свойство
    webkitdirectory?: string;
    directory?: string;
  }
}