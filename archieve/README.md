# ❄ Snowflakes — SAM Segmentation Demo

Интерактивная сегментация изображений снежинок с помощью Segment Anything (SAM).

## 🚀 Быстрый старт

### 1️⃣ Клонируем
```bash
git clone https://github.com/<your-org>/snowflakes.git
cd snowflakes
```

### 2️⃣ Устанавливаем окружение
```bash
conda env create -f environment.yml
conda activate snowflake
```

### 3️⃣ Скачиваем модель SAM (ViT-B)
[https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth)

Положить файл в:
```
backend/models/sam_vit_b_01ec64.pth
```

### 4️⃣ Запускаем бэкенд
```bash
cd backend
uvicorn main:app --reload
```

### 5️⃣ Запускаем фронтенд
```bash
cd ../frontend
npm install
npm run dev
```
Открыть [http://localhost:5173](http://localhost:5173)
