# ❄ Snowflakes — SAM Segmentation Demo

Интерактивная сегментация изображений снежинок с помощью Segment Anything Model 2 (SAM 2).

## 🚀 Быстрый старт

### 1️⃣ Клонируем
```bash
git clone https://github.com/katkoffff/snowflake.git # Замени на свой URL, если изменишь
cd snowflake

# Создаём и активируем conda-окружение
conda env create -f environment.yml
conda activate snowflake

# Устанавливаем PyTorch с поддержкой CUDA 12.6 (проверь совместимость версий)
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu126

# Устанавливаем SAM 2 из репозитория
git clone https://github.com/facebookresearch/sam2.git
cd sam2
pip install -e .
cd .. # Возвращаемся в корень проекта

Скачайте чекпоинт sam2.1_hiera_large.pt (или другой, если хочешь использовать меньшую модель).
https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt
Положить файл в:
backend/models/sam2.1_hiera_large.pt
Создайте папку backend/models, если её нет.

cd backend
uvicorn main:app --reload

cd frontend # или перейдите в папку фронтенда
npm install
npm run dev

Открыть http://localhost:5173
