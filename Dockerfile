FROM python:3.12-slim

# Отключаем буферизацию вывода Python (критично для логов в Docker!)
ENV PYTHONUNBUFFERED=1

# Устанавливаем базовую системную библиотеку, необходимую для бинарников rasterio
RUN apt-get update && apt-get install -y --no-install-recommends \
    libexpat1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app_root

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Копируем весь проект
COPY . .

# Прокидываем порты
EXPOSE 8000

# Запускаем FastAPI через uvicorn
WORKDIR /app_root/app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]