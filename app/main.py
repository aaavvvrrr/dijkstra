import os
import json
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from shapely.geometry import shape
import rasterio.features
from rasterio.transform import from_bounds

from core import SphericalRasterRouter

app = FastAPI(title="Maritime Routing API")

# Глобальные координаты для всей планеты
LAT_BOUNDS = (-90.0, 90.0)
LON_BOUNDS = (-180.0, 180.0)
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
GEOJSON_PATH = os.path.join(STATIC_DIR, "oceans-seas.geo.json")

def create_raster_from_geojson() -> np.ndarray:
    """Читает GeoJSON маску океанов и превращает её в numpy-растр."""
    if not os.path.exists(GEOJSON_PATH):
        raise FileNotFoundError(f"Файл {GEOJSON_PATH} не найден в папке static.")

    print("Загрузка и растеризация GeoJSON... (это займет пару секунд)")
    with open(GEOJSON_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Извлекаем все полигоны из GeoJSON
    if 'features' in data:
        geometries = [shape(feature['geometry']) for feature in data['features']]
    else:
        geometries = [shape(data)]

    # Задаем размер сетки. 900x1800 дает шаг сетки 0.2 градуса (~22 км на экваторе)
    # Этого достаточно для быстрого тестирования. 
    rows, cols = 900, 1800
    
    # Настраиваем трансформацию координат (west, south, east, north)
    transform = from_bounds(-180.0, -90.0, 180.0, 90.0, cols, rows)
    
    # Растеризуем геометрию: Море = 30.0 км/ч, Суша = 0.0
    raster = rasterio.features.rasterize(
        geometries,
        out_shape=(rows, cols),
        transform=transform,
        fill=0.0,             # Фон (суша)
        default_value=30.0,   # Значение внутри полигонов (океан)
        dtype=np.float32
    )
    print("Растеризация успешно завершена!")
    return raster

# Инициализируем маршрутизатор
router_instance = SphericalRasterRouter(create_raster_from_geojson(), LAT_BOUNDS, LON_BOUNDS)

# --- Модели API ---
class RouteRequest(BaseModel):
    start_lon: float
    start_lat: float
    end_lon: float
    end_lat: float

@app.post("/api/route")
async def calculate_route(req: RouteRequest):
    try:
        result = router_instance.find_route(
            (req.start_lon, req.start_lat),
            (req.end_lon, req.end_lat)
        )
        if not result:
            raise HTTPException(status_code=404, detail="Маршрут не найден. Возможно, путь прегражден сушей.")
        return result.to_geojson()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка сервера: {str(e)}")

# --- Раздача Frontend ---
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

@app.get("/")
async def serve_index():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))

if __name__ == "__main__":
    import uvicorn
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(script_dir)
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)