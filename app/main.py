import os
import warnings

# Гасим ворнинги от rio-tiler (про float32 и PNG)
warnings.filterwarnings("ignore", category=UserWarning)

from fastapi import FastAPI, HTTPException, Response
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from typing import Optional

from rio_tiler.io import Reader
from core import SphericalRasterRouter

app = FastAPI(title="Maritime Routing API")

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "processed"))

eff_vel_path = os.path.join(DATA_DIR, "effective_velocity_knots.tif")
eff_sd_path = os.path.join(DATA_DIR, "effective_sd_knots.tif")

try:
    router_instance = SphericalRasterRouter(eff_vel_path, eff_sd_path)
except Exception as e:
    print(f"ОШИБКА: {e}")
    router_instance = None

class RouteRequest(BaseModel):
    start_lon: float = Field(...)
    start_lat: float = Field(...)
    end_lon: float = Field(...)
    end_lat: float = Field(...)
    
    vessel_type: str = Field(default="all")
    vessel_length: Optional[float] = Field(default=None)
    vessel_width: Optional[float] = Field(default=None)
    vessel_draft: Optional[float] = Field(default=None)
    allow_suez: bool = Field(default=True)
    avoid_seca: bool = Field(default=False)

@app.post("/api/route")
async def calculate_route(req: RouteRequest):
    if not router_instance: raise HTTPException(status_code=500, detail="Бэкенд не инициализирован")

    try:
        # Вызываем с дефолтным таймаутом 15 секунд (можно передать max_search_time=20.0, если нужно больше)
        result = router_instance.find_route(
            (req.start_lon, req.start_lat),
            (req.end_lon, req.end_lat)
        )
        if not result: raise HTTPException(status_code=404, detail="Путь прегражден сушей.")
        
        return result.to_geojson(request_params=req.model_dump())
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except TimeoutError as e:
        # Перехватываем наш кастомный таймаут
        raise HTTPException(status_code=408, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/tiles/{layer}/{z}/{x}/{y}.png")
def get_tile(layer: str, z: int, x: int, y: int):
    filepath = eff_vel_path if layer == "speed" else eff_sd_path
    if not os.path.exists(filepath): return Response(status_code=404)
    try:
        with Reader(filepath) as src:
            img = src.tile(x, y, z)
            png_bytes = img.render(img_format="PNG", colormap_name="viridis", rescale=((0, 30),))
            return Response(content=png_bytes, media_type="image/png")
    except Exception:
        return Response(status_code=204)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

@app.get("/")
async def serve_index():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))

if __name__ == "__main__":
    import uvicorn
    # ИСПРАВЛЕНИЕ ДВОЙНОЙ ЗАГРУЗКИ: передаем инстанс app вместо строки "main:app"
    uvicorn.run(app, host="0.0.0.0", port=8000)