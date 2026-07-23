import os
import asyncio
import warnings
import io
warnings.filterwarnings("ignore", category=UserWarning)

from fastapi import FastAPI, HTTPException, Response, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from typing import Optional
from PIL import Image
import numpy as np

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
    
# =======================================================
# API ОТЛАДКИ РАСТРА
# =======================================================
@app.get("/api/debug/bounds")
def get_debug_bounds():
    if not router_instance: return {}
    return {
        "min_lat": router_instance.min_lat,
        "max_lat": router_instance.max_lat,
        "min_lon": router_instance.min_lon,
        "max_lon": router_instance.max_lon
    }

@app.get("/api/debug/coarse_grid.png")
def get_coarse_grid_png():
    if not router_instance: return Response(status_code=500)
    
    comp = router_instance.components
    main_id = router_instance.main_ocean_id
    
    img_arr = np.zeros((comp.shape[0], comp.shape[1], 4), dtype=np.uint8)
    # Синий для Мирового океана
    img_arr[comp == main_id] = [59, 130, 246, 180]
    # Красный для изолированных морей/озер
    img_arr[(comp > 0) & (comp != main_id)] = [239, 68, 68, 200]
    
    img = Image.fromarray(img_arr)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return Response(content=buf.getvalue(), media_type="image/png")

# =======================================================
# WEBSOCKETS И TILE SERVER
# =======================================================
@app.websocket("/api/ws/route")
async def websocket_route(websocket: WebSocket):
    await websocket.accept()
    if not router_instance:
        await websocket.send_json({"type": "error", "message": "Бэкенд не инициализирован"})
        await websocket.close()
        return

    cancel_flag = False
    def is_cancelled(): return cancel_flag

    async def on_progress(explored: int, current_path: list, time_val: float):
        clean_path = [[float(p[0]), float(p[1])] for p in current_path]
        await websocket.send_json({"type": "progress", "explored": explored, "current_time": float(time_val), "path": clean_path})

    async def listen_for_cancel():
        nonlocal cancel_flag
        try:
            while True:
                msg = await websocket.receive_json()
                if msg.get("action") == "cancel": cancel_flag = True
        except WebSocketDisconnect:
            cancel_flag = True

    try:
        msg = await websocket.receive_json()
        if msg.get("action") == "start":
            listener_task = asyncio.create_task(listen_for_cancel())
            try:
                result = await router_instance.find_route(
                    (msg["start_lon"], msg["start_lat"]),
                    (msg["end_lon"], msg["end_lat"]),
                    progress_callback=on_progress,
                    check_cancel_callback=is_cancelled
                )
                if result:
                    await websocket.send_json({"type": "result", "geojson": result.to_geojson(request_params=msg)})
            except InterruptedError as e:
                await websocket.send_json({"type": "info", "message": str(e)})
            except ValueError as e:
                await websocket.send_json({"type": "error", "message": str(e)})
            except Exception as e:
                await websocket.send_json({"type": "error", "message": f"Ошибка сервера: {str(e)}"})
            finally:
                listener_task.cancel()
    except WebSocketDisconnect: pass

# Замените эндпоинт get_tile на этот код:

@app.get("/tiles/{layer}/{z}/{x}/{y}.png")
def get_tile(layer: str, z: int, x: int, y: int):
    # Выбираем правильный файл
    if layer == "speed":
        filepath = eff_vel_path
    elif layer == "sd":
        filepath = eff_sd_path
    elif layer == "debug":
        filepath = router_instance.debug_tif_path if router_instance else None
    else:
        return Response(status_code=404)

    if not filepath or not os.path.exists(filepath): 
        return Response(status_code=404)

    try:
        with Reader(filepath) as src:
            # Магия здесь: nodata=0 автоматически маскирует нули (сушу) в прозрачность!
            img = src.tile(x, y, z, nodata=0)
            
            if layer == "debug":
                # Кастомная раскраска: Мировой океан - синий, изолированные озера - красные
                cmap = {0: (0, 0, 0, 0)} # Суша прозрачная
                for i in range(1, router_instance.num_features + 2):
                    if i == router_instance.main_ocean_id:
                        cmap[i] = (59, 130, 246, 180) # Синий
                    else:
                        cmap[i] = (239, 68, 68, 200)  # Красный
                png_bytes = img.render(img_format="PNG", colormap=cmap)
                
            elif layer == "speed":
                # turbo - от синего (медленно) к красному (быстро)
                png_bytes = img.render(img_format="PNG", colormap_name="turbo", rescale=((5, 30),))
            else:
                # plasma - от темно-фиолетового к желтому (высокая дисперсия)
                png_bytes = img.render(img_format="PNG", colormap_name="plasma", rescale=((0, 5),))
                
            return Response(content=png_bytes, media_type="image/png")
            
    except Exception as e:
        import traceback
        print(f"❌ ОШИБКА РЕНДЕРА ТАЙЛА '{layer}' (z={z}, x={x}, y={y}):")
        traceback.print_exc()
        return Response(status_code=204)    

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

@app.get("/")
async def serve_index(): return FileResponse(os.path.join(STATIC_DIR, "index.html"))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)