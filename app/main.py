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
from datetime import datetime
import matplotlib.cm as cm

from rio_tiler.io import Reader
from rio_tiler.colormap import cmap
from core import SphericalRasterRouter

app = FastAPI(title="Maritime Routing API")

import json

# Ищем конфиг в корне проекта или по пути из ENV
CONFIG_PATH = os.getenv("APP_CONFIG_PATH", os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "config.json")))
with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    APP_CONFIG = json.load(f)

STATIC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", APP_CONFIG["paths"]["static_dir"]))
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", APP_CONFIG["paths"]["processed_dir"]))

router_instance = None
SERVER_STATUS = {"status": "starting", "message": "Запуск FastAPI и подготовка окружения..."}

def _update_status(msg: str):
    SERVER_STATUS["message"] = msg

async def init_router_task():
    global router_instance, SERVER_STATUS
    try:
        loop = asyncio.get_running_loop()
        SERVER_STATUS["message"] = "Ожидание инициализации ядра..."
        # Запускаем тяжелый конструктор в отдельном потоке, чтобы API отвечал мгновенно
        router_instance = await loop.run_in_executor(None, lambda: SphericalRasterRouter(DATA_DIR, status_callback=_update_status))
        SERVER_STATUS = {"status": "ready", "message": "Сервер готов к работе"}
        print("✅ Бэкенд успешно инициализирован в фоне.")
    except Exception as e:
        import traceback
        traceback.print_exc()
        SERVER_STATUS = {"status": "error", "message": f"Ошибка: {str(e)}"}

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(init_router_task())

@app.get("/api/status")
def get_server_status():
    return SERVER_STATUS

class AutotestSaveRequest(BaseModel):
    test_name: str
    request_payload: dict
    expected_geojson: dict

class RouteRequest(BaseModel):
    start_lon: Optional[float] = Field(default=None, description="Долгота старта")
    start_lat: Optional[float] = Field(default=None, description="Широта старта")
    start_unlocode: Optional[str] = Field(default=None, description="UN/LOCODE старта")
    
    end_lon: Optional[float] = Field(default=None, description="Долгота финиша")
    end_lat: Optional[float] = Field(default=None, description="Широта финиша")
    end_unlocode: Optional[str] = Field(default=None, description="UN/LOCODE финиша")

    vessel_type: Optional[str] = Field(default="all", description="Тип судна")
    draft: Optional[float] = Field(default=None, description="Осадка в метрах")
    vessel_length: Optional[float] = Field(default=None, description="Длина судна в метрах")
    vessel_width: Optional[float] = Field(default=None, description="Ширина судна в метрах")
    forbidden_chokepoints: Optional[list[int]] = Field(default_factory=list, description="Список ID узких мест для запрета")
    average_speed: Optional[float] = Field(default=None, description="Заданная средняя скорость судна (узлы)")
    max_wave_height: Optional[float] = Field(default=None, description="Максимально допустимая высота волны (метры)")
    avoid_seca: bool = Field(default=False, description="Минимизировать движение по SECA")
    calc_seca: bool = Field(default=False, description="Считать дистанцию по SECA")
    timeout_seconds: Optional[int] = Field(default=None, description="Максимальное время поиска (сек)")
    
    reference_route: Optional[list[list[float]]] = Field(default=None, description="Оригинальный маршрут для корректировки [[lon, lat], ...]")
    rubber_band_weight: Optional[float] = Field(default=0.0, description="Сила притяжения к оригинальному маршруту")
    manual_storms: Optional[list[dict]] = Field(default_factory=list, description="Ручные зоны циклонов для отладки [{lat, lon, radius_km}]")


import csv

UNLOCODE_DB = {}
_locodes_path = os.path.join(os.path.dirname(__file__), "locodes.csv")

if os.path.exists(_locodes_path):
    with open(_locodes_path, "r", encoding="utf-8") as f:
        for row in csv.reader(f):
            if len(row) >= 7:
                try:
                    # Долгота - последний элемент, Широта - предпоследний
                    UNLOCODE_DB[row[0].strip().upper()] = (float(row[-1]), float(row[-2]))
                except ValueError:
                    pass

# Фоллбэк: Порт VADINAR (Индия) часто ищут по прямому названию, а не по коду INVAD
if "INVAD" in UNLOCODE_DB and "VADINAR" not in UNLOCODE_DB:
    UNLOCODE_DB["VADINAR"] = UNLOCODE_DB["INVAD"]
elif "VADINAR" not in UNLOCODE_DB:
    UNLOCODE_DB["VADINAR"] = (69.6752, 22.456428)

def resolve_coords(lon: Optional[float], lat: Optional[float], unlocode: Optional[str]):
    if lon is not None and lat is not None:
        return lon, lat
    if unlocode and unlocode.upper() in UNLOCODE_DB:
        return UNLOCODE_DB[unlocode.upper()]
    raise ValueError(f"Не удалось определить координаты для запроса (координаты не заданы, либо UN/LOCODE '{unlocode}' не найден).")

# =======================================================
# REST API 
# =======================================================
@app.post("/api/route")
async def calculate_route_api(req: RouteRequest):
    if SERVER_STATUS["status"] == "starting":
        raise HTTPException(status_code=503, detail="Сервер загружается, подождите...")
    if not router_instance:
        raise HTTPException(status_code=500, detail="Бэкенд не инициализирован")
    
    try:
        start_lon, start_lat = resolve_coords(req.start_lon, req.start_lat, req.start_unlocode)
        end_lon, end_lat = resolve_coords(req.end_lon, req.end_lat, req.end_unlocode)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
        
    try:
        req_dump = req.model_dump()
        if not req_dump.get("timeout_seconds"):
            req_dump["timeout_seconds"] = APP_CONFIG.get("routing", {}).get("timeout_seconds", 60)
            
        # Прокидываем путь к погоде для Этапа 3 (Телеметрия)
        wave_path = APP_CONFIG.get("waves", {}).get("output_file")
        if wave_path:
            req_dump["wave_path"] = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", wave_path))

        result = await router_instance.find_route(
            (start_lon, start_lat),
            (end_lon, end_lat),
            request_params=req_dump
        )
        if not result:
            raise HTTPException(status_code=404, detail="Маршрут не найден")
        return result.to_geojson(request_params=req.model_dump())
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Внутренняя ошибка сервера: {str(e)}")

@app.get("/api/chokepoints")
def get_chokepoints_config():
    if not router_instance or not router_instance.cp_config:
        return {}
    return router_instance.cp_config

@app.post("/api/autotests/save")
def save_autotest(req: AutotestSaveRequest):
    tests_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "tests", "cases"))
    os.makedirs(tests_dir, exist_ok=True)
    
    # Санитизация имени файла
    safe_name = "".join([c if c.isalnum() else "_" for c in req.test_name]).strip("_")
    if not safe_name: safe_name = "unnamed_test"
    
    filepath = os.path.join(tests_dir, f"{safe_name}.json")
    
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump({
            "name": req.test_name,
            "payload": req.request_payload,
            "expected_geojson": req.expected_geojson
        }, f, ensure_ascii=False, indent=2)
        
    return {"status": "success", "message": f"Тест '{safe_name}' успешно сохранен в папку tests/cases"}

# =======================================================
# API ГИДРОМЕТЕОРОЛОГИИ (WEATHER ROUTING)
# =======================================================
@app.post("/api/meteo/update")
def update_meteo_data():
    """
    Запускает асинхронное обновление метеоданных через шлюз.
    Вскоре будет интегрировано с Морским порталом и NOAA.
    """
    # Заготовка для фоновой задачи (Celery / BackgroundTasks)
    # gateway = NOAAMeteoGateway(output_dir)
    # gateway.fetch_forecast(target_date=datetime.now())
    return {"status": "success", "message": "Процесс обновления ГМУ запущен в фоне."}

@app.get("/api/meteo/status")
def get_meteo_status():
    """Возвращает статус текущих доступных слоев погоды и льда."""
    wave_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", APP_CONFIG.get("waves", {}).get("output_file", "")))
    waves_exist = os.path.exists(wave_path)
    
    return {
        "provider": "NOAA (GFS Wave) / Морской Портал",
        "layers": {
            "waves": {
                "available": waves_exist,
                "last_updated_utc": datetime.utcfromtimestamp(os.path.getmtime(wave_path)).isoformat() if waves_exist else None,
                "file": os.path.basename(wave_path)
            },
            "ice": {
                "available": False,
                "message": "Ожидается API Морского портала"
            }
        }
    }

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
    if SERVER_STATUS["status"] == "starting":
        await websocket.send_json({"type": "error", "message": "Сервер загружается. Пожалуйста, подождите..."})
        await websocket.close()
        return
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
            # Подмешиваем глобальный таймаут из конфига, если не передан свой
            if "timeout_seconds" not in msg or not msg["timeout_seconds"]:
                msg["timeout_seconds"] = APP_CONFIG.get("routing", {}).get("timeout_seconds", 60)
                
            # Прокидываем путь к погоде для Этапа 3 (Телеметрия)
            wave_path = APP_CONFIG.get("waves", {}).get("output_file")
            if wave_path:
                msg["wave_path"] = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", wave_path))

            listener_task = asyncio.create_task(listen_for_cancel())
            try:
                slon, slat = resolve_coords(msg.get("start_lon"), msg.get("start_lat"), msg.get("start_unlocode"))
                elon, elat = resolve_coords(msg.get("end_lon"), msg.get("end_lat"), msg.get("end_unlocode"))
                
                result = await router_instance.find_route(
                    (slon, slat),
                    (elon, elat),
                    progress_callback=on_progress,
                    check_cancel_callback=is_cancelled,
                    request_params=msg
                )
                if result:
                    await websocket.send_json({"type": "result", "geojson": result.to_geojson(request_params=msg)})
            except TimeoutError as e:
                await websocket.send_json({"type": "error", "message": str(e)})
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
def get_tile(layer: str, z: int, x: int, y: int, draft: Optional[float] = None, max_wave: Optional[float] = None):
    if layer == "speed":
        filepath = os.path.join(DATA_DIR, "eff_vel_all.tif")
    elif layer == "sd":
        filepath = os.path.join(DATA_DIR, "eff_sd_all.tif")
    elif layer == "seca":
        filepath = os.path.join(DATA_DIR, "seca_mask.tif")
    elif layer in ("depth", "depth_impassable"):
        filepath = os.path.join(DATA_DIR, "depth_meters.tif")
    elif layer == "chokepoints":
        filepath = os.path.join(DATA_DIR, "chokepoints.tif")
    elif layer in ("waves", "waves_impassable"):
        wave_path = APP_CONFIG["waves"]["output_file"]
        filepath = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", wave_path))
    elif layer == "debug":
        filepath = router_instance.debug_tif_path if router_instance else None
    else:
        return Response(status_code=404)

    if not filepath or not os.path.exists(filepath): 
        return Response(status_code=404)

    try:
        with Reader(filepath) as src:
            if layer in ("depth", "depth_impassable", "waves", "waves_impassable"):
                # Для батиметрии и погоды сохраняем оригинальный nodata (-32768 или 9999)
                img = src.tile(x, y, z, indexes=1)
            else:
                img = src.tile(x, y, z, indexes=1, nodata=0)
            
            if layer == "debug":
                _cmap = {0: (0, 0, 0, 0)}
                for i in range(1, router_instance.num_features + 2):
                    if i == router_instance.main_ocean_id:
                        _cmap[i] = (59, 130, 246, 180)
                    else:
                        _cmap[i] = (239, 68, 68, 200)
                png_bytes = img.render(img_format="PNG", colormap=_cmap)
                
            elif layer == "seca":
                _cmap = {1: (249, 115, 22, 150)}
                png_bytes = img.render(img_format="PNG", colormap=_cmap)
                
            elif layer == "chokepoints":
                _cmap = {i: (168, 85, 247, 200) for i in range(1, 1000)}
                png_bytes = img.render(img_format="PNG", colormap=_cmap)
                
            elif layer == "depth":
                img.rescale(in_range=((-5000, 0),))
                png_bytes = img.render(img_format="PNG", colormap=cmap.get("blues_r"))
                
            elif layer == "depth_impassable":
                if draft is None: return Response(status_code=204)
                raw_data = np.ma.getdata(img.data)
                if raw_data.ndim == 3: raw_data = raw_data[0]
                
                # Ищем воду (data < 0) мельче лимита
                mask = (raw_data > -(draft + 2.0)) & (raw_data < 0)
                
                rgba = np.zeros((mask.shape[0], mask.shape[1], 4), dtype=np.uint8)
                rgba[mask] = [239, 68, 68, 200] # Красная заливка
                
                buf = io.BytesIO()
                Image.fromarray(rgba).save(buf, format="PNG")
                png_bytes = buf.getvalue()

            elif layer == "waves":
                raw_data = np.ma.getdata(img.data)
                if raw_data.ndim == 3: raw_data = raw_data[0]
                
                raw_mask = np.ma.getdata(img.mask) if hasattr(img, 'mask') else np.ones_like(raw_data)
                if raw_mask.ndim == 3: raw_mask = raw_mask[0]
                
                # Валидные пиксели: там где нет суши (маска rio-tiler), нет ошибок (>100) и нет NaN
                valid = (raw_mask > 0) & (raw_data < 100) & (~np.isnan(raw_data))
                
                norm_data = np.clip(raw_data / 10.0, 0, 1)
                rgba = cm.viridis(norm_data, bytes=True)
                rgba[~valid] = [0, 0, 0, 0] # Суша и лед становятся абсолютно прозрачными!
                
                buf = io.BytesIO()
                Image.fromarray(rgba).save(buf, format="PNG")
                png_bytes = buf.getvalue()
                
            elif layer == "waves_impassable":
                if max_wave is None: return Response(status_code=204)
                raw_data = np.ma.getdata(img.data)
                if raw_data.ndim == 3: raw_data = raw_data[0]
                
                # Выделяем только штормы
                mask = (raw_data > max_wave) & (raw_data < 100) & (~np.isnan(raw_data))
                rgba = np.zeros((mask.shape[0], mask.shape[1], 4), dtype=np.uint8)
                rgba[mask] = [239, 68, 68, 200]
                
                buf = io.BytesIO()
                Image.fromarray(rgba).save(buf, format="PNG")
                png_bytes = buf.getvalue()

            elif layer == "speed":
                img.rescale(in_range=((0, 20),))
                png_bytes = img.render(img_format="PNG", colormap=cmap.get("turbo"))
            else:
                img.rescale(in_range=((0, 5),))
                png_bytes = img.render(img_format="PNG", colormap=cmap.get("plasma"))
                
            return Response(content=png_bytes, media_type="image/png")
            
    except Exception as e:
        # TileOutsideBounds - нормальная ситуация, когда Leaflet запрашивает пустоту за границами растра
        if type(e).__name__ == "TileOutsideBounds":
            return Response(status_code=204)
            
        import traceback
        print(f"❌ ОШИБКА РЕНДЕРА ТАЙЛА '{layer}' (z={z}, x={x}, y={y}): {e}")
        traceback.print_exc()
        return Response(status_code=204)    

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

@app.get("/")
async def serve_index(): return FileResponse(os.path.join(STATIC_DIR, "index.html"))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)