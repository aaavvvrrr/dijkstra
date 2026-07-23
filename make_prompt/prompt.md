# Промпт с кодовой базой

**Сгенерировано:** dijkstra

**Источник:** `/home/romanov/gitlab/dijkstra`

**Файлов найдено:** 7

## Оглавление

- [.gitignore](#gitignore)
- [app/core.py](#app/corepy)
- [app/main.py](#app/mainpy)
- [app/preprocessing/pipeline.py](#app/preprocessing/pipelinepy)
- [app/static/index.html](#app/static/indexhtml)
- [app/static/oceans-seas.geo.json](#app/static/oceans-seasgeojson)
- [requirements.txt](#requirementstxt)

---

<a id='gitignore'></a>

* .gitignore

```
data

```

---

<a id='app/corepy'></a>

* app/core.py

```python
import os
import math
import heapq
import time
import asyncio
from collections import deque
import numpy as np
import rasterio
from scipy.ndimage import label
from typing import Tuple, List, Optional, Dict, Any, Callable
from dataclasses import dataclass

EARTH_RADIUS_KM = 6371.0
KNOT_TO_KMH = 1.852

@dataclass
class RouteResult:
    path_coords: List[Tuple[float, float]]
    total_distance_km: float
    time_mean_hours: float
    segment_distances_km: List[float]
    segment_speeds_kmh: List[float]
    segment_times_hours: List[float]
    time_q05_hours: float  
    time_q95_hours: float  
    actual_start_coords: Tuple[float, float]
    actual_end_coords: Tuple[float, float]

    def to_geojson(self, request_params: dict = None) -> Dict[str, Any]:
        clean_coords = [[float(lon), float(lat)] for lon, lat in self.path_coords]
        return {
            "type": "Feature",
            "properties": {
                "distance_total_km": round(float(self.total_distance_km), 2),
                "time_total_hours": round(float(self.time_mean_hours), 2),
                "time_optimistic_hours": round(float(self.time_q05_hours), 2),
                "time_pessimistic_hours": round(float(self.time_q95_hours), 2),
                "avg_speed_kmh": round(float(self.total_distance_km / self.time_mean_hours), 2) if self.time_mean_hours > 0 else 0,
                "segments": {
                    "distances_km": [round(float(d), 3) for d in self.segment_distances_km],
                    "speeds_kmh": [round(float(s), 2) for s in self.segment_speeds_kmh],
                    "times_hours": [round(float(t), 3) for t in self.segment_times_hours],
                },
                "actual_start_lonlat": [round(self.actual_start_coords[0], 6), round(self.actual_start_coords[1], 6)],
                "actual_end_lonlat": [round(self.actual_end_coords[0], 6), round(self.actual_end_coords[1], 6)],
                "request_parameters": request_params or {}
            },
            "geometry": {"type": "LineString", "coordinates": clean_coords}
        }

class SphericalRasterRouter:
    def __init__(self, speed_tif_path: str, sd_tif_path: str):
        print(f"Загрузка растров в память: {speed_tif_path} ...")
        with rasterio.open(speed_tif_path) as src:
            self.transform = src.transform
            self.inv_transform = ~self.transform
            self.speed_raster = src.read(1).astype(np.float32) * KNOT_TO_KMH
            self.speed_raster = np.nan_to_num(self.speed_raster, nan=0.0)
            self.rows, self.cols = self.speed_raster.shape

        with rasterio.open(sd_tif_path) as src:
            self.sd_raster = src.read(1).astype(np.float32) * KNOT_TO_KMH
            self.sd_raster = np.nan_to_num(self.sd_raster, nan=0.1)

        self.min_lon = self.transform.c
        self.max_lat = self.transform.f
        self.d_lon = abs(self.transform.a)
        self.d_lat = abs(self.transform.e)
        self.dy_km = (math.radians(self.d_lat) * EARTH_RADIUS_KM)
        
        # Для фронтенда (Bounding Box)
        self.max_lon = self.min_lon + self.cols * self.d_lon
        self.min_lat = self.max_lat - self.rows * self.d_lat

        lats = np.array([self.max_lat - r * self.d_lat for r in range(self.rows)])
        self.cos_lats = np.cos(np.radians(lats)).astype(np.float32)
        self.dx_per_row = (math.radians(self.d_lon) * EARTH_RADIUS_KM) * self.cos_lats
        
        self.max_speed = float(np.max(self.speed_raster))
        if self.max_speed <= 0: self.max_speed = 30.0

        print("Создание грубой сетки (масштаб 1:20)...")
        self.scale = 20
        self.coarse_rows = self.rows // self.scale
        self.coarse_cols = self.cols // self.scale
        
        water_mask = self.speed_raster[:self.coarse_rows * self.scale, :self.coarse_cols * self.scale] > 0
        self.coarse_water = water_mask.reshape(self.coarse_rows, self.scale, self.coarse_cols, self.scale).any(axis=(1, 3))

        print("Анализ связности (Connected Components)...")
        # Ищем ВСЕ изолированные водоемы (8-way connectivity)
        structure = np.ones((3, 3), dtype=int)
        self.components, self.num_features = label(self.coarse_water, structure=structure)
        
        counts = np.bincount(self.components.ravel())
        self.main_ocean_id = np.argmax(counts[1:]) + 1 if len(counts) > 1 else 0
        print(f"Найдено {self.num_features} изолированных водоемов. Главный океан: ID {self.main_ocean_id}")

        self.coarse_dy = self.dy_km * self.scale
        c_lats = np.array([self.max_lat - r * self.d_lat * self.scale for r in range(self.coarse_rows)])
        self.coarse_dx_per_row = (math.radians(self.d_lon * self.scale) * EARTH_RADIUS_KM) * np.cos(np.radians(c_lats))

        debug_tif_path = os.path.join(os.path.dirname(speed_tif_path), "debug_connectivity.tif")
        coarse_transform = self.transform * rasterio.Affine.scale(self.scale)
        
        with rasterio.open(
            debug_tif_path, 'w',
            driver='GTiff',
            height=self.coarse_rows,
            width=self.coarse_cols,
            count=1,
            dtype=self.components.dtype,
            crs=src.crs,
            transform=coarse_transform
        ) as dst:
            dst.write(self.components, 1)
        
        self.debug_tif_path = debug_tif_path
        print("Роутер готов к работе! Отладочная сетка сгенерирована.")

    def _coord_to_index(self, lon: float, lat: float) -> Tuple[int, int]:
        col, row = self.inv_transform * (lon, lat)
        return max(0, min(self.rows - 1, int(row))), max(0, min(self.cols - 1, int(col)))

    def _index_to_coord(self, row: int, col: int) -> Tuple[float, float]:
        lon, lat = self.transform * (col + 0.5, row + 0.5)
        return float(lon), float(lat)

    def _haversine_distance(self, lon1: float, lat1: float, lon2: float, lat2: float) -> float:
        phi1, phi2 = math.radians(lat1), math.radians(lat2)
        dphi = math.radians(lat2 - lat1)
        dlambda = math.radians(lon2 - lon1)
        a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
        return 2 * EARTH_RADIUS_KM * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    def _get_nearest_water_point(self, start_row: int, start_col: int, max_radius: int = 300) -> Optional[Tuple[int, int, int]]:
        queue = deque([(start_row, start_col)])
        visited = {(start_row, start_col)}
        while queue:
            cr, cc = queue.popleft()
            if abs(cr - start_row) > max_radius or abs(cc - start_col) > max_radius: continue
            
            if self.speed_raster[cr, cc] > 0.5:
                cr_c, cc_c = cr // self.scale, cc // self.scale
                if 0 <= cr_c < self.coarse_rows and 0 <= cc_c < self.coarse_cols:
                    comp_id = self.components[cr_c, cc_c]
                    if comp_id > 0:  # Если это не суша
                        return (cr, cc, comp_id)

            for dr, dc in [(-1,0), (1,0), (0,-1), (0,1), (-1,-1), (-1,1), (1,-1), (1,1)]:
                nr, nc = cr + dr, cc + dc
                if 0 <= nr < self.rows and 0 <= nc < self.cols:
                    if (nr, nc) not in visited:
                        visited.add((nr, nc))
                        queue.append((nr, nc))
        return None

    def _build_coarse_heuristic(self, goal_r: int, goal_c: int, comp_id: int) -> np.ndarray:
        c_gr, c_gc = goal_r // self.scale, goal_c // self.scale
        coarse_h = np.full((self.coarse_rows, self.coarse_cols), np.inf, dtype=np.float32)
        coarse_h[c_gr, c_gc] = 0.0
        pq = [(0.0, c_gr, c_gc)]
        
        c_dx = self.coarse_dx_per_row
        c_dy = self.coarse_dy
        c_rows, c_cols = self.coarse_rows, self.coarse_cols
        max_s = self.max_speed
        
        while pq:
            t, r, c = heapq.heappop(pq)
            if t > coarse_h[r, c]: continue
            dx = c_dx[r]
            diag = math.sqrt(dx*dx + c_dy*c_dy)
            
            for dr, dc, dist in [(-1, 0, c_dy), (1, 0, c_dy), (0, -1, dx), (0, 1, dx), (-1, -1, diag), (-1, 1, diag), (1, -1, diag), (1, 1, diag)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < c_rows and 0 <= nc < c_cols:
                    # Двигаемся ТОЛЬКО внутри своего водоема
                    if self.components[nr, nc] == comp_id:
                        nt = t + (dist / max_s)
                        if nt < coarse_h[nr, nc]:
                            coarse_h[nr, nc] = nt
                            heapq.heappush(pq, (nt, nr, nc))
        return coarse_h

    async def find_route(
        self, start_coords: Tuple[float, float], end_coords: Tuple[float, float],
        progress_callback: Optional[Callable] = None, check_cancel_callback: Optional[Callable] = None
    ) -> Optional[RouteResult]:
        
        t_start = time.time()
        sr, sc = self._coord_to_index(*start_coords)
        gr, gc = self._coord_to_index(*end_coords)

        start_pt_info = self._get_nearest_water_point(sr, sc)
        goal_pt_info = self._get_nearest_water_point(gr, gc)
        
        if not start_pt_info: raise ValueError("Точка старта слишком далеко от воды.")
        if not goal_pt_info: raise ValueError("Точка финиша слишком далеко от воды.")

        sr, sc, start_comp = start_pt_info
        gr, gc, goal_comp = goal_pt_info

        # МГНОВЕННАЯ ПРОВЕРКА НА ИЗОЛИРОВАННОСТЬ
        if start_comp != goal_comp:
            raise ValueError(
                f"Путь невозможен: Старт и Финиш находятся в изолированных друг от друга водоемах "
                f"(ID {start_comp} и ID {goal_comp}). Включите слой 'Отладка: Связность морей' на карте, чтобы увидеть разрыв."
            )

        coarse_h = self._build_coarse_heuristic(gr, gc, goal_comp)
        if coarse_h[sr // self.scale, sc // self.scale] == np.inf:
            raise ValueError("Ошибка иерархической сетки: внутренний разрыв.")

        cols = self.rows
        cols = self.cols
        start_idx = sr * cols + sc
        goal_idx = gr * cols + gc

        pq = [(0.0, 0.0, sr, sc)]
        times_to_reach = {start_idx: 0.0}
        came_from = {start_idx: -1}

        speed_raster = self.speed_raster
        dx_per_row = self.dx_per_row
        dy_km = self.dy_km
        rows = self.rows
        
        TARGET_RADIUS_KM = 3.0 
        nodes_explored = 0

        while pq:
            _, current_time, r, c = heapq.heappop(pq)
            idx = r * cols + c

            if idx == goal_idx:
                print(f"✅ Точный финиш найден! Время: {time.time() - t_start:.2f} сек.")
                return self._reconstruct_and_simulate(came_from, gr, gc, sr, sc)

            if current_time > times_to_reach.get(idx, float('inf')): continue
            nodes_explored += 1
            
            if nodes_explored % 30000 == 0:
                if check_cancel_callback and check_cancel_callback(): raise InterruptedError("Поиск прерван.")
                if progress_callback:
                    fast_path = []
                    curr = idx
                    while curr != -1:
                        fast_path.append(self._index_to_coord(curr // cols, curr % cols))
                        curr = came_from.get(curr, -1)
                    await progress_callback(nodes_explored, fast_path[::5], current_time)
                await asyncio.sleep(0.001) 

            dx = dx_per_row[r]
            diag = math.sqrt(dx*dx + dy_km*dy_km)

            for dr, dc, dist in [
                (-1, 0, dy_km), (1, 0, dy_km), 
                (0, -1, dx), (0, 1, dx),
                (-1, -1, diag), (-1, 1, diag), (1, -1, diag), (1, 1, diag)
            ]:
                nr, nc = r + dr, c + dc 
                if not (0 <= nr < rows and 0 <= nc < cols): continue
                
                speed = speed_raster[nr, nc]
                if speed <= 0: continue
                
                cr_c, cc_c = nr // self.scale, nc // self.scale
                if cr_c >= self.coarse_rows or cc_c >= self.coarse_cols: continue
                
                h_time = coarse_h[cr_c, cc_c]
                if h_time == np.inf: continue 
                
                new_time = current_time + (dist / speed)
                n_idx = nr * cols + nc
                
                if new_time < times_to_reach.get(n_idx, float('inf')):
                    if abs(nr - gr) <= 15 and abs(nc - gc) <= 15:
                        dist_to_goal = math.hypot((nr - gr) * dx_per_row[nr], (nc - gc) * dy_km)
                        if dist_to_goal <= TARGET_RADIUS_KM:
                            came_from[n_idx] = idx
                            print(f"⚓ Захват рейда ({TARGET_RADIUS_KM} км до цели)! Время: {time.time() - t_start:.2f} сек.")
                            return self._reconstruct_and_simulate(came_from, nr, nc, sr, sc)
                    
                    times_to_reach[n_idx] = new_time
                    came_from[n_idx] = idx
                    
                    f_score = new_time + 1.2 * h_time
                    heapq.heappush(pq, (f_score, new_time, nr, nc))

        return None

    def _reconstruct_and_simulate(self, came_from: dict, gr: int, gc: int, sr: int, sc: int) -> RouteResult:
        path_coords = []
        curr_idx = gr * self.cols + gc
        while curr_idx != -1:
            path_coords.append(self._index_to_coord(curr_idx // self.cols, curr_idx % self.cols))
            curr_idx = came_from.get(curr_idx, -1)
        path_coords.reverse()
        
        smoothed = path_coords
        for _ in range(2):
            if len(smoothed) <= 2: break
            new_path = [smoothed[0]]
            for i in range(len(smoothed) - 1):
                p0, p1 = smoothed[i], smoothed[i+1]
                new_path.extend([(0.75 * p0[0] + 0.25 * p1[0], 0.75 * p0[1] + 0.25 * p1[1]),
                                 (0.25 * p0[0] + 0.75 * p1[0], 0.25 * p0[1] + 0.75 * p1[1])])
            new_path.append(smoothed[-1])
            smoothed = new_path

        N_SEGMENTS = len(smoothed) - 1
        dists_km = np.zeros(N_SEGMENTS) if N_SEGMENTS > 0 else np.zeros(0)
        means = np.zeros(N_SEGMENTS) if N_SEGMENTS > 0 else np.zeros(0)
        sds = np.zeros(N_SEGMENTS) if N_SEGMENTS > 0 else np.zeros(0)

        for i in range(N_SEGMENTS):
            p1, p2 = smoothed[i], smoothed[i+1]
            dists_km[i] = self._haversine_distance(*p1, *p2)
            mr, mc = self._coord_to_index((p1[0] + p2[0]) / 2.0, (p1[1] + p2[1]) / 2.0)
            means[i] = max(self.speed_raster[mr, mc], 1.0)
            sds[i] = max(self.sd_raster[mr, mc], 0.1)

        time_mean, time_q05, time_q95 = 0.0, 0.0, 0.0
        if N_SEGMENTS > 0:
            shape = (means / sds) ** 2
            scale = (sds ** 2) / means
            sim_speeds = np.maximum(np.random.gamma(shape[:, None], scale[:, None], size=(N_SEGMENTS, 10000)), 1.0)
            sim_times = dists_km[:, None] / sim_speeds
            total_times = np.sum(sim_times, axis=0)
            time_mean = float(np.mean(total_times))
            time_q05 = float(np.percentile(total_times, 5))
            time_q95 = float(np.percentile(total_times, 95))

        return RouteResult(
            path_coords=smoothed,
            total_distance_km=float(np.sum(dists_km)),
            time_mean_hours=time_mean,
            time_q05_hours=time_q05,
            time_q95_hours=time_q95,
            segment_distances_km=dists_km.tolist(),
            segment_speeds_kmh=means.tolist(),
            segment_times_hours=(dists_km / means).tolist() if N_SEGMENTS > 0 else [],
            actual_start_coords=self._index_to_coord(sr, sc),
            actual_end_coords=self._index_to_coord(gr, gc)
        )
```

---

<a id='app/mainpy'></a>

* app/main.py

```python
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
            img = src.tile(x, y, z)
            
            # Делаем нулевые пиксели прозрачными
            img.mask = np.where(img.data[0] == 0, 0, 255).astype(np.uint8)
            
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
        print(f"Ошибка рендера тайла {layer}: {e}")
        return Response(status_code=204)
    
    
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

@app.get("/")
async def serve_index(): return FileResponse(os.path.join(STATIC_DIR, "index.html"))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

---

<a id='app/preprocessing/pipelinepy'></a>

* app/preprocessing/pipeline.py

```python
import numpy as np
import rasterio
from rasterio.features import rasterize
import geopandas as gpd
from scipy.ndimage import gaussian_filter
from typing import Optional
import os

class RasterPreprocessor:
    """
    Пайплайн подготовки данных АИС.
    Заменяет шаги 1-5 из legacy .bat скриптов.
    Формирует итоговый растр эффективной скорости и сглаженный растр SD для Монте-Карло.
    """
    
    def __init__(self, base_dir: str):
        self.base_dir = base_dir
        self.meta = None
        self.transform = None
        self.shape = None

    def _read_raster(self, filepath: str) -> np.ndarray:
        """Чтение растра в формате float32."""
        with rasterio.open(filepath) as src:
            if self.meta is None:
                self.meta = src.meta.copy()
                self.transform = src.transform
                self.shape = (src.height, src.width)
            
            arr = src.read(1).astype(np.float32)
            # Обработка NoData
            nodata = src.nodata
            if nodata is not None:
                arr[arr == nodata] = 0.0
            return arr

    def _write_raster(self, arr: np.ndarray, filepath: str):
        """Сохранение итогового растра."""
        out_meta = self.meta.copy()
        out_meta.update({
            "driver": "GTiff",
            "dtype": 'float32',
            "compress": "deflate",
            "tiled": True,
            "blockxsize": 256,
            "blockysize": 256,
            "nodata": 0.0
        })
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with rasterio.open(filepath, 'w', **out_meta) as dest:
            dest.write(arr, 1)
        print(f"✅ Сохранен файл: {filepath}")

    def _create_land_mask_from_geojson(self, geojson_path: str) -> np.ndarray:
        """Создает маску суши (1 - суша, 0 - вода) из GeoJSON полигонов океанов."""
        print(f"🌊 Генерация маски суши из: {os.path.basename(geojson_path)}")
        gdf = gpd.read_file(geojson_path)
        
        # Полигоны в файле - это ОКЕАНЫ (вода). 
        # Значит, мы заливаем весь растр единицами (суша), 
        # а там где есть геометрия океана - прожигаем нулями (вода).
        shapes = ((geom, 0.0) for geom in gdf.geometry)
        
        land_mask = rasterize(
            shapes=shapes,
            out_shape=self.shape,
            transform=self.transform,
            fill=1.0,           # Фон по умолчанию = Суша
            all_touched=True,
            dtype=np.float32
        )
        return land_mask

    def process(
        self, count_tif: str, vel_tif: str, sd_tif: str, 
        out_eff_vel: str, out_sd: str, oceans_geojson: Optional[str] = None
    ):
        import gc
        print("🚀 Старт memory-optimized препроцессинга...")

        # --- ФАЗА 1: Маски и счетчики ---
        print("1. Загрузка count_tif...")
        count = self._read_raster(count_tif) # ~9.5 ГБ
        has_tracks = count > 0               # ~2.4 ГБ (boolean)

        print("2. Формирование гибридной маски суши...")
        if oceans_geojson and os.path.exists(oceans_geojson):
            land_mask = self._create_land_mask_from_geojson(oceans_geojson) # ~9.5 ГБ
            # In-place замена: где есть треки, делаем воду (0.0)
            np.copyto(land_mask, 0.0, where=has_tracks)
            print("🗺️ Применен гибридный подход.")
        else:
            land_mask = (~has_tracks).astype(np.float32)

        print("3. Сглаживание count (in-place)...")
        count_flt = gaussian_filter(count, sigma=1.2, mode='reflect')
        np.maximum(count, count_flt, out=count) # count = max(count, count_flt)
        del count_flt
        gc.collect()

        # Обнуляем счетчик на суше (count теперь является count_final)
        count[land_mask == 1.0] = 0.0

        print("4. Расчет сигмоиды (in-place)...")
        # Чтобы не тратить еще 9.5 ГБ, превращаем массив count в множитель сигмоиды
        count -= 20.0
        count *= -0.1962959
        np.clip(count, -50, 50, out=count)
        np.exp(count, out=count)
        count += 1.0
        np.reciprocal(count, out=count) # Теперь count = 1 / (1 + exp(...))

        # --- ФАЗА 2: Обработка Дисперсии (SD) ---
        print("5. Загрузка vel и sd для обработки дисперсии...")
        vel = self._read_raster(vel_tif)
        # Поднимаем минимальную скорость
        vel[has_tracks] = np.maximum(vel[has_tracks], 4.5)

        sd = self._read_raster(sd_tif)
        sd[has_tracks] = np.maximum(sd[has_tracks], 0.1 * vel[has_tracks])
        
        # Маска has_tracks больше не нужна, освобождаем 2.4 ГБ
        del has_tracks
        gc.collect()

        print("6. Сглаживание SD (in-place)...")
        sd_flt = gaussian_filter(sd, sigma=1.2, mode='reflect')
        np.maximum(sd, sd_flt, out=sd)
        del sd_flt
        gc.collect()

        # Обнуляем на суше, сохраняем, удаляем SD
        sd[land_mask == 1.0] = 0.0
        self._write_raster(sd, out_sd)
        del sd
        gc.collect()

        # --- ФАЗА 3: Обработка Скорости (Vel) ---
        print("7. Сглаживание скорости (in-place)...")
        vel_flt = gaussian_filter(vel, sigma=1.2, mode='reflect')
        np.maximum(vel, vel_flt, out=vel)
        del vel_flt
        gc.collect()

        print("8. Применение сигмоиды к скорости...")
        vel *= count  # Умножаем скорость на готовую сигмоиду
        del count
        gc.collect()

        # Обнуляем сушу, сохраняем, удаляем остатки
        vel[land_mask == 1.0] = 0.0
        self._write_raster(vel, out_eff_vel)
        del vel
        del land_mask
        gc.collect()

        print("🎯 Препроцессинг успешно завершен!")

if __name__ == "__main__":
    BASE_DIR = "./data"
    
    preprocessor = RasterPreprocessor(BASE_DIR)
    
    # Теперь мы передаем наш oceans-seas.geo.json
    preprocessor.process(
        count_tif=f"{BASE_DIR}/count.tif",
        vel_tif=f"{BASE_DIR}/mean_velocity_knots.tif",
        sd_tif=f"{BASE_DIR}/mean_velocity_sd_knots.tif",
        out_eff_vel=f"{BASE_DIR}/processed/effective_velocity_knots.tif",
        out_sd=f"{BASE_DIR}/processed/effective_sd_knots.tif",
        oceans_geojson=f"./app/static/oceans-seas.geo.json"
    )
```

---

<a id='app/static/indexhtml'></a>

* app/static/index.html

```html
<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Sea Routing FMM Demo</title>
    <!-- Leaflet CSS -->
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
    <style>
        :root {
            --primary: #2563eb;
            --primary-hover: #1d4ed8;
            --bg-color: #f8fafc;
            --surface: #ffffff;
            --text-main: #0f172a;
            --text-muted: #64748b;
            --border: #e2e8f0;
            --error: #ef4444;
        }

        * { box-sizing: border-box; margin: 0; padding: 0; }

        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            display: flex;
            height: 100vh;
            color: var(--text-main);
            background: var(--bg-color);
        }

        /* Sidebar UI */
        .sidebar {
            width: 380px;
            background: var(--surface);
            border-right: 1px solid var(--border);
            display: flex;
            flex-direction: column;
            padding: 24px;
            box-shadow: 4px 0 15px rgba(0,0,0,0.03);
            z-index: 1000;
        }

        .header h1 { font-size: 1.25rem; margin-bottom: 0.5rem; }
        .header p { font-size: 0.875rem; color: var(--text-muted); margin-bottom: 2rem; }

        .control-group { margin-bottom: 1.5rem; }
        .control-group label {
            display: block;
            font-size: 0.875rem;
            font-weight: 600;
            margin-bottom: 0.5rem;
            color: var(--text-main);
        }

        .input-row {
            display: flex;
            gap: 8px;
            margin-bottom: 8px;
        }

        input {
            width: 100%;
            padding: 10px 12px;
            border: 1px solid var(--border);
            border-radius: 6px;
            font-size: 0.875rem;
            background: var(--bg-color);
            transition: border-color 0.2s;
        }
        input[readonly] { cursor: not-allowed; opacity: 0.8; }

        .btn {
            width: 100%;
            padding: 12px;
            background: var(--primary);
            color: white;
            border: none;
            border-radius: 6px;
            font-size: 1rem;
            font-weight: 600;
            cursor: pointer;
            transition: background 0.2s, transform 0.1s;
        }
        .btn:hover { background: var(--primary-hover); }
        .btn:active { transform: scale(0.98); }
        .btn:disabled { background: var(--text-muted); cursor: not-allowed; }

        /* Results panel */
        .results {
            margin-top: 2rem;
            padding: 16px;
            background: var(--bg-color);
            border-radius: 8px;
            border: 1px solid var(--border);
            display: none;
        }
        .results.active { display: block; }
        .results h3 { font-size: 1rem; margin-bottom: 12px; }
        .stat-row { display: flex; justify-content: space-between; margin-bottom: 8px; font-size: 0.875rem; }
        .stat-value { font-weight: 600; }

        .error-msg {
            color: var(--error);
            font-size: 0.875rem;
            margin-top: 1rem;
            display: none;
        }

        /* Map area */
        #map { flex: 1; height: 100%; background: #c5e8ff; }

        /* Helper instruction */
        .instruction { font-size: 0.8rem; color: var(--primary); margin-top: 4px; display: block; }
    </style>
</head>
<body>

    <aside class="sidebar">
        <div class="header">
            <h1>Maritime Route Finder</h1>
            <p>Тестовый интерфейс алгоритма поверх растра АИС (A* Spherical)</p>
        </div>

        <div class="control-group">
            <label>Точка А (Старт)</label>
            <div class="input-row">
                <input type="text" id="start-lat" placeholder="Широта" readonly>
                <input type="text" id="start-lon" placeholder="Долгота" readonly>
            </div>
            <span class="instruction">Кликните на карту, чтобы задать</span>
        </div>

        <div class="control-group">
            <label>Точка Б (Финиш)</label>
            <div class="input-row">
                <input type="text" id="end-lat" placeholder="Широта" readonly>
                <input type="text" id="end-lon" placeholder="Долгота" readonly>
            </div>
            <span class="instruction">Кликните на карту, чтобы задать</span>
        </div>

        <div style="display: flex; gap: 8px;">
            <button class="btn" id="calc-btn" disabled>Построить маршрут</button>
            <button class="btn" id="cancel-btn" style="background: var(--error); display: none;">Прервать</button>
        </div>
        <div id="error-box" class="error-msg"></div>

        <div class="results" id="results-panel">
            <h3>Атрибуты маршрута</h3>
            <div class="stat-row">
                <span>Протяженность:</span>
                <span class="stat-value" id="res-dist">0 км</span>
            </div>
            <div class="stat-row">
                <span>Средняя скорость:</span>
                <span class="stat-value" id="res-speed">0 км/ч</span>
            </div>

            <h3 style="margin-top: 16px; border-top: 1px solid var(--border); padding-top: 12px;">Прогноз времени (Монте-Карло)</h3>
            <div class="stat-row" style="color: #10b981;">
                <span>Оптимистичный (5%):</span>
                <span class="stat-value" id="res-time-opt">0 ч</span>
            </div>
            <div class="stat-row" style="font-size: 1rem; color: var(--primary);">
                <span>Ожидаемое (Среднее):</span>
                <span class="stat-value" id="res-time-mean">0 ч</span>
            </div>
            <div class="stat-row" style="color: #ef4444;">
                <span>Пессимистичный (95%):</span>
                <span class="stat-value" id="res-time-pess">0 ч</span>
            </div>
        </div>
    </aside>

    <main id="map"></main>

    <!-- Leaflet JS -->
    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
    <script>
        // Инициализация карты
        const map = L.map('map').setView([45.0, 10.0], 4);

        // Базовая подложка
        const baseMap = L.tileLayer('https://{s}.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}{r}.png', {
            attribution: '&copy; CARTO'
        }).addTo(map);

        // Подключаем наши динамические растры из FastAPI Tile Server
        const effSpeedLayer = L.tileLayer('/tiles/speed/{z}/{x}/{y}.png', { opacity: 0.8, maxZoom: 10 });
        const effSdLayer = L.tileLayer('/tiles/sd/{z}/{x}/{y}.png', { opacity: 0.8, maxZoom: 10 });
        const debugGridLayer = L.tileLayer('/tiles/debug/{z}/{x}/{y}.png', { opacity: 0.7, maxZoom: 10 });

        const overlayMaps = {
            "Детализация: Скорость": effSpeedLayer,
            "Детализация: Дисперсия": effSdLayer,
            "Связность морей (Грубая)": debugGridLayer
        };
        L.control.layers({"Базовая карта": baseMap}, overlayMaps, {collapsed: false}).addTo(map);

        // По умолчанию включим скорость
        effSpeedLayer.addTo(map);

        // Загрузка векторной маски (Контуры берегов)
        fetch('/static/oceans-seas.geo.json')
            .then(res => res.json())
            .then(data => {
                L.geoJSON(data, {
                    style: {
                        color: '#3b82f6',
                        weight: 1,
                        fillColor: 'transparent', // Убрал синюю заливку, чтобы не мешала смотреть растры!
                        fillOpacity: 0
                    }
                }).addTo(map);
            })
            .catch(err => console.error("Ошибка загрузки маски:", err));

        // --- МАГИЯ ОТЛАДКИ СВЯЗНОСТИ МОРЕЙ ---
        // fetch('/api/debug/bounds')
        //     .then(res => res.json())
        //     .then(b => {
        //         const bounds = [[b.min_lat, b.min_lon], [b.max_lat, b.max_lon]];
        //         const coarseLayer = L.imageOverlay('/api/debug/coarse_grid.png', bounds, {opacity: 0.6});
        //         layerControl.addOverlay(coarseLayer, "Отладка: Связность морей (Грубая)");
        //         // Раскомментируйте ниже, чтобы слой включался сразу
        //         // coarseLayer.addTo(map); 
        //     })
        //     .catch(err => console.error("Ошибка загрузки грубой сетки:", err));

        // State UI
        let currentStep = 'start'; 
        const coords = { start: null, end: null };
        const markers = { start: null, end: null };
        let routeLayer = null;
        let progressPolyline = null; // Слой для "щупальца"
        let ws = null;

        const inputs = {
            startLat: document.getElementById('start-lat'),
            startLon: document.getElementById('start-lon'),
            endLat: document.getElementById('end-lat'),
            endLon: document.getElementById('end-lon')
        };
        const calcBtn = document.getElementById('calc-btn');
        const cancelBtn = document.getElementById('cancel-btn');
        const resultsPanel = document.getElementById('results-panel');
        const errorBox = document.getElementById('error-box');

        // Обработка клика
        map.on('click', (e) => {
            let lng = e.latlng.lng;
            while (lng > 180) lng -= 360;
            while (lng < -180) lng += 360;

            const lat = Math.max(-90, Math.min(90, e.latlng.lat)).toFixed(4);
            const lon = lng.toFixed(4);

            // Если мы только зашли на страницу ИЛИ если уже построили маршрут (3-й клик)
            if (currentStep === 'start' || currentStep === 'ready') {
                
                // Если это 3-й клик (сброс старого маршрута)
                if (currentStep === 'ready') {
                    if (markers.end) map.removeLayer(markers.end);
                    coords.end = null;
                    inputs.endLat.value = '';
                    inputs.endLon.value = '';
                    calcBtn.disabled = true;
                    
                    // Убираем старые линии
                    if (routeLayer) map.removeLayer(routeLayer);
                    if (progressPolyline) { map.removeLayer(progressPolyline); progressPolyline = null; }
                    resultsPanel.classList.remove('active');
                }

                coords.start = { lat: parseFloat(lat), lon: parseFloat(lon) };
                inputs.startLat.value = lat;
                inputs.startLon.value = lon;
                
                if (markers.start) map.removeLayer(markers.start);
                markers.start = L.marker([lat, lon]).addTo(map).bindPopup("Старт").openPopup();
                
                currentStep = 'end';
            } 
            // Второй клик (Установка финиша)
            else if (currentStep === 'end') {
                coords.end = { lat: parseFloat(lat), lon: parseFloat(lon) };
                inputs.endLat.value = lat;
                inputs.endLon.value = lon;
                
                if (markers.end) map.removeLayer(markers.end);
                markers.end = L.marker([lat, lon]).addTo(map).bindPopup("Финиш").openPopup();
                
                currentStep = 'ready';
                calcBtn.disabled = false;
                errorBox.style.display = 'none';
            }
        });

        // Отправка сигнала отмены
        cancelBtn.addEventListener('click', () => {
            if (ws && ws.readyState === WebSocket.OPEN) {
                ws.send(JSON.stringify({ action: 'cancel' }));
            }
        });

        // Запуск расчета через WebSocket
        calcBtn.addEventListener('click', () => {
            if (!coords.start || !coords.end) return;

            // UI сброс
            if (routeLayer) map.removeLayer(routeLayer);
            if (progressPolyline) { map.removeLayer(progressPolyline); progressPolyline = null; }
            resultsPanel.classList.remove('active');
            errorBox.style.display = 'none';
            
            calcBtn.style.display = 'none';
            cancelBtn.style.display = 'block';

            // Открываем WebSocket
            const protocol = window.location.protocol === "https:" ? "wss" : "ws";
            ws = new WebSocket(`${protocol}://${window.location.host}/api/ws/route`);

            ws.onopen = () => {
                // Показываем плашку результатов в режиме "Поиск..."
                document.getElementById('res-dist').innerText = 'Идет поиск...';
                document.getElementById('res-speed').innerText = '0 узлов';
                document.getElementById('res-time-opt').innerText = '-';
                document.getElementById('res-time-mean').innerText = '-';
                document.getElementById('res-time-pess').innerText = '-';
                resultsPanel.classList.add('active');

                // Отправляем старт
                ws.send(JSON.stringify({
                    action: "start",
                    start_lon: coords.start.lon, start_lat: coords.start.lat,
                    end_lon: coords.end.lon, end_lat: coords.end.lat
                }));
            };

            ws.onmessage = (event) => {
                const data = JSON.parse(event.data);

                if (data.type === 'progress') {
                    // Обновляем статистику поиска
                    document.getElementById('res-speed').innerText = data.explored.toLocaleString() + ' проверок';
                    document.getElementById('res-time-mean').innerText = `Изучаемый путь: ${data.current_time.toFixed(1)} ч.`;

                    // Отрисовываем "Щупальце"
                    const latlngs = data.path.map(p => [p[1], p[0]]);
                    if (progressPolyline) {
                        progressPolyline.setLatLngs(latlngs);
                    } else {
                        progressPolyline = L.polyline(latlngs, {color: '#f43f5e', weight: 3, dashArray: '5, 5'}).addTo(map);
                    }
                } 
                else if (data.type === 'result') {
                    if (progressPolyline) { map.removeLayer(progressPolyline); progressPolyline = null; }
                    
                    const geojson = data.geojson;
                    routeLayer = L.geoJSON(geojson, {
                        style: { color: 'var(--primary)', weight: 5, opacity: 0.9, dashArray: '8, 8' }
                    }).addTo(map);

                    map.fitBounds(routeLayer.getBounds(), { padding: [50, 50] });

                    const props = geojson.properties;
                    document.getElementById('res-dist').innerText = props.distance_total_km + ' км';
                    document.getElementById('res-speed').innerText = props.avg_speed_kmh + ' км/ч';
                    document.getElementById('res-time-opt').innerText = props.time_optimistic_hours + ' ч';
                    document.getElementById('res-time-mean').innerText = props.time_total_hours + ' ч';
                    document.getElementById('res-time-pess').innerText = props.time_pessimistic_hours + ' ч';

                    if (props.actual_start_lonlat) {
                        markers.start.setLatLng([props.actual_start_lonlat[1], props.actual_start_lonlat[0]]);
                    }
                    if (props.actual_end_lonlat) {
                        markers.end.setLatLng([props.actual_end_lonlat[1], props.actual_end_lonlat[0]]);
                    }
                    ws.close();
                } 
                else if (data.type === 'error' || data.type === 'info') {
                    if (progressPolyline) { map.removeLayer(progressPolyline); progressPolyline = null; }
                    errorBox.innerText = data.message;
                    errorBox.style.display = 'block';
                    errorBox.style.color = data.type === 'info' ? '#eab308' : 'var(--error)';
                    resultsPanel.classList.remove('active');
                    ws.close();
                }
            };

            ws.onclose = () => {
                calcBtn.style.display = 'block';
                cancelBtn.style.display = 'none';
                calcBtn.innerText = 'Построить маршрут';
                currentStep = 'start'; 
            };
        });
    </script>
</body>
</html>
```

---

<a id='app/static/oceans-seasgeojson'></a>

* app/static/oceans-seas.geo.json

```json
⚠️ Файл пропущен: размер 1,098,654 байт превышает лимит 204,800 байт
```

---

<a id='requirementstxt'></a>

* requirements.txt

```

fastapi
uvicorn[standard]
pydantic
numpy
scipy
shapely
rasterio
rio-tiler
Pillow
```

---

