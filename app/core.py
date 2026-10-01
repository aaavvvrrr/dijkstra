import os
import math
import heapq
import time
import json
import asyncio
from collections import deque
import numpy as np
import rasterio
from scipy.ndimage import label
from typing import Tuple, List, Optional, Dict, Any, Callable
from dataclasses import dataclass
from shapely.geometry import LineString

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
    seca_distance_km: float = 0.0
    segment_depths_m: List[float] = None
    segment_waves_m: List[float] = None

    def to_geojson(self, request_params: dict = None) -> Dict[str, Any]:
        clean_coords = [[float(lon), float(lat)] for lon, lat in self.path_coords]
        
        props = {
            "distance_total_km": round(float(self.total_distance_km), 2),
            "time_total_hours": round(float(self.time_mean_hours), 2),
            "time_optimistic_hours": round(float(self.time_q05_hours), 2),
            "time_pessimistic_hours": round(float(self.time_q95_hours), 2),
            "avg_speed_kmh": round(float(self.total_distance_km / self.time_mean_hours), 2) if self.time_mean_hours > 0 else 0,
            "segments": {
                "distances_km": [round(float(d), 3) for d in self.segment_distances_km],
                "speeds_kmh": [round(float(s), 2) for s in self.segment_speeds_kmh],
                "times_hours": [round(float(t), 3) for t in self.segment_times_hours],
                "depths_m": [round(float(d), 1) for d in self.segment_depths_m] if self.segment_depths_m else [],
                "waves_m": [round(float(w), 2) for w in self.segment_waves_m] if self.segment_waves_m else [],
            },
            "actual_start_lonlat": [round(self.actual_start_coords[0], 6), round(self.actual_start_coords[1], 6)],
            "actual_end_lonlat": [round(self.actual_end_coords[0], 6), round(self.actual_end_coords[1], 6)],
            "request_parameters": request_params or {}
        }
        
        if request_params and request_params.get("calc_seca", False):
            props["seca_distance_km"] = round(self.seca_distance_km, 2)
            
        return {
            "type": "Feature",
            "properties": props,
            "geometry": {"type": "LineString", "coordinates": clean_coords}
        }

class SphericalRasterRouter:
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.rasters_cache = {}
        
        base_speed = os.path.join(data_dir, "eff_vel_all.tif")
        seca_tif_path = os.path.join(data_dir, "seca_mask.tif")
        depth_tif_path = os.path.join(data_dir, "depth_meters.tif")
        cp_tif_path = os.path.join(data_dir, "chokepoints.tif")
        cp_config_path = os.path.join(data_dir, "chokepoints_config.json")

        print(f"Загрузка базового растра для инициализации сетки: {base_speed} ...")
        
        self.seca_raster = None
        if os.path.exists(seca_tif_path):
            print(f"Загрузка маски SECA: {seca_tif_path} ...")
            with rasterio.open(seca_tif_path) as src:
                self.seca_raster = src.read(1)

        self.depth_raster = None
        if os.path.exists(depth_tif_path):
            print(f"Загрузка батиметрии (GEBCO): {depth_tif_path} ...")
            with rasterio.open(depth_tif_path) as src:
                self.depth_raster = src.read(1).astype(np.int16)

        self.cp_raster = None
        self.cp_config = {}
        self.cp_coarse_map = {}
        if os.path.exists(cp_tif_path):
            print(f"Загрузка растра узких мест: {cp_tif_path} ...")
            with rasterio.open(cp_tif_path) as src:
                self.cp_raster = src.read(1).astype(np.uint16)
        if os.path.exists(cp_config_path):
            with open(cp_config_path, "r", encoding="utf-8") as f:
                self.cp_config = json.load(f)

        # Подгружаем растры "all" в кэш по умолчанию
        self._load_vessel_rasters("all")
        self.speed_raster = self.rasters_cache["all"]["speed"]
        self.sd_raster = self.rasters_cache["all"]["sd"]

        with rasterio.open(base_speed) as src:
            self.transform = src.transform
            self.inv_transform = ~self.transform
            self.rows, self.cols = src.height, src.width

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
        structure = np.ones((3, 3), dtype=int)
        self.components, self.num_features = label(self.coarse_water, structure=structure)
        
        counts = np.bincount(self.components.ravel())
        self.main_ocean_id = np.argmax(counts[1:]) + 1 if len(counts) > 1 else 0
        print(f"Найдено {self.num_features} изолированных водоемов. Главный океан: ID {self.main_ocean_id}")

        if self.cp_raster is not None:
            unique_cps = np.unique(self.cp_raster)
            for cp_id in unique_cps:
                if cp_id == 0: continue
                rr, cc = np.where(self.cp_raster == cp_id)
                self.cp_coarse_map[cp_id] = set(zip(rr // self.scale, cc // self.scale))

        self.coarse_dy = self.dy_km * self.scale
        c_lats = np.array([self.max_lat - r * self.d_lat * self.scale for r in range(self.coarse_rows)])
        self.coarse_dx_per_row = (math.radians(self.d_lon * self.scale) * EARTH_RADIUS_KM) * np.cos(np.radians(c_lats))

        debug_tif_path = os.path.join(self.data_dir, "debug_connectivity.tif")
        coarse_transform = self.transform * rasterio.Affine.scale(self.scale)
        
        with rasterio.open(
            debug_tif_path, 'w',
            driver='GTiff',
            height=self.coarse_rows,
            width=self.coarse_cols,
            count=1,
            dtype=self.components.dtype,
            crs="EPSG:4326",
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

    def _load_vessel_rasters(self, vessel_type: str):
        if vessel_type in self.rasters_cache: return
            
        speed_path = os.path.join(self.data_dir, f"eff_vel_{vessel_type}.tif")
        sd_path = os.path.join(self.data_dir, f"eff_sd_{vessel_type}.tif")
        
        if not os.path.exists(speed_path):
            print(f"⚠️ Растры для '{vessel_type}' не найдены. Откат на 'all'.")
            speed_path = os.path.join(self.data_dir, "eff_vel_all.tif")
            sd_path = os.path.join(self.data_dir, "eff_sd_all.tif")
            
        print(f"Кэширование в RAM растров типа '{vessel_type}' ...")
        with rasterio.open(speed_path) as src:
            speed_arr = src.read(1).astype(np.float32) * KNOT_TO_KMH
            speed_arr = np.nan_to_num(speed_arr, nan=0.0)
        with rasterio.open(sd_path) as src:
            sd_arr = src.read(1).astype(np.float32) * KNOT_TO_KMH
            sd_arr = np.nan_to_num(sd_arr, nan=0.1)
            
        self.rasters_cache[vessel_type] = {"speed": speed_arr, "sd": sd_arr}

    def _get_nearest_water_point(self, start_row: int, start_col: int, max_radius: int = 300, required_draft: Optional[float] = None, speed_raster: Optional[np.ndarray] = None) -> Optional[Tuple[int, int, int]]:
        queue = deque([(start_row, start_col)])
        visited = {(start_row, start_col)}
        
        # Запас 2 метра "под килем" (Under Keel Clearance)
        min_depth = (required_draft + 2.0) if required_draft else None
        
        current_speed_raster = speed_raster if speed_raster is not None else self.speed_raster

        while queue:
            cr, cc = queue.popleft()
            if abs(cr - start_row) > max_radius or abs(cc - start_col) > max_radius: continue
            
            if current_speed_raster[cr, cc] > 0.5:
                is_deep_enough = True
                
                if min_depth and self.depth_raster is not None:
                    # В GEBCO океан - это отрицательные высоты (напр. -15 = 15м глубины).
                    # Инвертируем знак, чтобы суша (высота > 0) уходила в минус и безопасно отсекалась.
                    water_depth = -self.depth_raster[cr, cc]
                    if water_depth < min_depth:
                        is_deep_enough = False
                        
                if is_deep_enough:
                    cr_c, cc_c = cr // self.scale, cc // self.scale
                    if 0 <= cr_c < self.coarse_rows and 0 <= cc_c < self.coarse_cols:
                        comp_id = self.components[cr_c, cc_c]
                        if comp_id > 0:
                            return (cr, cc, comp_id)

            for dr, dc in [(-1,0), (1,0), (0,-1), (0,1), (-1,-1), (-1,1), (1,-1), (1,1)]:
                nr, nc = cr + dr, cc + dc
                if 0 <= nr < self.rows and 0 <= nc < self.cols:
                    if (nr, nc) not in visited:
                        visited.add((nr, nc))
                        queue.append((nr, nc))
        return None

    def _build_coarse_heuristic(self, goal_r: int, goal_c: int, comp_id: int, forbidden_cp: set) -> np.ndarray:
        c_gr, c_gc = goal_r // self.scale, goal_c // self.scale
        
        # ИСПРАВЛЕНИЕ: Меняем dtype на np.float64, чтобы избежать конфликта точности с Python float!
        coarse_h = np.full((self.coarse_rows, self.coarse_cols), np.inf, dtype=np.float64)
        
        coarse_h[c_gr, c_gc] = 0.0
        pq = [(0.0, c_gr, c_gc)]
        
        c_dx = self.coarse_dx_per_row
        c_dy = self.coarse_dy
        c_rows, c_cols = self.coarse_rows, self.coarse_cols
        max_s = self.max_speed
        
        blocked_coarse = set()
        if forbidden_cp:
            for cp_id in forbidden_cp:
                if cp_id in self.cp_coarse_map:
                    blocked_coarse.update(self.cp_coarse_map[cp_id])
        
        while pq:
            t, r, c = heapq.heappop(pq)
            if t > coarse_h[r, c]: continue
            dx = c_dx[r]
            diag = math.sqrt(dx*dx + c_dy*c_dy)
            
            for dr, dc, dist in [(-1, 0, c_dy), (1, 0, c_dy), (0, -1, dx), (0, 1, dx), (-1, -1, diag), (-1, 1, diag), (1, -1, diag), (1, 1, diag)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < c_rows and 0 <= nc < c_cols:
                    if (nr, nc) in blocked_coarse: continue
                    if self.components[nr, nc] == comp_id:
                        nt = t + (dist / max_s)
                        if nt < coarse_h[nr, nc]:
                            coarse_h[nr, nc] = nt
                            heapq.heappush(pq, (nt, nr, nc))
        return coarse_h

    async def find_route(
        self, start_coords: Tuple[float, float], end_coords: Tuple[float, float],
        progress_callback: Optional[Callable] = None, check_cancel_callback: Optional[Callable] = None,
        request_params: Optional[dict] = None
    ) -> Optional[RouteResult]:
        
        t_start = time.time()
        sr, sc = self._coord_to_index(*start_coords)
        gr, gc = self._coord_to_index(*end_coords)

        request_params = request_params or {}
        vessel_draft = request_params.get("draft")
        vessel_type = request_params.get("vessel_type", "all")

        self._load_vessel_rasters(vessel_type)
        req_speed_raster = self.rasters_cache[vessel_type]["speed"]
        req_sd_raster = self.rasters_cache[vessel_type]["sd"]

        start_pt_info = self._get_nearest_water_point(sr, sc, required_draft=vessel_draft, speed_raster=req_speed_raster)
        goal_pt_info = self._get_nearest_water_point(gr, gc, required_draft=vessel_draft, speed_raster=req_speed_raster)
        
        if not start_pt_info: raise ValueError("Точка старта слишком далеко от воды (или недостаточная глубина).")
        if not goal_pt_info: raise ValueError("Точка финиша слишком далеко от воды (или недостаточная глубина).")

        sr, sc, start_comp = start_pt_info
        gr, gc, goal_comp = goal_pt_info

        if start_comp != goal_comp:
            raise ValueError(
                f"Путь невозможен: Старт и Финиш находятся в изолированных друг от друга водоемах "
                f"(ID {start_comp} и ID {goal_comp}). Включите слой 'Отладка: Связность морей' на карте, чтобы увидеть разрыв."
            )

        forbidden_cp = set(request_params.get("forbidden_chokepoints", []))
        v_len = request_params.get("vessel_length")
        v_wid = request_params.get("vessel_width")
        
        if self.cp_config:
            for cp_id_str, cp_data in self.cp_config.items():
                cp_id = int(cp_id_str)
                if v_len and cp_data.get("max_length") and v_len > cp_data["max_length"]:
                    forbidden_cp.add(cp_id)
                elif v_wid and cp_data.get("max_width") and v_wid > cp_data["max_width"]:
                    forbidden_cp.add(cp_id)
                elif vessel_draft and cp_data.get("max_draft") and vessel_draft > cp_data["max_draft"]:
                    forbidden_cp.add(cp_id)

        coarse_h = self._build_coarse_heuristic(gr, gc, goal_comp, forbidden_cp)
        if coarse_h[sr // self.scale, sc // self.scale] == np.inf:
            raise ValueError("Ошибка иерархической сетки: внутренний разрыв. Возможно, из-за габаритов судна перекрыты все доступные пути.")

        cols = self.cols
        start_idx = sr * cols + sc
        goal_idx = gr * cols + gc

        pq = [(0.0, 0.0, sr, sc)]
        costs_to_reach = {start_idx: 0.0}
        came_from = {start_idx: -1}
        
        avoid_seca = request_params.get("avoid_seca", False)
        calc_seca = request_params.get("calc_seca", False)
        min_depth = (vessel_draft + 2.0) if vessel_draft else None

        custom_speed_knots = request_params.get("average_speed")
        custom_speed_kmh = custom_speed_knots * KNOT_TO_KMH if custom_speed_knots else None
        
        # Метео-маршрутизация: Подготовка данных о волнении
        max_wave_height = request_params.get("max_wave_height")
        wave_path = request_params.get("wave_path")
        wave_arr, wave_inv, wave_h, wave_w = None, None, 0, 0
        if max_wave_height is not None and wave_path and os.path.exists(wave_path):
            with rasterio.open(wave_path) as ds:
                wave_arr = ds.read(1)
                wave_inv = ~ds.transform
                wave_h, wave_w = ds.height, ds.width
        
        max_timeout = request_params.get("timeout_seconds")
        if not max_timeout:
            max_timeout = 60

        speed_raster = req_speed_raster
        dx_per_row = self.dx_per_row
        dy_km = self.dy_km
        rows = self.rows
        
        TARGET_RADIUS_KM = 3.0 
        nodes_explored = 0

        while pq:
            _, current_cost, r, c = heapq.heappop(pq)
            idx = r * cols + c

            if idx == goal_idx:
                print(f"✅ Точный финиш найден! Время: {time.time() - t_start:.2f} сек.")
                return self._reconstruct_and_simulate(came_from, gr, gc, sr, sc, calc_seca, custom_speed_kmh, req_speed_raster, req_sd_raster, request_params)

            if current_cost > costs_to_reach.get(idx, float('inf')): continue

            nodes_explored += 1
            
            if nodes_explored % 30000 == 0:
                if time.time() - t_start > max_timeout:
                    raise TimeoutError(f"Превышено максимальное время расчета ({max_timeout} сек). Сложная конфигурация узких мест.")
                if check_cancel_callback and check_cancel_callback(): raise InterruptedError("Поиск прерван.")
                if progress_callback:
                    fast_path = []
                    curr = idx
                    while curr != -1:
                        fast_path.append(self._index_to_coord(curr // cols, curr % cols))
                        curr = came_from.get(curr, -1)
                    await progress_callback(nodes_explored, fast_path[::5], current_cost)
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
                
                # Переопределяем скорость, если задана явно
                if custom_speed_kmh: 
                    speed = custom_speed_kmh
                
                # O(1) Проверка батиметрии
                if min_depth and self.depth_raster is not None:
                    water_depth = -self.depth_raster[nr, nc]
                    if water_depth < min_depth: continue

                # O(1) Проверка узких мест
                if self.cp_raster is not None and len(forbidden_cp) > 0:
                    cp_val = self.cp_raster[nr, nc]
                    if cp_val > 0 and cp_val in forbidden_cp:
                        continue
                        
                # O(1) Проверка погоды (Огибание штормов)
                if wave_arr is not None:
                    # Быстрый пересчет координат пикселя сетки A* в пиксель матрицы GRIB2
                    lon, lat = self.transform * (nc + 0.5, nr + 0.5)
                    wc, wr = wave_inv * (lon, lat)
                    wr, wc = int(wr), int(wc)
                    if 0 <= wr < wave_h and 0 <= wc < wave_w:
                        w_val = wave_arr[wr, wc]
                        # Если значение валидное (меньше 100м) и превышает лимит судна - обходим стороной
                        if w_val < 100.0 and w_val > max_wave_height:
                            continue
                
                cr_c, cc_c = nr // self.scale, nc // self.scale
                if cr_c >= self.coarse_rows or cc_c >= self.coarse_cols: continue
                
                h_time = coarse_h[cr_c, cc_c]
                if h_time == np.inf: continue 
                
                time_step = dist / speed
                penalty = 5.0 if (avoid_seca and self.seca_raster is not None and self.seca_raster[nr, nc] > 0) else 1.0
                new_cost = current_cost + (time_step * penalty)
                
                n_idx = nr * cols + nc
                
                if new_cost < costs_to_reach.get(n_idx, float('inf')):
                    costs_to_reach[n_idx] = new_cost
                    came_from[n_idx] = idx

                    if abs(nr - gr) <= 15 and abs(nc - gc) <= 15:
                        dist_to_goal = math.hypot((nr - gr) * dx_per_row[nr], (nc - gc) * dy_km)
                        if dist_to_goal <= TARGET_RADIUS_KM:
                            print(f"⚓ Захват рейда ({TARGET_RADIUS_KM} км до цели)! Время: {time.time() - t_start:.2f} сек.")
                            return self._reconstruct_and_simulate(came_from, nr, nc, sr, sc, calc_seca, custom_speed_kmh, req_speed_raster, req_sd_raster, request_params)

                    # h_time - это эвристика времени. Умножение на 1.2 делает Weighted A*
                    f_score = new_cost + 1.2 * h_time
                    heapq.heappush(pq, (f_score, new_cost, nr, nc))

        return None

    def _reconstruct_and_simulate(self, came_from: dict, gr: int, gc: int, sr: int, sc: int, calc_seca: bool = False, custom_speed_kmh: Optional[float] = None, speed_raster: Optional[np.ndarray] = None, sd_raster: Optional[np.ndarray] = None, request_params: Optional[dict] = None) -> RouteResult:
        request_params = request_params or {}
        path_coords = []
        curr_idx = gr * self.cols + gc
        while curr_idx != -1:
            path_coords.append(self._index_to_coord(curr_idx // self.cols, curr_idx % self.cols))
            curr_idx = came_from.get(curr_idx, -1)
        path_coords.reverse()
        
        current_speed_raster = speed_raster if speed_raster is not None else self.speed_raster
        current_sd_raster = sd_raster if sd_raster is not None else self.sd_raster
        
        # 1. Точный расчет времени, дистанции и SECA по ПЛОТНОЙ сетке
        N_DENSE = len(path_coords) - 1
        dense_dists = np.zeros(N_DENSE) if N_DENSE > 0 else np.zeros(0)
        dense_means = np.zeros(N_DENSE) if N_DENSE > 0 else np.zeros(0)
        dense_sds = np.zeros(N_DENSE) if N_DENSE > 0 else np.zeros(0)
        seca_distance_total = 0.0

        for i in range(N_DENSE):
            p1, p2 = path_coords[i], path_coords[i+1]
            d_km = self._haversine_distance(*p1, *p2)
            dense_dists[i] = d_km
            mr, mc = self._coord_to_index((p1[0] + p2[0]) / 2.0, (p1[1] + p2[1]) / 2.0)
            
            if custom_speed_kmh:
                dense_means[i] = custom_speed_kmh
                dense_sds[i] = max(custom_speed_kmh * 0.1, 0.1) # Имитируем небольшую дисперсию (10%)
            else:
                dense_means[i] = max(current_speed_raster[mr, mc], 1.0)
                dense_sds[i] = max(current_sd_raster[mr, mc], 0.1)
            
            if calc_seca and self.seca_raster is not None and self.seca_raster[mr, mc] > 0:
                seca_distance_total += d_km

        time_mean, time_q05, time_q95 = 0.0, 0.0, 0.0
        if N_DENSE > 0:
            shape = (dense_means / dense_sds) ** 2
            scale = (dense_sds ** 2) / dense_means
            sim_speeds = np.maximum(np.random.gamma(shape[:, None], scale[:, None], size=(N_DENSE, 10000)), 1.0)
            sim_times = dense_dists[:, None] / sim_speeds
            total_times = np.sum(sim_times, axis=0)
            time_mean = float(np.mean(total_times))
            time_q05 = float(np.percentile(total_times, 5))
            time_q95 = float(np.percentile(total_times, 95))
            
        total_dist = float(np.sum(dense_dists))

        # 2. Упрощение сырой геометрии (алгоритм Дугласа-Пекера, ~2-3 км толерантность)
        if len(path_coords) > 2:
            path_coords = list(LineString(path_coords).simplify(0.02, preserve_topology=False).coords)

        # 3. Алгоритм Чайкина для скругления оставшихся поворотов
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
            
        # 4. Финальное удаление избыточных точек после сглаживания
        if len(smoothed) > 2:
            smoothed = list(LineString(smoothed).simplify(0.005, preserve_topology=False).coords)

        # 5. Грубые сегменты для контракта API (чтобы не отдавать пустые массивы)
        N_SEGMENTS = len(smoothed) - 1
        dists_km = np.zeros(N_SEGMENTS) if N_SEGMENTS > 0 else np.zeros(0)
        means = np.zeros(N_SEGMENTS) if N_SEGMENTS > 0 else np.zeros(0)
        depths = np.zeros(N_SEGMENTS) if N_SEGMENTS > 0 else np.zeros(0)
        waves = np.zeros(N_SEGMENTS) if N_SEGMENTS > 0 else np.zeros(0)

        wave_path = request_params.get("wave_path")
        wave_dataset = None
        if wave_path and os.path.exists(wave_path):
            try: wave_dataset = rasterio.open(wave_path)
            except: pass

        for i in range(N_SEGMENTS):
            p1, p2 = smoothed[i], smoothed[i+1]
            d_km = self._haversine_distance(*p1, *p2)
            dists_km[i] = d_km
            
            mlon, mlat = (p1[0] + p2[0]) / 2.0, (p1[1] + p2[1]) / 2.0
            mr, mc = self._coord_to_index(mlon, mlat)
            means[i] = custom_speed_kmh if custom_speed_kmh else max(current_speed_raster[mr, mc], 1.0)
            
            # Телеметрия: Глубина
            if self.depth_raster is not None:
                # В GEBCO океан отрицательный (-15 = 15м)
                depths[i] = max(0.0, float(-self.depth_raster[mr, mc]))
                
            # Телеметрия: Волнение (сэмплинг на лету из GRIB2/TIF)
            if wave_dataset is not None:
                try:
                    val = list(wave_dataset.sample([(mlon, mlat)]))[0][0]
                    # Игнорируем nodata (обычно для волн > 100м это мусор)
                    waves[i] = float(val) if val < 100 else 0.0
                except:
                    waves[i] = 0.0

        if wave_dataset is not None:
            wave_dataset.close()

        return RouteResult(
            path_coords=smoothed,
            total_distance_km=total_dist,
            time_mean_hours=time_mean,
            time_q05_hours=time_q05,
            time_q95_hours=time_q95,
            segment_distances_km=dists_km.tolist(),
            segment_speeds_kmh=means.tolist(),
            segment_times_hours=(dists_km / means).tolist() if N_SEGMENTS > 0 else [],
            segment_depths_m=depths.tolist(),
            segment_waves_m=waves.tolist(),
            actual_start_coords=self._index_to_coord(sr, sc),
            actual_end_coords=self._index_to_coord(gr, gc),
            seca_distance_km=seca_distance_total
        )