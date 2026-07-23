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
                    if comp_id > 0:
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
        
        # ИСПРАВЛЕНИЕ: Меняем dtype на np.float64, чтобы избежать конфликта точности с Python float!
        coarse_h = np.full((self.coarse_rows, self.coarse_cols), np.inf, dtype=np.float64)
        
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