import math
import heapq
import time
from collections import deque
import numpy as np
import rasterio
from typing import Tuple, List, Optional, Dict, Any
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
        properties = {
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
        }
        return {
            "type": "Feature",
            "properties": properties,
            "geometry": {"type": "LineString", "coordinates": clean_coords}
        }

class SphericalRasterRouter:
    def __init__(self, speed_tif_path: str, sd_tif_path: str):
        print(f"Загрузка растров в память: {speed_tif_path} ...")
        with rasterio.open(speed_tif_path) as src:
            self.transform = src.transform
            self.inv_transform = ~self.transform
            self.speed_raster = src.read(1).astype(np.float32) * KNOT_TO_KMH
            self.rows, self.cols = self.speed_raster.shape

        with rasterio.open(sd_tif_path) as src:
            self.sd_raster = src.read(1).astype(np.float32) * KNOT_TO_KMH

        self.min_lon = self.transform.c
        self.max_lat = self.transform.f
        self.d_lon = abs(self.transform.a)
        self.d_lat = abs(self.transform.e)
        self.dy_km = (math.radians(self.d_lat) * EARTH_RADIUS_KM)

        # Предрасчет для экстремальной скорости A*
        lats = np.array([self.max_lat - r * self.d_lat for r in range(self.rows)])
        self.cos_lats = np.cos(np.radians(lats)).astype(np.float32)
        # Массив dx_km для каждой строки (чтобы не считать косинусы внутри цикла)
        self.dx_per_row = (math.radians(self.d_lon) * EARTH_RADIUS_KM) * self.cos_lats

        print("Роутер готов к работе!")

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

    def _get_nearest_water_point(self, start_row: int, start_col: int, max_radius: int = 150) -> Optional[Tuple[int, int]]:
        """Ищет ближайший валидный пиксель методом BFS (Спираль)."""
        if self.speed_raster[start_row, start_col] > 0:
            return (start_row, start_col)

        queue = deque([(start_row, start_col)])
        visited = {(start_row, start_col)}
        
        while queue:
            cr, cc = queue.popleft()
            if abs(cr - start_row) > max_radius or abs(cc - start_col) > max_radius:
                continue

            for dr, dc in [(-1,0), (1,0), (0,-1), (0,1), (-1,-1), (-1,1), (1,-1), (1,1)]:
                nr, nc = cr + dr, cc + dc
                # Оборачивание по долготе
                nc = nc % self.cols 
                if 0 <= nr < self.rows:
                    if (nr, nc) not in visited:
                        if self.speed_raster[nr, nc] > 0:
                            return (nr, nc)
                        visited.add((nr, nc))
                        queue.append((nr, nc))
        return None

    def find_route(self, start_coords: Tuple[float, float], end_coords: Tuple[float, float], max_search_time: float = 30.0) -> Optional[RouteResult]:
        t_start = time.time()
        
        sr, sc = self._coord_to_index(*start_coords)
        gr, gc = self._coord_to_index(*end_coords)

        start_pt = self._get_nearest_water_point(sr, sc)
        goal_pt = self._get_nearest_water_point(gr, gc)
        
        if not start_pt or not goal_pt:
            raise ValueError("Не удалось найти точку на воде. Увеличьте радиус поиска или выберите другую точку.")

        sr, sc = start_pt
        gr, gc = goal_pt

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

        gl, glat = self._index_to_coord(gr, gc)
        g_phi = math.radians(glat)
        g_lam = math.radians(gl)
        cos_g_phi = math.cos(g_phi)

        max_speed = float(np.max(speed_raster))
        W = 1.5 
        
        nodes_explored = 0

        while pq:
            _, current_time, r, c = heapq.heappop(pq)
            idx = r * cols + c

            if idx == goal_idx:
                print(f"✅ Маршрут найден! Время поиска: {time.time() - t_start:.2f} сек. Исследовано узлов: {nodes_explored}")
                return self._reconstruct_and_simulate(came_from, gr, gc, sr, sc)

            if current_time > times_to_reach.get(idx, float('inf')):
                continue

            nodes_explored += 1
            
            # --- БЛОК ТАЙМАУТА ---
            # Проверяем время каждые 50 000 узлов, чтобы не тормозить цикл вызовами time.time()
            if nodes_explored % 50000 == 0:
                elapsed = time.time() - t_start
                print(f"⏳ Поиск... исследовано {nodes_explored} узлов ({elapsed:.1f} сек.)")
                if elapsed > max_search_time:
                    raise TimeoutError(f"Превышено время поиска ({max_search_time} сек). Вероятно, точки маршрута изолированы сушей друг от друга.")
            # ---------------------

            dx = dx_per_row[r]
            diag = math.hypot(dx, dy_km)

            for dr, dc, dist in [
                (-1, 0, dy_km), (1, 0, dy_km), 
                (0, -1, dx), (0, 1, dx),
                (-1, -1, diag), (-1, 1, diag), (1, -1, diag), (1, 1, diag)
            ]:
                nr = r + dr
                nc = (c + dc) % cols 

                if not (0 <= nr < rows):
                    continue
                
                speed = speed_raster[nr, nc]
                if speed <= 0:
                    continue
                
                new_time = current_time + (dist / speed)
                n_idx = nr * cols + nc
                
                if new_time < times_to_reach.get(n_idx, float('inf')):
                    times_to_reach[n_idx] = new_time
                    came_from[n_idx] = idx
                    
                    n_phi = math.radians(self.max_lat - nr * self.d_lat)
                    n_lam = math.radians(self.min_lon + nc * self.d_lon)
                    
                    dphi = g_phi - n_phi
                    dlam = g_lam - n_lam
                    a = math.sin(dphi*0.5)**2 + math.cos(n_phi)*cos_g_phi*math.sin(dlam*0.5)**2
                    if a < 0: a = 0.0
                    if a > 1: a = 1.0
                    h_dist = 2.0 * EARTH_RADIUS_KM * math.atan2(math.sqrt(a), math.sqrt(1 - a))
                    
                    f_score = new_time + W * (h_dist / max_speed)
                    heapq.heappush(pq, (f_score, new_time, nr, nc))

        return None

    def _reconstruct_and_simulate(self, came_from: dict, gr: int, gc: int, sr: int, sc: int) -> RouteResult:
        # Восстанавливаем 1D индексы в координаты
        path_coords = []
        curr_idx = gr * self.cols + gc
        while curr_idx != -1:
            r = curr_idx // self.cols
            c = curr_idx % self.cols
            path_coords.append(self._index_to_coord(r, c))
            curr_idx = came_from[curr_idx]
        path_coords.reverse()
        
        # Сглаживание алгоритмом Чайкина
        smoothed = path_coords
        for _ in range(2):
            if len(smoothed) <= 2: break
            new_path = [smoothed[0]]
            for i in range(len(smoothed) - 1):
                p0, p1 = smoothed[i], smoothed[i+1]
                # Коррекция перескока через 180 градус при сглаживании
                dx = p1[0] - p0[0]
                if dx > 180: p1 = (p1[0] - 360, p1[1])
                elif dx < -180: p1 = (p1[0] + 360, p1[1])
                
                new_path.extend([(0.75 * p0[0] + 0.25 * p1[0], 0.75 * p0[1] + 0.25 * p1[1]),
                                 (0.25 * p0[0] + 0.75 * p1[0], 0.25 * p0[1] + 0.75 * p1[1])])
            new_path.append(smoothed[-1])
            smoothed = [(lon if lon <= 180 else lon - 360, lat) for lon, lat in new_path]

        N_SEGMENTS = len(smoothed) - 1
        dists_km = np.zeros(N_SEGMENTS)
        means = np.zeros(N_SEGMENTS)
        sds = np.zeros(N_SEGMENTS)

        for i in range(N_SEGMENTS):
            p1, p2 = smoothed[i], smoothed[i+1]
            dists_km[i] = self._haversine_distance(*p1, *p2)
            
            mid_lon = (p1[0] + p2[0]) / 2.0
            if abs(p1[0] - p2[0]) > 180: mid_lon = (mid_lon + 180) % 360 - 180
                
            mid_lat = (p1[1] + p2[1]) / 2.0
            mr, mc = self._coord_to_index(mid_lon, mid_lat)
            
            means[i] = max(self.speed_raster[mr, mc], 1.0)
            sds[i] = max(self.sd_raster[mr, mc], 0.1)

        N_SIM = 10000
        shape = (means / sds) ** 2
        scale = (sds ** 2) / means
        sim_speeds = np.random.gamma(shape[:, None], scale[:, None], size=(N_SEGMENTS, N_SIM))
        sim_speeds = np.maximum(sim_speeds, 1.0)
        sim_times = dists_km[:, None] / sim_speeds
        total_times = np.sum(sim_times, axis=0)
        segment_times = dists_km / means

        return RouteResult(
            path_coords=smoothed,
            total_distance_km=float(np.sum(dists_km)),
            time_mean_hours=float(np.mean(total_times)),
            time_q05_hours=float(np.percentile(total_times, 5)),
            time_q95_hours=float(np.percentile(total_times, 95)),
            segment_distances_km=dists_km.tolist(),
            segment_speeds_kmh=means.tolist(),
            segment_times_hours=segment_times.tolist(),
            actual_start_coords=self._index_to_coord(sr, sc),
            actual_end_coords=self._index_to_coord(gr, gc)
        )